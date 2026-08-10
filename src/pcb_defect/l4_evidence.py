"""Verify a private L4 package and derive path-free public metadata evidence."""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcb_defect.benchmark import (
    ARTIFACT_FIELDS,
    BENCHMARK_BACKENDS,
    CANONICAL_CYCLES,
    CANONICAL_WARMUP,
    PROTOCOL_FIELDS,
    REPORT_FIELDS,
    TIMING_FIELDS,
    _timings_match,
)
from pcb_defect.prediction_parity import prediction_parity_is_complete
from pcb_defect.result_package import PackageError, verify_verifiable_zip

_REPORT_PATH_RE = re.compile(r"benchmark_l4/[0-9a-f]{12}/benchmark_l4\.json")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+")
_HARDWARE_FIELDS = frozenset({"gpu", "driver", "torch_cuda", "cudnn", "tensorrt"})
_RUNTIME_FIELDS = frozenset({"onnxruntime_providers"})
_EXPECTED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_EXPECTED_SCOPE = "predecoded PIL image; preprocess + inference + postprocess; CUDA synchronized"


class EvidencePromotionError(RuntimeError):
    """Raised when private evidence cannot support a safe public metadata derivation."""


@dataclass(frozen=True, slots=True)
class PromotedL4Evidence:
    summary: dict[str, Any]
    raw_timings: dict[str, Any]
    prediction_parity: dict[str, Any]


def promote_l4_package(package: Path) -> PromotedL4Evidence:
    """Return verified summary, raw timing, and pseudonymized per-box parity documents."""
    package = package.resolve()
    try:
        manifest = verify_verifiable_zip(package)
        report_rows = [row for row in manifest["files"] if _REPORT_PATH_RE.fullmatch(row["path"])]
        if len(report_rows) != 1:
            raise EvidencePromotionError(
                "private L4 package must contain exactly one benchmark report"
            )
        report_row = report_rows[0]
        with zipfile.ZipFile(package) as archive:
            report = json.loads(archive.read(report_row["path"]).decode("utf-8"))
    except EvidencePromotionError:
        raise
    except (PackageError, OSError, UnicodeError, KeyError, json.JSONDecodeError) as exc:
        raise EvidencePromotionError("private L4 package verification failed") from exc
    _validate_raw_report(report)

    package_receipt = {
        "filename": package.name,
        "bytes": package.stat().st_size,
        "sha256": manifest["package_sha256"],
    }
    source = {
        "package": package_receipt,
        "raw_report_sha256": report_row["sha256"],
    }
    timing_summaries = {
        backend: {key: value for key, value in timing.items() if key != "raw_ms"}
        for backend, timing in report["timings"].items()
    }
    parity_summary = {
        **{
            key: value for key, value in report["prediction_parity"].items() if key != "comparisons"
        },
        "comparisons": {
            backend: {key: value for key, value in comparison.items() if key != "per_image"}
            for backend, comparison in report["prediction_parity"]["comparisons"].items()
        },
    }
    provenance = {
        "runner_git_sha": report["runner_git_sha"],
        "experiment_git_sha": report["experiment_git_sha"],
        "deployment_gate_sha256": report["deployment_gate_sha256"],
        "dataset_sha256": report["dataset_sha256"],
        "manifest_sha256": report["manifest_sha256"],
    }
    hardware = _copy_fields(report["hardware"], _HARDWARE_FIELDS)
    runtime = _copy_fields(report["runtime"], _RUNTIME_FIELDS)
    protocol = _copy_fields(report["protocol"], PROTOCOL_FIELDS)
    artifacts = _copy_fields(report["artifacts"], ARTIFACT_FIELDS)
    fidelity = _copy_fields(report["fidelity"], set(report["fidelity"]))
    raw_backend_timings = {
        backend: _copy_fields(report["timings"][backend], TIMING_FIELDS)
        for backend in sorted(BENCHMARK_BACKENDS)
    }
    summary = {
        "schema_version": "2.0",
        "status": "complete",
        "evidence_visibility": "public_metadata_from_private_unreleased_package",
        **source,
        "provenance": provenance,
        "hardware": hardware,
        "runtime": runtime,
        "protocol": protocol,
        "artifacts": artifacts,
        "fidelity": fidelity,
        "prediction_parity": parity_summary,
        "timings": timing_summaries,
        "limitations": [
            (
                "One L4 session does not estimate between-session, machine, driver, "
                "or thermal variance."
            ),
            (
                "The benchmark uses calibration images only and is not a final-test "
                "or production-SLA measurement."
            ),
            "TensorRT engine bytes remain private, hardware-stack-bound, and non-portable.",
        ],
    }
    raw_timings = {
        "schema_version": "1.0",
        "status": "complete",
        **source,
        "provenance": provenance,
        "hardware": hardware,
        "runtime": runtime,
        "protocol": protocol,
        "timings": raw_backend_timings,
        "statistical_scope": {
            "sessions": report["protocol"]["sessions"],
            "descriptive_only": True,
            "between_session_uncertainty_estimated": False,
        },
    }
    prediction_parity = _sanitize_prediction_parity(report["prediction_parity"], source, provenance)
    return PromotedL4Evidence(summary, raw_timings, prediction_parity)


def _validate_raw_report(report: object) -> None:
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
        raise EvidencePromotionError("raw L4 benchmark report schema is incomplete")
    protocol = report["protocol"]
    artifacts = report["artifacts"]
    fidelity = report["fidelity"]
    hardware = report["hardware"]
    runtime = report["runtime"]
    if (
        report["schema_version"] != "3.0"
        or report["status"] != "complete"
        or not _git_sha(report["runner_git_sha"])
        or not _git_sha(report["experiment_git_sha"])
        or any(
            not _sha256(report[field])
            for field in ("deployment_gate_sha256", "dataset_sha256", "manifest_sha256")
        )
        or not isinstance(protocol, dict)
        or set(protocol) != PROTOCOL_FIELDS
        or protocol.get("split") != "calibration"
        or protocol.get("warmup") != CANONICAL_WARMUP
        or protocol.get("cycles") != CANONICAL_CYCLES
        or protocol.get("batch") != 1
        or protocol.get("timing_schedule") != "interleaved-rotating-backend-order"
        or protocol.get("sessions") != 1
        or not isinstance(protocol.get("images"), int)
        or isinstance(protocol.get("images"), bool)
        or protocol["images"] < 1
        or not _sha256(protocol.get("image_list_sha256"))
        or not _sha256(protocol.get("image_content_sha256"))
        or protocol.get("confidence") != 0.25
        or protocol.get("scope") != _EXPECTED_SCOPE
        or not isinstance(artifacts, dict)
        or set(artifacts) != ARTIFACT_FIELDS
        or artifacts.get("engine_committable") is not False
        or any(
            not _sha256(artifacts.get(field))
            for field in (
                "source_checkpoint_sha256",
                "onnx_sha256",
                "tensorrt_engine_sha256",
            )
        )
        or not isinstance(fidelity, dict)
        or fidelity.get("split") != "calibration"
        or fidelity.get("passed") is not True
        or not _fidelity_is_consistent(fidelity)
        or not prediction_parity_is_complete(report["prediction_parity"])
        or report["prediction_parity"].get("required_images") != protocol["images"]
        or not _timings_match(report["timings"], protocol["images"] * CANONICAL_CYCLES)
        or set(report["timings"]) != BENCHMARK_BACKENDS
        or not isinstance(hardware, dict)
        or set(hardware) != _HARDWARE_FIELDS
        or hardware.get("gpu") != "NVIDIA L4"
        or not _version(hardware.get("driver"))
        or not _version(hardware.get("torch_cuda"))
        or not _version(hardware.get("tensorrt"))
        or not isinstance(hardware.get("cudnn"), int)
        or isinstance(hardware.get("cudnn"), bool)
        or hardware["cudnn"] <= 0
        or not isinstance(runtime, dict)
        or set(runtime) != _RUNTIME_FIELDS
        or runtime.get("onnxruntime_providers") != _EXPECTED_PROVIDERS
    ):
        raise EvidencePromotionError("raw L4 benchmark evidence failed consistency checks")


def _fidelity_is_consistent(fidelity: dict[str, Any]) -> bool:
    fields = {
        "split",
        "source_map50_95",
        "onnx_map50_95",
        "tensorrt_fp16_map50_95",
        "tensorrt_minus_source",
        "absolute_delta_threshold",
        "passed",
    }
    if set(fidelity) != fields:
        return False
    values = [
        fidelity["source_map50_95"],
        fidelity["onnx_map50_95"],
        fidelity["tensorrt_fp16_map50_95"],
        fidelity["tensorrt_minus_source"],
        fidelity["absolute_delta_threshold"],
    ]
    if any(not _finite(value) for value in values):
        return False
    expected_delta = fidelity["tensorrt_fp16_map50_95"] - fidelity["source_map50_95"]
    return (
        math.isclose(fidelity["tensorrt_minus_source"], expected_delta, rel_tol=0.0, abs_tol=1e-12)
        and abs(expected_delta) <= fidelity["absolute_delta_threshold"]
    )


def _sanitize_prediction_parity(
    report: dict[str, Any], source: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    first_backend = report["candidate_backends"][0]
    stems = sorted(report["comparisons"][first_backend]["per_image"])
    image_ids = {stem: f"image_{index:03d}" for index, stem in enumerate(stems, start=1)}
    comparisons: dict[str, Any] = {}
    for backend in report["candidate_backends"]:
        raw_comparison = report["comparisons"][backend]
        per_image = {
            image_ids[stem]: _sanitize_image_parity(raw_comparison["per_image"][stem])
            for stem in stems
        }
        comparisons[backend] = {
            **{key: value for key, value in raw_comparison.items() if key != "per_image"},
            "per_image": per_image,
        }
    return {
        "schema_version": "1.0",
        "status": "complete",
        **source,
        "provenance": provenance,
        "image_identity_policy": (
            "lexicographically ordered source stems replaced by image_001..image_NNN"
        ),
        "split": report["split"],
        "reference_backend": report["reference_backend"],
        "candidate_backends": report["candidate_backends"],
        "required_images": report["required_images"],
        "n_images": report["n_images"],
        "config_sha256": report["config_sha256"],
        "thresholds": report["thresholds"],
        "comparisons": comparisons,
        "passed": report["passed"],
    }


def _sanitize_image_parity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_detections": report["reference_detections"],
        "candidate_detections": report["candidate_detections"],
        "matched_detections": report["matched_detections"],
        "unmatched_reference_detections": report["unmatched_reference_detections"],
        "unmatched_candidate_detections": report["unmatched_candidate_detections"],
        "min_iou": report["min_iou"],
        "max_conf_delta": report["max_conf_delta"],
        "matches": [
            {
                "class_id": match["class_id"],
                "iou": match["iou"],
                "confidence_delta": match["confidence_delta"],
            }
            for match in report["matches"]
        ],
        "unmatched_reference_class_ids": [box["class_id"] for box in report["unmatched_reference"]],
        "unmatched_candidate_class_ids": [box["class_id"] for box in report["unmatched_candidate"]],
        "passed": report["passed"],
    }


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _git_sha(value: object) -> bool:
    return isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _version(value: object) -> bool:
    return isinstance(value, str) and _VERSION_RE.fullmatch(value) is not None


def _copy_fields(source: dict[str, Any], fields: set[str] | frozenset[str]) -> dict[str, Any]:
    return {field: source[field] for field in sorted(fields)}
