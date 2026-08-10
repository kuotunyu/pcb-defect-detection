from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pcb_defect.benchmark import summarize_latencies
from pcb_defect.l4_evidence import EvidencePromotionError, promote_l4_package
from pcb_defect.prediction_parity import (
    ParityThresholds,
    compare_backend_predictions,
)
from pcb_defect.result_package import create_verifiable_zip
from pcb_defect.viz import Box


def _raw_report() -> dict:
    parity = compare_backend_predictions(
        {
            "01_detected": [Box(1, (1.0, 2.0, 11.0, 12.0), 0.8)],
            "01_empty": [],
        },
        {
            "onnxruntime_cuda_fp32": {
                "01_detected": [Box(1, (1.0, 2.0, 11.0, 12.0), 0.79)],
                "01_empty": [],
            },
            "tensorrt_fp16": {
                "01_detected": [Box(1, (1.0, 2.0, 11.0, 12.0), 0.78)],
                "01_empty": [],
            },
        },
        reference_backend="pytorch_fp32",
        split="calibration",
        thresholds=ParityThresholds(0.25, 0.5, 0.9, 0.15),
        required_images=2,
        config_sha256="f" * 64,
    )
    timings = {}
    for index, backend in enumerate(
        ("pytorch_fp32", "onnxruntime_cuda_fp32", "tensorrt_fp16"), start=1
    ):
        raw = [float(index + offset) for offset in range(8)]
        timings[backend] = {**summarize_latencies(raw), "raw_ms": raw}
    return {
        "schema_version": "3.0",
        "status": "complete",
        "started_at_utc": "2026-08-10T00:00:00Z",
        "completed_at_utc": "2026-08-10T00:10:00Z",
        "command": ["private", "/content/private/path"],
        "runner_git_sha": "a" * 40,
        "experiment_git_sha": "b" * 40,
        "deployment_gate_sha256": "c" * 64,
        "dataset_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "environment": {"python": "3.11.15", "packages": {"ultralytics": "8.4.89"}},
        "runtime": {"onnxruntime_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]},
        "runtime_contract": {"before": {"state": "same"}, "after": {"state": "same"}},
        "hardware": {
            "gpu": "NVIDIA L4",
            "driver": "580.82.07",
            "torch_cuda": "12.6",
            "cudnn": 9000,
            "tensorrt": "10.13.3.9",
        },
        "protocol": {
            "split": "calibration",
            "images": 2,
            "image_list_sha256": "1" * 64,
            "image_content_sha256": "2" * 64,
            "cycles": 4,
            "warmup": 30,
            "batch": 1,
            "confidence": 0.25,
            "scope": (
                "predecoded PIL image; preprocess + inference + postprocess; CUDA synchronized"
            ),
            "timing_schedule": "interleaved-rotating-backend-order",
            "sessions": 1,
        },
        "artifacts": {
            "source_checkpoint_sha256": "3" * 64,
            "onnx_sha256": "4" * 64,
            "tensorrt_engine_sha256": "5" * 64,
            "engine_committable": False,
        },
        "fidelity": {
            "split": "calibration",
            "source_map50_95": 0.12,
            "onnx_map50_95": 0.11,
            "tensorrt_fp16_map50_95": 0.11,
            "tensorrt_minus_source": -0.01,
            "absolute_delta_threshold": 0.02,
            "passed": True,
        },
        "prediction_parity": parity,
        "timings": timings,
        "int8": {"status": "not_run", "reason": "fixture"},
    }


def _package(tmp_path: Path, report: dict | None = None) -> Path:
    root = tmp_path / "workspace"
    report_path = root / "benchmark_l4" / ("a" * 12) / "benchmark_l4.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(report or _raw_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    package = tmp_path / "paired-results-l4-fixture.zip"
    create_verifiable_zip(root, [report_path.relative_to(root)], package)
    return package


def test_promote_l4_package_keeps_raw_timings_and_sanitizes_per_box_parity(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    evidence = promote_l4_package(package)

    assert evidence.summary["schema_version"] == "2.0"
    assert evidence.summary["evidence_visibility"] == (
        "public_metadata_from_private_unreleased_package"
    )
    assert evidence.summary["package"] == {
        "filename": package.name,
        "bytes": package.stat().st_size,
        "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
    }
    assert "raw_ms" not in evidence.summary["timings"]["pytorch_fp32"]
    assert evidence.raw_timings["timings"]["pytorch_fp32"]["raw_ms"] == [
        float(value) for value in range(1, 9)
    ]
    parity = evidence.prediction_parity
    assert parity["passed"] is True
    per_image = parity["comparisons"]["onnxruntime_cuda_fp32"]["per_image"]
    assert list(per_image) == ["image_001", "image_002"]
    assert per_image["image_001"]["matches"] == [
        {"class_id": 1, "iou": 1.0, "confidence_delta": pytest.approx(0.01)}
    ]
    serialized = json.dumps(
        {
            "summary": evidence.summary,
            "raw_timings": evidence.raw_timings,
            "prediction_parity": evidence.prediction_parity,
        },
        sort_keys=True,
    )
    assert "01_detected" not in serialized
    assert "xyxy" not in serialized
    assert "/content/" not in serialized


@pytest.mark.parametrize(
    "mutation",
    [
        "parity",
        "timing_count",
        "schema",
        "hardware_extra",
        "runtime_extra",
        "protocol_extra",
        "artifacts_extra",
        "hardware_path",
        "protocol_path",
    ],
)
def test_promote_l4_package_rejects_incomplete_raw_evidence(tmp_path: Path, mutation: str) -> None:
    report = copy.deepcopy(_raw_report())
    if mutation == "parity":
        report["prediction_parity"]["comparisons"]["tensorrt_fp16"]["matched_detections"] = 99
    elif mutation == "timing_count":
        report["timings"]["pytorch_fp32"]["raw_ms"].pop()
    elif mutation == "schema":
        report["schema_version"] = "2.0"
    elif mutation == "hardware_extra":
        report["hardware"]["private_path"] = "/content/private"
    elif mutation == "runtime_extra":
        report["runtime"]["api_token"] = "secret"
    elif mutation == "protocol_extra":
        report["protocol"]["workspace"] = "/content/private"
    elif mutation == "artifacts_extra":
        report["artifacts"]["checkpoint_path"] = "/content/private/best.pt"
    elif mutation == "hardware_path":
        report["hardware"]["driver"] = "/content/private/driver"
    elif mutation == "protocol_path":
        report["protocol"]["scope"] = "/content/private/protocol"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(EvidencePromotionError):
        promote_l4_package(_package(tmp_path, report))
