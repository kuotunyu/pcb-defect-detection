"""Colab L4 PyTorch/ONNX Runtime/TensorRT FP16 benchmark with raw timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from pcb_defect.experiment import _environment, _sha256_file
from pcb_defect.l4_contract import L4ContractError, L4RunIdentity, verify_l4_inputs
from pcb_defect.prediction_parity import (
    ParityConfig,
    ParityError,
    compare_backend_predictions,
    load_parity_config,
    prediction_parity_is_complete,
)
from pcb_defect.runtime_contract import (
    RuntimeContractError,
    configure_hermetic_ultralytics,
    onnxruntime_state,
)
from pcb_defect.viz import Box, boxes_from_ultralytics

configure_hermetic_ultralytics()


CANONICAL_WARMUP = 30
CANONICAL_CYCLES = 4
COLAB_L4_DEVICE_NAMES = frozenset({"nvidia l4"})
BENCHMARK_BACKENDS = frozenset({"pytorch_fp32", "onnxruntime_cuda_fp32", "tensorrt_fp16"})
TIMING_FIELDS = frozenset(
    {
        "n_runs",
        "mean_ms",
        "std_ms",
        "p50_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "fps_from_p50",
        "raw_ms",
    }
)
FLOAT_SUMMARY_FIELDS = frozenset(
    {"mean_ms", "std_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "fps_from_p50"}
)
PROTOCOL_FIELDS = frozenset(
    {
        "split",
        "images",
        "image_list_sha256",
        "image_content_sha256",
        "cycles",
        "warmup",
        "batch",
        "confidence",
        "scope",
        "timing_schedule",
        "sessions",
    }
)
ARTIFACT_FIELDS = frozenset(
    {"source_checkpoint_sha256", "onnx_sha256", "tensorrt_engine_sha256", "engine_committable"}
)
REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "started_at_utc",
        "completed_at_utc",
        "command",
        "runner_git_sha",
        "experiment_git_sha",
        "deployment_gate_sha256",
        "dataset_sha256",
        "manifest_sha256",
        "environment",
        "runtime",
        "runtime_contract",
        "hardware",
        "protocol",
        "artifacts",
        "fidelity",
        "prediction_parity",
        "timings",
        "int8",
    }
)


class BenchmarkError(RuntimeError):
    """The benchmark environment, source artifacts, or fidelity gate is invalid."""


@dataclass(frozen=True, slots=True)
class _GpuRuntime:
    """The narrow import boundary for GPU-only benchmark dependencies."""

    torch: Any
    yolo: Any
    onnx_model: Any
    tensorrt_version: str


def _load_gpu_runtime() -> _GpuRuntime:
    import tensorrt
    import torch
    from ultralytics import YOLO

    from pcb_defect.e2e_onnx import OnnxYoloModel

    return _GpuRuntime(torch, YOLO, OnnxYoloModel, tensorrt.__version__)


def _require_cuda_onnxruntime_state() -> dict[str, object]:
    try:
        return onnxruntime_state(require_cuda_provider=True)
    except RuntimeContractError as exc:
        raise BenchmarkError(str(exc)) from exc


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float | int]:
    if not latencies_ms:
        raise BenchmarkError("latency summary requires at least one timing")
    ordered = sorted(latencies_ms)
    p50 = _quantile(ordered, 0.50)
    return {
        "n_runs": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "p50_ms": p50,
        "p95_ms": _quantile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "fps_from_p50": 1000.0 / p50,
    }


def _ultralytics_boxes(model: Any, image: object, confidence: float) -> list[Box]:
    results = model.predict(image, conf=confidence, verbose=False)
    if not results:
        return []
    if not isinstance(results, list) or len(results) != 1:
        raise BenchmarkError("Ultralytics parity inference must return exactly one result")
    return boxes_from_ultralytics(results[0])


def _collect_predictions(
    image_paths: list[Path],
    images: list[object],
    inference: Callable[[object], list[Box]],
    synchronize: Callable[[], None],
) -> dict[str, list[Box]]:
    if len(image_paths) != len(images):
        raise BenchmarkError("prediction parity image paths and decoded images differ")
    predictions: dict[str, list[Box]] = {}
    for path, image in zip(image_paths, images, strict=True):
        if path.stem in predictions:
            raise BenchmarkError(f"duplicate prediction parity image stem: {path.stem}")
        boxes = inference(image)
        synchronize()
        predictions[path.stem] = boxes
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-runner-git-sha", required=True)
    parser.add_argument("--expected-experiment-git-sha", required=True)
    parser.add_argument("--expected-deployment-gate-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-onnx-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args(argv)
    identity = L4RunIdentity.parse(
        runner_git_sha=args.expected_runner_git_sha,
        experiment_git_sha=args.expected_experiment_git_sha,
        deployment_gate_sha256=args.expected_deployment_gate_sha256,
        checkpoint_sha256=args.expected_checkpoint_sha256,
        onnx_sha256=args.expected_onnx_sha256,
    )
    benchmark(
        args.repo.resolve(),
        args.workspace.resolve(),
        args.dataset.resolve(),
        identity,
        warmup=args.warmup,
        cycles=args.cycles,
    )
    return 0


def benchmark(
    repo: Path,
    workspace: Path,
    dataset_root: Path,
    identity: L4RunIdentity,
    *,
    warmup: int,
    cycles: int,
) -> Path:
    """Benchmark verified parent artifacts only after every L4 runtime gate passes."""

    verified = verify_l4_inputs(repo, workspace, dataset_root, identity)
    try:
        gpu = _load_gpu_runtime()
    except ImportError as exc:
        raise BenchmarkError("TensorRT runtime is unavailable") from exc
    if not gpu.torch.cuda.is_available():
        raise BenchmarkError("CUDA is unavailable")
    device_name = gpu.torch.cuda.get_device_name(0)
    if not _is_colab_l4_device(device_name):
        raise BenchmarkError(f"benchmark requires a Colab L4, found {device_name!r}")
    runtime_before = _require_cuda_onnxruntime_state()
    if not gpu.tensorrt_version:
        raise BenchmarkError("TensorRT runtime is unavailable")
    if (warmup, cycles) != (CANONICAL_WARMUP, CANONICAL_CYCLES):
        raise BenchmarkError("benchmark requires exactly warmup=30 and cycles=4")
    parity_config_path = repo / "configs" / "backend_parity.yaml"
    try:
        parity_config = load_parity_config(parity_config_path)
    except ParityError as exc:
        raise BenchmarkError(str(exc)) from exc
    if len(verified.parent.calibration_images) != parity_config.required_images:
        raise BenchmarkError(
            "calibration image count does not match the frozen prediction-parity config"
        )
    started_at_utc = _utc_now()

    benchmark_root = workspace / "benchmark_l4"
    final_output_dir = benchmark_root / identity.runner_git_sha[:12]
    final_report_path = final_output_dir / "benchmark_l4.json"
    if final_report_path.is_file():
        try:
            report = json.loads(final_report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"invalid existing benchmark report: {final_report_path}") from exc
        if not isinstance(report, dict):
            raise BenchmarkError(f"invalid existing benchmark report: {final_report_path}")
        if benchmark_is_complete(repo, workspace, dataset_root, identity, report):
            print(f"SKIP completed L4 benchmark: {final_report_path}")
            return final_report_path
        raise BenchmarkError(f"existing benchmark evidence is incomplete: {final_report_path}")
    if final_output_dir.exists():
        raise BenchmarkError(f"partial benchmark directory exists: {final_output_dir}")

    benchmark_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(
        tempfile.mkdtemp(prefix=f".attempt-{identity.runner_git_sha[:12]}-", dir=benchmark_root)
    )
    report_path = output_dir / "benchmark_l4.json"
    engine_path = _export_engine(verified.parent.checkpoint_path, output_dir)
    engine_metrics = gpu.yolo(str(engine_path)).val(
        data=str(verified.parent.calibration_yaml),
        split="val",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
    )
    engine_map50_95 = float(engine_metrics.box.map)
    source_map50_95 = float(verified.parent.gate["fidelity"]["pt"]["map50_95"])
    engine_delta = engine_map50_95 - source_map50_95

    image_paths = list(verified.parent.calibration_images)
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    pt_model = gpu.yolo(str(verified.parent.checkpoint_path))
    trt_model = gpu.yolo(str(engine_path))
    ort_model = gpu.onnx_model(verified.parent.onnx_path, providers=["CUDAExecutionProvider"])
    providers = ort_model.session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise BenchmarkError(f"ONNX Runtime did not activate CUDAExecutionProvider: {providers}")

    backends = {
        "pytorch_fp32": lambda image: pt_model.predict(
            image, conf=parity_config.thresholds.confidence, verbose=False
        ),
        "onnxruntime_cuda_fp32": lambda image: ort_model.predict(
            image, conf=parity_config.thresholds.confidence
        ),
        "tensorrt_fp16": lambda image: trt_model.predict(
            image, conf=parity_config.thresholds.confidence, verbose=False
        ),
    }
    timings = _time_backends_interleaved(
        backends, images, gpu.torch.cuda.synchronize, warmup, cycles
    )
    prediction_functions: dict[str, Callable[[object], list[Box]]] = {
        "pytorch_fp32": lambda image: _ultralytics_boxes(
            pt_model, image, parity_config.thresholds.confidence
        ),
        "onnxruntime_cuda_fp32": lambda image: ort_model.predict(
            image, conf=parity_config.thresholds.confidence
        ),
        "tensorrt_fp16": lambda image: _ultralytics_boxes(
            trt_model, image, parity_config.thresholds.confidence
        ),
    }
    predictions = {
        backend: _collect_predictions(
            image_paths, images, prediction_functions[backend], gpu.torch.cuda.synchronize
        )
        for backend in (
            parity_config.reference_backend,
            *parity_config.candidate_backends,
        )
    }
    try:
        prediction_parity = compare_backend_predictions(
            predictions[parity_config.reference_backend],
            {backend: predictions[backend] for backend in parity_config.candidate_backends},
            reference_backend=parity_config.reference_backend,
            split=parity_config.split,
            thresholds=parity_config.thresholds,
            required_images=parity_config.required_images,
            config_sha256=_sha256_file(parity_config_path),
        )
    except ParityError as exc:
        raise BenchmarkError(str(exc)) from exc
    runtime_after = _require_cuda_onnxruntime_state()
    if runtime_after != runtime_before:
        raise BenchmarkError("ONNX Runtime state changed during the L4 benchmark")
    fidelity_passed = abs(engine_delta) <= float(verified.parent.gate["fidelity"]["threshold"])
    if not fidelity_passed:
        raise BenchmarkError("TensorRT FP16 calibration fidelity failed")
    report = {
        "schema_version": "3.0",
        "status": "complete",
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_now(),
        "command": list(sys.argv),
        "runner_git_sha": identity.runner_git_sha,
        "experiment_git_sha": identity.parent.experiment_git_sha,
        "deployment_gate_sha256": identity.parent.deployment_gate_sha256,
        "dataset_sha256": verified.parent.gate["dataset_sha256"],
        "manifest_sha256": verified.parent.gate["manifest_sha256"],
        "environment": _environment(),
        "runtime": {"onnxruntime_providers": providers},
        "runtime_contract": {"before": runtime_before, "after": runtime_after},
        "hardware": _hardware(gpu.torch, device_name, gpu.tensorrt_version),
        "protocol": {
            "split": "calibration",
            "images": len(images),
            "image_list_sha256": _image_list_sha256(image_paths),
            "image_content_sha256": _image_content_sha256(image_paths),
            "cycles": cycles,
            "warmup": warmup,
            "batch": 1,
            "confidence": parity_config.thresholds.confidence,
            "scope": (
                "predecoded PIL image; preprocess + inference + postprocess; CUDA synchronized"
            ),
            "timing_schedule": "interleaved-rotating-backend-order",
            "sessions": 1,
        },
        "artifacts": {
            "source_checkpoint_sha256": identity.parent.checkpoint_sha256,
            "onnx_sha256": identity.parent.onnx_sha256,
            "tensorrt_engine_sha256": _sha256_file(engine_path),
            "engine_committable": False,
        },
        "fidelity": {
            "split": "calibration",
            "source_map50_95": source_map50_95,
            "onnx_map50_95": verified.parent.gate["fidelity"]["onnx"]["map50_95"],
            "tensorrt_fp16_map50_95": engine_map50_95,
            "tensorrt_minus_source": engine_delta,
            "absolute_delta_threshold": verified.parent.gate["fidelity"]["threshold"],
            "passed": fidelity_passed,
        },
        "prediction_parity": prediction_parity,
        "timings": timings,
        "int8": {
            "status": "not_run",
            "reason": (
                "Legacy exploration showed no speed advantage over FP16 and a material accuracy "
                "loss; INT8 adds no portfolio value without a deployment requirement."
            ),
        },
    }
    temporary_report = report_path.with_name(".benchmark_l4.json.tmp")
    with temporary_report.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_report, report_path)
    if final_output_dir.exists():
        raise BenchmarkError(f"refusing to overwrite L4 benchmark: {final_output_dir}")
    try:
        output_dir.rename(final_output_dir)
    except OSError as exc:
        raise BenchmarkError(
            f"completed L4 benchmark could not publish final directory: {final_output_dir}"
        ) from exc
    print(f"L4 BENCHMARK COMPLETE: {final_report_path}")
    return final_report_path


def benchmark_is_complete(
    repo: Path,
    workspace: Path,
    dataset_root: Path,
    identity: L4RunIdentity,
    report: Any,
) -> bool:
    """Verify every persisted dependency before accepting a completed benchmark."""
    if not isinstance(report, dict):
        return False
    try:
        verified = verify_l4_inputs(repo, workspace, dataset_root, identity)
        gpu = _load_gpu_runtime()
        if not gpu.torch.cuda.is_available():
            return False
        device_name = gpu.torch.cuda.get_device_name(0)
        if not _is_colab_l4_device(device_name) or not gpu.tensorrt_version:
            return False
        engine = workspace / "benchmark_l4" / identity.runner_git_sha[:12] / "best_fp16.engine"
        image_paths = list(verified.parent.calibration_images)
        current_runtime = _require_cuda_onnxruntime_state()
        current_hardware = _hardware(gpu.torch, device_name, gpu.tensorrt_version)
        parity_config_path = repo / "configs" / "backend_parity.yaml"
        parity_config = load_parity_config(parity_config_path)
        return _report_matches_complete_evidence(
            report,
            identity,
            verified.parent.gate,
            image_paths,
            engine,
            current_runtime,
            current_hardware,
            parity_config,
            _sha256_file(parity_config_path),
        )
    except (
        BenchmarkError,
        L4ContractError,
        ParityError,
        ImportError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        AttributeError,
        UnicodeError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ):
        return False


def _report_matches_complete_evidence(
    report: dict[str, Any],
    identity: L4RunIdentity,
    gate: dict[str, Any],
    image_paths: list[Path],
    engine: Path,
    current_runtime: dict[str, object],
    current_hardware: dict[str, Any],
    parity_config: ParityConfig,
    parity_config_sha256: str,
) -> bool:
    if set(report) != REPORT_FIELDS:
        return False
    try:
        started = _parse_utc_timestamp(report["started_at_utc"])
        completed = _parse_utc_timestamp(report["completed_at_utc"])
        expected_protocol = _benchmark_protocol(image_paths)
        expected_artifacts = {
            "source_checkpoint_sha256": identity.parent.checkpoint_sha256,
            "onnx_sha256": identity.parent.onnx_sha256,
            "tensorrt_engine_sha256": _sha256_file(engine),
            "engine_committable": False,
        }
        return (
            report.get("schema_version") == "3.0"
            and report.get("status") == "complete"
            and completed >= started
            and _command_matches(report["command"])
            and report["runner_git_sha"] == identity.runner_git_sha
            and report["experiment_git_sha"] == identity.parent.experiment_git_sha
            and report["deployment_gate_sha256"] == identity.parent.deployment_gate_sha256
            and report["dataset_sha256"] == gate["dataset_sha256"]
            and report["manifest_sha256"] == gate["manifest_sha256"]
            and engine.is_file()
            and report["environment"] == _environment()
            and report["hardware"] == current_hardware
            and _protocol_matches(report["protocol"], expected_protocol)
            and _artifacts_match(report["artifacts"], expected_artifacts)
            and _runtime_matches(report["runtime"], current_runtime)
            and _runtime_contract_matches(report["runtime_contract"], current_runtime)
            and _fidelity_matches(report["fidelity"], gate)
            and _prediction_parity_matches(
                report["prediction_parity"], parity_config, parity_config_sha256
            )
            and _timings_match(report["timings"], len(image_paths) * CANONICAL_CYCLES)
            and report["int8"]
            == {
                "status": "not_run",
                "reason": (
                    "Legacy exploration showed no speed advantage over FP16 and a material "
                    "accuracy "
                    "loss; INT8 adds no portfolio value without a deployment requirement."
                ),
            }
        )
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return False


def _normalize_device_name(device_name: str) -> str:
    return " ".join(device_name.split()).casefold()


def _is_colab_l4_device(device_name: Any) -> bool:
    return (
        isinstance(device_name, str)
        and _normalize_device_name(device_name) in COLAB_L4_DEVICE_NAMES
    )


def _parse_utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("benchmark timestamp must be a UTC ISO-8601 string")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _benchmark_protocol(image_paths: list[Path]) -> dict[str, Any]:
    return {
        "split": "calibration",
        "images": len(image_paths),
        "image_list_sha256": _image_list_sha256(image_paths),
        "image_content_sha256": _image_content_sha256(image_paths),
        "cycles": CANONICAL_CYCLES,
        "warmup": CANONICAL_WARMUP,
        "batch": 1,
        "confidence": 0.25,
        "scope": "predecoded PIL image; preprocess + inference + postprocess; CUDA synchronized",
        "timing_schedule": "interleaved-rotating-backend-order",
        "sessions": 1,
    }


def _command_matches(command: Any) -> bool:
    return (
        isinstance(command, list)
        and bool(command)
        and all(isinstance(argument, str) and argument for argument in command)
    )


def _protocol_matches(protocol: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_FIELDS:
        return False
    integer_fields = ("images", "cycles", "warmup", "batch", "sessions")
    return (
        all(type(protocol[field]) is int for field in integer_fields)
        and type(protocol["confidence"]) is float
        and all(
            isinstance(protocol[field], str)
            for field in (
                "split",
                "image_list_sha256",
                "image_content_sha256",
                "scope",
                "timing_schedule",
            )
        )
        and protocol == expected
    )


def _artifacts_match(artifacts: Any, expected: dict[str, Any]) -> bool:
    return (
        isinstance(artifacts, dict)
        and set(artifacts) == ARTIFACT_FIELDS
        and all(
            isinstance(artifacts[field], str)
            for field in (
                "source_checkpoint_sha256",
                "onnx_sha256",
                "tensorrt_engine_sha256",
            )
        )
        and artifacts["engine_committable"] is False
        and artifacts == expected
    )


def _runtime_matches(runtime: Any, current_runtime: dict[str, object]) -> bool:
    if not isinstance(runtime, dict) or set(runtime) != {"onnxruntime_providers"}:
        return False
    providers = runtime["onnxruntime_providers"]
    observed = current_runtime["available_providers"]
    return (
        isinstance(providers, list)
        and bool(providers)
        and all(isinstance(provider, str) and provider for provider in providers)
        and providers[0] == "CUDAExecutionProvider"
        and isinstance(observed, list)
        and all(isinstance(provider, str) and provider for provider in observed)
        and len(providers) == len(set(providers))
        and set(providers).issubset(set(observed))
    )


def _runtime_contract_matches(runtime_contract: Any, current_runtime: dict[str, object]) -> bool:
    return (
        isinstance(runtime_contract, dict)
        and set(runtime_contract) == {"before", "after"}
        and isinstance(runtime_contract["before"], dict)
        and isinstance(runtime_contract["after"], dict)
        and runtime_contract["before"] == runtime_contract["after"] == current_runtime
    )


def _fidelity_matches(fidelity: Any, gate: dict[str, Any]) -> bool:
    if not isinstance(fidelity, dict):
        return False
    required = {
        "split",
        "source_map50_95",
        "onnx_map50_95",
        "tensorrt_fp16_map50_95",
        "tensorrt_minus_source",
        "absolute_delta_threshold",
        "passed",
    }
    if set(fidelity) != required:
        return False
    threshold = gate["fidelity"]["threshold"]
    source = gate["fidelity"]["pt"]["map50_95"]
    onnx = gate["fidelity"]["onnx"]["map50_95"]
    engine = fidelity["tensorrt_fp16_map50_95"]
    delta = fidelity["tensorrt_minus_source"]
    numeric_values = (threshold, source, onnx, engine, delta)
    return (
        fidelity["split"] == "calibration"
        and all(_is_finite_number(value) for value in numeric_values)
        and fidelity["source_map50_95"] == source
        and fidelity["onnx_map50_95"] == onnx
        and fidelity["absolute_delta_threshold"] == threshold
        and delta == engine - source
        and fidelity["passed"] is (abs(delta) <= float(threshold))
    )


def _prediction_parity_matches(report: Any, config: ParityConfig, config_sha256: str) -> bool:
    return (
        isinstance(report, dict)
        and report.get("split") == config.split
        and report.get("reference_backend") == config.reference_backend
        and report.get("candidate_backends") == list(config.candidate_backends)
        and report.get("required_images") == config.required_images
        and report.get("n_images") == config.required_images
        and report.get("config_sha256") == config_sha256
        and report.get("thresholds")
        == {
            "confidence": config.thresholds.confidence,
            "match_iou": config.thresholds.match_iou,
            "required_min_iou": config.thresholds.required_min_iou,
            "allowed_max_conf_delta": config.thresholds.allowed_max_conf_delta,
        }
        and prediction_parity_is_complete(report)
    )


def _timings_match(timings: Any, expected_runs: int) -> bool:
    if not isinstance(timings, dict) or set(timings) != BENCHMARK_BACKENDS:
        return False
    for timing in timings.values():
        if not isinstance(timing, dict) or set(timing) != TIMING_FIELDS:
            return False
        raw = timing["raw_ms"]
        if (
            not isinstance(raw, list)
            or len(raw) != expected_runs
            or any(not _is_finite_number(value) or value <= 0 for value in raw)
        ):
            return False
        expected_summary = summarize_latencies(raw)
        if type(timing["n_runs"]) is not int or timing["n_runs"] != expected_summary["n_runs"]:
            return False
        if any(not _is_finite_number(timing[field]) for field in FLOAT_SUMMARY_FIELDS):
            return False
        if any(timing[key] != value for key, value in expected_summary.items()):
            return False
    return True


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _load_verified_deployment_inputs(
    workspace: Path,
) -> tuple[Path, dict[str, Any], Path, Path]:
    deployment_dir = workspace / "deployment"
    gate_path = deployment_dir / "deployment_gate.json"
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("passed") is not True:
            raise BenchmarkError("ONNX deployment gate has not passed")
        onnx_path = deployment_dir / "best.onnx"
        if _sha256_file(onnx_path) != gate["artifacts"]["onnx_sha256"]:
            raise BenchmarkError("ONNX bytes no longer match the deployment gate")
        source_weights = (workspace / gate["artifacts"]["source_checkpoint"]).resolve()
        source_weights.relative_to(workspace.resolve())
        if _sha256_file(source_weights) != gate["artifacts"]["source_checkpoint_sha256"]:
            raise BenchmarkError("source checkpoint bytes no longer match the deployment gate")
    except BenchmarkError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkError("deployment evidence is missing or malformed") from exc
    return gate_path, gate, onnx_path, source_weights


def _benchmark_image_paths(calibration_yaml: Path) -> list[Path]:
    """Load only the predeclared calibration set; final-test images are never benchmark input."""
    try:
        payload = yaml.safe_load(calibration_yaml.read_text(encoding="utf-8"))
        calibration_list = Path(payload["val"])
        paths = [
            Path(line)
            for line in calibration_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise BenchmarkError("calibration benchmark input is missing or malformed") from exc
    if not paths or any(not path.is_file() for path in paths):
        raise BenchmarkError("calibration benchmark input contains missing images")
    return paths


def _image_list_sha256(image_paths: list[Path]) -> str:
    payload = "".join(f"{path.stem}\n" for path in image_paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _image_content_sha256(image_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in image_paths:
        digest.update(path.stem.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _time_backend(
    function: Callable[[Image.Image], Any],
    images: list[Image.Image],
    synchronize: Callable[[], None],
    warmup: int,
    cycles: int,
) -> dict[str, Any]:
    for index in range(warmup):
        function(images[index % len(images)])
    synchronize()
    latencies = []
    for _ in range(cycles):
        for image in images:
            synchronize()
            started = time.perf_counter()
            function(image)
            synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
    return {**summarize_latencies(latencies), "raw_ms": latencies}


def _time_backends_interleaved(
    backends: dict[str, Callable[[object], Any]],
    images: list[object],
    synchronize: Callable[[], None],
    warmup: int,
    cycles: int,
) -> dict[str, dict[str, Any]]:
    """Rotate backend order per image so each backend occupies each timing position equally."""
    names = tuple(backends)
    if set(names) != BENCHMARK_BACKENDS or len(names) != len(BENCHMARK_BACKENDS):
        raise BenchmarkError("interleaved timing requires all canonical backends exactly once")
    if not images:
        raise BenchmarkError("interleaved timing requires at least one image")

    for index in range(warmup):
        image = images[index % len(images)]
        for name in _rotated(names, index):
            backends[name](image)
            synchronize()

    latencies: dict[str, list[float]] = {name: [] for name in names}
    for cycle in range(cycles):
        for image_index, image in enumerate(images):
            schedule_index = cycle * len(images) + image_index
            for name in _rotated(names, schedule_index):
                synchronize()
                started = time.perf_counter()
                backends[name](image)
                synchronize()
                latencies[name].append((time.perf_counter() - started) * 1000)
    return {
        name: {**summarize_latencies(values), "raw_ms": values}
        for name, values in latencies.items()
    }


def _rotated(values: tuple[str, ...], index: int) -> tuple[str, ...]:
    offset = index % len(values)
    return values[offset:] + values[:offset]


def _export_engine(source_weights: Path, output_dir: Path) -> Path:
    from ultralytics import YOLO

    destination = output_dir / "best_fp16.engine"
    if destination.exists() or destination.is_symlink():
        raise BenchmarkError(f"refusing to overwrite existing TensorRT engine: {destination}")
    scratch = Path(tempfile.mkdtemp(prefix=".export-", dir=output_dir))
    try:
        scratch_weights = scratch / source_weights.name
        shutil.copy2(source_weights, scratch_weights)
        generated = Path(
            YOLO(str(scratch_weights)).export(
                format="engine",
                imgsz=640,
                batch=1,
                dynamic=False,
                half=True,
                workspace=4,
            )
        ).resolve()
        try:
            generated.relative_to(scratch.resolve())
        except ValueError as exc:
            raise BenchmarkError(
                "TensorRT export escaped its runner-owned scratch directory"
            ) from exc
        if not generated.is_file():
            raise BenchmarkError(
                "TensorRT export did not produce an engine in its scratch directory"
            )
        try:
            with generated.open("rb") as source, destination.open("xb") as published:
                shutil.copyfileobj(source, published)
        except FileExistsError as exc:
            raise BenchmarkError(
                f"refusing to overwrite existing TensorRT engine: {destination}"
            ) from exc
        return destination
    finally:
        shutil.rmtree(scratch)


def _hardware(torch: Any, device_name: str, tensorrt_version: str) -> dict[str, Any]:
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "gpu": device_name,
        "driver": driver,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tensorrt": tensorrt_version,
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
