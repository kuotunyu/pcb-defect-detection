"""Load committed evaluation evidence and derive the app's runtime mode."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import AppMode, AppState, EvidenceSummary, Metric


class EvidenceError(RuntimeError):
    """A committed evidence file is missing or violates the expected schema."""


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvidenceError(f"Cannot read committed evidence: {path}") from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"Invalid JSON in committed evidence: {path}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"Committed evidence must be a JSON object: {path}")
    return payload


def load_evidence(repo_root: Path) -> EvidenceSummary:
    """Return the small, high-signal metric set displayed by the portfolio."""

    final_metrics_path = repo_root / "reports" / "paired_a100" / "final_metrics.json"
    benchmark_path = repo_root / "reports" / "benchmark_l4.json"
    parity_path = repo_root / "reports" / "backend_parity_l4.json"

    final_metrics = _read_json(final_metrics_path)
    benchmark = _read_json(benchmark_path)
    parity = _read_json(parity_path)

    try:
        arms = final_metrics["aggregate"]["by_arm"]
        grouped = float(arms["grouped"]["map50"]["mean"])
        leaky = float(arms["leaky_control"]["map50"]["mean"])
        ort_p50 = float(benchmark["timings"]["onnxruntime_cuda_fp32"]["p50_ms"])
        strict_parity_passed = bool(parity["passed"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"Committed evidence schema mismatch: {error}") from error

    return EvidenceSummary(
        defect_classes=6,
        grouped_map50=Metric(
            label="Grouped mAP50",
            display_value=f"{grouped:.2%}",
            context="held-out Board 08 · 3 seeds",
            evidence_path="reports/paired_a100/final_metrics.json",
        ),
        leakage_effect=Metric(
            label="Leakage Effect",
            display_value=f"+{(leaky - grouped) * 100:.1f} pp",
            context="frozen same-board sibling exposure effect",
            evidence_path="reports/paired_a100/final_metrics.json",
        ),
        ort_cuda_p50=Metric(
            label="ORT CUDA p50",
            display_value=f"{ort_p50:.2f} ms",
            context="ONNX Runtime CUDA FP32 · NVIDIA L4 · calibration-only",
            evidence_path="reports/benchmark_l4.json",
        ),
        strict_parity_passed=strict_parity_passed,
    )


def build_app_state(repo_root: Path, contract: dict | None = None) -> AppState:
    """Resolve the portfolio mode without loading a model or importing ONNX Runtime."""

    try:
        loaded_contract = contract or _read_json(repo_root / "app" / "model_contract.json")
    except EvidenceError as error:
        return AppState(
            mode=AppMode.DEGRADED,
            evidence=None,
            contract={},
            status_title="Evidence unavailable",
            status_detail=str(error),
            inference_enabled=False,
            errors=(str(error),),
        )

    try:
        evidence = load_evidence(repo_root)
    except EvidenceError as error:
        return AppState(
            mode=AppMode.DEGRADED,
            evidence=None,
            contract=loaded_contract,
            status_title="Evidence unavailable",
            status_detail=str(error),
            inference_enabled=False,
            errors=(str(error),),
        )

    if loaded_contract.get("status") != "passed":
        detail = str(loaded_contract.get("reason", "Model promotion is blocked"))
        return AppState(
            mode=AppMode.EVIDENCE,
            evidence=evidence,
            contract=loaded_contract,
            status_title="Recorded evidence mode",
            status_detail=detail,
            inference_enabled=False,
        )

    return AppState(
        mode=AppMode.LIVE,
        evidence=evidence,
        contract=loaded_contract,
        status_title="Live inference candidate",
        status_detail="Model contract passed; runtime verification required",
        inference_enabled=True,
    )
