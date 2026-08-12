from __future__ import annotations

import json
from pathlib import Path

from app.evidence import build_app_state, load_evidence
from app.models import AppMode

ROOT = Path(__file__).resolve().parents[1]


def test_load_evidence_uses_committed_metrics() -> None:
    evidence = load_evidence(ROOT)

    assert evidence.defect_classes == 6
    assert evidence.grouped_map50.display_value == "63.30%"
    assert evidence.leakage_effect.display_value == "+21.3 pp"
    assert evidence.ort_cuda_p50.display_value == "20.28 ms"
    assert evidence.strict_parity_passed is False


def test_blocked_contract_enters_evidence_mode_without_model() -> None:
    state = build_app_state(ROOT)

    assert state.mode is AppMode.EVIDENCE
    assert state.inference_enabled is False
    assert "strict" in state.status_detail.lower()


def test_missing_report_enters_degraded_mode(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "model_contract.json").write_text(
        '{"schema_version":"1.0","status":"blocked","reason":"blocked"}',
        encoding="utf-8",
    )

    state = build_app_state(tmp_path)

    assert state.mode is AppMode.DEGRADED
    assert state.inference_enabled is False
    assert state.errors


def test_non_boolean_parity_status_enters_degraded_mode(tmp_path: Path) -> None:
    _write_minimal_evidence(tmp_path, parity_passed="false")

    state = build_app_state(tmp_path, _blocked_contract())

    assert state.mode is AppMode.DEGRADED
    assert "passed must be a boolean" in state.status_detail


def test_non_finite_or_out_of_range_metrics_enter_degraded_mode(tmp_path: Path) -> None:
    _write_minimal_evidence(tmp_path, grouped_map50=float("nan"))

    state = build_app_state(tmp_path, _blocked_contract())

    assert state.mode is AppMode.DEGRADED
    assert "grouped map50" in state.status_detail.lower()


def test_passed_contract_with_failed_committed_parity_enters_degraded_mode() -> None:
    contract = {
        "schema_version": "1.0",
        "status": "passed",
        "onnx_sha256": "a" * 64,
        "hf_repo_id": "owner/model",
        "hf_revision": "b" * 40,
    }

    state = build_app_state(ROOT, contract)

    assert state.mode is AppMode.DEGRADED
    assert state.inference_enabled is False
    assert "contradicts" in state.status_detail


def _blocked_contract() -> dict:
    return {"schema_version": "1.0", "status": "blocked", "reason": "blocked"}


def _write_minimal_evidence(
    root: Path,
    *,
    grouped_map50: float = 0.633,
    parity_passed: object = False,
) -> None:
    (root / "reports" / "paired_a100").mkdir(parents=True)
    (root / "reports" / "paired_a100" / "final_metrics.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "by_arm": {
                        "grouped": {"map50": {"mean": grouped_map50}},
                        "leaky_control": {"map50": {"mean": 0.8456}},
                    }
                }
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    (root / "reports" / "benchmark_l4.json").write_text(
        json.dumps({"timings": {"onnxruntime_cuda_fp32": {"p50_ms": 20.28}}}),
        encoding="utf-8",
    )
    (root / "reports" / "backend_parity_l4.json").write_text(
        json.dumps({"passed": parity_passed}),
        encoding="utf-8",
    )
