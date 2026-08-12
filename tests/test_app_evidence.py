from __future__ import annotations

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
