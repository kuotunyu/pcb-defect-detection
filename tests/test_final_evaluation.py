from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pcb_defect.final_evaluation import (
    FinalEvaluationError,
    begin_final_test_once,
    choose_grouped_deployment_seed,
    final_evaluation_is_complete,
    paired_bootstrap_ci,
)


def test_paired_bootstrap_is_deterministic_and_preserves_direction() -> None:
    deltas = [0.10, 0.20, 0.30, 0.40]

    first = paired_bootstrap_ci(deltas, n_resamples=2_000, seed=42)
    second = paired_bootstrap_ci(deltas, n_resamples=2_000, seed=42)

    assert first == second
    assert first["mean_delta"] == pytest.approx(0.25)
    assert first["ci95_low"] > 0
    assert first["unit"] == "final-test-image"


def test_deployment_seed_is_selected_only_from_grouped_validation_metrics() -> None:
    validation = {
        42: {"map50_95": 0.40},
        43: {"map50_95": 0.45},
        44: {"map50_95": 0.45},
    }

    assert choose_grouped_deployment_seed(validation) == 43

    validation[42]["map50_95"] = float("nan")
    with pytest.raises(FinalEvaluationError, match="finite"):
        choose_grouped_deployment_seed(validation)


def test_final_test_marker_prevents_accidental_second_evaluation(tmp_path: Path) -> None:
    marker = tmp_path / "FINAL_TEST_STARTED.json"
    payload = {"git_sha": "a" * 40, "manifest_sha256": "b" * 64}

    begin_final_test_once(marker, payload)

    with pytest.raises(FinalEvaluationError, match="already exists"):
        begin_final_test_once(marker, payload)


def test_final_completion_requires_untampered_metrics_bytes(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    metrics = final_dir / "final_metrics.json"
    payload = b'{"status": "complete"}\n'
    metrics.write_bytes(payload)
    record = final_dir / "finalization_record.json"
    record.write_text(
        "{\n"
        '  "status": "complete",\n'
        '  "results": "final_metrics.json",\n'
        f'  "results_sha256": "{hashlib.sha256(payload).hexdigest()}",\n'
        f'  "results_bytes": {len(payload)}\n'
        "}\n",
        encoding="utf-8",
    )

    assert final_evaluation_is_complete(final_dir)

    metrics.write_text('{"status": "tampered"}\n', encoding="utf-8")
    assert not final_evaluation_is_complete(final_dir)
