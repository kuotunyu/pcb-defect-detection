from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pcb_defect.deployment import DeploymentError, gate_passes, verified_deployment_selection


def _report() -> dict:
    return {
        "fidelity": {
            "threshold": 0.02,
            "delta_map50": -0.01,
            "delta_map50_95": 0.005,
        },
        "parity": {
            "n_images": 60,
            "required_images": 60,
            "n_failed": 0,
            "min_iou": 0.95,
            "required_min_iou": 0.90,
            "max_conf_delta": 0.10,
            "allowed_max_conf_delta": 0.15,
        },
    }


def test_gate_requires_both_metric_fidelity_and_all_image_parity() -> None:
    report = _report()
    assert gate_passes(report)

    report["parity"]["n_failed"] = 1
    assert not gate_passes(report)


def test_gate_rejects_fidelity_or_incomplete_calibration_set() -> None:
    report = _report()
    report["fidelity"]["delta_map50"] = -0.021
    assert not gate_passes(report)

    report = _report()
    report["parity"]["n_images"] = 59
    assert not gate_passes(report)


def test_runtime_parity_builds_both_backends_from_the_same_onnx(tmp_path: Path) -> None:
    from pcb_defect import deployment

    onnx_path = tmp_path / "best.onnx"
    onnx_path.write_bytes(b"onnx")
    observed: dict[str, Path] = {}

    def reference_factory(path: str) -> object:
        observed["reference"] = Path(path)
        return object()

    def standalone_factory(path: Path) -> object:
        observed["standalone"] = Path(path)
        return object()

    build_models = getattr(deployment, "_build_runtime_parity_models", None)
    assert build_models is not None
    build_models(
        onnx_path,
        reference_factory=reference_factory,
        standalone_factory=standalone_factory,
    )

    assert observed == {"reference": onnx_path, "standalone": onnx_path}


def test_parity_predicate_keeps_existing_threshold_contract() -> None:
    from pcb_defect.deployment import parity_passes

    parity = _report()["parity"]
    assert parity_passes(parity)
    parity["min_iou"] = 0.899
    assert not parity_passes(parity)


def test_deployment_selection_must_equal_hash_verified_final_metrics(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    selection = {"arm": "grouped", "seed": 42, "selected_before_final_test": True}
    metrics = final_dir / "final_metrics.json"
    metrics.write_text(json.dumps({"deployment_selection": selection}) + "\n", encoding="utf-8")
    (final_dir / "finalization_record.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "results": "final_metrics.json",
                "results_bytes": metrics.stat().st_size,
                "results_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (final_dir / "deployment_selection.json").write_text(json.dumps(selection), encoding="utf-8")

    assert verified_deployment_selection(final_dir) == selection

    (final_dir / "deployment_selection.json").write_text(
        json.dumps({**selection, "seed": 43}), encoding="utf-8"
    )
    with pytest.raises(DeploymentError, match="sidecar differs"):
        verified_deployment_selection(final_dir)


def test_deployment_module_forces_ultralytics_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import os

    import pcb_defect.deployment as deployment

    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.delenv("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", raising=False)
    importlib.reload(deployment)

    assert os.environ["YOLO_AUTOINSTALL"] == "false"
    assert os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] == "1"
