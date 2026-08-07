from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcb_defect.evidence import artifact_ref
from pcb_defect.experiment import (
    ExperimentError,
    InputLock,
    _environment,
    _load_base_model_contract,
    _verify_base_model,
    freeze_or_verify_input_lock,
    planned_runs,
    run_is_complete,
)


def _lock(base_model_sha256: str = "a" * 64, config_sha256: str = "c" * 64) -> InputLock:
    return InputLock(
        git_sha="b" * 40,
        config_sha256=config_sha256,
        dataset_sha256="d" * 64,
        manifest_sha256="e" * 64,
        base_model_contract_sha256="f" * 64,
        base_model_sha256=base_model_sha256,
    )


def test_planned_runs_finish_grouped_before_leaky_control() -> None:
    assert planned_runs((42, 43, 44)) == [
        ("grouped", 42),
        ("grouped", 43),
        ("grouped", 44),
        ("leaky_control", 42),
        ("leaky_control", 43),
        ("leaky_control", 44),
    ]


def test_environment_records_tensorrt_cu12_when_generic_distribution_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pcb_defect import experiment

    def version(name: str) -> str:
        if name == "tensorrt-cu12":
            return "10.13.3.9"
        raise experiment.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(experiment.importlib.metadata, "version", version)

    assert _environment()["packages"]["tensorrt"] == "10.13.3.9"


def test_input_lock_is_created_once_and_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "input_lock.json"

    assert freeze_or_verify_input_lock(path, _lock()) == "created"
    assert freeze_or_verify_input_lock(path, _lock()) == "verified"

    with pytest.raises(ExperimentError, match="base_model_sha256"):
        freeze_or_verify_input_lock(path, _lock(base_model_sha256="f" * 64))


def test_base_model_contract_rejects_mutable_or_mismatched_bytes(tmp_path: Path) -> None:
    contract_path = tmp_path / "base_model.yaml"
    contract_path.write_text(
        "source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt\n"
        "revision: v8.4.0\n"
        "filename: yolo26n.pt\n"
        f"sha256: {'a' * 64}\n"
        "bytes: 5\n",
        encoding="utf-8",
    )
    contract = _load_base_model_contract(contract_path)
    weights = tmp_path / "base_model.pt"
    weights.write_bytes(b"wrong")

    with pytest.raises(ExperimentError, match="SHA-256"):
        _verify_base_model(weights, contract)

    contract_path.write_text(
        "source: https://github.com/ultralytics/assets/releases/latest/download/yolo26n.pt\n"
        "revision: v8.4.0\n"
        "filename: yolo26n.pt\n"
        f"sha256: {'a' * 64}\n"
        "bytes: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentError, match="immutable release revision"):
        _load_base_model_contract(contract_path)


def test_resume_gate_stages_checkpoint_and_stops_trainer_cleanly(tmp_path: Path) -> None:
    from pcb_defect import experiment

    stage_resume_checkpoint = getattr(experiment, "_stage_resume_checkpoint", None)
    assert stage_resume_checkpoint is not None

    source = tmp_path / "resume_smoke" / "weights" / "last.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"optimizer-bearing checkpoint")
    destination = tmp_path / "resume_checkpoint.pt"
    trainer = SimpleNamespace(epoch=0, last=source, stop=False)

    stage_resume_checkpoint(trainer, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert trainer.stop is True


def test_resume_gate_rejects_checkpoint_without_optimizer_state() -> None:
    from pcb_defect import experiment

    validate_resume_payload = getattr(experiment, "_validate_resume_checkpoint_payload", None)
    assert validate_resume_payload is not None

    assert validate_resume_payload({"epoch": 0, "optimizer": {"state": {}}}) == {
        "checkpoint_epoch": 0,
        "optimizer_state_present": True,
    }
    with pytest.raises(ExperimentError, match="optimizer state"):
        validate_resume_payload({"epoch": 0, "optimizer": None})


def test_complete_run_is_skipped_only_when_lock_and_artifact_hashes_match(tmp_path: Path) -> None:
    run_dir = tmp_path / "grouped" / "seed42"
    checkpoint = run_dir / "weights" / "best.pt"
    metrics = run_dir / "metrics" / "validation.json"
    manifest = run_dir / "inputs" / "manifest.json"
    config = run_dir / "inputs" / "train.yaml"
    base_contract = run_dir / "inputs" / "base_model.yaml"
    checkpoint.parent.mkdir(parents=True)
    metrics.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    metrics.write_text('{"map50": 0.5}\n', encoding="utf-8")
    manifest.write_bytes(b"m")
    config.write_bytes(b"c")
    base_contract.write_bytes(b"base-contract")
    manifest_ref = artifact_ref(manifest, relative_to=run_dir)
    config_ref = artifact_ref(config, relative_to=run_dir)
    base_contract_ref = artifact_ref(base_contract, relative_to=run_dir)
    checkpoint_ref = artifact_ref(checkpoint, relative_to=run_dir)
    metrics_ref = artifact_ref(metrics, relative_to=run_dir)
    record = {
        "schema_version": "1.0",
        "run_id": "grouped-seed42",
        "arm": "grouped",
        "seed": 42,
        "status": "complete",
        "timestamps": {
            "created_at_utc": "2026-08-03T00:00:00Z",
            "updated_at_utc": "2026-08-03T00:00:00Z",
        },
        "provenance": {
            "git_sha": "b" * 40,
            "git_dirty": False,
            "command": ["python", "-m", "pcb_defect.experiment", "train"],
            "environment": {"python": "3.11", "platform": "Linux", "packages": {}},
        },
        "protocol": {
            "version": "paired-board-sensitivity-v1",
            "manifest": manifest_ref,
            "manifest_sha256": "e" * 64,
            "dataset_sha256": "d" * 64,
        },
        "training": {
            "config": config_ref,
            "resolved": {"epochs": 100},
            "base_model": {
                "source": "https://example.invalid/v1/yolo26n.pt",
                "revision": "v1",
                "filename": "yolo26n.pt",
                "sha256": "a" * 64,
                "bytes": 10,
                "contract": base_contract_ref,
            },
        },
        "artifacts": {"best_checkpoint": checkpoint_ref},
        "metrics": {"validation": metrics_ref},
        "failure": None,
    }
    (run_dir / "run_record.json").write_text(json.dumps(record), encoding="utf-8")

    lock = InputLock(
        **{
            **_lock(config_sha256=str(config_ref["sha256"])).as_dict(),
            "base_model_contract_sha256": str(base_contract_ref["sha256"]),
        }
    )
    assert run_is_complete(run_dir, lock)

    checkpoint.write_bytes(b"tampered")
    assert not run_is_complete(run_dir, lock)
