from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcb_defect.evidence import (
    EvidenceError,
    artifact_ref,
    validate_run_record,
    write_run_record,
)


def _ref(path: str = "configs/train.yaml") -> dict:
    return {"path": path, "sha256": "a" * 64, "bytes": 123}


def _record(status: str = "planned") -> dict:
    return {
        "schema_version": "1.0",
        "run_id": "grouped-seed42",
        "arm": "grouped",
        "seed": 42,
        "status": status,
        "timestamps": {
            "created_at_utc": "2026-08-03T00:00:00Z",
            "updated_at_utc": "2026-08-03T00:00:00Z",
        },
        "provenance": {
            "git_sha": "b" * 40,
            "git_dirty": False,
            "command": ["python", "-m", "pcb_defect.experiment", "train"],
            "environment": {
                "python": "3.11.9",
                "platform": "Linux-6.1-x86_64",
                "packages": {"ultralytics": "8.4.89"},
            },
        },
        "protocol": {
            "version": "paired-board-sensitivity-v1",
            "manifest": _ref("reports/protocol/paired_split_manifest.json"),
            "manifest_sha256": "c" * 64,
            "dataset_sha256": "d" * 64,
        },
        "training": {
            "config": _ref(),
            "resolved": {"epochs": 100, "imgsz": 640},
            "base_model": {
                "source": "https://example.invalid/v1/yolo26n.pt",
                "revision": "v1",
                "filename": "yolo26n.pt",
                "sha256": "e" * 64,
                "bytes": 123,
                "contract": _ref("inputs/base_model.yaml"),
            },
        },
        "artifacts": {},
        "metrics": {},
        "failure": None,
    }


def test_planned_run_record_satisfies_contract() -> None:
    validate_run_record(_record())


def test_complete_run_requires_hashed_checkpoint_and_validation_metrics() -> None:
    record = _record(status="complete")

    with pytest.raises(EvidenceError, match="best_checkpoint"):
        validate_run_record(record)

    record["artifacts"]["best_checkpoint"] = _ref("weights/grouped/seed42/best.pt")
    with pytest.raises(EvidenceError, match="validation"):
        validate_run_record(record)

    record["metrics"]["validation"] = _ref("runs/grouped/seed42/validation.json")
    validate_run_record(record)


def test_record_rejects_absolute_or_parent_traversal_artifact_paths() -> None:
    for unsafe in ("C:/Users/name/best.pt", "/content/best.pt", "../outside.pt"):
        record = _record()
        record["training"]["config"]["path"] = unsafe
        with pytest.raises(EvidenceError, match="portable relative path"):
            validate_run_record(record)


def test_artifact_ref_hashes_bytes_and_uses_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "run"
    artifact = root / "metrics" / "validation.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"map50": 0.5}\n', encoding="utf-8", newline="\n")

    ref = artifact_ref(artifact, relative_to=root)

    assert ref == {
        "path": "metrics/validation.json",
        "sha256": "1a9868a09d4de694ffd6616feff14f1ec6778a14d758a2452bc3edd893d42569",
        "bytes": 15,
    }


def test_write_run_record_is_canonical_and_newline_terminated(tmp_path: Path) -> None:
    destination = tmp_path / "run_record.json"

    write_run_record(_record(), destination)

    payload = destination.read_bytes()
    assert payload.endswith(b"\n")
    assert json.loads(payload) == _record()
