"""The committed six-board replication evidence must stay bound to its packages and analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pcb_defect.lobo import aggregate_folds, load_registry

ROOT = Path(__file__).resolve().parent.parent
LOBO = ROOT / "reports" / "lobo"
RUNNER_SNAPSHOT = "0a82c89b3037c935d1aa6f2da9658c97fbdcd252"
PACKAGE_SHA256 = {
    "05": "a94818e6191e4e642816aa27abb4eac4b63bce1c9c403e997f216e7b4cd2058b",
    "07": "25a509d4e69c43017b27619c30eeace4ec004021b54059190875b6dc5a4ccb64",
    "09": "f0b51b27b66a721b59273ff693e1ef759b878f752072bd61527ebe0758bfc853",
    "11": "c702bf1c562a67fb554c817f007f335fc3f529f85bcb03ad33abb88fa233064a",
    "12": "39f837faea558ba7ef6103a3b4be0952a60bb85fe00484d813fa38ba5b78ab99",
}
EXPECTED_DELTA_MAP50 = {
    "05": 0.1942,
    "07": 0.0677,
    "08": 0.2126,
    "09": 0.0731,
    "11": 0.1996,
    "12": 0.0185,
}
PUBLIC_FILES = {
    "deployment_gate.public.json",
    "deployment_selection.json",
    "final_metrics.json",
    "finalization_record.json",
    "gate_report.json",
    "input_lock.json",
    "package_manifest.json",
    "result_package_receipt.json",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_promoted_fold_is_bound_to_its_package_registry_and_snapshot() -> None:
    registry = {fold["board"]: fold for fold in load_registry(ROOT)["folds"]}

    for board, package_sha256 in PACKAGE_SHA256.items():
        evidence = ROOT / registry[board]["evidence"]
        assert {path.name for path in evidence.iterdir()} == PUBLIC_FILES
        receipt = _read_json(evidence / "result_package_receipt.json")
        assert receipt["package"]["sha256"] == package_sha256
        assert receipt["package"]["name"] == f"paired-results-a100-0a82c89b3037-board{board}.zip"
        assert receipt["held_out_board"] == board
        assert receipt["manifest_sha256"] == registry[board]["manifest_sha256"]
        assert receipt["source_git_sha"] == RUNNER_SNAPSHOT
        assert receipt["verification"] == {"sidecar_match": True, "internal_manifest_passed": True}
        lock = _read_json(evidence / "input_lock.json")
        assert lock["git_sha"] == RUNNER_SNAPSHOT
        assert lock["manifest_sha256"] == registry[board]["manifest_sha256"]
        metrics = _read_json(evidence / "final_metrics.json")
        assert metrics["final_test_board"] == board
        assert metrics["manifest_sha256"] == registry[board]["manifest_sha256"]
        assert metrics["status"] == "complete"
        assert set(metrics["runs"]) == {
            f"{arm}-seed{seed}" for arm in ("grouped", "leaky_control") for seed in (42, 43, 44)
        }
        assert _read_json(evidence / "gate_report.json")["passed"] is True
        gate = _read_json(evidence / "deployment_gate.public.json")
        assert "command" not in gate
        assert gate["_provenance"]["source_entry"] == "deployment/deployment_gate.json"
        manifest_rows = {
            row["path"]: row for row in _read_json(evidence / "package_manifest.json")["files"]
        }
        assert (
            gate["_provenance"]["source_sha256"]
            == manifest_rows["deployment/deployment_gate.json"]["sha256"]
        )
        assert len(manifest_rows) == 52
        for path in evidence.iterdir():
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in ("/content/", "/root/", "MyDrive", "C:\\"))


def test_committed_summary_matches_the_pre_registered_analysis_recomputed_from_evidence() -> None:
    summary = _read_json(LOBO / "summary.json")

    assert summary["status"] == "complete"
    assert summary["wording"] == "all_positive"
    assert summary["n_boards_total"] == summary["n_boards_complete"] == 6
    assert [row["board"] for row in summary["boards"]] == ["05", "07", "08", "09", "11", "12"]
    prereg = ROOT / "docs" / "lobo-preregistration.md"
    assert summary["preregistration"]["sha256"] == hashlib.sha256(prereg.read_bytes()).hexdigest()
    for row in summary["boards"]:
        assert row["status"] == "complete"
        assert row["n_seeds"] == 3
        assert row["delta_map50"] == pytest.approx(EXPECTED_DELTA_MAP50[row["board"]], abs=5e-5)
        assert row["delta_map50"] > 0
    stats = summary["statistics"]["delta_map50"]
    assert stats["n_boards"] == 6
    assert stats["n_positive"] == 6
    assert stats["mean"] == pytest.approx(0.1276, abs=5e-5)
    assert stats["sample_std"] == pytest.approx(0.0840, abs=5e-5)
    assert stats["min"] == pytest.approx(0.0185, abs=5e-5)
    assert stats["max"] == pytest.approx(0.2126, abs=5e-5)
    assert stats["board_bootstrap"]["n_resamples"] == 10_000
    assert stats["board_bootstrap"]["seed"] == 20_260_803
    assert stats["board_bootstrap"]["ci95_low"] == pytest.approx(0.0651, abs=5e-5)
    assert stats["board_bootstrap"]["ci95_high"] == pytest.approx(0.1832, abs=5e-5)

    assert aggregate_folds(ROOT) == summary
