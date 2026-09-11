from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from pcb_defect.lobo import (
    LoboError,
    aggregate_folds,
    delta_statistics,
    main,
    promote_paired_package,
    write_lobo_summary,
)
from pcb_defect.result_package import create_verifiable_zip

MANIFEST = "d" * 64
GIT = "a" * 40


def _workspace(root: Path, *, board: str = "05", gate_extra: dict | None = None) -> list[Path]:
    """A minimal A100-style workspace with the members the promotion reads."""
    files = {
        "inputs/input_lock.json": {"git_sha": GIT, "manifest_sha256": MANIFEST},
        "gates/gate_report.json": {"passed": True, **(gate_extra or {})},
        "final/FINAL_TEST_STARTED.json": {"started": True},
        "final/deployment_selection.json": {"arm": "grouped", "seed": 42},
        "final/final_metrics.json": {
            "final_test_board": board,
            "manifest_sha256": MANIFEST,
            "git_sha": GIT,
            "aggregate": {},
        },
        "final/finalization_record.json": {"status": "complete"},
        "deployment/deployment_gate.json": {
            "passed": True,
            "command": ["/content/venv/bin/python", "-m", "pcb_defect.deployment"],
            "artifacts": {"onnx_sha256": "b" * 64},
        },
        "deployment/model_contract.candidate.json": {"status": "passed"},
        "runs/grouped/seed42/inputs/paired_split_manifest.json": {
            "manifest_sha256": MANIFEST,
            "board_roles": {"final_test_and_exposure": board},
        },
    }
    for name, document in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    (root / "deployment" / "calibration.yaml").write_text("train: x\n", encoding="utf-8")
    (root / "deployment" / "best.onnx").write_bytes(b"onnx-bytes")
    return [Path(name) for name in files] + [
        Path("deployment/calibration.yaml"),
        Path("deployment/best.onnx"),
    ]


def _package(tmp_path: Path, **kwargs: object) -> Path:
    root = tmp_path / "workspace"
    files = _workspace(root, **kwargs)
    package = tmp_path / "packages" / "paired-results-a100-abcdef123456-board05.zip"
    create_verifiable_zip(root, files, package)
    return package


def test_promote_writes_path_free_public_evidence_bound_to_the_fold(tmp_path: Path) -> None:
    package = _package(tmp_path)
    output = tmp_path / "reports" / "lobo" / "board05"

    receipt = promote_paired_package(
        package, board="05", expected_manifest_sha256=MANIFEST, output=output
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "deployment_gate.public.json",
        "deployment_selection.json",
        "final_metrics.json",
        "finalization_record.json",
        "gate_report.json",
        "input_lock.json",
        "package_manifest.json",
        "result_package_receipt.json",
    ]
    gate = json.loads((output / "deployment_gate.public.json").read_text(encoding="utf-8"))
    embedded = json.loads((output / "package_manifest.json").read_text(encoding="utf-8"))
    raw = next(row for row in embedded["files"] if row["path"] == "deployment/deployment_gate.json")
    assert "command" not in gate
    assert gate["_provenance"] == {
        "source_entry": "deployment/deployment_gate.json",
        "source_bytes": raw["bytes"],
        "source_sha256": raw["sha256"],
        "removed_fields": ["command"],
    }
    written_receipt = json.loads(
        (output / "result_package_receipt.json").read_text(encoding="utf-8")
    )
    assert written_receipt == receipt
    assert receipt["package"]["name"] == package.name
    assert receipt["package"]["sha256"] == hashlib.sha256(package.read_bytes()).hexdigest()
    assert receipt["held_out_board"] == "05"
    assert receipt["manifest_sha256"] == MANIFEST
    assert receipt["package_manifest"]["verified_entries"] == len(embedded["files"])
    for path in output.iterdir():
        assert "/content/" not in path.read_text(encoding="utf-8")


def test_promote_fails_closed_on_wrong_board_hash_paths_or_existing_output(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)

    with pytest.raises(LoboError, match="holds out board"):
        promote_paired_package(
            package, board="07", expected_manifest_sha256=MANIFEST, output=tmp_path / "out1"
        )
    with pytest.raises(LoboError, match="differs from the fold registry"):
        promote_paired_package(
            package, board="05", expected_manifest_sha256="e" * 64, output=tmp_path / "out2"
        )
    occupied = tmp_path / "out3"
    occupied.mkdir()
    (occupied / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LoboError, match="non-empty evidence directory"):
        promote_paired_package(
            package, board="05", expected_manifest_sha256=MANIFEST, output=occupied
        )

    leaky = _package(tmp_path / "leaky", gate_extra={"log": "/content/drive/MyDrive/x.log"})
    with pytest.raises(LoboError, match="private path token"):
        promote_paired_package(
            leaky, board="05", expected_manifest_sha256=MANIFEST, output=tmp_path / "out4"
        )


def _metrics(board: str, grouped: float, leaky: float) -> dict:
    def arm(value: float) -> dict:
        return {
            "map50": {"n_seeds": 3, "mean": value, "std": 0.05},
            "map50_95": {"n_seeds": 3, "mean": value / 2, "std": 0.02},
        }

    return {
        "final_test_board": board,
        "manifest_sha256": f"{board}{board}".ljust(64, "0"),
        "aggregate": {
            "by_arm": {"grouped": arm(grouped), "leaky_control": arm(leaky)},
            "paired_bootstrap": {
                "leaky_minus_grouped_f1": {"mean_delta": 0.2, "ci95_low": 0.1, "ci95_high": 0.3}
            },
        },
    }


def _lobo_repo(tmp_path: Path, results: dict[str, tuple[float, float] | None]) -> Path:
    repo = tmp_path / "repo"
    folds = []
    for board, values in results.items():
        evidence = f"reports/lobo/board{board}"
        folds.append(
            {
                "board": board,
                "config": f"configs/lobo/board{board}.yaml",
                "artifacts": f"reports/protocol/lobo/board{board}",
                "evidence": evidence,
                "manifest_sha256": f"{board}{board}".ljust(64, "0"),
                "run": board != "08",
            }
        )
        if values is not None:
            path = repo / evidence / "final_metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_metrics(board, *values)), encoding="utf-8")
    registry = repo / "configs" / "lobo" / "folds.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        yaml.safe_dump({"schema_version": "1.0", "parent_board": "08", "folds": folds}),
        encoding="utf-8",
    )
    prereg = repo / "docs" / "lobo-preregistration.md"
    prereg.parent.mkdir(parents=True, exist_ok=True)
    prereg.write_text("# pre-registration\n", encoding="utf-8")
    return repo


def test_aggregate_reports_every_board_and_the_pre_registered_statistics(tmp_path: Path) -> None:
    repo = _lobo_repo(tmp_path, {"05": (0.60, 0.80), "08": (0.6330, 0.8456), "12": (0.70, 0.75)})

    summary = aggregate_folds(repo, resamples=500)

    assert summary["status"] == "complete"
    assert summary["wording"] == "all_positive"
    assert [row["board"] for row in summary["boards"]] == ["05", "08", "12"]
    deltas = [row["delta_map50"] for row in summary["boards"]]
    assert deltas == pytest.approx([0.20, 0.2126, 0.05])
    stats = summary["statistics"]["delta_map50"]
    assert stats["n_boards"] == 3
    assert stats["mean"] == pytest.approx(sum(deltas) / 3)
    assert stats["min"] == pytest.approx(0.05)
    assert stats["max"] == pytest.approx(0.2126)
    assert stats["n_positive"] == 3
    ci = stats["board_bootstrap"]
    assert ci["n_resamples"] == 500 and ci["seed"] == 20_260_803
    assert stats["min"] <= ci["ci95_low"] <= ci["ci95_high"] <= stats["max"]
    assert (
        summary["preregistration"]["sha256"] == hashlib.sha256(b"# pre-registration\n").hexdigest()
    )
    assert aggregate_folds(repo, resamples=500) == summary  # deterministic

    summary_path, readme_path = write_lobo_summary(repo, summary)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    readme = readme_path.read_text(encoding="utf-8")
    assert "| 08 | 0.6330 ± 0.0500 | 0.8456 ± 0.0500 | +21.3 | +10.6 |" in readme
    assert "**all_positive**" in readme


def test_aggregate_handles_mixed_directions_incomplete_folds_and_mismatches(
    tmp_path: Path,
) -> None:
    repo = _lobo_repo(tmp_path, {"05": (0.60, 0.55), "08": (0.6330, 0.8456), "12": None})

    with pytest.raises(LoboError, match="missing final metrics for boards \\['12'\\]"):
        aggregate_folds(repo, resamples=200)

    summary = aggregate_folds(repo, incomplete={"12": "Colab session lost"}, resamples=200)
    assert summary["status"] == "incomplete"
    assert summary["wording"] == "incomplete"
    assert summary["boards"][2] == {
        "board": "12",
        "status": "incomplete",
        "reason": "Colab session lost",
    }
    assert summary["statistics"]["delta_map50"]["n_positive"] == 1

    complete = _lobo_repo(tmp_path / "complete", {"05": (0.60, 0.55), "08": (0.6330, 0.8456)})
    assert aggregate_folds(complete, resamples=200)["wording"] == "direction_varies"

    wrong = _lobo_repo(tmp_path / "wrong", {"05": (0.60, 0.80)})
    metrics = wrong / "reports" / "lobo" / "board05" / "final_metrics.json"
    metrics.write_text(json.dumps(_metrics("07", 0.60, 0.80)), encoding="utf-8")
    with pytest.raises(LoboError, match="different board"):
        aggregate_folds(wrong, resamples=200)


def test_delta_statistics_single_board_has_no_standard_deviation() -> None:
    stats = delta_statistics([0.2], resamples=100, seed=1)

    assert stats["sample_std"] is None
    assert stats["board_bootstrap"]["ci95_low"] == stats["board_bootstrap"]["ci95_high"] == 0.2
    assert delta_statistics([], resamples=100, seed=1) == {"n_boards": 0}


def test_cli_promote_and_aggregate_round_trip(tmp_path: Path) -> None:
    repo = _lobo_repo(tmp_path, {"05": None, "08": (0.6330, 0.8456)})
    package = _package(tmp_path / "pkg")
    registry = yaml.safe_load((repo / "configs" / "lobo" / "folds.yaml").read_text("utf-8"))
    registry["folds"][0]["manifest_sha256"] = MANIFEST
    (repo / "configs" / "lobo" / "folds.yaml").write_text(yaml.safe_dump(registry), "utf-8")

    assert main(["promote", "--repo", str(repo), "--package", str(package), "--board", "05"]) == 0
    assert (repo / "reports" / "lobo" / "board05" / "final_metrics.json").is_file()

    with pytest.raises(SystemExit, match="BOARD=REASON"):
        main(["aggregate", "--repo", str(repo), "--incomplete", "oops"])
    with pytest.raises(LoboError):
        main(["aggregate", "--repo", str(repo)])  # board05 metrics lack aggregate fields
