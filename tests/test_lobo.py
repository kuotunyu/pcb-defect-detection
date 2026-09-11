from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pcb_defect.constants import CLASSES
from pcb_defect.data_prep.paired import discover_converted_samples, write_protocol_artifacts
from pcb_defect.lobo import (
    LoboError,
    build_fold_registry,
    eligible_boards,
    load_registry,
    main,
    verify_folds,
    verify_registry_consistency,
    write_folds,
)
from pcb_defect.paired_protocol import PairedProtocolConfig, build_paired_protocol

ROOT = Path(__file__).resolve().parent.parent
BOARDS = ("01", "04", "05", "06", "07", "08", "09", "10", "11", "12")


def _dataset(root: Path) -> Path:
    """Ten boards: 01 = 4/class (validation), 10 = 1/class, 06 = one extra image, others 2/class."""
    for board in BOARDS:
        count = {"01": 4, "10": 1}.get(board, 2)
        for cls_id, class_name in enumerate(CLASSES):
            extra = 1 if board == "06" and class_name == "short" else 0
            for index in range(1, count + extra + 1):
                stem = f"{board}_{class_name}_{index:02d}"
                image = root / "images" / "source" / f"{stem}.jpg"
                label = root / "labels" / "source" / f"{stem}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"image:{stem}".encode())
                label.write_text(f"{cls_id} 0.5 0.5 0.1 0.1\n", encoding="ascii")
    return root


def _parent_protocol_mapping() -> dict[str, object]:
    return {
        "final_board": "08",
        "validation_board": "01",
        "final_per_class": 1,
        "exposure_per_class": 1,
        "validation_per_class": 2,
        "calibration_per_class": 2,
        "protocol_seed": 42,
        "expected_boards": 10,
        "training_seeds": [42, 43, 44],
    }


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal repo with a frozen parent protocol whose artifacts are committed."""
    dataset = _dataset(tmp_path / "dataset")
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    samples = discover_converted_samples(dataset)
    protocol = build_paired_protocol(samples, PairedProtocolConfig(**_parent_protocol_mapping()))
    write_protocol_artifacts(protocol, repo / "reports" / "protocol")
    (repo / "configs" / "paired_protocol.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol_version": "paired-board-sensitivity-v1",
                "selection": {"legacy_test_board_excluded": "04"},
                "protocol": _parent_protocol_mapping(),
                "frozen_hashes": {
                    "dataset_sha256": protocol.dataset_sha256,
                    "manifest_sha256": protocol.manifest_sha256,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return repo, dataset


def test_eligibility_rule_excludes_validation_legacy_uneven_and_small_boards(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    samples = discover_converted_samples(dataset)

    eligible, excluded = eligible_boards(
        samples, PairedProtocolConfig(**_parent_protocol_mapping()), frozenset({"04"})
    )

    assert eligible == ["05", "07", "08", "09", "11", "12"]
    assert excluded["01"] == "validation and calibration board"
    assert excluded["04"].startswith("legacy-excluded board")
    assert "class short has 3 images" in excluded["06"]
    assert "protocol requires exactly 2" in excluded["06"]
    assert "has 1 images" in excluded["10"]


def test_fold_registry_reuses_the_parent_board_and_writes_frozen_folds(tmp_path: Path) -> None:
    repo, dataset = _repo(tmp_path)

    registry_path = write_folds(repo, dataset)
    registry = load_registry(repo)

    assert registry_path == repo / "configs" / "lobo" / "folds.yaml"
    assert [fold["board"] for fold in registry["folds"]] == ["05", "07", "08", "09", "11", "12"]
    parent = next(fold for fold in registry["folds"] if fold["board"] == "08")
    assert parent["run"] is False
    assert parent["config"] == "configs/paired_protocol.yaml"
    assert parent["evidence"] == "reports/paired_a100"
    assert parent["manifest_sha256"] == registry["parent_manifest_sha256"]
    for fold in registry["folds"]:
        if fold["board"] == "08":
            continue
        assert fold["run"] is True
        spec = yaml.safe_load((repo / fold["config"]).read_text(encoding="utf-8"))
        assert spec["protocol"]["final_board"] == fold["board"]
        assert spec["protocol"]["validation_board"] == "01"
        assert spec["frozen_hashes"]["manifest_sha256"] == fold["manifest_sha256"]
        manifest = json.loads(
            (repo / fold["artifacts"] / "paired_split_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["manifest_sha256"] == fold["manifest_sha256"]
        assert manifest["board_roles"]["final_test_and_exposure"] == fold["board"]
        assert (
            manifest["counts"]["grouped_train"]["images"]
            == manifest["counts"]["leaky_train"]["images"]
        )
    assert verify_registry_consistency(repo)["parent_board"] == "08"
    assert verify_folds(repo, dataset)["parent_board"] == "08"


def test_fold_verification_fails_closed_on_tampered_manifest_or_registry(tmp_path: Path) -> None:
    repo, dataset = _repo(tmp_path)
    write_folds(repo, dataset)
    manifest = repo / "reports" / "protocol" / "lobo" / "board05" / "paired_split_manifest.json"
    original = manifest.read_bytes()

    manifest.write_bytes(original.replace(b'"05_', b'"55_', 1))
    with pytest.raises(LoboError, match="differ from the regenerated protocol"):
        verify_folds(repo, dataset)
    manifest.write_bytes(original)

    registry = repo / "configs" / "lobo" / "folds.yaml"
    registry.write_text(
        registry.read_text(encoding="utf-8").replace("- board: '07'", "- board: '06'"),
        encoding="utf-8",
    )
    with pytest.raises(LoboError, match="differ from the eligibility rule"):
        verify_registry_consistency(repo)


def test_fold_registry_rejects_a_parent_that_does_not_reproduce_its_frozen_hash(
    tmp_path: Path,
) -> None:
    repo, dataset = _repo(tmp_path)
    config = repo / "configs" / "paired_protocol.yaml"
    spec = yaml.safe_load(config.read_text(encoding="utf-8"))
    spec["frozen_hashes"]["manifest_sha256"] = "0" * 64
    config.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    with pytest.raises(LoboError, match="does not reproduce its frozen manifest"):
        build_fold_registry(repo, discover_converted_samples(dataset))


def test_folds_cli_writes_then_verifies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo, dataset = _repo(tmp_path)

    assert main(["folds", "--repo", str(repo), "--dataset", str(dataset), "--write"]) == 0
    assert main(["folds", "--repo", str(repo)]) == 0

    output = capsys.readouterr().out
    assert "pending Colab runs=05, 07, 09, 11, 12" in output


def test_committed_fold_registry_is_consistent_with_the_parent_protocol() -> None:
    registry = verify_registry_consistency(ROOT)

    assert [fold["board"] for fold in registry["folds"]] == ["05", "07", "08", "09", "11", "12"]
    assert registry["parent_board"] == "08"
    assert registry["validation_board"] == "01"
    assert registry["legacy_excluded"] == ["04"]
    assert set(registry["excluded"]) == {"01", "04", "06", "10"}
    assert "class short has 11 images" in registry["excluded"]["06"]
    assert [fold["run"] for fold in registry["folds"]] == [True, True, False, True, True, True]
