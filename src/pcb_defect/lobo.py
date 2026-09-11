"""Leave-one-board-out replication of the paired board-leakage experiment.

The parent protocol (``configs/paired_protocol.yaml``) holds out Board 08. This module applies one
eligibility rule to every board, derives one frozen protocol per eligible board with only
``final_board`` changed, and records the result in ``configs/lobo/folds.yaml``. The parent board
keeps its already-published evidence; every other fold is trained on Colab with the unchanged
pipeline and promoted here.

Subcommands:

* ``folds``     generate (``--write``) or verify the per-board configs, manifests, and registry
* ``promote``   verify a returned A100 package and derive path-free public fold evidence
* ``aggregate`` compute the pre-registered multi-board endpoints into ``reports/lobo/``
* ``handoff``   create the Colab handoff directory for every fold marked ``run: true``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from pcb_defect.constants import CLASSES
from pcb_defect.data_prep.paired import (
    _load_spec,
    discover_converted_samples,
    write_protocol_artifacts,
)
from pcb_defect.paired_protocol import (
    PROTOCOL_VERSION,
    PairedProtocol,
    PairedProtocolConfig,
    ProtocolSample,
    build_paired_protocol,
)

REGISTRY_RELATIVE = Path("configs/lobo/folds.yaml")
PARENT_CONFIG_RELATIVE = Path("configs/paired_protocol.yaml")
PARENT_ARTIFACTS_RELATIVE = Path("reports/protocol")
PARENT_EVIDENCE_RELATIVE = Path("reports/paired_a100")
FOLD_CONFIG_DIR = Path("configs/lobo")
FOLD_ARTIFACTS_DIR = Path("reports/protocol/lobo")
FOLD_EVIDENCE_DIR = Path("reports/lobo")
MANIFEST_NAME = "paired_split_manifest.json"
ELIGIBILITY_RULE = (
    "every board with exactly final_per_class + exposure_per_class images in every class, "
    "excluding the validation board and the legacy-excluded board"
)


class LoboError(RuntimeError):
    """The fold registry, a returned package, or the evidence set violates the frozen contract."""


# --------------------------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------------------------


def eligible_boards(
    samples: Iterable[ProtocolSample],
    config: PairedProtocolConfig,
    legacy_excluded: frozenset[str],
) -> tuple[list[str], dict[str, str]]:
    """Apply the single eligibility rule and explain every excluded board."""
    per_board: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        per_board[sample.board_id][sample.class_name] += 1
    required = config.final_per_class + config.exposure_per_class
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for board in sorted(per_board):
        counts = per_board[board]
        if board == config.validation_board:
            excluded[board] = "validation and calibration board"
            continue
        if board in legacy_excluded:
            excluded[board] = "legacy-excluded board (selection.legacy_test_board_excluded)"
            continue
        problems = [
            f"class {class_name} has {counts.get(class_name, 0)} images"
            for class_name in CLASSES
            if counts.get(class_name, 0) != required
        ]
        if problems:
            excluded[board] = f"{'; '.join(problems)}; protocol requires exactly {required}"
            continue
        eligible.append(board)
    return eligible, excluded


def _fold_relative_paths(board: str) -> dict[str, str]:
    return {
        "config": (FOLD_CONFIG_DIR / f"board{board}.yaml").as_posix(),
        "artifacts": (FOLD_ARTIFACTS_DIR / f"board{board}").as_posix(),
        "evidence": (FOLD_EVIDENCE_DIR / f"board{board}").as_posix(),
    }


def fold_config_document(parent_spec: dict[str, Any], board: str, protocol: PairedProtocol) -> dict:
    """Return the YAML document for one fold: the parent protocol with ``final_board`` swapped."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "lobo": {
            "parent_config": PARENT_CONFIG_RELATIVE.as_posix(),
            "held_out_board": board,
            "eligibility": ELIGIBILITY_RULE,
        },
        "protocol": {**parent_spec["protocol"], "final_board": board},
        "frozen_hashes": {
            "dataset_sha256": protocol.dataset_sha256,
            "manifest_sha256": protocol.manifest_sha256,
        },
    }


def _legacy_excluded(parent_spec: dict[str, Any]) -> frozenset[str]:
    selection = parent_spec.get("selection") or {}
    value = selection.get("legacy_test_board_excluded")
    return frozenset({str(value)}) if value is not None else frozenset()


def build_fold_registry(
    repo: Path, samples: list[ProtocolSample]
) -> tuple[dict[str, Any], dict[str, PairedProtocol]]:
    """Derive the registry and the protocols of every fold that needs new artifacts."""
    repo = repo.resolve()
    parent_spec = _load_spec(repo / PARENT_CONFIG_RELATIVE)
    parent_config = PairedProtocolConfig(**parent_spec["protocol"])
    parent_manifest_sha256 = parent_spec["frozen_hashes"]["manifest_sha256"]
    legacy = _legacy_excluded(parent_spec)
    eligible, excluded = eligible_boards(samples, parent_config, legacy)
    if parent_config.final_board not in eligible:
        raise LoboError("the parent protocol board is not eligible under the fold rule")
    folds: list[dict[str, Any]] = []
    protocols: dict[str, PairedProtocol] = {}
    for board in eligible:
        config = PairedProtocolConfig(**{**parent_spec["protocol"], "final_board": board})
        protocol = build_paired_protocol(samples, config)
        if board == parent_config.final_board:
            if protocol.manifest_sha256 != parent_manifest_sha256:
                raise LoboError(
                    "regenerating the parent fold does not reproduce its frozen manifest"
                )
            folds.append(
                {
                    "board": board,
                    "config": PARENT_CONFIG_RELATIVE.as_posix(),
                    "artifacts": PARENT_ARTIFACTS_RELATIVE.as_posix(),
                    "evidence": PARENT_EVIDENCE_RELATIVE.as_posix(),
                    "manifest_sha256": protocol.manifest_sha256,
                    "run": False,
                }
            )
            continue
        folds.append(
            {
                "board": board,
                **_fold_relative_paths(board),
                "manifest_sha256": protocol.manifest_sha256,
                "run": True,
            }
        )
        protocols[board] = protocol
    registry = {
        "schema_version": "1.0",
        "parent_config": PARENT_CONFIG_RELATIVE.as_posix(),
        "parent_board": parent_config.final_board,
        "parent_manifest_sha256": parent_manifest_sha256,
        "dataset_sha256": parent_spec["frozen_hashes"]["dataset_sha256"],
        "validation_board": parent_config.validation_board,
        "legacy_excluded": sorted(legacy),
        "eligibility": ELIGIBILITY_RULE,
        "excluded": excluded,
        "folds": folds,
    }
    return registry, protocols


class _BoardSafeDumper(yaml.SafeDumper):
    """Quote digit-only strings such as board ids so no YAML parser reads them as integers."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "'" if data.isdigit() else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BoardSafeDumper.add_representer(str, _represent_str)


def _yaml_text(document: dict[str, Any]) -> str:
    return yaml.dump(
        document, Dumper=_BoardSafeDumper, sort_keys=False, allow_unicode=True, width=100
    )


def _manifest_bytes(protocol: PairedProtocol) -> bytes:
    return (json.dumps(protocol.to_manifest(), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_folds(repo: Path, dataset: Path) -> Path:
    """Write every fold config, manifest set, and the registry (regeneration is deterministic)."""
    repo = repo.resolve()
    samples = discover_converted_samples(dataset)
    registry, protocols = build_fold_registry(repo, samples)
    parent_spec = _load_spec(repo / PARENT_CONFIG_RELATIVE)
    for fold in registry["folds"]:
        if not fold["run"]:
            continue
        board = fold["board"]
        config_path = repo / fold["config"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            _yaml_text(fold_config_document(parent_spec, board, protocols[board])),
            encoding="utf-8",
            newline="\n",
        )
        write_protocol_artifacts(protocols[board], repo / fold["artifacts"])
    registry_path = repo / REGISTRY_RELATIVE
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(_yaml_text(registry), encoding="utf-8", newline="\n")
    return registry_path


def verify_folds(repo: Path, dataset: Path) -> dict[str, Any]:
    """Regenerate every fold from the dataset and require byte-identical committed files."""
    repo = repo.resolve()
    samples = discover_converted_samples(dataset)
    registry, protocols = build_fold_registry(repo, samples)
    parent_spec = _load_spec(repo / PARENT_CONFIG_RELATIVE)
    mismatches: list[str] = []
    registry_path = repo / REGISTRY_RELATIVE
    if _read_text(registry_path) != _yaml_text(registry):
        mismatches.append(REGISTRY_RELATIVE.as_posix())
    for fold in registry["folds"]:
        if not fold["run"]:
            continue
        board = fold["board"]
        expected_config = _yaml_text(fold_config_document(parent_spec, board, protocols[board]))
        if _read_text(repo / fold["config"]) != expected_config:
            mismatches.append(fold["config"])
        artifacts = repo / fold["artifacts"]
        if _read_bytes(artifacts / MANIFEST_NAME) != _manifest_bytes(protocols[board]):
            mismatches.append(f"{fold['artifacts']}/{MANIFEST_NAME}")
    if mismatches:
        raise LoboError(f"fold files differ from the regenerated protocol: {mismatches}")
    return registry


def load_registry(repo: Path) -> dict[str, Any]:
    path = repo.resolve() / REGISTRY_RELATIVE
    try:
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LoboError(f"fold registry is missing or malformed: {path}") from exc
    if not isinstance(registry, dict) or registry.get("schema_version") != "1.0":
        raise LoboError("fold registry schema is unsupported")
    folds = registry.get("folds")
    if not isinstance(folds, list) or not folds:
        raise LoboError("fold registry lists no folds")
    return registry


def verify_registry_consistency(repo: Path) -> dict[str, Any]:
    """Check the committed registry against committed configs and manifests without a dataset."""
    repo = repo.resolve()
    registry = load_registry(repo)
    parent_spec = _load_spec(repo / registry["parent_config"])
    parent_config = PairedProtocolConfig(**parent_spec["protocol"])
    parent_manifest = _read_json(repo / PARENT_ARTIFACTS_RELATIVE / MANIFEST_NAME)
    samples = [ProtocolSample(**row) for row in parent_manifest["dataset"]["samples"]]
    eligible, excluded = eligible_boards(samples, parent_config, _legacy_excluded(parent_spec))
    boards = [fold["board"] for fold in registry["folds"]]
    if boards != eligible:
        raise LoboError(f"registry boards {boards} differ from the eligibility rule {eligible}")
    if registry.get("excluded") != excluded:
        raise LoboError("registry exclusion reasons differ from the eligibility rule")
    if registry.get("parent_manifest_sha256") != parent_spec["frozen_hashes"]["manifest_sha256"]:
        raise LoboError("registry parent manifest hash differs from the parent protocol")
    for fold in registry["folds"]:
        spec = _load_spec(repo / fold["config"])
        if spec["protocol"].get("final_board") != fold["board"]:
            raise LoboError(f"fold {fold['board']} config holds out a different board")
        if spec["frozen_hashes"]["manifest_sha256"] != fold["manifest_sha256"]:
            raise LoboError(f"fold {fold['board']} config hash differs from the registry")
        if spec["frozen_hashes"]["dataset_sha256"] != registry["dataset_sha256"]:
            raise LoboError(f"fold {fold['board']} dataset hash differs from the registry")
        manifest_path = repo / fold["artifacts"] / MANIFEST_NAME
        manifest_bytes = _read_bytes(manifest_path)
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if manifest.get("manifest_sha256") != fold["manifest_sha256"]:
            raise LoboError(f"fold {fold['board']} manifest hash differs from the registry")
        if manifest["board_roles"]["final_test_and_exposure"] != fold["board"]:
            raise LoboError(f"fold {fold['board']} manifest holds out a different board")
        sidecar = _read_text(manifest_path.with_suffix(".sha256"))
        expected_sidecar = f"{hashlib.sha256(manifest_bytes).hexdigest()}  {MANIFEST_NAME}\n"
        if sidecar != expected_sidecar:
            raise LoboError(f"fold {fold['board']} manifest sidecar differs from the manifest")
        expected_run = fold["board"] != registry["parent_board"]
        if fold["run"] is not expected_run:
            raise LoboError(f"fold {fold['board']} run flag is inconsistent with the parent board")
    return registry


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoboError(f"JSON evidence is missing or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise LoboError(f"JSON evidence must be an object: {path}")
    return value


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    folds = subparsers.add_parser("folds", help="generate or verify the fold registry")
    folds.add_argument("--repo", type=Path, default=Path.cwd())
    folds.add_argument("--dataset", type=Path, default=None)
    folds.add_argument("--write", action="store_true", help="write fold files (requires --dataset)")

    args = parser.parse_args(argv)
    if args.command == "folds":
        return _folds_command(args.repo, args.dataset, args.write)
    raise AssertionError(f"unknown command: {args.command}")


def _folds_command(repo: Path, dataset: Path | None, write: bool) -> int:
    if write:
        if dataset is None:
            raise SystemExit("--write requires --dataset")
        registry_path = write_folds(repo, dataset)
        print(f"fold registry written: {registry_path}")
    registry = verify_registry_consistency(repo)
    if dataset is not None:
        verify_folds(repo, dataset)
    boards = ", ".join(fold["board"] for fold in registry["folds"])
    pending = ", ".join(fold["board"] for fold in registry["folds"] if fold["run"])
    print(f"folds verified: boards={boards}; pending Colab runs={pending or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
