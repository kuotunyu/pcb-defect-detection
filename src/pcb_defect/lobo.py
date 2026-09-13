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
import math
import os
import random
import shutil
import statistics
import subprocess
import tempfile
import zipfile
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
from pcb_defect.handoff import (
    HandoffError,
    _remove_readonly,
    create_clean_bundle,
    project_handoff_metadata,
    render_notebook,
)
from pcb_defect.paired_protocol import (
    PROTOCOL_VERSION,
    PairedProtocol,
    PairedProtocolConfig,
    ProtocolSample,
    build_paired_protocol,
)
from pcb_defect.result_package import PackageError, verify_verifiable_zip

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
# promote
# --------------------------------------------------------------------------------------------

PUBLIC_MEMBERS = (
    "inputs/input_lock.json",
    "gates/gate_report.json",
    "final/deployment_selection.json",
    "final/final_metrics.json",
    "final/finalization_record.json",
)
DEPLOYMENT_GATE_MEMBER = "deployment/deployment_gate.json"
FOLD_MANIFEST_MEMBER = "runs/grouped/seed42/inputs/paired_split_manifest.json"
EMBEDDED_MANIFEST_MEMBER = "package_manifest.json"
FORBIDDEN_TOKENS = ("/content/", "/root/", "MyDrive", "C:\\")
PREREGISTRATION_RELATIVE = Path("docs/lobo-preregistration.md")
SUMMARY_RELATIVE = FOLD_EVIDENCE_DIR / "summary.json"
SUMMARY_README_RELATIVE = FOLD_EVIDENCE_DIR / "README.md"


def promote_paired_package(
    package: Path, *, board: str, expected_manifest_sha256: str, output: Path
) -> dict[str, Any]:
    """Verify a returned A100 package for one fold and write its path-free public evidence."""
    package = package.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise LoboError(f"refusing to write into a non-empty evidence directory: {output}")
    try:
        manifest = verify_verifiable_zip(package)
    except PackageError as exc:
        raise LoboError(f"returned package failed verification: {exc}") from exc
    rows = {row["path"]: row for row in manifest["files"]}
    outputs: dict[str, bytes] = {}
    with zipfile.ZipFile(package) as archive:

        def member(name: str) -> bytes:
            if name not in rows:
                raise LoboError(f"returned package lacks the member {name}")
            return archive.read(name)

        fold_manifest = _member_json(member(FOLD_MANIFEST_MEMBER), FOLD_MANIFEST_MEMBER)
        held_out = fold_manifest.get("board_roles", {}).get("final_test_and_exposure")
        if held_out != board:
            raise LoboError(f"package holds out board {held_out!r}, expected {board!r}")
        if fold_manifest.get("manifest_sha256") != expected_manifest_sha256:
            raise LoboError("package protocol manifest hash differs from the fold registry")
        lock = _member_json(member("inputs/input_lock.json"), "inputs/input_lock.json")
        if lock.get("manifest_sha256") != expected_manifest_sha256:
            raise LoboError("package input lock manifest hash differs from the fold registry")
        metrics = _member_json(member("final/final_metrics.json"), "final/final_metrics.json")
        if metrics.get("final_test_board") != board:
            raise LoboError("package final metrics were evaluated on a different board")
        if metrics.get("manifest_sha256") != expected_manifest_sha256:
            raise LoboError("package final metrics manifest hash differs from the fold registry")
        if metrics.get("git_sha") != lock.get("git_sha"):
            raise LoboError("package final metrics and input lock disagree on the source commit")
        for name in PUBLIC_MEMBERS:
            outputs[Path(name).name] = member(name)
        gate = _member_json(member(DEPLOYMENT_GATE_MEMBER), DEPLOYMENT_GATE_MEMBER)
        public_gate = {key: value for key, value in gate.items() if key != "command"}
        public_gate["_provenance"] = {
            "source_entry": DEPLOYMENT_GATE_MEMBER,
            "source_bytes": rows[DEPLOYMENT_GATE_MEMBER]["bytes"],
            "source_sha256": rows[DEPLOYMENT_GATE_MEMBER]["sha256"],
            "removed_fields": ["command"],
        }
        outputs["deployment_gate.public.json"] = _json_bytes(public_gate)
        embedded = archive.read(EMBEDDED_MANIFEST_MEMBER)
        outputs[EMBEDDED_MANIFEST_MEMBER] = embedded
    receipt = {
        "schema_version": "1.0",
        "package": {
            "name": package.name,
            "bytes": package.stat().st_size,
            "sha256": manifest["package_sha256"],
            "sidecar": package.name + ".sha256",
        },
        "source_git_sha": lock["git_sha"],
        "held_out_board": board,
        "manifest_sha256": expected_manifest_sha256,
        "package_manifest": {
            "path": EMBEDDED_MANIFEST_MEMBER,
            "sha256": hashlib.sha256(embedded).hexdigest(),
            "verified_entries": len(rows),
            "failed_entries": 0,
        },
        "verification": {"sidecar_match": True, "internal_manifest_passed": True},
    }
    outputs["result_package_receipt.json"] = _json_bytes(receipt)
    for name, payload in outputs.items():
        text = payload.decode("utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                raise LoboError(f"public evidence {name} would contain a private path token")
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        with (output / name).open("xb") as handle:
            handle.write(payload)
    return receipt


def _member_json(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LoboError(f"package member {name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LoboError(f"package member {name} must be a JSON object")
    return value


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------------------------
# aggregate (pre-registered endpoints)
# --------------------------------------------------------------------------------------------

PRIMARY_ENDPOINT = (
    "per held-out board, the three-seed mean final-test mAP50 of the leaky-control arm minus "
    "that of the grouped arm; reported per board, as mean and sample standard deviation across "
    "boards, and with a percentile bootstrap over boards"
)
SECONDARY_ENDPOINTS = [
    "the same delta on mAP50-95",
    "per-board grouped mAP50 (board difficulty spread)",
    "the number of boards with a positive mAP50 delta",
    "per-board paired image-bootstrap F1 deltas from final_evaluation",
]
WORDING_RULE = {
    "all_positive": (
        "same-board sibling exposure increased final-test mAP50 on every held-out board"
    ),
    "direction_varies": "the direction of the exposure effect varies across held-out boards",
    "incomplete": "the multi-board replication is incomplete; no cross-board claim is made",
}
SUMMARY_LIMITATIONS = [
    "Boards 01, 04, 06, and 10 are never held out; the estimate covers the 60-image boards only.",
    "With six boards the board-level bootstrap interval is approximate.",
    "Each fold is one frozen 30-image final test on one board; seeds capture training noise only.",
]


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def delta_statistics(deltas: list[float], *, resamples: int, seed: int) -> dict[str, Any]:
    """Descriptive statistics and a percentile board bootstrap for one endpoint."""
    if not deltas:
        return {"n_boards": 0}
    rng = random.Random(seed)
    n = len(deltas)
    # math.fsum is correctly rounded on every Python version; the built-in sum changed its
    # float accumulation in 3.12, which moved bootstrap percentiles by one ulp.
    means = sorted(
        math.fsum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return {
        "n_boards": n,
        "mean": statistics.fmean(deltas),
        "sample_std": statistics.stdev(deltas) if n > 1 else None,
        "min": min(deltas),
        "max": max(deltas),
        "n_positive": sum(delta > 0 for delta in deltas),
        "board_bootstrap": {
            "unit": "held-out board",
            "n_resamples": resamples,
            "seed": seed,
            "ci95_low": _quantile(means, 0.025),
            "ci95_high": _quantile(means, 0.975),
        },
    }


def aggregate_folds(
    repo: Path,
    *,
    incomplete: dict[str, str] | None = None,
    resamples: int = 10_000,
    seed: int = 20_260_803,
) -> dict[str, Any]:
    """Compute the pre-registered endpoints from every fold's promoted final metrics."""
    repo = repo.resolve()
    registry = load_registry(repo)
    incomplete = dict(incomplete or {})
    prereg_path = repo / PREREGISTRATION_RELATIVE
    if not prereg_path.is_file():
        raise LoboError("the pre-registration document is missing; it must exist before analysis")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for fold in registry["folds"]:
        board = fold["board"]
        if board in incomplete:
            rows.append({"board": board, "status": "incomplete", "reason": incomplete[board]})
            continue
        metrics_path = repo / fold["evidence"] / "final_metrics.json"
        if not metrics_path.is_file():
            missing.append(board)
            continue
        metrics = _read_json(metrics_path)
        if metrics.get("final_test_board") != board:
            raise LoboError(f"fold {board} final metrics were evaluated on a different board")
        if metrics.get("manifest_sha256") != fold["manifest_sha256"]:
            raise LoboError(f"fold {board} final metrics manifest hash differs from the registry")
        rows.append(_fold_row(board, fold["evidence"], metrics))
    if missing:
        raise LoboError(
            f"missing final metrics for boards {missing}; promote their packages or mark them "
            "--incomplete with a reason"
        )
    complete = [row for row in rows if row["status"] == "complete"]
    status = "complete" if len(complete) == len(rows) else "incomplete"
    deltas_50 = [row["delta_map50"] for row in complete]
    if status != "complete":
        wording = "incomplete"
    elif all(delta > 0 for delta in deltas_50):
        wording = "all_positive"
    else:
        wording = "direction_varies"
    return {
        "schema_version": "1.0",
        "status": status,
        "unit": "held-out board",
        "registry": REGISTRY_RELATIVE.as_posix(),
        "parent_board": registry["parent_board"],
        "preregistration": {
            "path": PREREGISTRATION_RELATIVE.as_posix(),
            "sha256": hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
        },
        "endpoints": {"primary": PRIMARY_ENDPOINT, "secondary": SECONDARY_ENDPOINTS},
        "n_boards_total": len(rows),
        "n_boards_complete": len(complete),
        "boards": rows,
        "statistics": {
            "delta_map50": delta_statistics(deltas_50, resamples=resamples, seed=seed),
            "delta_map50_95": delta_statistics(
                [row["delta_map50_95"] for row in complete], resamples=resamples, seed=seed
            ),
            "grouped_map50": delta_statistics(
                [row["grouped_map50"]["mean"] for row in complete], resamples=resamples, seed=seed
            ),
        },
        "wording_rule": WORDING_RULE,
        "wording": wording,
        "limitations": SUMMARY_LIMITATIONS,
    }


def _fold_row(board: str, evidence: str, metrics: dict[str, Any]) -> dict[str, Any]:
    try:
        by_arm = metrics["aggregate"]["by_arm"]
        grouped, leaky = by_arm["grouped"], by_arm["leaky_control"]
        f1 = metrics["aggregate"]["paired_bootstrap"]["leaky_minus_grouped_f1"]
        return {
            "board": board,
            "status": "complete",
            "evidence": evidence,
            "n_seeds": grouped["map50"]["n_seeds"],
            "grouped_map50": {"mean": grouped["map50"]["mean"], "std": grouped["map50"]["std"]},
            "leaky_map50": {"mean": leaky["map50"]["mean"], "std": leaky["map50"]["std"]},
            "delta_map50": leaky["map50"]["mean"] - grouped["map50"]["mean"],
            "grouped_map50_95": {
                "mean": grouped["map50_95"]["mean"],
                "std": grouped["map50_95"]["std"],
            },
            "leaky_map50_95": {"mean": leaky["map50_95"]["mean"], "std": leaky["map50_95"]["std"]},
            "delta_map50_95": leaky["map50_95"]["mean"] - grouped["map50_95"]["mean"],
            "f1_delta_image_bootstrap": {
                "mean_delta": f1["mean_delta"],
                "ci95_low": f1["ci95_low"],
                "ci95_high": f1["ci95_high"],
            },
        }
    except (KeyError, TypeError) as exc:
        raise LoboError(f"fold {board} final metrics lack the aggregate fields") from exc


def write_lobo_summary(repo: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write the machine-readable summary and a Markdown table beside it."""
    repo = repo.resolve()
    summary_path = repo / SUMMARY_RELATIVE
    readme_path = repo / SUMMARY_README_RELATIVE
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(_json_bytes(summary))
    readme_path.write_text(_summary_markdown(summary), encoding="utf-8", newline="\n")
    return summary_path, readme_path


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-board replication of the paired leakage experiment",
        "",
        f"Status: **{summary['status']}** · unit: held-out board · "
        f"{summary['n_boards_complete']}/{summary['n_boards_total']} boards complete.",
        "",
        f"Pre-registered analysis: `{summary['preregistration']['path']}` "
        f"(SHA-256 `{summary['preregistration']['sha256']}`).",
        "",
        "| Board | Grouped mAP50 | Leaky control mAP50 | Δ mAP50 (pp) | Δ mAP50-95 (pp) "
        "| Evidence |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["boards"]:
        if row["status"] != "complete":
            lines.append(f"| {row['board']} | incomplete | | | | {row['reason']} |")
            continue
        lines.append(
            f"| {row['board']} "
            f"| {row['grouped_map50']['mean']:.4f} ± {row['grouped_map50']['std']:.4f} "
            f"| {row['leaky_map50']['mean']:.4f} ± {row['leaky_map50']['std']:.4f} "
            f"| {row['delta_map50'] * 100:+.1f} "
            f"| {row['delta_map50_95'] * 100:+.1f} "
            f"| `{row['evidence']}/` |"
        )
    stats = summary["statistics"]["delta_map50"]
    if stats.get("n_boards"):
        std = stats["sample_std"]
        std_text = f"{std * 100:.1f}" if std is not None else "n/a"
        lines += [
            "",
            f"Δ mAP50 across boards: mean {stats['mean'] * 100:+.1f} pp, sample SD {std_text} pp, "
            f"range {stats['min'] * 100:+.1f} to {stats['max'] * 100:+.1f} pp, "
            f"{stats['n_positive']}/{stats['n_boards']} boards positive; "
            f"board bootstrap 95% interval {stats['board_bootstrap']['ci95_low'] * 100:+.1f} to "
            f"{stats['board_bootstrap']['ci95_high'] * 100:+.1f} pp "
            f"({stats['board_bootstrap']['n_resamples']} resamples, approximate for n = "
            f"{stats['n_boards']}).",
        ]
    lines += [
        "",
        f"Pre-registered wording outcome: **{summary['wording']}** — "
        f"{summary['wording_rule'][summary['wording']]}.",
        "",
        "Limitations:",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# handoff
# --------------------------------------------------------------------------------------------

NOTEBOOK_NAME = "lobo_experiment_a100.ipynb"
HANDOFF_FILES = frozenset({NOTEBOOK_NAME, "handoff_manifest.json", "pcb-defect-source.bundle"})
DRIVE_HANDOFF_ROOT = "/content/drive/MyDrive/pcb-defect-paired/handoff-lobo"


def create_lobo_handoff(repo: Path, output_root: Path) -> Path:
    """Create one immutable Colab handoff directory covering every pending fold."""
    repo = repo.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    registry = verify_registry_consistency(repo)
    pending = [fold["board"] for fold in registry["folds"] if fold["run"]]
    if not pending:
        raise LoboError("no fold is pending a Colab run")
    template = repo / "notebooks" / NOTEBOOK_NAME
    staging = Path(tempfile.mkdtemp(prefix=".lobo-handoff-stage-", dir=output_root))
    content = staging / "content"
    try:
        metadata = {
            **project_handoff_metadata(repo),
            "stage": "lobo-replication",
            "lobo_boards": pending,
            "lobo_registry_sha256": _sha256_file(repo / REGISTRY_RELATIVE),
        }
        try:
            result = create_clean_bundle(repo, content, metadata)
        except HandoffError as exc:
            raise LoboError(str(exc)) from exc
        snapshot = result["snapshot_git_sha"]
        drive_directory = f"{DRIVE_HANDOFF_ROOT}/{snapshot[:12]}"
        notebook_sha256 = render_notebook(
            template,
            content / NOTEBOOK_NAME,
            {
                "PASTE_FINAL_BUNDLE_SHA256": result["bundle_sha256"],
                "PASTE_FINAL_GIT_SHA": snapshot,
                "PASTE_LOBO_HANDOFF_DIRECTORY": drive_directory,
            },
        )
        manifest_path = content / "handoff_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "stage": "lobo-replication",
                "lobo_notebook": NOTEBOOK_NAME,
                "lobo_template_sha256": _sha256_file(template),
                "lobo_notebook_sha256": notebook_sha256,
                "drive_handoff_directory": drive_directory,
                "lobo_boards": pending,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        _verify_lobo_handoff(content, snapshot, pending)
        final = output_root / f"colab-handoff-lobo-{snapshot[:12]}"
        if final.exists():
            raise LoboError(f"refusing to overwrite handoff directory: {final}")
        os.rename(content, final)
        return final
    finally:
        if staging.exists():
            shutil.rmtree(staging, onerror=_remove_readonly)


def _verify_lobo_handoff(content: Path, snapshot: str, pending: list[str]) -> None:
    if {path.name for path in content.iterdir()} != HANDOFF_FILES:
        raise LoboError("LOBO handoff must contain exactly the bundle, notebook, and manifest")
    manifest = _read_json(content / "handoff_manifest.json")
    bundle = content / "pcb-defect-source.bundle"
    notebook = content / NOTEBOOK_NAME
    if manifest.get("bundle_sha256") != _sha256_file(bundle):
        raise LoboError("LOBO handoff bundle hash differs from its manifest")
    if manifest.get("lobo_notebook_sha256") != _sha256_file(notebook):
        raise LoboError("LOBO handoff notebook hash differs from its manifest")
    if manifest.get("snapshot_git_sha") != snapshot or manifest.get("lobo_boards") != pending:
        raise LoboError("LOBO handoff manifest identity differs from the rendered values")
    source = notebook.read_text(encoding="utf-8")
    if "PASTE_" in source:
        raise LoboError("LOBO handoff notebook contains an unresolved placeholder")
    for index, cell in enumerate(json.loads(source)["cells"]):
        if cell.get("outputs"):
            raise LoboError("LOBO handoff notebook contains persisted outputs")
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), f"{notebook}:cell-{index}", "exec")
    if snapshot not in source or manifest["bundle_sha256"] not in source:
        raise LoboError("LOBO handoff notebook does not bind the bundle and snapshot identities")
    temporary = Path(tempfile.mkdtemp(prefix=".lobo-handoff-verify-"))
    try:
        clone = temporary / "clone"
        _git(temporary, "clone", "--quiet", str(bundle), str(clone))
        _git(clone, "checkout", "--quiet", "--detach", snapshot)
        if _git(clone, "rev-parse", "HEAD") != snapshot:
            raise LoboError("LOBO bundle snapshot Git SHA mismatch")
        if _git(clone, "rev-list", "--count", "HEAD") != "1":
            raise LoboError("LOBO bundle must contain exactly one reachable commit")
        bundled = yaml.safe_load((clone / REGISTRY_RELATIVE).read_text(encoding="utf-8"))
        bundled_pending = [fold["board"] for fold in bundled["folds"] if fold["run"]]
        if bundled_pending != pending:
            raise LoboError("LOBO bundle fold registry differs from the handoff manifest")
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise LoboError("LOBO bundle verification failed") from exc
    finally:
        shutil.rmtree(temporary, onerror=_remove_readonly)


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LoboError(f"git command failed during LOBO handoff verification: {args}") from exc
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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

    promote = subparsers.add_parser("promote", help="promote a returned A100 package")
    promote.add_argument("--repo", type=Path, default=Path.cwd())
    promote.add_argument("--package", type=Path, required=True)
    promote.add_argument("--board", required=True)
    promote.add_argument("--output", type=Path, default=None)

    aggregate = subparsers.add_parser("aggregate", help="compute the pre-registered endpoints")
    aggregate.add_argument("--repo", type=Path, default=Path.cwd())
    aggregate.add_argument(
        "--incomplete",
        action="append",
        default=[],
        metavar="BOARD=REASON",
        help="report a fold as incomplete with the given reason",
    )

    handoff = subparsers.add_parser("handoff", help="create the Colab handoff for pending folds")
    handoff.add_argument("--repo", type=Path, default=Path.cwd())
    handoff.add_argument("--output-root", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "folds":
        return _folds_command(args.repo, args.dataset, args.write)
    if args.command == "promote":
        return _promote_command(args.repo, args.package, args.board, args.output)
    if args.command == "aggregate":
        return _aggregate_command(args.repo, args.incomplete)
    if args.command == "handoff":
        return _handoff_command(args.repo, args.output_root)
    raise AssertionError(f"unknown command: {args.command}")


def _handoff_command(repo: Path, output_root: Path) -> int:
    handoff = create_lobo_handoff(repo, output_root)
    print(f"lobo_handoff_dir={handoff}")
    return 0


def _promote_command(repo: Path, package: Path, board: str, output: Path | None) -> int:
    registry = load_registry(repo)
    fold = next((fold for fold in registry["folds"] if fold["board"] == board), None)
    if fold is None:
        raise SystemExit(f"board {board} is not in the fold registry")
    destination = repo.resolve() / fold["evidence"] if output is None else output
    receipt = promote_paired_package(
        package,
        board=board,
        expected_manifest_sha256=fold["manifest_sha256"],
        output=destination,
    )
    print(f"promoted board {board}: {destination} ({receipt['package']['sha256']})")
    return 0


def _aggregate_command(repo: Path, incomplete_arguments: list[str]) -> int:
    incomplete: dict[str, str] = {}
    for argument in incomplete_arguments:
        board, separator, reason = argument.partition("=")
        if not separator or not board or not reason:
            raise SystemExit(f"--incomplete expects BOARD=REASON, got {argument!r}")
        incomplete[board] = reason
    summary = aggregate_folds(repo, incomplete=incomplete)
    summary_path, readme_path = write_lobo_summary(repo, summary)
    print(f"summary written: {summary_path} and {readme_path} (status={summary['status']})")
    return 0


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
