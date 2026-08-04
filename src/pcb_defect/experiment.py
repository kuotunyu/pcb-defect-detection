"""Resumable, provenance-first runner for the frozen paired experiment.

This module is intentionally the source of truth used by Colab.  The notebook
only installs the locked environment and invokes these subcommands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pcb_defect.data_prep.paired import (
    _load_spec,
    _verify_frozen_hashes,
    discover_converted_samples,
    render_runtime_datasets,
)
from pcb_defect.evidence import artifact_ref, validate_run_record, write_run_record
from pcb_defect.paired_protocol import PROTOCOL_VERSION, PairedProtocolConfig, build_paired_protocol

ARMS = ("grouped", "leaky_control")
REQUIRED_TRAIN_KEYS = {
    "model",
    "epochs",
    "imgsz",
    "batch",
    "optimizer",
    "patience",
    "workers",
    "amp",
    "deterministic",
    "cache",
    "close_mosaic",
    "cos_lr",
    "rect",
    "plots",
    "save",
    "save_period",
    "val",
    "device",
}


class ExperimentError(RuntimeError):
    """An experiment gate failed; no further work should be attempted."""


@dataclass(frozen=True)
class InputLock:
    """The immutable source, data, config, and initialization shared by all six runs."""

    git_sha: str
    config_sha256: str
    dataset_sha256: str
    manifest_sha256: str
    base_model_contract_sha256: str
    base_model_sha256: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def planned_runs(training_seeds: tuple[int, ...]) -> list[tuple[str, int]]:
    """Fixed execution order: finish all grouped runs before leaky controls."""
    return [(arm, seed) for arm in ARMS for seed in training_seeds]


def freeze_or_verify_input_lock(path: Path, expected: InputLock) -> str:
    """Create the pre-training lock once, or fail on any later mismatch."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(expected.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return "created"
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"input lock is unreadable or corrupt: {path}") from exc
    for key, value in expected.as_dict().items():
        if observed.get(key) != value:
            raise ExperimentError(
                f"input lock mismatch for {key}: expected {value}, found {observed.get(key)}"
            )
    if set(observed) != set(expected.as_dict()):
        raise ExperimentError("input lock contains unexpected or missing fields")
    return "verified"


def run_is_complete(run_dir: Path, expected: InputLock) -> bool:
    """Return true only for a complete record whose referenced bytes still match."""
    record_path = run_dir / "run_record.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        validate_run_record(record)
        if record["status"] != "complete":
            return False
        checks = {
            "git_sha": record["provenance"]["git_sha"],
            "config_sha256": record["training"]["config"]["sha256"],
            "dataset_sha256": record["protocol"]["dataset_sha256"],
            "manifest_sha256": record["protocol"]["manifest_sha256"],
            "base_model_contract_sha256": record["training"]["base_model"]["contract"]["sha256"],
            "base_model_sha256": record["training"]["base_model"]["sha256"],
        }
        if checks != expected.as_dict():
            return False
        refs = [
            record["protocol"]["manifest"],
            record["training"]["config"],
            record["training"]["base_model"]["contract"],
        ]
        refs.extend(record["artifacts"].values())
        refs.extend(record["metrics"].values())
        return all(_ref_matches(run_dir, ref) for ref in refs)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve-base", help="download/copy and freeze base weights")
    _add_common_paths(resolve, needs_dataset=False, needs_base=False)

    preflight = subparsers.add_parser("preflight", help="verify immutable inputs and GPU runtime")
    _add_common_paths(preflight)
    preflight.add_argument("--required-gpu", default="A100")

    gates = subparsers.add_parser("gates", help="run tiny train/resume/speed gates")
    _add_common_paths(gates)
    gates.add_argument("--required-gpu", default="A100")

    train = subparsers.add_parser("train", help="train or resume one arm/seed")
    _add_common_paths(train)
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--seed", type=int, required=True)

    train_all = subparsers.add_parser("train-all", help="run all six jobs in frozen order")
    _add_common_paths(train_all)

    args = parser.parse_args(argv)
    if args.command == "resolve-base":
        _resolve_base(args.repo, args.workspace)
        return 0
    if args.command in {"preflight", "gates"}:
        _assert_gpu(args.required_gpu)
    else:
        _assert_gpu("A100")
    context = _prepare_context(args.repo, args.dataset, args.workspace, args.base_model)
    if args.command == "preflight":
        _print_preflight(context)
        return 0
    if args.command == "gates":
        _run_gates(context)
        return 0
    if args.command == "train":
        _train_one(context, args.arm, args.seed)
        return 0
    for arm, seed in planned_runs(tuple(context["protocol"].config.training_seeds)):
        _train_one(context, arm, seed)
    return 0


def _add_common_paths(
    parser: argparse.ArgumentParser, *, needs_dataset: bool = True, needs_base: bool = True
) -> None:
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    if needs_dataset:
        parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    if needs_base:
        parser.add_argument("--base-model", type=Path, required=True)


def _prepare_context(
    repo: Path, dataset: Path, workspace: Path, base_model: Path
) -> dict[str, Any]:
    repo = repo.resolve()
    workspace = workspace.resolve()
    git_sha, git_dirty = _git_provenance(repo)
    if git_dirty:
        raise ExperimentError("repository is dirty; commit or discard changes before preflight")
    train_config_path = repo / "configs" / "train_paired.yaml"
    train_config = _load_train_config(train_config_path)
    base_contract_path = repo / "configs" / "base_model.yaml"
    base_contract = _load_base_model_contract(base_contract_path)
    if train_config["model"] != base_contract["filename"]:
        raise ExperimentError("training config model does not match the base-model contract")
    spec = _load_spec(repo / "configs" / "paired_protocol.yaml")
    protocol = build_paired_protocol(
        discover_converted_samples(dataset), PairedProtocolConfig(**spec["protocol"])
    )
    _verify_frozen_hashes(protocol, spec)
    runtime = workspace / "runtime_data"
    render_runtime_datasets(dataset, protocol, runtime)
    if not base_model.is_file():
        raise ExperimentError(f"base model is missing: {base_model}")
    _verify_base_model(base_model, base_contract)
    lock = InputLock(
        git_sha=git_sha,
        config_sha256=_sha256_file(train_config_path),
        dataset_sha256=protocol.dataset_sha256,
        manifest_sha256=protocol.manifest_sha256,
        base_model_contract_sha256=_sha256_file(base_contract_path),
        base_model_sha256=_sha256_file(base_model),
    )
    lock_state = freeze_or_verify_input_lock(workspace / "inputs" / "input_lock.json", lock)
    return {
        "repo": repo,
        "dataset": dataset.resolve(),
        "workspace": workspace,
        "base_model": base_model.resolve(),
        "protocol": protocol,
        "train_config": train_config,
        "train_config_path": train_config_path,
        "base_contract": base_contract,
        "base_contract_path": base_contract_path,
        "runtime": runtime,
        "lock": lock,
        "lock_state": lock_state,
    }


def _resolve_base(repo: Path, workspace: Path) -> Path:
    repo = repo.resolve()
    config = _load_train_config(repo / "configs" / "train_paired.yaml")
    contract = _load_base_model_contract(repo / "configs" / "base_model.yaml")
    if config["model"] != contract["filename"]:
        raise ExperimentError("training config model does not match the base-model contract")
    destination = workspace.resolve() / "inputs" / "base_model.pt"
    if destination.is_file():
        _verify_base_model(destination, contract)
        print(f"base model already verified: {destination} sha256={_sha256_file(destination)}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    if temporary.exists():
        raise ExperimentError(f"partial base-model download exists; inspect it: {temporary}")
    request = urllib.request.Request(
        contract["source"], headers={"User-Agent": "pcb-defect-evidence-runner/1.0"}
    )
    try:
        with urllib.request.urlopen(request) as response, temporary.open("xb") as handle:
            shutil.copyfileobj(response, handle)
    except (OSError, urllib.error.URLError) as exc:
        raise ExperimentError(f"base-model download failed from {contract['source']}") from exc
    _verify_base_model(temporary, contract)
    temporary.replace(destination)
    print(f"base model resolved: {destination} sha256={_sha256_file(destination)}")
    return destination


def _train_one(context: dict[str, Any], arm: str, seed: int) -> None:
    protocol = context["protocol"]
    if seed not in protocol.config.training_seeds:
        raise ExperimentError(
            f"seed {seed} is outside frozen seeds {protocol.config.training_seeds}"
        )
    run_dir = context["workspace"] / "runs" / arm / f"seed{seed}"
    if run_is_complete(run_dir, context["lock"]):
        print(f"SKIP complete hash-matching run: {arm} seed={seed}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    record = _initial_record(context, arm, seed, run_dir)
    write_run_record(record, run_dir / "run_record.json")
    record["status"] = "running"
    record["timestamps"]["updated_at_utc"] = _utc_now()
    write_run_record(record, run_dir / "run_record.json")
    try:
        _execute_training(context, arm, seed, run_dir, record)
    except Exception as exc:
        record["status"] = "failed"
        record["timestamps"]["updated_at_utc"] = _utc_now()
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        write_run_record(record, run_dir / "run_record.json")
        raise


def _run_gates(context: dict[str, Any]) -> None:
    """Exercise one interrupted tiny run, checkpoint reload, resume, and timing."""
    import torch
    from ultralytics import YOLO

    gate_dir = context["workspace"] / "gates"
    report_path = gate_dir / "gate_report.json"
    if report_path.is_file():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                existing.get("passed") is True
                and existing.get("input_lock") == context["lock"].as_dict()
            ):
                print("SKIP complete hash-matching GPU gates")
                return
        except json.JSONDecodeError:
            pass
    if gate_dir.exists():
        raise ExperimentError(
            f"incomplete gate directory exists at {gate_dir}; inspect it before retrying"
        )
    gate_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "input_lock": context["lock"].as_dict(),
        "started_at_utc": _utc_now(),
        "passed": False,
        "checks": {
            "locked_environment": True,
            "dataset_manifest": True,
            "gpu": True,
            "tiny_train": False,
            "resume": False,
            "speed_probe": False,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    model = YOLO(str(context["base_model"]))
    resume_checkpoint = gate_dir / "resume_checkpoint.pt"

    def stop_after_first_save(trainer) -> None:
        _stage_resume_checkpoint(trainer, resume_checkpoint)

    model.add_callback("on_model_save", stop_after_first_save)
    smoke_args = dict(context["train_config"])
    smoke_args.pop("model")
    smoke_args.update(
        {
            "epochs": 2,
            "imgsz": 320,
            "batch": 2,
            "workers": 1,
            "plots": False,
            "save_period": -1,
        }
    )
    started = time.perf_counter()
    model.train(
        data=str(context["runtime"] / "grouped" / "data.yaml"),
        seed=42,
        fraction=0.05,
        project=str(gate_dir),
        name="resume_smoke",
        exist_ok=True,
        **smoke_args,
    )
    last = resume_checkpoint
    if not last.is_file():
        raise ExperimentError("tiny train did not stage a checkpoint before its clean stop")
    report["checks"]["tiny_train"] = True
    try:
        resume_payload = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ExperimentError(f"staged resume checkpoint is unreadable: {last}") from exc
    report["resume_checkpoint"] = _validate_resume_checkpoint_payload(resume_payload)
    YOLO(str(last)).train(resume=True)
    elapsed = time.perf_counter() - started
    results_csv = gate_dir / "resume_smoke" / "results.csv"
    if not results_csv.is_file() or len(results_csv.read_text(encoding="utf-8").splitlines()) < 3:
        raise ExperimentError("resume smoke did not complete two epochs")
    report["checks"]["resume"] = True
    report["checks"]["speed_probe"] = True
    report["speed_probe"] = {
        "two_tiny_epochs_seconds": round(elapsed, 3),
        "note": "Gate-only 5% data, imgsz=320, batch=2; not a production benchmark.",
    }
    report["completed_at_utc"] = _utc_now()
    report["passed"] = all(report["checks"].values())
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"GPU GATES PASS: {report_path}")


def _stage_resume_checkpoint(trainer: Any, destination: Path) -> None:
    """Preserve the epoch-zero optimizer state, then ask Ultralytics to stop cleanly."""
    if trainer.epoch != 0:
        return
    source = Path(trainer.last)
    if not source.is_file():
        raise ExperimentError(f"trainer callback could not find checkpoint: {source}")
    if destination.exists():
        raise ExperimentError(f"refusing to overwrite staged resume checkpoint: {destination}")
    temporary = destination.with_name(f".{destination.name}.stage")
    if temporary.exists():
        raise ExperimentError(f"partial staged resume checkpoint exists: {temporary}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    trainer.stop = True


def _validate_resume_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Prove that the staged epoch-zero checkpoint can exercise a real optimizer resume."""
    if not isinstance(payload, dict) or payload.get("epoch") != 0:
        raise ExperimentError("resume checkpoint must contain the completed epoch-zero state")
    if payload.get("optimizer") is None:
        raise ExperimentError("resume checkpoint is missing optimizer state")
    return {"checkpoint_epoch": 0, "optimizer_state_present": True}


def _initial_record(context: dict[str, Any], arm: str, seed: int, run_dir: Path) -> dict[str, Any]:
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    config_copy = inputs / "train_paired.yaml"
    base_contract_copy = inputs / "base_model.yaml"
    manifest_copy = inputs / "paired_split_manifest.json"
    shutil.copy2(context["train_config_path"], config_copy)
    shutil.copy2(context["base_contract_path"], base_contract_copy)
    shutil.copy2(context["repo"] / "reports" / "protocol" / manifest_copy.name, manifest_copy)
    now = _utc_now()
    return {
        "schema_version": "1.0",
        "run_id": f"{arm}-seed{seed}",
        "arm": arm,
        "seed": seed,
        "status": "planned",
        "timestamps": {"created_at_utc": now, "updated_at_utc": now},
        "provenance": {
            "git_sha": context["lock"].git_sha,
            "git_dirty": False,
            "command": list(sys.argv),
            "environment": _environment(),
        },
        "protocol": {
            "version": PROTOCOL_VERSION,
            "manifest": artifact_ref(manifest_copy, relative_to=run_dir),
            "manifest_sha256": context["lock"].manifest_sha256,
            "dataset_sha256": context["lock"].dataset_sha256,
        },
        "training": {
            "config": artifact_ref(config_copy, relative_to=run_dir),
            "resolved": {**context["train_config"], "seed": seed, "arm": arm},
            "base_model": {
                "source": context["base_contract"]["source"],
                "revision": context["base_contract"]["revision"],
                "filename": context["base_contract"]["filename"],
                "sha256": context["lock"].base_model_sha256,
                "bytes": context["base_contract"]["bytes"],
                "contract": artifact_ref(base_contract_copy, relative_to=run_dir),
            },
        },
        "artifacts": {},
        "metrics": {},
        "failure": None,
    }


def _execute_training(
    context: dict[str, Any], arm: str, seed: int, run_dir: Path, record: dict[str, Any]
) -> None:
    from ultralytics import YOLO

    data_yaml = context["runtime"] / arm / "data.yaml"
    scratch = run_dir / "scratch"
    last = scratch / "train" / "weights" / "last.pt"
    if last.is_file():
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(str(context["base_model"]))
        train_args = dict(context["train_config"])
        train_args.pop("model")
        model.train(
            data=str(data_yaml),
            seed=seed,
            project=str(scratch),
            name="train",
            exist_ok=True,
            **train_args,
        )
    source_weights = scratch / "train" / "weights"
    best = source_weights / "best.pt"
    last = source_weights / "last.pt"
    if not best.is_file() or not last.is_file():
        raise ExperimentError("training ended without readable best.pt and last.pt")
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, weights_dir / "best.pt")
    shutil.copy2(last, weights_dir / "last.pt")

    metrics = _validate_checkpoint(weights_dir / "best.pt", data_yaml, run_dir)
    if not _all_finite(metrics):
        raise ExperimentError("validation metrics contain NaN or Inf")
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    validation_path = metrics_dir / "validation.json"
    validation_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    record["artifacts"] = {
        "best_checkpoint": artifact_ref(weights_dir / "best.pt", relative_to=run_dir),
        "last_checkpoint": artifact_ref(weights_dir / "last.pt", relative_to=run_dir),
    }
    record["metrics"] = {"validation": artifact_ref(validation_path, relative_to=run_dir)}
    record["status"] = "complete"
    record["timestamps"]["updated_at_utc"] = _utc_now()
    write_run_record(record, run_dir / "run_record.json")


def _validate_checkpoint(weights: Path, data_yaml: Path, run_dir: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        plots=False,
        project=str(run_dir / "scratch"),
        name="validation",
        exist_ok=True,
    )
    return {
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "per_class": {
            model.names[index]: {
                "ap50": float(result.box.ap50[index]),
                "ap50_95": float(result.box.ap[index]),
                "precision": float(result.box.p[index]),
                "recall": float(result.box.r[index]),
            }
            for index in range(len(model.names))
        },
    }


def _print_preflight(context: dict[str, Any]) -> None:
    print(f"git_sha={context['lock'].git_sha}")
    print(f"input_lock={context['lock_state']}")
    print(f"dataset_sha256={context['lock'].dataset_sha256}")
    print(f"manifest_sha256={context['lock'].manifest_sha256}")
    print(f"base_model_sha256={context['lock'].base_model_sha256}")
    print("PREFLIGHT PASS")


def _assert_gpu(required_name: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise ExperimentError(
            "CUDA is unavailable; this command must run in the selected Colab GPU"
        )
    name = torch.cuda.get_device_name(0)
    if required_name.lower() not in name.lower():
        raise ExperimentError(f"required GPU containing {required_name!r}, found {name!r}")
    print(f"gpu={name}")


def _load_train_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ExperimentError(f"training config is not a mapping: {path}")
    if set(config) != REQUIRED_TRAIN_KEYS:
        missing = sorted(REQUIRED_TRAIN_KEYS - set(config))
        extra = sorted(set(config) - REQUIRED_TRAIN_KEYS)
        raise ExperimentError(f"training config keys changed; missing={missing}, extra={extra}")
    if config["deterministic"] is not True or config["val"] is not True:
        raise ExperimentError("training must remain deterministic with validation enabled")
    return config


def _load_base_model_contract(path: Path) -> dict[str, Any]:
    """Load a content-addressed model contract pinned to an immutable release URL."""
    try:
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentError(f"base-model contract is unreadable: {path}") from exc
    required = {"source", "revision", "filename", "sha256", "bytes"}
    if not isinstance(contract, dict) or set(contract) != required:
        raise ExperimentError(f"base-model contract keys must be exactly {sorted(required)}")
    if not all(isinstance(contract[key], str) and contract[key] for key in required - {"bytes"}):
        raise ExperimentError("base-model contract string fields must be non-empty")
    if not isinstance(contract["bytes"], int) or contract["bytes"] <= 0:
        raise ExperimentError("base-model contract bytes must be a positive integer")
    if len(contract["sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in contract["sha256"]
    ):
        raise ExperimentError("base-model contract SHA-256 must be lowercase hexadecimal")
    immutable_segment = f"/releases/download/{contract['revision']}/"
    if (
        not contract["source"].startswith("https://github.com/")
        or immutable_segment not in contract["source"]
        or contract["source"].rsplit("/", 1)[-1] != contract["filename"]
    ):
        raise ExperimentError("base-model source must pin its immutable release revision")
    return contract


def _verify_base_model(path: Path, contract: dict[str, Any]) -> None:
    """Fail closed unless local initialization bytes exactly match the frozen contract."""
    if not path.is_file():
        raise ExperimentError(f"base model is missing: {path}")
    if path.stat().st_size != contract["bytes"]:
        raise ExperimentError(
            f"base-model byte count mismatch: expected {contract['bytes']}, "
            f"found {path.stat().st_size}"
        )
    observed = _sha256_file(path)
    if observed != contract["sha256"]:
        raise ExperimentError(
            f"base-model SHA-256 mismatch: expected {contract['sha256']}, found {observed}"
        )


def _git_provenance(repo: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExperimentError(f"cannot read Git provenance from {repo}") from exc
    return sha, bool(status.strip())


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "ultralytics",
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "pyyaml",
        "onnxruntime",
        "onnxruntime-gpu",
        "tensorrt",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _ref_matches(root: Path, ref: dict[str, Any]) -> bool:
    path = (root / ref["path"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return (
        path.is_file()
        and path.stat().st_size == ref["bytes"]
        and _sha256_file(path) == ref["sha256"]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
