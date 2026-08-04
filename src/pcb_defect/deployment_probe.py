"""Read-only same-ONNX deployment parity probe for immutable failed deployments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml

from pcb_defect.deployment import _standalone_parity, parity_passes
from pcb_defect.experiment import ExperimentError, InputLock, _git_provenance, _sha256_file
from pcb_defect.runtime_contract import configure_hermetic_ultralytics, onnxruntime_state

configure_hermetic_ultralytics()

FROZEN_GATE_VALUES = {
    "calibration_images": 60,
    "parity_confidence": 0.25,
    "parity_match_iou": 0.5,
    "parity_min_iou": 0.90,
    "parity_max_confidence_delta": 0.15,
}


class ProbeError(RuntimeError):
    """The probe inputs or output destination are unsafe or incomplete."""


@dataclass(frozen=True)
class ProbeInputs:
    """Hash-verified evidence that the probe may read, but never alter."""

    parent_workspace: Path
    input_lock: InputLock
    failed_gate_path: Path
    failed_gate: dict[str, Any]
    onnx_path: Path
    source_checkpoint: Path
    calibration_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _OutputReservation:
    """An invocation-owned exclusive claim on an external report destination."""

    destination: Path
    sidecar: Path
    reservation: Path


@dataclass(frozen=True)
class _StagedOnnx:
    """A uniquely named external ONNX copy owned by one probe invocation."""

    directory: Path
    path: Path


def verify_probe_inputs(
    parent_workspace: Path,
    *,
    expected_parent_git_sha: str,
    expected_gate_sha256: str,
    expected_onnx_sha256: str,
) -> ProbeInputs:
    """Fail closed unless the frozen parent evidence matches every supplied identity."""
    parent = parent_workspace.resolve()
    if not parent.is_dir():
        raise ProbeError(f"parent workspace is missing: {parent}")

    lock_path = parent / "inputs" / "input_lock.json"
    try:
        input_lock = InputLock(**_read_json_object(lock_path, "input lock"))
    except TypeError as exc:
        raise ProbeError("input lock is malformed") from exc
    if input_lock.git_sha != expected_parent_git_sha:
        raise ProbeError("parent input-lock Git SHA-1 does not match the expected parent SHA")

    deployment = parent / "deployment"
    failed_gate_path = deployment / "deployment_gate.json"
    failed_gate, observed_gate_sha256 = _read_hashed_json_object(
        failed_gate_path, "deployment gate"
    )
    if observed_gate_sha256 != expected_gate_sha256:
        raise ProbeError("deployment-gate SHA-256 does not match the expected failed gate")
    if failed_gate.get("passed") is not False:
        raise ProbeError("deployment gate must be an explicitly failed report")

    artifacts = failed_gate.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ProbeError("deployment gate artifacts are malformed")
    onnx_path = _verified_artifact_path(deployment, artifacts.get("onnx"), "ONNX")
    observed_onnx_sha256 = _sha256_file_required(onnx_path, "ONNX")
    report_onnx_sha256 = artifacts.get("onnx_sha256")
    if (
        not isinstance(report_onnx_sha256, str)
        or observed_onnx_sha256 != report_onnx_sha256
        or observed_onnx_sha256 != expected_onnx_sha256
    ):
        raise ProbeError("ONNX SHA-256 does not match the deployment report and expected value")

    source_checkpoint = _verified_artifact_path(
        parent, artifacts.get("source_checkpoint"), "source checkpoint"
    )
    observed_checkpoint_sha256 = _sha256_file_required(source_checkpoint, "source checkpoint")
    report_checkpoint_sha256 = artifacts.get("source_checkpoint_sha256")
    if (
        not isinstance(report_checkpoint_sha256, str)
        or observed_checkpoint_sha256 != report_checkpoint_sha256
    ):
        raise ProbeError("source checkpoint SHA-256 does not match the deployment report")

    calibration_paths = _load_calibration_paths(parent, deployment / "calibration.yaml")
    return ProbeInputs(
        parent_workspace=parent,
        input_lock=input_lock,
        failed_gate_path=failed_gate_path.resolve(),
        failed_gate=failed_gate,
        onnx_path=onnx_path,
        source_checkpoint=source_checkpoint,
        calibration_paths=calibration_paths,
    )


def run_probe(
    repo: Path,
    parent_workspace: Path,
    output: Path,
    *,
    expected_parent_git_sha: str,
    expected_gate_sha256: str,
    expected_onnx_sha256: str,
) -> dict[str, Any]:
    """Run same-ONNX CPU parity and atomically record a new external diagnostic report."""
    repo = repo.resolve()
    try:
        probe_git_sha, probe_dirty = _git_provenance(repo)
    except ExperimentError as exc:
        raise ProbeError("cannot verify probe repository Git provenance") from exc
    if probe_dirty:
        raise ProbeError("probe repository must be clean")
    inputs = verify_probe_inputs(
        parent_workspace,
        expected_parent_git_sha=expected_parent_git_sha,
        expected_gate_sha256=expected_gate_sha256,
        expected_onnx_sha256=expected_onnx_sha256,
    )
    config_path = repo / "configs" / "deployment_gate.yaml"
    config = _read_yaml_object(config_path, "deployment-gate config")
    _verify_frozen_gate_values(config)
    runtime_before = onnxruntime_state()
    reservation = _prepare_output(output, inputs.parent_workspace)
    staged = _stage_verified_onnx(
        inputs.onnx_path,
        expected_onnx_sha256,
        reservation.destination.parent,
        inputs.parent_workspace,
    )
    parity = _standalone_parity(staged.path, list(inputs.calibration_paths), config)
    _verify_onnx_unchanged_after_inference(staged.path, inputs.onnx_path, expected_onnx_sha256)
    runtime_after = onnxruntime_state()
    if runtime_after != runtime_before:
        raise ProbeError("ONNX Runtime state changed during parity inference")
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "passed": _probe_parity_passes(parity, expected_onnx_sha256),
        "probe_git_sha": probe_git_sha,
        "parent": {
            "experiment_git_sha": inputs.input_lock.git_sha,
            "deployment_gate_sha256": expected_gate_sha256,
            "onnx_sha256": expected_onnx_sha256,
            "parity_onnx_sha256": expected_onnx_sha256,
            "source_checkpoint_sha256": inputs.failed_gate["artifacts"]["source_checkpoint_sha256"],
        },
        "config_sha256": _sha256_file_required(config_path, "deployment-gate config"),
        "parity": parity,
        "runtime_contract": {"before": runtime_before, "after": runtime_after},
    }
    _write_report_no_replace(report, reservation)
    return report


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"{label} is missing or malformed") from exc
    if not isinstance(data, dict):
        raise ProbeError(f"{label} must be a JSON object")
    return data


def _read_hashed_json_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    """Read, hash, and parse one exact byte sequence from immutable evidence."""
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"{label} is missing or malformed") from exc
    if not isinstance(data, dict):
        raise ProbeError(f"{label} must be a JSON object")
    return data, hashlib.sha256(raw).hexdigest()


def _read_yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProbeError(f"{label} is missing or malformed") from exc
    if not isinstance(data, dict):
        raise ProbeError(f"{label} must be a mapping")
    return data


def _verify_frozen_gate_values(config: dict[str, Any]) -> None:
    for key, expected in FROZEN_GATE_VALUES.items():
        observed = config.get(key)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ProbeError(f"frozen deployment-gate configuration is missing {key}")
        if observed != expected:
            raise ProbeError(
                f"frozen deployment-gate configuration changed {key}: "
                f"expected {expected}, found {observed}"
            )


def _probe_parity_passes(parity: dict[str, Any], expected_onnx_sha256: str) -> bool:
    """Accept metrics only when their backend and ONNX provenance are the frozen contract."""
    try:
        return (
            parity.get("reference_backend") == "ultralytics-onnx"
            and parity.get("candidate_backend") == "standalone-onnxruntime"
            and parity.get("onnx_sha256") == expected_onnx_sha256
            and _valid_parity_aggregates(parity)
            and parity_passes(parity)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _valid_parity_aggregates(parity: dict[str, Any]) -> bool:
    """Reject coercible, non-finite, or physically invalid aggregate parity values."""
    return (
        _exact_integer(parity.get("n_images")) == FROZEN_GATE_VALUES["calibration_images"]
        and _exact_integer(parity.get("required_images"))
        == FROZEN_GATE_VALUES["calibration_images"]
        and _exact_integer(parity.get("n_failed")) == 0
        and _bounded_metric(parity.get("min_iou"), 0.0, 1.0)
        and _bounded_metric(parity.get("required_min_iou"), 0.0, 1.0)
        and float(parity["required_min_iou"]) == FROZEN_GATE_VALUES["parity_min_iou"]
        and _bounded_metric(parity.get("max_conf_delta"), 0.0, 1.0)
        and _bounded_metric(parity.get("allowed_max_conf_delta"), 0.0, 1.0)
        and float(parity["allowed_max_conf_delta"])
        == FROZEN_GATE_VALUES["parity_max_confidence_delta"]
    )


def _exact_integer(value: Any) -> int | None:
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) else None


def _bounded_metric(value: Any, lower: float, upper: float) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and lower <= float(value) <= upper
    )


def _sha256_file_required(path: Path, label: str) -> str:
    if not path.is_file():
        raise ProbeError(f"{label} is missing: {path}")
    try:
        return _sha256_file(path)
    except OSError as exc:
        raise ProbeError(f"cannot hash {label}: {path}") from exc


def _verified_artifact_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProbeError(f"{label} path is missing from the deployment gate")
    return _resolve_under(root, value, label)


def _resolve_under(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ProbeError(f"{label} path escapes its declared workspace: {value}") from exc
    return resolved


def _load_calibration_paths(parent: Path, calibration_yaml: Path) -> tuple[Path, ...]:
    config = _read_yaml_object(calibration_yaml, "calibration YAML")
    declared = config.get("val")
    if isinstance(declared, str):
        calibration_list = _resolve_under(parent, declared, "calibration list")
        if not calibration_list.is_file():
            raise ProbeError(f"calibration list is missing: {calibration_list}")
        try:
            raw_paths = calibration_list.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ProbeError(f"cannot read calibration list: {calibration_list}") from exc
    elif isinstance(declared, list):
        raw_paths = declared
    else:
        raise ProbeError("calibration YAML val must declare a calibration path list")
    if len(raw_paths) != 60 or not all(isinstance(item, str) and item for item in raw_paths):
        raise ProbeError("calibration set must contain exactly 60 declared image paths")
    # The immutable list lives under the parent workspace, but its images are source-dataset
    # files and are intentionally absolute paths outside that workspace.
    resolved_paths = tuple(_resolve_calibration_image(parent, item) for item in raw_paths)
    if len(set(resolved_paths)) != 60:
        raise ProbeError("calibration set must contain 60 distinct image paths")
    if not all(path.is_file() for path in resolved_paths):
        raise ProbeError("calibration set contains a missing image path")
    return resolved_paths


def _resolve_calibration_image(parent: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (parent / candidate).resolve()


def _prepare_output(output: Path, parent_workspace: Path) -> _OutputReservation:
    destination = output.resolve()
    try:
        destination.relative_to(parent_workspace)
    except ValueError:
        pass
    else:
        raise ProbeError("probe output must be outside the immutable parent workspace")
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    reservation = destination.with_name(f".{destination.name}.probe-reservation")
    if any(path.exists() for path in (destination, sidecar, reservation)):
        raise ProbeError(
            f"refusing to overwrite or reserve an existing probe output: {destination}"
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProbeError(f"cannot create probe output directory: {destination.parent}") from exc
    try:
        descriptor = os.open(str(reservation), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            handle.write("deployment parity probe reservation\n")
    except FileExistsError as exc:
        raise ProbeError(f"probe output reservation already exists: {reservation}") from exc
    except OSError as exc:
        raise ProbeError(f"cannot exclusively reserve probe output: {destination}") from exc
    return _OutputReservation(destination, sidecar, reservation)


def _stage_verified_onnx(
    parent_onnx: Path, expected_onnx_sha256: str, external_root: Path, parent_workspace: Path
) -> _StagedOnnx:
    """Copy the verified parent model into an invocation-owned external directory."""
    try:
        directory = Path(
            tempfile.mkdtemp(prefix=".deployment-parity-stage-", dir=external_root)
        ).resolve()
    except OSError as exc:
        raise ProbeError(f"cannot create external ONNX staging directory: {external_root}") from exc
    try:
        directory.relative_to(parent_workspace)
    except ValueError:
        pass
    else:
        raise ProbeError("ONNX staging directory must be outside the immutable parent workspace")
    staged = _StagedOnnx(directory, directory / "best.onnx")
    try:
        shutil.copyfile(parent_onnx, staged.path)
        if _sha256_file_required(staged.path, "staged ONNX") != expected_onnx_sha256:
            raise ProbeError("staged ONNX SHA-256 does not match the expected ONNX")
    except (OSError, ProbeError):
        # Staging names are private and unguessable, but portable replacement-safe cleanup is
        # unavailable after a failure. Leave the evidence rather than risk deleting another writer.
        raise
    return staged


def _verify_onnx_unchanged_after_inference(
    staged_onnx: Path, parent_onnx: Path, expected_onnx_sha256: str
) -> None:
    if _sha256_file_required(staged_onnx, "staged ONNX") != expected_onnx_sha256:
        raise ProbeError("ONNX changed during parity inference (staged copy)")
    if _sha256_file_required(parent_onnx, "parent ONNX") != expected_onnx_sha256:
        raise ProbeError("ONNX changed during parity inference (parent evidence)")


def _write_report_no_replace(report: dict[str, Any], reservation: _OutputReservation) -> None:
    """Publish prepared bytes through portable exclusive creation, never replacement."""
    try:
        report_bytes = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProbeError("cannot serialize probe report as standards-compliant JSON") from exc
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    sidecar_bytes = f"{report_sha256}  {reservation.destination.name}\n".encode("ascii")
    try:
        _write_exclusive_bytes(reservation.destination, report_bytes)
        _write_exclusive_bytes(reservation.sidecar, sidecar_bytes)
    except OSError as exc:
        # The sidecar is completion evidence. Never roll back any public path: an interloper may
        # have replaced it between creation and cleanup, and a partial state must fail closed.
        raise ProbeError(
            f"cannot publish probe report without overwrite: {reservation.destination}"
        ) from exc
    try:
        if (
            reservation.destination.read_bytes() != report_bytes
            or reservation.sidecar.read_bytes() != sidecar_bytes
        ):
            raise ProbeError("published probe report pair changed after exclusive publication")
    except OSError as exc:
        raise ProbeError("cannot verify published probe report pair") from exc


def _write_exclusive_bytes(destination: Path, payload: bytes) -> None:
    """Write one public file with the portable O_EXCL no-replace primitive."""
    descriptor = os.open(str(destination), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--parent-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parent-git-sha", required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--expected-onnx-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_probe(
            args.repo,
            args.parent_workspace,
            args.output,
            expected_parent_git_sha=args.expected_parent_git_sha,
            expected_gate_sha256=args.expected_gate_sha256,
            expected_onnx_sha256=args.expected_onnx_sha256,
        )
    except ProbeError as exc:
        parser.error(str(exc))
    print(f"PARITY PROBE: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
