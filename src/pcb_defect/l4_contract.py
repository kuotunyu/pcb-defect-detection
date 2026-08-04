"""Immutable identity and evidence verification for L4 benchmark handoffs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pcb_defect.constants import CLASSES


class L4ContractError(RuntimeError):
    """The L4 handoff evidence does not satisfy the immutable contract."""


def _require_lowercase_hex(value: str, length: int, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise L4ContractError(f"{label} must be {length} lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class L4ParentIdentity:
    """Content identities inherited from the parent experiment workspace."""

    experiment_git_sha: str
    deployment_gate_sha256: str
    checkpoint_sha256: str
    onnx_sha256: str

    @classmethod
    def parse(
        cls,
        *,
        experiment_git_sha: str,
        deployment_gate_sha256: str,
        checkpoint_sha256: str,
        onnx_sha256: str,
    ) -> L4ParentIdentity:
        _require_lowercase_hex(experiment_git_sha, 40, "parent experiment Git SHA")
        _require_lowercase_hex(deployment_gate_sha256, 64, "deployment-gate SHA-256")
        _require_lowercase_hex(checkpoint_sha256, 64, "checkpoint SHA-256")
        _require_lowercase_hex(onnx_sha256, 64, "ONNX SHA-256")
        return cls(experiment_git_sha, deployment_gate_sha256, checkpoint_sha256, onnx_sha256)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class L4RunIdentity:
    """Runner and parent identities required for an L4 benchmark run."""

    runner_git_sha: str
    parent: L4ParentIdentity

    @classmethod
    def parse(cls, *, runner_git_sha: str, **parent_values: str) -> L4RunIdentity:
        _require_lowercase_hex(runner_git_sha, 40, "runner Git SHA")
        parent = L4ParentIdentity.parse(**parent_values)
        if runner_git_sha == parent.experiment_git_sha:
            raise L4ContractError("runner and parent experiment Git SHAs must differ")
        return cls(runner_git_sha, parent)


@dataclass(frozen=True, slots=True)
class VerifiedL4ParentInputs:
    """Verified parent evidence available to an L4 benchmark runner."""

    experiment_git_sha: str
    gate_path: Path
    gate: dict[str, Any]
    checkpoint_path: Path
    onnx_path: Path
    calibration_yaml: Path
    calibration_images: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class VerifiedL4Inputs:
    """Verified runner identity together with verified parent evidence."""

    runner_git_sha: str
    parent: VerifiedL4ParentInputs


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _contained_path(workspace: Path, candidate: Path, label: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise L4ContractError(f"{label} path escapes parent workspace") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file")
    return resolved


def _read_simple_calibration_yaml(path: Path) -> dict[str, Any]:
    """Read only the canonical safe subset emitted by the deployment stage."""

    lines = path.read_text(encoding="utf-8").splitlines()
    scalar_keys = ("train", "val", "test")
    forbidden = set("[]{}&*!|>@'\"#")
    if len(lines) < len(scalar_keys) + 2:
        raise ValueError("calibration YAML is not the canonical deployment mapping")
    values: dict[str, Any] = {}
    for key, line in zip(scalar_keys, lines, strict=False):
        prefix = f"{key}: "
        if not line.startswith(prefix):
            raise ValueError("calibration YAML is not the canonical deployment mapping")
        value = line.removeprefix(prefix)
        if (
            not value
            or value != value.strip()
            or any(character in forbidden for character in value)
        ):
            raise ValueError("calibration YAML is not the canonical deployment mapping")
        values[key] = value
    if lines[len(scalar_keys)] != "names:":
        raise ValueError("calibration YAML is not the canonical deployment mapping")
    names: dict[int, str] = {}
    for expected_index, line in enumerate(lines[len(scalar_keys) + 1 :]):
        match = re.fullmatch(r"  (0|[1-9][0-9]*): ([a-z][a-z0-9_]{0,127})", line)
        if match is None or int(match.group(1)) != expected_index:
            raise ValueError("calibration YAML names mapping is unsafe or noncanonical")
        name = match.group(2)
        if name in {"null", "true", "false", "yes", "no", "on", "off"} or name in names.values():
            raise ValueError("calibration YAML names mapping is unsafe or noncanonical")
        names[expected_index] = name
    if names != dict(enumerate(CLASSES)):
        raise ValueError("calibration YAML names mapping is not the canonical class mapping")
    values["names"] = names
    return values


def _read_canonical_calibration_list(workspace: Path, path: Path) -> tuple[Path, ...]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise L4ContractError("calibration list bytes mismatch") from error
    if not text or "\r" in text or not text.endswith("\n"):
        raise L4ContractError("calibration list bytes mismatch")
    lines = text[:-1].split("\n")
    if not lines or any(not line or not Path(line).is_absolute() for line in lines):
        raise L4ContractError("calibration list bytes mismatch")
    paths = tuple(_contained_path(workspace, Path(line), "calibration image") for line in lines)
    expected = "".join(f"{image}\n" for image in paths).encode("utf-8")
    if raw != expected:
        raise L4ContractError("calibration list bytes mismatch")
    return paths


def _verify_l4_parent_inputs(workspace: Path, parent: L4ParentIdentity) -> VerifiedL4ParentInputs:
    workspace = workspace.resolve()
    if workspace.name != parent.experiment_git_sha[:12]:
        raise L4ContractError("parent workspace is not derived from the parent experiment SHA")
    gate_path = _contained_path(
        workspace, workspace / "deployment" / "deployment_gate.json", "gate"
    )
    if _sha256_file(gate_path) != parent.deployment_gate_sha256:
        raise L4ContractError("raw deployment-gate SHA-256 mismatch")
    gate = _read_json_object(gate_path, "deployment gate")
    if gate.get("passed") is not True or gate.get("git_sha") != parent.experiment_git_sha:
        raise L4ContractError("deployment gate did not pass for the expected parent experiment")
    manifest_path = workspace / "inputs" / "paired_split_manifest.json"
    manifest = _read_json_object(manifest_path, "protocol manifest")
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if _sha256_json(manifest_payload) != manifest.get("manifest_sha256"):
        raise L4ContractError("protocol-manifest SHA-256 mismatch")
    if gate["dataset_sha256"] != manifest["dataset"]["sha256"]:
        raise L4ContractError("dataset SHA-256 mismatch")
    if gate["manifest_sha256"] != manifest["manifest_sha256"]:
        raise L4ContractError("protocol-manifest identity mismatch")
    artifacts = gate["artifacts"]
    checkpoint_path = _contained_path(
        workspace, workspace / artifacts["source_checkpoint"], "checkpoint"
    )
    onnx_path = _contained_path(workspace, workspace / "deployment" / artifacts["onnx"], "ONNX")
    if _sha256_file(checkpoint_path) != parent.checkpoint_sha256:
        raise L4ContractError("checkpoint SHA-256 mismatch")
    if _sha256_file(onnx_path) != parent.onnx_sha256:
        raise L4ContractError("ONNX SHA-256 mismatch")
    if artifacts["source_checkpoint_sha256"] != parent.checkpoint_sha256:
        raise L4ContractError("deployment gate checkpoint SHA-256 mismatch")
    if artifacts["onnx_sha256"] != parent.onnx_sha256:
        raise L4ContractError("deployment gate ONNX SHA-256 mismatch")
    calibration_yaml = _contained_path(
        workspace, workspace / "deployment" / "calibration.yaml", "calibration"
    )
    calibration_config = _read_simple_calibration_yaml(calibration_yaml)
    expected_list = _contained_path(
        workspace, workspace / "runtime_data" / "grouped" / "calibration.txt", "calibration list"
    )
    if any(calibration_config[split] != str(expected_list) for split in ("train", "val", "test")):
        raise L4ContractError("calibration list path mismatch")
    calibration_images = _read_canonical_calibration_list(workspace, expected_list)
    observed_stems = [path.stem for path in calibration_images]
    expected_stems = manifest["partitions"]["calibration"]
    if observed_stems != expected_stems or set(observed_stems) & set(
        manifest["partitions"]["final_test"]
    ):
        raise L4ContractError("calibration partition mismatch")
    sample_by_stem = {row["stem"]: row for row in manifest["dataset"]["samples"]}
    for path in calibration_images:
        if _sha256_file(path) != sample_by_stem[path.stem]["image_sha256"]:
            raise L4ContractError("calibration image SHA-256 mismatch")
    return VerifiedL4ParentInputs(
        parent.experiment_git_sha,
        gate_path,
        gate,
        checkpoint_path,
        onnx_path,
        calibration_yaml,
        calibration_images,
    )


def verify_l4_parent_inputs(workspace: Path, parent: L4ParentIdentity) -> VerifiedL4ParentInputs:
    """Fail closed unless all parent evidence agrees with the supplied identity."""

    try:
        return _verify_l4_parent_inputs(workspace, parent)
    except L4ContractError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise L4ContractError("L4 parent evidence is missing or malformed") from error


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise L4ContractError("runner repository identity or cleanliness mismatch") from error
    return result.stdout.strip()


def verify_l4_inputs(repo: Path, workspace: Path, identity: L4RunIdentity) -> VerifiedL4Inputs:
    """Verify a clean runner checkout and all immutable parent evidence."""

    repo = repo.resolve()
    observed_runner = _git(repo, "rev-parse", "HEAD")
    if observed_runner != identity.runner_git_sha or _git(repo, "status", "--porcelain"):
        raise L4ContractError("runner repository identity or cleanliness mismatch")
    parent = verify_l4_parent_inputs(workspace, identity.parent)
    return VerifiedL4Inputs(observed_runner, parent)
