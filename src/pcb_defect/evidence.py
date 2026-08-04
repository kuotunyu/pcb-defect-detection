"""Machine-readable run evidence with portable, content-addressed artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA_VERSION = "1.0"
RUN_STATUSES = {"planned", "running", "complete", "failed"}
ARMS = {"grouped", "leaky_control"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class EvidenceError(ValueError):
    """A run record cannot be treated as verifiable evidence."""


def artifact_ref(path: Path, *, relative_to: Path) -> dict[str, str | int]:
    """Return a portable path, byte count and SHA-256 for an existing file."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(relative_to.resolve())
    except ValueError as exc:
        raise EvidenceError(f"artifact is outside evidence root: {resolved}") from exc
    if not resolved.is_file():
        raise EvidenceError(f"artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": relative.as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": resolved.stat().st_size,
    }


def validate_run_record(record: dict[str, Any]) -> None:
    """Fail closed unless a record satisfies the release evidence contract."""
    _require_equal(record, "schema_version", SCHEMA_VERSION)
    _require_text(record, "run_id")
    if record.get("arm") not in ARMS:
        raise EvidenceError(f"arm must be one of {sorted(ARMS)}")
    if not isinstance(record.get("seed"), int):
        raise EvidenceError("seed must be an integer")
    status = record.get("status")
    if status not in RUN_STATUSES:
        raise EvidenceError(f"status must be one of {sorted(RUN_STATUSES)}")

    timestamps = _require_mapping(record, "timestamps")
    for key in ("created_at_utc", "updated_at_utc"):
        value = timestamps.get(key)
        if not isinstance(value, str) or not UTC_RE.fullmatch(value):
            raise EvidenceError(f"timestamps.{key} must be an ISO-8601 UTC value ending in Z")

    provenance = _require_mapping(record, "provenance")
    git_sha = provenance.get("git_sha")
    if not isinstance(git_sha, str) or not GIT_SHA_RE.fullmatch(git_sha):
        raise EvidenceError("provenance.git_sha must be a 40-character lowercase Git SHA")
    if not isinstance(provenance.get("git_dirty"), bool):
        raise EvidenceError("provenance.git_dirty must be boolean")
    command = provenance.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise EvidenceError("provenance.command must be a non-empty argument list")
    environment = _require_mapping(provenance, "environment", prefix="provenance")
    _require_text(environment, "python", prefix="provenance.environment")
    _require_text(environment, "platform", prefix="provenance.environment")
    if not isinstance(environment.get("packages"), dict):
        raise EvidenceError("provenance.environment.packages must be a mapping")

    protocol = _require_mapping(record, "protocol")
    _require_text(protocol, "version", prefix="protocol")
    _validate_ref(_require_mapping(protocol, "manifest", prefix="protocol"), "protocol.manifest")
    _validate_sha256(protocol.get("manifest_sha256"), "protocol.manifest_sha256")
    _validate_sha256(protocol.get("dataset_sha256"), "protocol.dataset_sha256")

    training = _require_mapping(record, "training")
    _validate_ref(_require_mapping(training, "config", prefix="training"), "training.config")
    if not isinstance(training.get("resolved"), dict):
        raise EvidenceError("training.resolved must be a mapping")
    base_model = _require_mapping(training, "base_model", prefix="training")
    _require_text(base_model, "source", prefix="training.base_model")
    _require_text(base_model, "revision", prefix="training.base_model")
    _require_text(base_model, "filename", prefix="training.base_model")
    _validate_sha256(base_model.get("sha256"), "training.base_model.sha256")
    if not isinstance(base_model.get("bytes"), int) or base_model["bytes"] <= 0:
        raise EvidenceError("training.base_model.bytes must be a positive integer")
    _validate_ref(
        _require_mapping(base_model, "contract", prefix="training.base_model"),
        "training.base_model.contract",
    )

    artifacts = _require_mapping(record, "artifacts")
    metrics = _require_mapping(record, "metrics")
    for name, ref in artifacts.items():
        if not isinstance(ref, dict):
            raise EvidenceError(f"artifacts.{name} must be an artifact reference")
        _validate_ref(ref, f"artifacts.{name}")
    for name, ref in metrics.items():
        if not isinstance(ref, dict):
            raise EvidenceError(f"metrics.{name} must be an artifact reference")
        _validate_ref(ref, f"metrics.{name}")

    if status == "complete":
        if "best_checkpoint" not in artifacts:
            raise EvidenceError("complete run is missing artifacts.best_checkpoint")
        if "validation" not in metrics:
            raise EvidenceError("complete run is missing metrics.validation")
    failure = record.get("failure")
    if status == "failed" and not isinstance(failure, dict):
        raise EvidenceError("failed run must include a failure mapping")
    if status != "failed" and failure is not None:
        raise EvidenceError("failure must be null unless status is failed")


def write_run_record(record: dict[str, Any], destination: Path) -> None:
    """Validate and write stable UTF-8 JSON with LF and a final newline."""
    validate_run_record(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_ref(ref: dict[str, Any], name: str) -> None:
    path = ref.get("path")
    if not isinstance(path, str) or not _is_portable_relative(path):
        raise EvidenceError(f"{name}.path must be a portable relative path")
    _validate_sha256(ref.get("sha256"), f"{name}.sha256")
    size = ref.get("bytes")
    if not isinstance(size, int) or size < 0:
        raise EvidenceError(f"{name}.bytes must be a non-negative integer")


def _is_portable_relative(value: str) -> bool:
    if "\\" in value:
        return False
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        bool(value)
        and not posix.is_absolute()
        and not windows.is_absolute()
        and ".." not in posix.parts
    )


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise EvidenceError(f"{name} must be a 64-character lowercase SHA-256")


def _require_mapping(
    mapping: dict[str, Any], key: str, *, prefix: str | None = None
) -> dict[str, Any]:
    value = mapping.get(key)
    name = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must be a mapping")
    return value


def _require_text(mapping: dict[str, Any], key: str, *, prefix: str | None = None) -> str:
    value = mapping.get(key)
    name = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{name} must be non-empty text")
    return value


def _require_equal(mapping: dict[str, Any], key: str, expected: Any) -> None:
    if mapping.get(key) != expected:
        raise EvidenceError(f"{key} must equal {expected!r}")
