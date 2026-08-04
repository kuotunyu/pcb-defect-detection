"""Small, testable runtime primitives used by the Colab handoff notebooks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class NotebookRuntimeError(RuntimeError):
    """Raised when notebook command evidence or parity completion checks fail."""


def _emit_line(line: str) -> None:
    print(line, end="", flush=True)


def run_streaming_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    label: str,
    popen: Callable[..., Any] = subprocess.Popen,
    emit: Callable[[str], None] = _emit_line,
) -> None:
    """Tee one long-running combined stream to live output and an append-only log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {label} attempt started {attempt} =====\n")
        handle.flush()
        process = popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            process.wait()
            raise NotebookRuntimeError(f"{label} did not expose a combined output stream")
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            emit(line)
        returncode = process.wait()
        handle.write(f"===== {label} attempt finished returncode={returncode} =====\n")
        handle.flush()
    if returncode:
        raise NotebookRuntimeError(f"{label} failed with returncode={returncode}; log: {log_path}")


def run_captured_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    label: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Run a short command after exclusively reserving its diagnostic log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = log_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise NotebookRuntimeError(
            f"{label} command log already exists; refusing to overwrite: {log_path}"
        ) from exc
    with handle:
        result = runner(list(command), cwd=cwd, text=True, capture_output=True)
        transcript = result.stdout + "\n--- STDERR ---\n" + result.stderr
        handle.write(transcript)
        handle.flush()
    if result.returncode:
        _emit_line(transcript)
        raise NotebookRuntimeError(
            f"{label} failed with returncode={result.returncode}; log: {log_path}"
        )
    return result


def verify_probe_result(
    report_path: Path,
    *,
    expected_parent_git_sha: str,
    expected_gate_sha256: str,
    expected_onnx_sha256: str,
) -> dict[str, object]:
    """Verify the exact published probe pair and its deployment-probe completion contract."""
    sidecar = report_path.with_suffix(report_path.suffix + ".sha256")
    try:
        report_bytes = report_path.read_bytes()
        sidecar_bytes = sidecar.read_bytes()
    except OSError as exc:
        raise NotebookRuntimeError("PARITY PROBE report or SHA-256 sidecar is missing") from exc
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    expected_sidecar = f"{report_sha256}  {report_path.name}\n".encode("ascii")
    if sidecar_bytes != expected_sidecar:
        raise NotebookRuntimeError(
            "PARITY PROBE report SHA-256 sidecar bytes, name, or hash are invalid"
        )
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookRuntimeError("PARITY PROBE report is not valid JSON") from exc
    if not isinstance(report, dict):
        raise NotebookRuntimeError("PARITY PROBE report must be a JSON object")
    if report.get("status") != "complete" or report.get("passed") is not True:
        raise NotebookRuntimeError("PARITY PROBE report does not confirm completion")
    parent = _required_mapping(report, "parent")
    _require_equal(
        parent, "experiment_git_sha", expected_parent_git_sha, "parent experiment Git SHA"
    )
    _require_equal(
        parent, "deployment_gate_sha256", expected_gate_sha256, "parent failed-gate SHA-256"
    )
    _require_equal(parent, "onnx_sha256", expected_onnx_sha256, "parent ONNX SHA-256")
    _require_equal(parent, "parity_onnx_sha256", expected_onnx_sha256, "parent parity ONNX SHA-256")
    parity = _required_mapping(report, "parity")
    _require_equal(parity, "reference_backend", "ultralytics-onnx", "reference backend")
    _require_equal(parity, "candidate_backend", "standalone-onnxruntime", "candidate backend")
    _require_equal(parity, "onnx_sha256", expected_onnx_sha256, "parity ONNX SHA-256")
    _require_exact_integer(parity, "n_images", 60)
    _require_exact_integer(parity, "required_images", 60)
    _require_exact_integer(parity, "n_failed", 0)
    per_image = parity.get("per_image")
    if not isinstance(per_image, dict) or len(per_image) != 60:
        raise NotebookRuntimeError("PARITY PROBE requires 60 per-image records")
    return report


def _required_mapping(report: dict[str, object], key: str) -> dict[str, object]:
    value = report.get(key)
    if not isinstance(value, dict):
        raise NotebookRuntimeError(f"PARITY PROBE report has no {key} object")
    return value


def _require_equal(mapping: dict[str, object], key: str, expected: str, label: str) -> None:
    if mapping.get(key) != expected:
        raise NotebookRuntimeError(f"PARITY PROBE {label} mismatch")


def _require_exact_integer(mapping: dict[str, object], key: str, expected: int) -> None:
    value = mapping.get(key)
    if type(value) is not int or value != expected:
        raise NotebookRuntimeError(f"PARITY PROBE {key} must be the exact integer {expected}")
