"""Create or verify the private evidence package for one completed L4 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcb_defect.benchmark import benchmark_is_complete
from pcb_defect.l4_contract import L4RunIdentity, verify_l4_inputs
from pcb_defect.result_package import (
    PackageError,
    _package_manifest,
    create_verifiable_zip,
    verify_verifiable_zip,
)

_PIXEL_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-runner-git-sha", required=True)
    parser.add_argument("--expected-experiment-git-sha", required=True)
    parser.add_argument("--expected-deployment-gate-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-onnx-sha256", required=True)
    args = parser.parse_args(argv)
    identity = L4RunIdentity.parse(
        runner_git_sha=args.expected_runner_git_sha,
        experiment_git_sha=args.expected_experiment_git_sha,
        deployment_gate_sha256=args.expected_deployment_gate_sha256,
        checkpoint_sha256=args.expected_checkpoint_sha256,
        onnx_sha256=args.expected_onnx_sha256,
    )
    package = create_or_verify_l4_package(
        args.repo.resolve(),
        args.workspace.resolve(),
        args.dataset.resolve(),
        args.output_root.resolve(),
        identity,
    )
    print(f"L4 RESULT PACKAGE: {package}")
    return 0


def l4_package_name(identity: L4RunIdentity) -> str:
    """Derive the only allowed package filename from both verified Git identities."""
    return (
        f"paired-results-l4-{identity.parent.experiment_git_sha[:12]}-"
        f"runner-{identity.runner_git_sha[:12]}.zip"
    )


def collect_l4_files(
    repo: Path, workspace: Path, dataset_root: Path, identity: L4RunIdentity
) -> list[Path]:
    """Return exactly the non-pixel private inputs required to audit an L4 result."""
    repo = repo.resolve()
    workspace = workspace.resolve()
    dataset_root = dataset_root.resolve()
    verified = verify_l4_inputs(repo, workspace, dataset_root, identity)
    report_path = workspace / "benchmark_l4" / identity.runner_git_sha[:12] / "benchmark_l4.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError("L4 benchmark report is missing or malformed") from exc
    if not isinstance(report, dict) or not benchmark_is_complete(
        repo, workspace, dataset_root, identity, report
    ):
        raise PackageError("L4 benchmark is incomplete or hash-mismatched")
    files = _l4_package_paths(workspace, identity, verified)
    for relative in files:
        resolved = (workspace / relative).resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise PackageError(f"L4 package path escapes workspace: {relative}") from exc
        if not resolved.is_file():
            raise PackageError(f"required L4 package file is missing: {relative}")
        if any(suffix.lower() in _PIXEL_SUFFIXES for suffix in resolved.suffixes):
            raise PackageError(f"dataset pixels are forbidden in the L4 package: {relative}")
    return files


def _l4_package_paths(workspace: Path, identity: L4RunIdentity, verified: object) -> list[Path]:
    parent = verified.parent
    try:
        manifest = parent.manifest_path.relative_to(workspace)
        onnx = parent.onnx_path.relative_to(workspace)
        checkpoint = parent.checkpoint_path.relative_to(workspace)
    except (AttributeError, ValueError) as exc:
        raise PackageError("verified L4 artifacts do not belong to the parent workspace") from exc
    report = Path("benchmark_l4") / identity.runner_git_sha[:12] / "benchmark_l4.json"
    return [
        Path("inputs/input_lock.json"),
        manifest,
        Path("deployment/calibration.yaml"),
        Path("deployment/deployment_gate.json"),
        Path("deployment/model_contract.candidate.json"),
        onnx,
        checkpoint,
        report,
        report.with_name("best_fp16.engine"),
        Path("l4_logs") / identity.runner_git_sha[:12] / "benchmark_command.log",
    ]


def create_or_verify_l4_package(
    repo: Path,
    workspace: Path,
    dataset_root: Path,
    output_root: Path,
    identity: L4RunIdentity,
) -> Path:
    """Reuse only a completely verified pair; otherwise create the identity-derived pair."""
    package = output_root.resolve() / l4_package_name(identity)
    sidecar = package.with_suffix(package.suffix + ".sha256")
    if package.exists() or sidecar.exists():
        if package.exists() != sidecar.exists():
            verify_verifiable_zip(package)
        files = collect_l4_files(
            repo.resolve(), workspace.resolve(), dataset_root.resolve(), identity
        )
        manifest = verify_verifiable_zip(package)
        expected, _ = _package_manifest(workspace.resolve(), files)
        if _manifest_inventory(manifest) != expected:
            raise PackageError("existing L4 package inventory does not match current workspace")
        return package
    files = collect_l4_files(repo.resolve(), workspace.resolve(), dataset_root.resolve(), identity)
    create_verifiable_zip(workspace.resolve(), files, package)
    manifest = verify_verifiable_zip(package)
    expected, _ = _package_manifest(workspace.resolve(), files)
    if _manifest_inventory(manifest) != expected:
        raise PackageError("created L4 package inventory does not match current workspace")
    return package


def _manifest_inventory(manifest: dict[str, object]) -> dict[str, object]:
    return {"schema_version": manifest.get("schema_version"), "files": manifest.get("files")}


if __name__ == "__main__":
    raise SystemExit(main())
