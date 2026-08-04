"""Create a one-commit, history-free Git bundle for the immutable Colab handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import yaml

RELEASE_NAME = "kuotunyu"
RELEASE_EMAIL = "61350295+kuotunyu@users.noreply.github.com"


class HandoffError(RuntimeError):
    """A safe clean-snapshot bundle could not be created."""


def create_clean_bundle(
    source_repo: Path, output_dir: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Collapse only HEAD's current tree into one new commit, excluding all source history."""
    source_repo = source_repo.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise HandoffError(f"refusing to overwrite handoff directory: {output_dir}")
    source_sha = _git(source_repo, "rev-parse", "HEAD")
    if _git(source_repo, "status", "--porcelain"):
        raise HandoffError("source repository must be clean before creating a handoff")
    source_date = _git(source_repo, "show", "-s", "--format=%cI", "HEAD")
    output_dir.mkdir(parents=True)
    archive_path = output_dir / ".source.zip"
    snapshot = output_dir / ".snapshot"
    subprocess.run(
        ["git", "archive", "--format=zip", "--output", str(archive_path), "HEAD"],
        cwd=source_repo,
        check=True,
    )
    snapshot.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(snapshot)
    provenance = {**metadata, "source_git_sha": source_sha}
    (snapshot / ".source_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    _git(snapshot, "init", "-b", "main")
    _git(snapshot, "add", "-A")
    commit_environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": RELEASE_NAME,
        "GIT_AUTHOR_EMAIL": RELEASE_EMAIL,
        "GIT_COMMITTER_NAME": RELEASE_NAME,
        "GIT_COMMITTER_EMAIL": RELEASE_EMAIL,
        "GIT_AUTHOR_DATE": source_date,
        "GIT_COMMITTER_DATE": source_date,
    }
    subprocess.run(
        ["git", "commit", "-m", "Immutable Colab source snapshot"],
        cwd=snapshot,
        check=True,
        capture_output=True,
        env=commit_environment,
    )
    snapshot_sha = _git(snapshot, "rev-parse", "HEAD")
    bundle_path = output_dir / "pcb-defect-source.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle_path), "HEAD", "main"],
        cwd=snapshot,
        check=True,
    )
    bundle_sha = _sha256_file(bundle_path)
    result = {
        "source_git_sha": source_sha,
        "snapshot_git_sha": snapshot_sha,
        "bundle_sha256": bundle_sha,
        "bundle_path": bundle_path,
    }
    (output_dir / "handoff_manifest.json").write_text(
        json.dumps(
            {
                **provenance,
                "snapshot_git_sha": snapshot_sha,
                "bundle": bundle_path.name,
                "bundle_sha256": bundle_sha,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    archive_path.unlink()
    shutil.rmtree(snapshot, onerror=_remove_readonly)
    return result


def render_notebook(
    template: Path,
    destination: Path,
    replacements: dict[str, str],
) -> str:
    """Create a ready-to-run notebook without hand-edited immutable placeholders."""
    if destination.exists():
        raise HandoffError(f"refusing to overwrite rendered notebook: {destination}")
    source = template.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        if source.count(placeholder) != 1:
            raise HandoffError(
                f"notebook must contain exactly one {placeholder} placeholder: {template}"
            )
        source = source.replace(placeholder, value)
    try:
        notebook = json.loads(source)
    except json.JSONDecodeError as exc:
        raise HandoffError(f"rendered notebook is invalid JSON: {template}") from exc
    if any(cell.get("outputs") for cell in notebook.get("cells", [])):
        raise HandoffError(f"refusing to render a notebook with persisted outputs: {template}")
    destination.write_text(source, encoding="utf-8", newline="\n")
    return _sha256_file(destination)


def project_handoff_metadata(repo: Path) -> dict[str, Any]:
    """Return shared immutable protocol and base-model handoff metadata."""
    protocol = json.loads(
        (repo / "reports" / "protocol" / "paired_split_manifest.json").read_text(encoding="utf-8")
    )
    base_contract_path = repo / "configs" / "base_model.yaml"
    base_contract = yaml.safe_load(base_contract_path.read_text(encoding="utf-8"))
    return {
        "protocol_version": protocol["protocol_version"],
        "dataset_sha256": protocol["dataset"]["sha256"],
        "manifest_sha256": protocol["manifest_sha256"],
        "base_model_contract_sha256": _sha256_file(base_contract_path),
        "base_model_source": base_contract["source"],
        "base_model_revision": base_contract["revision"],
        "base_model_sha256": base_contract["sha256"],
    }


def _write_l4_migration_notebook(destination: Path) -> str:
    """Write a sentinel-free legacy boundary that cannot start the private L4 stage."""
    if destination.exists():
        raise HandoffError(f"refusing to overwrite rendered notebook: {destination}")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# L4 stage-specific handoff required\n",
                    "\n",
                    "This legacy paired handoff cannot bind the reviewed parent checkpoint and "
                    "ONNX identities. Generate a dedicated private L4 handoff with "
                    "`python -m pcb_defect.l4_handoff` and the four reviewed parent values.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "raise RuntimeError(\n",
                    "    'Legacy paired handoff cannot run the L4 stage; use '\n",
                    "    'python -m pcb_defect.l4_handoff with reviewed parent identities'\n",
                    ")\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8", newline="\n")
    return _sha256_file(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-parent-git-sha")
    parser.add_argument("--probe-parent-deployment-gate-sha256")
    parser.add_argument("--probe-parent-onnx-sha256")
    args = parser.parse_args(argv)
    probe_values = (
        args.probe_parent_git_sha,
        args.probe_parent_deployment_gate_sha256,
        args.probe_parent_onnx_sha256,
    )
    probe_argument_presence = tuple(value is not None for value in probe_values)
    if any(probe_argument_presence) and not all(probe_argument_presence):
        raise HandoffError("probe arguments must be supplied together")
    has_probe_arguments = all(probe_argument_presence)
    if has_probe_arguments:
        _validate_lowercase_hex(args.probe_parent_git_sha, 40, "probe parent Git SHA")
        _validate_lowercase_hex(
            args.probe_parent_deployment_gate_sha256,
            64,
            "probe parent deployment gate SHA-256",
        )
        _validate_lowercase_hex(args.probe_parent_onnx_sha256, 64, "probe parent ONNX SHA-256")
    repo = args.repo.resolve()
    metadata = {
        **project_handoff_metadata(repo),
        "a100_notebook": "paired_experiment_a100.ipynb",
        "l4_notebook": "deployment_benchmark_l4.ipynb",
        "a100_template_sha256": _sha256_file(repo / "notebooks" / "paired_experiment_a100.ipynb"),
        "l4_template_sha256": _sha256_file(repo / "notebooks" / "deployment_benchmark_l4.ipynb"),
        "drive_bundle_path": (
            "/content/drive/MyDrive/pcb-defect-paired/handoff/pcb-defect-source.bundle"
        ),
    }
    if has_probe_arguments:
        metadata.update(
            {
                "probe_notebook": "deployment_parity_probe_a100.ipynb",
                "probe_parent_git_sha": args.probe_parent_git_sha,
                "probe_parent_deployment_gate_sha256": (args.probe_parent_deployment_gate_sha256),
                "probe_parent_onnx_sha256": args.probe_parent_onnx_sha256,
            }
        )
    result = create_clean_bundle(repo, args.output, metadata)
    base_replacements = {
        "PASTE_FINAL_BUNDLE_SHA256": result["bundle_sha256"],
        "PASTE_FINAL_GIT_SHA": result["snapshot_git_sha"],
    }
    a100_notebook = metadata["a100_notebook"]
    l4_notebook = metadata["l4_notebook"]
    notebook_hashes = {
        "a100_notebook_sha256": render_notebook(
            repo / "notebooks" / a100_notebook,
            args.output / a100_notebook,
            replacements=base_replacements,
        ),
        "l4_notebook_sha256": _write_l4_migration_notebook(args.output / l4_notebook),
    }
    if has_probe_arguments:
        probe_replacements = {
            **base_replacements,
            "PASTE_PARENT_EXPERIMENT_GIT_SHA": args.probe_parent_git_sha,
            "PASTE_PARENT_DEPLOYMENT_GATE_SHA256": args.probe_parent_deployment_gate_sha256,
            "PASTE_PARENT_ONNX_SHA256": args.probe_parent_onnx_sha256,
        }
        notebook_hashes["probe_notebook_sha256"] = render_notebook(
            repo / "notebooks" / metadata["probe_notebook"],
            args.output / metadata["probe_notebook"],
            replacements=probe_replacements,
        )
    manifest_path = args.output / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(notebook_hashes)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"source_git_sha={result['source_git_sha']}")
    print(f"snapshot_git_sha={result['snapshot_git_sha']}")
    print(f"bundle_sha256={result['bundle_sha256']}")
    print(f"a100_notebook_sha256={notebook_hashes['a100_notebook_sha256']}")
    if "probe_notebook_sha256" in notebook_hashes:
        print(f"probe_notebook_sha256={notebook_hashes['probe_notebook_sha256']}")
    print(f"handoff_dir={args.output.resolve()}")
    return 0


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HandoffError(f"Git command failed in {repo}: {' '.join(args)}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_lowercase_hex(value: str, length: int, label: str) -> None:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise HandoffError(f"{label} must be {length} lowercase hexadecimal characters")


def _remove_readonly(function, path: str, _error) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


if __name__ == "__main__":
    raise SystemExit(main())
