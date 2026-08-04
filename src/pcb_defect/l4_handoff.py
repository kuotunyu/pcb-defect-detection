"""Create an immutable stage-specific handoff for one private L4 benchmark."""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pcb_defect.handoff import (
    HandoffError,
    _remove_readonly,
    _sha256_file,
    create_clean_bundle,
    project_handoff_metadata,
    render_notebook,
)
from pcb_defect.l4_contract import L4ParentIdentity

_HANDOFF_FILES = {
    "deployment_benchmark_l4.ipynb",
    "handoff_manifest.json",
    "pcb-defect-source.bundle",
}
_PROJECT_METADATA_KEYS = {
    "protocol_version",
    "dataset_sha256",
    "manifest_sha256",
    "base_model_contract_sha256",
    "base_model_source",
    "base_model_revision",
    "base_model_sha256",
}
_PARENT_METADATA_KEYS = {
    "parent_experiment_git_sha",
    "parent_deployment_gate_sha256",
    "parent_checkpoint_sha256",
    "parent_onnx_sha256",
}
_PROVENANCE_KEYS = (
    _PROJECT_METADATA_KEYS
    | _PARENT_METADATA_KEYS
    | {
        "source_git_sha",
        "stage",
    }
)
_MANIFEST_KEYS = _PROVENANCE_KEYS | {
    "snapshot_git_sha",
    "bundle",
    "bundle_sha256",
    "l4_notebook",
    "l4_template_sha256",
    "l4_notebook_sha256",
    "drive_handoff_directory",
}
_ASSIGNMENT_ROLES = (
    "SOURCE_BUNDLE_SHA256",
    "RUNNER_GIT_SHA",
    "PARENT_EXPERIMENT_GIT_SHA",
    "PARENT_DEPLOYMENT_GATE_SHA256",
    "PARENT_CHECKPOINT_SHA256",
    "PARENT_ONNX_SHA256",
    "L4_HANDOFF_DIRECTORY",
)


class _ImmutableBindingCollector(ast.NodeVisitor):
    """Collect every Python binding or deletion of an immutable notebook role."""

    def __init__(self) -> None:
        self.bindings: dict[str, list[ast.AST]] = {role: [] for role in _ASSIGNMENT_ROLES}

    def _record(self, name: str | None, node: ast.AST) -> None:
        if name in self.bindings:
            self.bindings[name].append(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record(node.id, node)

    def visit_arg(self, node: ast.arg) -> None:
        self._record(node.arg, node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node.name, node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node.name, node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node.name, node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.asname or alias.name.split(".", maxsplit=1)[0], alias)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                for role in _ASSIGNMENT_ROLES:
                    self._record(role, alias)
            else:
                self._record(alias.asname or alias.name, alias)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._record(node.name, node)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._record(name, node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._record(name, node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self._record(node.name, node)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._record(node.name, node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self._record(node.rest, node)
        self.generic_visit(node)

    def _visit_type_parameter(self, node: ast.AST) -> None:
        name = getattr(node, "name", None)
        if not isinstance(name, str) or not name:
            raise HandoffError("L4 handoff notebook type parameter name is malformed")
        self._record(name, node)
        self.generic_visit(node)

    def visit_TypeVar(self, node: ast.AST) -> None:
        self._visit_type_parameter(node)

    def visit_TypeVarTuple(self, node: ast.AST) -> None:
        self._visit_type_parameter(node)

    def visit_ParamSpec(self, node: ast.AST) -> None:
        self._visit_type_parameter(node)


def create_l4_handoff(repo: Path, output_root: Path, parent: L4ParentIdentity) -> Path:
    """Atomically create and verify one runner-bound L4 handoff directory."""
    repo = repo.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".l4-handoff-stage-", dir=output_root))
    content = staging / "content"
    try:
        metadata = {
            **project_handoff_metadata(repo),
            **{f"parent_{key}": value for key, value in parent.as_dict().items()},
            "stage": "l4-benchmark",
        }
        result = create_clean_bundle(repo, content, metadata)
        runner_sha = result["snapshot_git_sha"]
        if runner_sha == parent.experiment_git_sha:
            raise HandoffError("runner and parent experiment Git SHAs must differ")
        final = output_root / f"colab-handoff-l4-{runner_sha[:12]}"
        drive_directory = f"/content/drive/MyDrive/pcb-defect-paired/handoff-l4/{runner_sha[:12]}"
        notebook_template = repo / "notebooks" / "deployment_benchmark_l4.ipynb"
        notebook_path = content / "deployment_benchmark_l4.ipynb"
        notebook_sha256 = render_notebook(
            notebook_template,
            notebook_path,
            {
                "PASTE_FINAL_BUNDLE_SHA256": result["bundle_sha256"],
                "PASTE_FINAL_GIT_SHA": runner_sha,
                "PASTE_PARENT_EXPERIMENT_GIT_SHA": parent.experiment_git_sha,
                "PASTE_PARENT_DEPLOYMENT_GATE_SHA256": parent.deployment_gate_sha256,
                "PASTE_PARENT_CHECKPOINT_SHA256": parent.checkpoint_sha256,
                "PASTE_PARENT_ONNX_SHA256": parent.onnx_sha256,
                "PASTE_L4_HANDOFF_DIRECTORY": drive_directory,
            },
        )
        manifest_path = content / "handoff_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "stage": "l4-benchmark",
                "l4_notebook": notebook_path.name,
                "l4_template_sha256": _sha256_file(notebook_template),
                "l4_notebook_sha256": notebook_sha256,
                "drive_handoff_directory": drive_directory,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _verify_l4_handoff(
            content,
            parent,
            expected_runner_sha=runner_sha,
            expected_bundle_sha256=result["bundle_sha256"],
            expected_drive_directory=drive_directory,
        )
        _publish_directory_no_replace(content, final)
        return final
    finally:
        if staging.exists():
            shutil.rmtree(staging, onerror=_remove_readonly)


def _verify_l4_handoff(
    content: Path,
    parent: L4ParentIdentity,
    *,
    expected_runner_sha: str,
    expected_bundle_sha256: str,
    expected_drive_directory: str,
) -> None:
    """Verify the complete handoff from its bundle and rendered file bytes."""
    if {path.name for path in content.iterdir()} != _HANDOFF_FILES:
        raise HandoffError("L4 handoff must contain exactly the three stage files")
    if any(not path.is_file() or path.is_symlink() for path in content.iterdir()):
        raise HandoffError("L4 handoff entries must be regular files")
    manifest_path = content / "handoff_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffError("L4 handoff manifest is missing or malformed") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise HandoffError("L4 handoff manifest fields are not exact")
    expected_parent = {f"parent_{key}": value for key, value in parent.as_dict().items()}
    if manifest.get("stage") != "l4-benchmark" or any(
        manifest.get(key) != value for key, value in expected_parent.items()
    ):
        raise HandoffError("L4 handoff parent identity mismatch")
    if manifest.get("snapshot_git_sha") != expected_runner_sha:
        raise HandoffError("L4 handoff runner identity mismatch")
    if manifest.get("bundle_sha256") != expected_bundle_sha256:
        raise HandoffError("L4 handoff expected bundle SHA-256 mismatch")
    if manifest.get("drive_handoff_directory") != expected_drive_directory:
        raise HandoffError("L4 handoff Drive handoff directory mismatch")
    bundle = content / "pcb-defect-source.bundle"
    notebook = content / "deployment_benchmark_l4.ipynb"
    if manifest.get("bundle") != bundle.name or manifest.get("l4_notebook") != notebook.name:
        raise HandoffError("L4 handoff manifest filenames mismatch")
    if manifest.get("bundle_sha256") != _sha256_file(bundle):
        raise HandoffError("L4 handoff bundle SHA-256 mismatch")
    if manifest.get("l4_notebook_sha256") != _sha256_file(notebook):
        raise HandoffError("L4 handoff notebook SHA-256 mismatch")
    _verify_rendered_notebook(
        notebook,
        {
            "SOURCE_BUNDLE_SHA256": expected_bundle_sha256,
            "RUNNER_GIT_SHA": expected_runner_sha,
            "PARENT_EXPERIMENT_GIT_SHA": parent.experiment_git_sha,
            "PARENT_DEPLOYMENT_GATE_SHA256": parent.deployment_gate_sha256,
            "PARENT_CHECKPOINT_SHA256": parent.checkpoint_sha256,
            "PARENT_ONNX_SHA256": parent.onnx_sha256,
            "L4_HANDOFF_DIRECTORY": expected_drive_directory,
        },
    )
    _verify_bundle_snapshot(bundle, manifest)


def _verify_rendered_notebook(path: Path, expected_assignments: dict[str, str]) -> None:
    source = path.read_text(encoding="utf-8")
    if "PASTE_" in source:
        raise HandoffError("L4 handoff notebook contains an unresolved placeholder")
    try:
        notebook = json.loads(source)
        cells = notebook["cells"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HandoffError("L4 handoff notebook is malformed") from exc
    if not isinstance(cells, list):
        raise HandoffError("L4 handoff notebook cells are malformed")
    if set(expected_assignments) != set(_ASSIGNMENT_ROLES) or len(
        set(expected_assignments.values())
    ) != len(_ASSIGNMENT_ROLES):
        raise HandoffError("L4 handoff immutable assignment expectations are not unique")
    code_cell_index = next(
        (index for index, cell in enumerate(cells) if cell.get("cell_type") == "code"), None
    )
    parsed_cells: list[tuple[int, ast.Module]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("outputs"):
            raise HandoffError("L4 handoff notebook contains persisted outputs")
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                raise HandoffError("L4 handoff notebook contains an execution count")
            try:
                tree = ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
            except (KeyError, TypeError, SyntaxError) as exc:
                raise HandoffError("L4 handoff notebook contains invalid Python") from exc
            parsed_cells.append((index, tree))
    if code_cell_index is None:
        raise HandoffError("L4 handoff notebook has no code cell")
    first_tree = next(tree for index, tree in parsed_cells if index == code_cell_index)
    approved_targets: dict[str, list[ast.Name]] = {role: [] for role in _ASSIGNMENT_ROLES}
    for statement in first_tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in approved_targets
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) is str
            and statement.value.value == expected_assignments[statement.targets[0].id]
        ):
            approved_targets[statement.targets[0].id].append(statement.targets[0])
    collector = _ImmutableBindingCollector()
    for _, tree in parsed_cells:
        collector.visit(tree)
    for role in expected_assignments:
        approved = approved_targets[role]
        if len(approved) != 1:
            raise HandoffError(f"L4 handoff notebook assignment mismatch: {role}")
        bindings = collector.bindings[role]
        if len(bindings) != 1 or bindings[0] is not approved[0]:
            raise HandoffError(f"L4 handoff notebook immutable binding mismatch: {role}")


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory only when no destination entry exists."""
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except OSError as exc:
            if os.path.lexists(destination):
                raise HandoffError(
                    f"refusing to overwrite handoff directory: {destination}"
                ) from exc
            raise HandoffError("atomic no-replace handoff publish failed") from exc
        return
    if not sys.platform.startswith("linux"):
        raise HandoffError("atomic no-replace handoff publish is unavailable")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise HandoffError("atomic no-replace handoff publish is unavailable") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise HandoffError(f"refusing to overwrite handoff directory: {destination}")
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise HandoffError("atomic no-replace handoff publish is unavailable")
    raise HandoffError(f"atomic no-replace handoff publish failed: errno={error_number}")


def _verify_bundle_snapshot(bundle: Path, manifest: dict[str, object]) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=".l4-handoff-verify-"))
    verifier = temporary / "verifier"
    clone = temporary / "clone"
    try:
        verifier.mkdir()
        _run_git(verifier, "init", "-b", "main")
        _run_git(verifier, "bundle", "verify", str(bundle))
        _run_git(temporary, "clone", str(bundle), str(clone))
        runner_sha = _require_manifest_string(manifest, "snapshot_git_sha")
        _run_git(clone, "checkout", "--detach", runner_sha)
        if _run_git(clone, "rev-list", "--count", "HEAD") != "1":
            raise HandoffError("L4 bundle must contain exactly one reachable commit")
        if _run_git(clone, "rev-parse", "HEAD") != runner_sha:
            raise HandoffError("L4 bundle runner Git SHA mismatch")
        if _run_git(clone, "status", "--porcelain"):
            raise HandoffError("L4 bundle checkout is dirty")
        symbolic = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=clone,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if symbolic.returncode == 0:
            raise HandoffError("L4 bundle verification checkout must be detached")
        provenance = json.loads((clone / ".source_provenance.json").read_text(encoding="utf-8"))
        expected_provenance = {key: manifest[key] for key in _PROVENANCE_KEYS}
        if provenance != expected_provenance:
            raise HandoffError("L4 bundle source provenance mismatch")
        template_sha256 = _git_blob_sha256(clone, "HEAD:notebooks/deployment_benchmark_l4.ipynb")
        if manifest.get("l4_template_sha256") != template_sha256:
            raise HandoffError("L4 notebook template SHA-256 mismatch")
    except (OSError, UnicodeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HandoffError("L4 bundle verification failed") from exc
    finally:
        shutil.rmtree(temporary, onerror=_remove_readonly)


def _require_manifest_string(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str):
        raise HandoffError(f"L4 handoff manifest {key} must be a string")
    return value


def _run_git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise HandoffError(
            f"Git command failed during L4 handoff verification: {' '.join(args)}"
        ) from exc


def _git_blob_sha256(repo: Path, revision_path: str) -> str:
    try:
        blob = subprocess.run(
            ["git", "show", revision_path],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise HandoffError("Git blob verification failed during L4 handoff creation") from exc
    return hashlib.sha256(blob).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-experiment-git-sha", required=True)
    parser.add_argument("--parent-deployment-gate-sha256", required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--parent-onnx-sha256", required=True)
    args = parser.parse_args(argv)
    parent = L4ParentIdentity.parse(
        experiment_git_sha=args.parent_experiment_git_sha,
        deployment_gate_sha256=args.parent_deployment_gate_sha256,
        checkpoint_sha256=args.parent_checkpoint_sha256,
        onnx_sha256=args.parent_onnx_sha256,
    )
    output = create_l4_handoff(args.repo, args.output_root, parent)
    print(f"l4_handoff_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
