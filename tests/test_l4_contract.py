from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import pcb_defect.deployment as deployment_module
from pcb_defect.constants import CLASSES
from pcb_defect.l4_contract import (
    L4ContractError,
    L4RunIdentity,
    verify_l4_inputs,
    verify_l4_parent_inputs,
)


def test_l4_identity_requires_distinct_runner_and_parent() -> None:
    with pytest.raises(L4ContractError, match="runner and parent experiment Git SHAs must differ"):
        L4RunIdentity.parse(
            runner_git_sha="a" * 40,
            experiment_git_sha="a" * 40,
            deployment_gate_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            onnx_sha256="d" * 64,
        )


@pytest.mark.parametrize("bad", ["", "A" * 64, "g" * 64, "a" * 63])
def test_l4_identity_rejects_malformed_sha256(bad: str) -> None:
    with pytest.raises(L4ContractError, match="64 lowercase hexadecimal"):
        L4RunIdentity.parse(
            runner_git_sha="a" * 40,
            experiment_git_sha="b" * 40,
            deployment_gate_sha256=bad,
            checkpoint_sha256="c" * 64,
            onnx_sha256="d" * 64,
        )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_clean_runner_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "runner"
    repo.mkdir()
    commands = (
        ("init",),
        ("config", "user.name", "L4 Test"),
        ("config", "user.email", "l4@test.invalid"),
    )
    for args in commands:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("runner\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"], cwd=repo, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "clean runner"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return repo, result.stdout.strip()


def _manifest_paths(workspace: Path) -> tuple[Path, ...]:
    return tuple(
        workspace / "runs" / arm / f"seed{seed}" / "inputs" / "paired_split_manifest.json"
        for arm in ("grouped", "leaky_control")
        for seed in (42, 43, 44)
    )


def _write_manifest_copies(workspace: Path, manifest: dict[str, Any]) -> None:
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    for path in _manifest_paths(workspace):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(manifest_bytes)


def _write_parent_workspace(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    workspace = tmp_path / ("b" * 12)
    dataset_root = tmp_path / "dataset" / "pcb"
    deployment = workspace / "deployment"
    runtime = workspace / "runtime_data" / "grouped"
    images = dataset_root / "images"
    for directory in (deployment, runtime, images):
        directory.mkdir(parents=True, exist_ok=True)
    calibration = images / "01_missing_hole_02.jpg"
    calibration.write_bytes(b"calibration-image")
    final_test = images / "08_open_circuit_01.jpg"
    final_test.write_bytes(b"final-test-image")
    calibration_list = runtime / "calibration.txt"
    calibration_list.write_text(f"{calibration}\n", encoding="utf-8", newline="\n")
    deployment_module._write_calibration_yaml(deployment / "calibration.yaml", calibration_list)
    checkpoint = workspace / "runs" / "grouped" / "seed42" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    onnx = deployment / "best.onnx"
    onnx.write_bytes(b"onnx")
    samples = [
        {
            "stem": calibration.stem,
            "board_id": "01",
            "class_name": "missing_hole",
            "image_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
            "label_sha256": "0" * 64,
        },
        {
            "stem": final_test.stem,
            "board_id": "08",
            "class_name": "open_circuit",
            "image_sha256": hashlib.sha256(final_test.read_bytes()).hexdigest(),
            "label_sha256": "1" * 64,
        },
    ]
    dataset_sha256 = _sha256_json(samples)
    manifest_payload = {
        "protocol_version": "paired-board-sensitivity-v1",
        "config": {},
        "dataset": {"sample_count": 2, "sha256": dataset_sha256, "samples": samples},
        "board_roles": {
            "final_test_and_exposure": "08",
            "validation_and_calibration": "01",
            "grouped_train": [],
        },
        "partitions": {"calibration": [calibration.stem], "final_test": [final_test.stem]},
        "counts": {},
    }
    manifest_sha256 = _sha256_json(manifest_payload)
    _write_manifest_copies(workspace, {**manifest_payload, "manifest_sha256": manifest_sha256})
    experiment_git_sha = "b" * 40
    gate = {
        "passed": True,
        "git_sha": experiment_git_sha,
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "artifacts": {
            "source_checkpoint": checkpoint.relative_to(workspace).as_posix(),
            "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "onnx": "best.onnx",
            "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        },
        "fidelity": {"pt": {"map50_95": 0.12}, "onnx": {"map50_95": 0.11}, "threshold": 0.02},
    }
    gate_path = deployment / "deployment_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return (
        workspace,
        dataset_root,
        {
            "deployment_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        },
    )


def _valid_contract_fixture(tmp_path: Path) -> tuple[Path, Path, Path, L4RunIdentity]:
    repo, runner_sha = _write_clean_runner_repo(tmp_path)
    workspace, dataset_root, values = _write_parent_workspace(tmp_path)
    identity = L4RunIdentity.parse(
        runner_git_sha=runner_sha,
        experiment_git_sha="b" * 40,
        deployment_gate_sha256=values["deployment_gate_sha256"],
        checkpoint_sha256=values["checkpoint_sha256"],
        onnx_sha256=values["onnx_sha256"],
    )
    return repo, workspace, dataset_root, identity


def test_verify_l4_inputs_binds_runner_parent_and_calibration(tmp_path: Path) -> None:
    repo, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)

    verified = verify_l4_inputs(repo, workspace, dataset_root, identity)

    assert verified.runner_git_sha == identity.runner_git_sha
    assert verified.parent.experiment_git_sha == "b" * 40
    assert [path.stem for path in verified.parent.calibration_images] == ["01_missing_hole_02"]
    assert verified.parent.checkpoint_path.is_relative_to(workspace)
    assert verified.parent.onnx_path.is_relative_to(workspace)
    assert verified.parent.manifest_path == (
        workspace / "runs" / "grouped" / "seed42" / "inputs" / "paired_split_manifest.json"
    )


def test_verify_l4_parent_inputs_accepts_deployment_calibration_yaml_schema(
    tmp_path: Path,
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    calibration_list = workspace / "runtime_data" / "grouped" / "calibration.txt"
    deployment_module._write_calibration_yaml(
        workspace / "deployment" / "calibration.yaml", calibration_list
    )

    verified = verify_l4_parent_inputs(workspace, dataset_root, identity.parent)

    assert verified.calibration_yaml == workspace / "deployment" / "calibration.yaml"
    assert verified.calibration_images == (dataset_root / "images" / "01_missing_hole_02.jpg",)
    assert verified.manifest_path == (
        workspace / "runs" / "grouped" / "seed42" / "inputs" / "paired_split_manifest.json"
    )


def test_verify_l4_parent_inputs_requires_existing_dataset_root(tmp_path: Path) -> None:
    _, workspace, _, identity = _valid_contract_fixture(tmp_path)

    with pytest.raises(L4ContractError, match="dataset root"):
        verify_l4_parent_inputs(workspace, tmp_path / "missing-dataset", identity.parent)


def test_verify_l4_parent_inputs_requires_dataset_root_distinct_from_workspace(
    tmp_path: Path,
) -> None:
    _, workspace, _, identity = _valid_contract_fixture(tmp_path)

    with pytest.raises(L4ContractError, match="dataset root"):
        verify_l4_parent_inputs(workspace, workspace, identity.parent)


def test_verify_l4_parent_inputs_rejects_dataset_root_nested_beneath_workspace(
    tmp_path: Path,
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    nested_dataset_root = workspace / "dataset" / "pcb"
    nested_calibration = nested_dataset_root / "images" / "01_missing_hole_02.jpg"
    nested_calibration.parent.mkdir(parents=True)
    nested_calibration.write_bytes((dataset_root / "images" / nested_calibration.name).read_bytes())
    (workspace / "runtime_data" / "grouped" / "calibration.txt").write_text(
        f"{nested_calibration}\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(L4ContractError, match="dataset root"):
        verify_l4_parent_inputs(workspace, nested_dataset_root, identity.parent)


def test_verify_l4_parent_inputs_rejects_dataset_root_containing_workspace(
    tmp_path: Path,
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    ancestor_dataset_root = tmp_path
    ancestor_calibration = ancestor_dataset_root / "images" / "01_missing_hole_02.jpg"
    ancestor_calibration.parent.mkdir()
    ancestor_calibration.write_bytes(
        (dataset_root / "images" / ancestor_calibration.name).read_bytes()
    )
    (workspace / "runtime_data" / "grouped" / "calibration.txt").write_text(
        f"{ancestor_calibration}\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(L4ContractError, match="dataset root"):
        verify_l4_parent_inputs(workspace, ancestor_dataset_root, identity.parent)


@pytest.mark.parametrize("split", ["train", "test"])
def test_verify_l4_parent_inputs_requires_every_split_to_use_calibration_list(
    tmp_path: Path, split: str
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    calibration_list = workspace / "runtime_data" / "grouped" / "calibration.txt"
    final_list = workspace / "runtime_data" / "grouped" / "final_test.txt"
    final_list.write_text("/forbidden/final-test.jpg\n", encoding="utf-8", newline="\n")
    payload = {
        "train": str(calibration_list.resolve()),
        "val": str(calibration_list.resolve()),
        "test": str(calibration_list.resolve()),
        "names": dict(enumerate(CLASSES)),
    }
    payload[split] = str(final_list.resolve())
    calibration_yaml = deployment_module.yaml.safe_dump(payload, sort_keys=False)
    (workspace / "deployment" / "calibration.yaml").write_text(
        calibration_yaml, encoding="utf-8", newline="\n"
    )

    with pytest.raises(L4ContractError, match="calibration list path mismatch"):
        verify_l4_parent_inputs(workspace, dataset_root, identity.parent)


def test_verify_l4_parent_inputs_rejects_noncanonical_equivalent_split_path(
    tmp_path: Path,
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    calibration_list = workspace / "runtime_data" / "grouped" / "calibration.txt"
    payload = {
        "train": str(calibration_list.parent / ".." / "grouped" / calibration_list.name),
        "val": str(calibration_list.resolve()),
        "test": str(calibration_list.resolve()),
        "names": dict(enumerate(CLASSES)),
    }
    (workspace / "deployment" / "calibration.yaml").write_text(
        deployment_module.yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(L4ContractError, match="calibration list path mismatch"):
        verify_l4_parent_inputs(workspace, dataset_root, identity.parent)


@pytest.mark.parametrize(
    "names_block",
    [
        "names:\n",
        "names:\n  0: missing_hole\n  2: open_circuit\n",
        "names:\n  00: missing_hole\n",
        "names:\n  0: '&alias'\n",
        "names:\n  0: null\n",
        "names: {0: missing_hole}\n",
    ],
)
def test_verify_l4_parent_inputs_rejects_unsafe_or_noncanonical_names_mapping(
    tmp_path: Path, names_block: str
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    calibration_list = (workspace / "runtime_data" / "grouped" / "calibration.txt").resolve()
    content = (
        f"train: {calibration_list}\n"
        f"val: {calibration_list}\n"
        f"test: {calibration_list}\n"
        f"{names_block}"
    )
    (workspace / "deployment" / "calibration.yaml").write_text(
        content, encoding="utf-8", newline="\n"
    )

    with pytest.raises(L4ContractError, match="missing or malformed"):
        verify_l4_parent_inputs(workspace, dataset_root, identity.parent)


@pytest.mark.parametrize(
    "names",
    [
        [
            "mouse_bite",
            "missing_hole",
            "open_circuit",
            "short",
            "spur",
            "spurious_copper",
        ],
        ["missing_hole", "mouse_bite", "open_circuit", "short", "spur"],
        [
            "missing_hole",
            "mouse_bite",
            "open_circuit",
            "short",
            "spur",
            "spurious_copper",
            "bogus",
        ],
        ["missing_hole", "mouse_bite", "bogus", "short", "spur", "spurious_copper"],
    ],
    ids=["swapped", "truncated", "extended", "single-safe-but-unknown"],
)
def test_verify_l4_parent_inputs_requires_canonical_class_mapping(
    tmp_path: Path, names: list[str]
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    calibration_list = (workspace / "runtime_data" / "grouped" / "calibration.txt").resolve()
    payload = {
        "train": str(calibration_list),
        "val": str(calibration_list),
        "test": str(calibration_list),
        "names": dict(enumerate(names)),
    }
    (workspace / "deployment" / "calibration.yaml").write_text(
        deployment_module.yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(L4ContractError, match="missing or malformed"):
        verify_l4_parent_inputs(workspace, dataset_root, identity.parent)


def test_verify_l4_parent_inputs_imports_and_passes_without_third_party_modules(
    tmp_path: Path,
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    source = Path(__file__).parents[1] / "src"
    script = f"""
import sys
from pathlib import Path

class StandardLibraryOnly:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.', 1)[0]
        if root == 'pcb_defect' or root in sys.stdlib_module_names:
            return None
        raise ImportError(f'non-standard-library import blocked: {{fullname}}')

sys.path.insert(0, {str(source)!r})
sys.meta_path.insert(0, StandardLibraryOnly())
from pcb_defect.l4_contract import L4ParentIdentity, verify_l4_parent_inputs
identity = L4ParentIdentity.parse(
    experiment_git_sha={identity.parent.experiment_git_sha!r},
    deployment_gate_sha256={identity.parent.deployment_gate_sha256!r},
    checkpoint_sha256={identity.parent.checkpoint_sha256!r},
    onnx_sha256={identity.parent.onnx_sha256!r},
)
verified = verify_l4_parent_inputs(
    Path({str(workspace)!r}), Path({str(dataset_root)!r}), identity
)
assert len(verified.calibration_images) == 1
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def _rewrite_manifest_and_identity(
    workspace: Path, identity: L4RunIdentity, manifest: dict[str, Any]
) -> L4RunIdentity:
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = _sha256_json(manifest_payload)
    _write_manifest_copies(workspace, manifest)
    gate_path = workspace / "deployment" / "deployment_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["manifest_sha256"] = manifest["manifest_sha256"]
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return L4RunIdentity.parse(
        runner_git_sha=identity.runner_git_sha,
        experiment_git_sha=identity.parent.experiment_git_sha,
        deployment_gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        checkpoint_sha256=identity.parent.checkpoint_sha256,
        onnx_sha256=identity.parent.onnx_sha256,
    )


def _apply_mutation(
    repo: Path,
    workspace: Path,
    dataset_root: Path,
    identity: L4RunIdentity,
    mutation: str,
) -> L4RunIdentity:
    gate_path = workspace / "deployment" / "deployment_gate.json"
    calibration_list = workspace / "runtime_data" / "grouped" / "calibration.txt"
    calibration_image = dataset_root / "images" / "01_missing_hole_02.jpg"
    final_test_image = dataset_root / "images" / "08_open_circuit_01.jpg"
    manifest_paths = _manifest_paths(workspace)
    if mutation == "dirty_runner":
        (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    elif mutation == "gate_bytes":
        gate_path.write_bytes(gate_path.read_bytes() + b"\n")
    elif mutation == "checkpoint_bytes":
        checkpoint = workspace / "runs" / "grouped" / "seed42" / "weights" / "best.pt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"mutated")
    elif mutation == "onnx_bytes":
        onnx = workspace / "deployment" / "best.onnx"
        onnx.write_bytes(onnx.read_bytes() + b"mutated")
    elif mutation == "missing_manifest":
        manifest_paths[-1].unlink()
    elif mutation == "byte_different_manifest":
        manifest_paths[-1].write_bytes(manifest_paths[-1].read_bytes() + b"\n")
    elif mutation == "invalid_manifest_self_hash":
        manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
        manifest["manifest_sha256"] = "0" * 64
        _write_manifest_copies(workspace, manifest)
    elif mutation == "calibration_path_outside_dataset":
        outside_image = workspace.parent / "outside" / calibration_image.name
        outside_image.parent.mkdir()
        outside_image.write_bytes(calibration_image.read_bytes())
        calibration_list.write_text(f"{outside_image}\n", encoding="utf-8", newline="\n")
    elif mutation == "calibration_symlink_escape":
        outside_image = workspace.parent / "outside-symlink-target.jpg"
        outside_image.write_bytes(calibration_image.read_bytes())
        calibration_image.unlink()
        try:
            calibration_image.symlink_to(outside_image)
        except OSError as error:
            pytest.skip(f"symlink creation is unavailable: {error}")
    elif mutation == "calibration_list_blank_line":
        calibration_list.write_bytes(calibration_list.read_bytes() + b"\n")
    elif mutation == "calibration_list_crlf":
        calibration_list.write_bytes(calibration_list.read_bytes().replace(b"\n", b"\r\n"))
    elif mutation == "calibration_image_bytes":
        calibration_image.write_bytes(calibration_image.read_bytes() + b"mutated")
    elif mutation == "wrong_order":
        manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
        manifest["partitions"]["calibration"] = [
            calibration_image.stem,
            final_test_image.stem,
        ]
        manifest["partitions"]["final_test"] = []
        identity = _rewrite_manifest_and_identity(workspace, identity, manifest)
        calibration_list.write_text(
            f"{final_test_image}\n{calibration_image}\n", encoding="utf-8", newline="\n"
        )
    elif mutation == "duplicate_stem":
        manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
        manifest["partitions"]["calibration"] = [calibration_image.stem] * 2
        identity = _rewrite_manifest_and_identity(workspace, identity, manifest)
        calibration_list.write_text(
            f"{calibration_image}\n{calibration_image}\n", encoding="utf-8", newline="\n"
        )
    elif mutation == "final_test_overlap":
        manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
        manifest["partitions"]["final_test"].append(calibration_image.stem)
        identity = _rewrite_manifest_and_identity(workspace, identity, manifest)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    return identity


@pytest.mark.parametrize(
    "mutation",
    [
        "dirty_runner",
        "gate_bytes",
        "checkpoint_bytes",
        "onnx_bytes",
        "missing_manifest",
        "byte_different_manifest",
        "invalid_manifest_self_hash",
        "calibration_path_outside_dataset",
        "calibration_symlink_escape",
        "calibration_list_blank_line",
        "calibration_list_crlf",
        "calibration_image_bytes",
        "wrong_order",
        "duplicate_stem",
        "final_test_overlap",
    ],
)
def test_verify_l4_inputs_rejects_mutation(tmp_path: Path, mutation: str) -> None:
    repo, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    identity = _apply_mutation(repo, workspace, dataset_root, identity, mutation)

    with pytest.raises(L4ContractError):
        verify_l4_inputs(repo, workspace, dataset_root, identity)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("git_sha", "c" * 40, "deployment gate did not pass"),
        ("dataset_sha256", "0" * 64, "dataset SHA-256 mismatch"),
        ("manifest_sha256", "0" * 64, "protocol-manifest identity mismatch"),
    ],
)
def test_verify_l4_parent_inputs_rejects_rehashed_gate_identity_mismatch(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    gate_path = workspace / "deployment" / "deployment_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate[field] = replacement
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    rehashed_identity = L4RunIdentity.parse(
        runner_git_sha=identity.runner_git_sha,
        experiment_git_sha=identity.parent.experiment_git_sha,
        deployment_gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        checkpoint_sha256=identity.parent.checkpoint_sha256,
        onnx_sha256=identity.parent.onnx_sha256,
    )

    with pytest.raises(L4ContractError, match=message):
        verify_l4_parent_inputs(workspace, dataset_root, rehashed_identity.parent)


def test_verify_l4_parent_inputs_rejects_checkpoint_escape(tmp_path: Path) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    gate_path = workspace / "deployment" / "deployment_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["artifacts"]["source_checkpoint"] = "../escaped.pt"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    escaped_identity = L4RunIdentity.parse(
        runner_git_sha=identity.runner_git_sha,
        experiment_git_sha=identity.parent.experiment_git_sha,
        deployment_gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        checkpoint_sha256=identity.parent.checkpoint_sha256,
        onnx_sha256=identity.parent.onnx_sha256,
    )

    with pytest.raises(L4ContractError, match="checkpoint path escapes parent workspace"):
        verify_l4_parent_inputs(workspace, dataset_root, escaped_identity.parent)


@pytest.mark.parametrize(
    "content",
    [
        "val: [list]\n",
        "val: one\nval: two\n",
        "val: &anchor value\n",
        "unknown: value\nval: value\n",
    ],
)
def test_verify_l4_parent_inputs_rejects_non_scalar_or_ambiguous_calibration_yaml(
    tmp_path: Path, content: str
) -> None:
    _, workspace, dataset_root, identity = _valid_contract_fixture(tmp_path)
    (workspace / "deployment" / "calibration.yaml").write_text(content, encoding="utf-8")

    with pytest.raises(L4ContractError, match="missing or malformed"):
        verify_l4_parent_inputs(workspace, dataset_root, identity.parent)
