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


def _write_parent_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path / ("b" * 12)
    deployment = workspace / "deployment"
    inputs = workspace / "inputs"
    runtime = workspace / "runtime_data" / "grouped"
    images = workspace / "dataset" / "images"
    for directory in (deployment, inputs, runtime, images):
        directory.mkdir(parents=True, exist_ok=True)
    calibration = images / "01_missing_hole_02.jpg"
    calibration.write_bytes(b"calibration-image")
    calibration_list = runtime / "calibration.txt"
    calibration_list.write_text(f"{calibration}\n", encoding="utf-8", newline="\n")
    deployment_module._write_calibration_yaml(deployment / "calibration.yaml", calibration_list)
    checkpoint = workspace / "runs" / "grouped" / "seed42" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    onnx = deployment / "best.onnx"
    onnx.write_bytes(b"onnx")
    sample = {
        "stem": calibration.stem,
        "board_id": "01",
        "class_name": "missing_hole",
        "image_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "label_sha256": "0" * 64,
    }
    dataset_sha256 = _sha256_json([sample])
    manifest_payload = {
        "protocol_version": "paired-board-sensitivity-v1",
        "config": {},
        "dataset": {"sample_count": 1, "sha256": dataset_sha256, "samples": [sample]},
        "board_roles": {
            "final_test_and_exposure": "08",
            "validation_and_calibration": "01",
            "grouped_train": [],
        },
        "partitions": {"calibration": [calibration.stem], "final_test": []},
        "counts": {},
    }
    manifest_sha256 = _sha256_json(manifest_payload)
    (inputs / "paired_split_manifest.json").write_text(
        json.dumps({**manifest_payload, "manifest_sha256": manifest_sha256}), encoding="utf-8"
    )
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
    return workspace, {
        "deployment_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
    }


def _valid_contract_fixture(tmp_path: Path) -> tuple[Path, Path, L4RunIdentity]:
    repo, runner_sha = _write_clean_runner_repo(tmp_path)
    workspace, values = _write_parent_workspace(tmp_path)
    identity = L4RunIdentity.parse(
        runner_git_sha=runner_sha,
        experiment_git_sha="b" * 40,
        deployment_gate_sha256=values["deployment_gate_sha256"],
        checkpoint_sha256=values["checkpoint_sha256"],
        onnx_sha256=values["onnx_sha256"],
    )
    return repo, workspace, identity


def test_verify_l4_inputs_binds_runner_parent_and_calibration(tmp_path: Path) -> None:
    repo, workspace, identity = _valid_contract_fixture(tmp_path)

    verified = verify_l4_inputs(repo, workspace, identity)

    assert verified.runner_git_sha == identity.runner_git_sha
    assert verified.parent.experiment_git_sha == "b" * 40
    assert [path.stem for path in verified.parent.calibration_images] == ["01_missing_hole_02"]
    assert verified.parent.checkpoint_path.is_relative_to(workspace)
    assert verified.parent.onnx_path.is_relative_to(workspace)


def test_verify_l4_parent_inputs_accepts_deployment_calibration_yaml_schema(
    tmp_path: Path,
) -> None:
    _, workspace, identity = _valid_contract_fixture(tmp_path)
    calibration_list = workspace / "runtime_data" / "grouped" / "calibration.txt"
    deployment_module._write_calibration_yaml(
        workspace / "deployment" / "calibration.yaml", calibration_list
    )

    verified = verify_l4_parent_inputs(workspace, identity.parent)

    assert verified.calibration_yaml == workspace / "deployment" / "calibration.yaml"
    assert verified.calibration_images == (
        workspace / "dataset" / "images" / "01_missing_hole_02.jpg",
    )


@pytest.mark.parametrize("split", ["train", "test"])
def test_verify_l4_parent_inputs_requires_every_split_to_use_calibration_list(
    tmp_path: Path, split: str
) -> None:
    _, workspace, identity = _valid_contract_fixture(tmp_path)
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
        verify_l4_parent_inputs(workspace, identity.parent)


def test_verify_l4_parent_inputs_rejects_noncanonical_equivalent_split_path(
    tmp_path: Path,
) -> None:
    _, workspace, identity = _valid_contract_fixture(tmp_path)
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
        verify_l4_parent_inputs(workspace, identity.parent)


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
    _, workspace, identity = _valid_contract_fixture(tmp_path)
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
        verify_l4_parent_inputs(workspace, identity.parent)


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
    _, workspace, identity = _valid_contract_fixture(tmp_path)
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
        verify_l4_parent_inputs(workspace, identity.parent)


def test_verify_l4_parent_inputs_imports_and_passes_without_third_party_modules(
    tmp_path: Path,
) -> None:
    _, workspace, identity = _valid_contract_fixture(tmp_path)
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
verified = verify_l4_parent_inputs(Path({str(workspace)!r}), identity)
assert len(verified.calibration_images) == 1
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def _apply_mutation(repo: Path, workspace: Path, mutation: str) -> None:
    gate_path = workspace / "deployment" / "deployment_gate.json"
    calibration_list = workspace / "runtime_data" / "grouped" / "calibration.txt"
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
    elif mutation == "manifest_partition":
        manifest_path = workspace / "inputs" / "paired_split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["partitions"]["calibration"] = []
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "final_test_in_calibration":
        manifest_path = workspace / "inputs" / "paired_split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["partitions"]["final_test"] = ["01_missing_hole_02"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "calibration_list_blank_line":
        calibration_list.write_bytes(calibration_list.read_bytes() + b"\n")
    elif mutation == "calibration_list_crlf":
        calibration_list.write_bytes(calibration_list.read_bytes().replace(b"\n", b"\r\n"))
    elif mutation == "calibration_image_bytes":
        image = workspace / "dataset" / "images" / "01_missing_hole_02.jpg"
        image.write_bytes(image.read_bytes() + b"mutated")
    else:
        raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    [
        "dirty_runner",
        "gate_bytes",
        "checkpoint_bytes",
        "onnx_bytes",
        "manifest_partition",
        "final_test_in_calibration",
        "calibration_list_blank_line",
        "calibration_list_crlf",
        "calibration_image_bytes",
    ],
)
def test_verify_l4_inputs_rejects_mutation(tmp_path: Path, mutation: str) -> None:
    repo, workspace, identity = _valid_contract_fixture(tmp_path)
    _apply_mutation(repo, workspace, mutation)

    with pytest.raises(L4ContractError):
        verify_l4_inputs(repo, workspace, identity)


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
    _, workspace, identity = _valid_contract_fixture(tmp_path)
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
        verify_l4_parent_inputs(workspace, rehashed_identity.parent)


def test_verify_l4_parent_inputs_rejects_checkpoint_escape(tmp_path: Path) -> None:
    _, workspace, identity = _valid_contract_fixture(tmp_path)
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
        verify_l4_parent_inputs(workspace, escaped_identity.parent)


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
    _, workspace, identity = _valid_contract_fixture(tmp_path)
    (workspace / "deployment" / "calibration.yaml").write_text(content, encoding="utf-8")

    with pytest.raises(L4ContractError, match="missing or malformed"):
        verify_l4_parent_inputs(workspace, identity.parent)
