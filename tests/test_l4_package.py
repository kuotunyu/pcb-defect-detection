from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pcb_defect.l4_package as l4_package
from pcb_defect.l4_contract import L4ContractError, L4RunIdentity
from pcb_defect.l4_package import (
    collect_l4_files,
    create_or_verify_l4_package,
    l4_package_name,
)
from pcb_defect.result_package import PackageError


def test_l4_package_name_contains_parent_and_runner_prefixes() -> None:
    identity = _identity(runner="a" * 40, parent="b" * 40)

    assert l4_package_name(identity) == ("paired-results-l4-bbbbbbbbbbbb-runner-aaaaaaaaaaaa.zip")


def test_l4_collector_includes_only_private_verification_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)

    files = collect_l4_files(repo, workspace, dataset_root, identity)

    assert Path("benchmark_l4/aaaaaaaaaaaa/benchmark_l4.json") in files
    assert Path("benchmark_l4/aaaaaaaaaaaa/best_fp16.engine") in files
    assert Path("deployment/best.onnx") in files
    assert Path("runs/grouped/seed42/weights/best.pt") in files
    assert Path("runs/grouped/seed42/inputs/paired_split_manifest.json") in files
    assert Path("inputs/paired_split_manifest.json") not in files
    assert not any(
        suffix.lower() in {".jpg", ".jpeg", ".png"} for path in files for suffix in path.suffixes
    )


@pytest.mark.parametrize("filename", ["evidence.JPG", "evidence.jpg.pt", "evidence.PNG.engine"])
def test_l4_collector_rejects_dataset_pixels_in_any_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    pixel = workspace / "deployment" / filename
    pixel.write_bytes(b"pixels")
    monkeypatch.setattr(
        l4_package, "_l4_package_paths", lambda *_args: [pixel.relative_to(workspace)]
    )

    with pytest.raises(PackageError, match="dataset pixels are forbidden"):
        collect_l4_files(repo, workspace, dataset_root, identity)


@pytest.mark.parametrize("dataset_state", ["missing", "wrong", "mutated"])
def test_l4_package_creation_rejects_unverified_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset_state: str
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    if dataset_state == "missing":
        dataset_root = tmp_path / "missing-dataset" / "pcb"
    elif dataset_state == "wrong":
        dataset_root = tmp_path / "different-dataset" / "pcb"
        (dataset_root / "images").mkdir(parents=True)
        (dataset_root / "images" / "calibration.jpg").write_bytes(b"pixels")
    else:
        (dataset_root / "images" / "calibration.jpg").write_bytes(b"changed")

    output = tmp_path / "packages"
    with pytest.raises(L4ContractError, match="dataset root"):
        create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)
    assert not output.exists()


def test_l4_package_reuse_rejects_mutated_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    output = tmp_path / "packages"
    package = create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)
    original_package = package.read_bytes()
    (dataset_root / "images" / "calibration.jpg").write_bytes(b"changed")

    with pytest.raises(L4ContractError, match="dataset root"):
        create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)
    assert package.read_bytes() == original_package


@pytest.mark.parametrize("existing", ["package", "sidecar"])
def test_l4_package_rejects_partial_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    output = tmp_path / "packages"
    output.mkdir()
    package = output / l4_package_name(identity)
    target = package if existing == "package" else package.with_suffix(".zip.sha256")
    target.write_bytes(b"partial")

    with pytest.raises(PackageError, match="must exist together"):
        create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)


def test_l4_package_reuses_unchanged_current_evidence_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    output = tmp_path / "packages"
    package = create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)
    assert create_or_verify_l4_package(repo, workspace, dataset_root, output, identity) == package


@pytest.mark.parametrize(
    "foreign",
    ["deployment/evidence.jpg", "runs/leaky_control/seed42/weights/best.pt"],
)
def test_l4_package_rejects_self_consistent_foreign_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign: str
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    foreign_path = workspace / foreign
    foreign_path.parent.mkdir(parents=True, exist_ok=True)
    foreign_path.write_bytes(b"foreign")
    output = tmp_path / "packages"
    output.mkdir()
    package = output / l4_package_name(identity)
    l4_package.create_verifiable_zip(workspace, [Path(foreign)], package)

    with pytest.raises(PackageError, match="inventory"):
        create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)


def test_l4_package_rejects_current_workspace_mutation_on_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    output = tmp_path / "packages"
    create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)
    (workspace / "l4_logs" / "aaaaaaaaaaaa" / "benchmark_command.log").write_bytes(b"changed\n")

    with pytest.raises(PackageError, match="inventory"):
        create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)


def test_l4_package_rejects_mutated_existing_pair_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity = _complete_l4_workspace(tmp_path, monkeypatch)
    output = tmp_path / "packages"
    package = create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)
    sidecar = package.with_suffix(".zip.sha256")
    original = sidecar.read_bytes()
    sidecar.write_bytes(b"0" * len(original))

    with pytest.raises(PackageError, match="sidecar bytes, name, or hash are invalid"):
        create_or_verify_l4_package(repo, workspace, dataset_root, output, identity)
    assert sidecar.read_bytes() == b"0" * len(original)


def test_l4_package_cli_rejects_arbitrary_output_filename() -> None:
    with pytest.raises(SystemExit):
        l4_package.main(
            [
                "--repo",
                "repo",
                "--workspace",
                "workspace",
                "--dataset",
                "dataset",
                "--output-root",
                "packages",
                "--expected-runner-git-sha",
                "a" * 40,
                "--expected-experiment-git-sha",
                "b" * 40,
                "--expected-deployment-gate-sha256",
                "c" * 64,
                "--expected-checkpoint-sha256",
                "d" * 64,
                "--expected-onnx-sha256",
                "e" * 64,
                "--output",
                "anything.zip",
            ]
        )


def test_l4_package_cli_requires_dataset_root() -> None:
    with pytest.raises(SystemExit) as error:
        l4_package.main(
            [
                "--repo",
                "repo",
                "--workspace",
                "workspace",
                "--output-root",
                "packages",
                "--expected-runner-git-sha",
                "a" * 40,
                "--expected-experiment-git-sha",
                "b" * 40,
                "--expected-deployment-gate-sha256",
                "c" * 64,
                "--expected-checkpoint-sha256",
                "d" * 64,
                "--expected-onnx-sha256",
                "e" * 64,
            ]
        )

    assert error.value.code == 2


def _identity(*, runner: str, parent: str) -> L4RunIdentity:
    return L4RunIdentity.parse(
        runner_git_sha=runner,
        experiment_git_sha=parent,
        deployment_gate_sha256="c" * 64,
        checkpoint_sha256="d" * 64,
        onnx_sha256="e" * 64,
    )


def _complete_l4_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, L4RunIdentity]:
    repo = tmp_path / "repo"
    workspace = tmp_path / ("b" * 12)
    dataset_root = tmp_path / "dataset" / "pcb"
    repo.mkdir()
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "images" / "calibration.jpg").write_bytes(b"pixels")
    identity = _identity(runner="a" * 40, parent="b" * 40)
    paths = {
        "inputs/input_lock.json": b"{}\n",
        "runs/grouped/seed42/inputs/paired_split_manifest.json": b"{}\n",
        "deployment/calibration.yaml": b"val: calibration.txt\n",
        "deployment/deployment_gate.json": b"{}\n",
        "deployment/model_contract.candidate.json": b"{}\n",
        "deployment/best.onnx": b"onnx",
        "runs/grouped/seed42/weights/best.pt": b"checkpoint",
        "benchmark_l4/aaaaaaaaaaaa/benchmark_l4.json": json.dumps({"status": "complete"}).encode(),
        "benchmark_l4/aaaaaaaaaaaa/best_fp16.engine": b"engine",
        "l4_logs/aaaaaaaaaaaa/benchmark_command.log": b"command\n",
    }
    for relative, contents in paths.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    verified = SimpleNamespace(
        parent=SimpleNamespace(
            manifest_path=(
                workspace / "runs" / "grouped" / "seed42" / "inputs" / "paired_split_manifest.json"
            ),
            onnx_path=workspace / "deployment" / "best.onnx",
            checkpoint_path=workspace / "runs" / "grouped" / "seed42" / "weights" / "best.pt",
        )
    )

    def verify_inputs(
        _repo: Path, _workspace: Path, observed_dataset_root: Path, _identity: L4RunIdentity
    ) -> object:
        calibration = observed_dataset_root / "images" / "calibration.jpg"
        if observed_dataset_root.resolve() != dataset_root.resolve() or (
            not calibration.is_file() or calibration.read_bytes() != b"pixels"
        ):
            raise L4ContractError("dataset root evidence mismatch")
        return verified

    def benchmark_complete(
        _repo: Path,
        _workspace: Path,
        observed_dataset_root: Path,
        _identity: L4RunIdentity,
        _report: object,
    ) -> bool:
        return observed_dataset_root.resolve() == dataset_root.resolve()

    monkeypatch.setattr(l4_package, "verify_l4_inputs", verify_inputs)
    monkeypatch.setattr(l4_package, "benchmark_is_complete", benchmark_complete)
    return repo, workspace, dataset_root, identity
