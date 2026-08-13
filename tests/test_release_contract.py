from __future__ import annotations

import ast
import builtins
import hashlib
import json
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PAIRED_A100 = ROOT / "reports" / "paired_a100"
GPU_NOTEBOOKS = (
    "notebooks/paired_experiment_a100.ipynb",
    "notebooks/deployment_parity_probe_a100.ipynb",
    "notebooks/deployment_benchmark_l4.ipynb",
)
ENVIRONMENT_CONTROLS = (
    "os.environ['YOLO_AUTOINSTALL'] = 'false'",
    "os.environ['ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS'] = '1'",
)
FORBIDDEN_FIRST_CELL_ACTIONS = (
    "from google.colab import drive",
    "drive.mount(",
    "subprocess.",
    "git clone",
    "git checkout",
    "pip install",
    "uv sync",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def _first_code(notebook: dict) -> str:
    return "".join(
        next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")["source"]
    )


def _code_cell_containing(notebook: dict, text: str) -> str:
    return "".join(
        next(
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and text in "".join(cell["source"])
        )["source"]
    )


def _assert_in_order(source: str, *tokens: str) -> None:
    offset = -1
    for token in tokens:
        next_offset = source.index(token, offset + 1)
        assert offset < next_offset, token
        offset = next_offset


def _assert_controls_precede_external_actions(source: str) -> None:
    for action in FORBIDDEN_FIRST_CELL_ACTIONS:
        action_offset = source.find(action)
        if action_offset >= 0:
            for control in ENVIRONMENT_CONTROLS:
                assert source.index(control) < action_offset, action


def test_promoted_a100_package_receipt_is_portable_and_complete() -> None:
    receipt = _read_json(PAIRED_A100 / "result_package_receipt.json")

    assert receipt == {
        "schema_version": "1.0",
        "package": {
            "name": "paired-results-a100-9e3a1ed5827a.zip",
            "bytes": 75767467,
            "sha256": "c6158e5017bbec02c089ad42f55f83b20a46fb8ebd3bd1c35115d2940bb74c92",
            "sidecar": "paired-results-a100-9e3a1ed5827a.zip.sha256",
        },
        "source_git_sha": "9e3a1ed5827ac3759cbb15632f041e3e5c183b51",
        "package_manifest": {
            "path": "package_manifest.json",
            "sha256": "2387496eb57a8126515f13cde72a83fab799e4a5b9e7a8d93ee1a680d6a7132d",
            "verified_entries": 52,
            "failed_entries": 0,
        },
        "verification": {
            "sidecar_match": True,
            "internal_manifest_passed": True,
        },
    }


def test_promoted_a100_input_and_finalization_chain_is_hash_bound() -> None:
    input_lock = _read_json(PAIRED_A100 / "input_lock.json")
    protocol = yaml.safe_load(
        (ROOT / "configs" / "paired_protocol.yaml").read_text(encoding="utf-8")
    )
    gate = _read_json(PAIRED_A100 / "gate_report.json")
    selection = _read_json(PAIRED_A100 / "deployment_selection.json")
    metrics = _read_json(PAIRED_A100 / "final_metrics.json")
    finalization = _read_json(PAIRED_A100 / "finalization_record.json")

    assert input_lock["git_sha"] == "9e3a1ed5827ac3759cbb15632f041e3e5c183b51"
    assert input_lock["dataset_sha256"] == protocol["frozen_hashes"]["dataset_sha256"]
    assert input_lock["manifest_sha256"] == protocol["frozen_hashes"]["manifest_sha256"]
    assert gate["passed"] is True
    assert all(gate["checks"].values())
    assert gate["input_lock"] == input_lock
    assert selection["arm"] == "grouped"
    assert selection["seed"] == 42
    assert selection["selected_before_final_test"] is True
    assert metrics["status"] == "complete"
    assert metrics["deployment_selection"] == selection
    assert metrics["git_sha"] == input_lock["git_sha"]
    assert finalization["status"] == "complete"
    assert finalization["results"] == "final_metrics.json"
    assert finalization["results_bytes"] == (PAIRED_A100 / "final_metrics.json").stat().st_size
    assert finalization["results_sha256"] == _sha256(PAIRED_A100 / "final_metrics.json")


def test_historical_training_recipe_resolves_augmentation_without_rewriting_evidence() -> None:
    recipe = _read_json(ROOT / "reports" / "training_recipe.json")
    input_lock = _read_json(PAIRED_A100 / "input_lock.json")
    train_config_path = ROOT / "configs" / "train_paired.yaml"
    train_config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))

    assert recipe["schema_version"] == "1.0"
    assert recipe["evidence_scope"] == "historical-metadata-reconstruction"
    assert recipe["historical_evidence"] == {
        "experiment_git_sha": input_lock["git_sha"],
        "train_config_sha256": input_lock["config_sha256"],
        "ultralytics_version": "8.4.89",
        "ultralytics_wheel_sha256": (
            "3b50379a0a0d99f9accab640b2dbaa6b7fdc947f71a0cb56266bc2d87426e5be"
        ),
    }
    assert recipe["historical_evidence"]["train_config_sha256"] == _sha256(train_config_path)
    assert recipe["explicit_training_config"] == train_config
    assert recipe["training_seeds"] == [42, 43, 44]
    assert recipe["augmentation"]["resolved_values"] == {
        "bgr": 0.0,
        "copy_paste": 0.0,
        "cutmix": 0.0,
        "degrees": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mixup": 0.0,
        "mosaic": 1.0,
        "perspective": 0.0,
        "scale": 0.5,
        "shear": 0.0,
        "translate": 0.1,
    }
    assert recipe["augmentation"]["value_origin"] == "ultralytics-v8.4.89-default.yaml"
    assert recipe["augmentation"]["close_mosaic"] == {
        "value": train_config["close_mosaic"],
        "origin": "explicit-train-config",
    }
    assert recipe["data_access_contract"] == {
        "augmentation_inputs": ["grouped_train", "leaky_train"],
        "excluded_from_augmentation": ["validation", "calibration", "final_test"],
        "source_dataset_mutated": False,
        "augmented_training_samples_exported_to_evaluation": False,
        "runtime_preview_plots_may_be_written": True,
    }
    assert recipe["reproducibility"]["deterministic_requested"] is True
    assert recipe["reproducibility"]["bitwise_reproducibility_claimed"] is False
    assert recipe["limitations"]


def test_current_release_metadata_is_v0_2_0_and_preserves_v0_1_0_provenance() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = _read_json(ROOT / ".zenodo.json")
    research_package = (ROOT / "docs" / "research-package.md").read_text(encoding="utf-8")

    assert citation["cff-version"] == "1.2.0"
    assert citation["authors"] == [{"alias": "kuotunyu", "family-names": "kuotunyu"}]
    assert citation["repository-code"] == "https://github.com/kuotunyu/pcb-defect-detection"
    assert citation["license"] == "AGPL-3.0-or-later"
    assert project["version"] == "0.2.0"
    assert citation["version"] == project["version"]
    assert citation["doi"] == "10.5281/zenodo.21912370"
    assert str(citation["date-released"]) == "2026-08-13"
    assert zenodo["creators"] == [{"name": "kuotunyu"}]
    assert zenodo["upload_type"] == "software"
    assert zenodo["access_right"] == "open"
    assert zenodo["license"] == citation["license"]
    assert zenodo["version"] == project["version"]
    assert "No dataset pixels, model weights, ONNX exports, or TensorRT engines" in zenodo["notes"]
    assert "strict per-box parity failed" in zenodo["notes"]
    assert "python -m pcb_defect.research_package" in research_package
    assert "https://github.com/kuotunyu/pcb-defect-detection/releases/tag/v0.1.0" in (
        research_package
    )
    assert "https://doi.org/10.5281/zenodo.21877497" in research_package
    assert "https://doi.org/10.5281/zenodo.21877496" in research_package
    assert "https://doi.org/10.5281/zenodo.21912370" in research_package
    assert "7fc1777d306584fc1f3ffe0c05989296370fe6df" in research_package
    assert "89d82a6ab8737193f8c59614d2a04c68f07b02fca3bc7d3ee7178c56ff882f29" in (research_package)
    assert "1,699,878 bytes" in research_package
    assert "56c086206eab9be1a9c6a4e36410fd13ed42f5ec" in research_package
    assert "21abbe3c71c5f7b962a8c33a8bc649dbe98757199a6ae17b5a6af0bbe27998e1" in (research_package)
    assert "not yet published" not in research_package
    assert "Running this command from any other commit produces a different" in research_package
    assert "not present that result as the `v0.1.0` asset" in research_package
    assert "dataset pixels" in research_package
    assert "TensorRT engines" in research_package


def test_public_deployment_gate_is_path_free_and_bound_to_raw_manifest_entry() -> None:
    public_path = PAIRED_A100 / "deployment_gate.public.json"
    public_text = public_path.read_text(encoding="utf-8")
    gate = json.loads(public_text)
    manifest = _read_json(PAIRED_A100 / "package_manifest.json")
    raw = next(
        item for item in manifest["files"] if item["path"] == "deployment/deployment_gate.json"
    )

    assert gate["_provenance"] == {
        "source_entry": "deployment/deployment_gate.json",
        "source_bytes": 18615,
        "source_sha256": "466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803",
        "removed_fields": ["command"],
    }
    assert gate["_provenance"]["source_bytes"] == raw["bytes"]
    assert gate["_provenance"]["source_sha256"] == raw["sha256"]
    assert "command" not in gate
    assert not any(token in public_text for token in ("/content/", "/root/", "MyDrive", "C:\\"))


def test_gpu_notebooks_disable_ultralytics_install_before_external_actions() -> None:
    for relative in GPU_NOTEBOOKS:
        _assert_controls_precede_external_actions(_first_code(_notebook(relative)))


def test_first_cell_action_guard_rejects_every_forbidden_predecessor() -> None:
    controls = "\n".join(ENVIRONMENT_CONTROLS)
    for action in FORBIDDEN_FIRST_CELL_ACTIONS:
        with pytest.raises(AssertionError):
            _assert_controls_precede_external_actions(f"{action}\n{controls}")


def test_full_gpu_notebooks_gate_runtime_before_and_after_onnx_stages() -> None:
    a100 = _code_cell_containing(
        _notebook("notebooks/paired_experiment_a100.ipynb"), "deployment_result = run_logged("
    )
    _assert_in_order(
        a100,
        "DEPLOYMENT_RUNTIME_BEFORE = runtime_contract_state(",
        "deployment_result = run_logged(",
        "DEPLOYMENT_RUNTIME_AFTER = runtime_contract_state(",
        "if DEPLOYMENT_RUNTIME_AFTER != DEPLOYMENT_RUNTIME_BEFORE:",
        "if deployment_result.returncode:",
    )

    l4 = _code_cell_containing(
        _notebook("notebooks/deployment_benchmark_l4.ipynb"), "benchmark_command = ["
    )
    _assert_in_order(
        l4,
        "VERIFY_BENCHMARK_SCRIPT = r'''",
        "benchmark_verification = run_project_json(",
        "if benchmark_verification == {'complete': True}:",
        "benchmark_log = (",
    )
    runtime_branch = l4[l4.index("BENCHMARK_RUNTIME_BEFORE = runtime_contract_state(") :]
    _assert_in_order(
        runtime_branch,
        "BENCHMARK_RUNTIME_BEFORE = runtime_contract_state(",
        "run_streaming_command(\n        benchmark_command,",
        "BENCHMARK_RUNTIME_AFTER = runtime_contract_state(",
        "if BENCHMARK_RUNTIME_AFTER != BENCHMARK_RUNTIME_BEFORE:",
        "benchmark_verification = run_project_json(",
    )


def test_l4_notebook_separates_runner_and_parent_and_never_trains() -> None:
    notebook = _notebook("notebooks/deployment_benchmark_l4.ipynb")
    code = _code(notebook)
    first = _first_code(notebook)

    assignments = (
        'SOURCE_BUNDLE_SHA256 = "PASTE_FINAL_BUNDLE_SHA256"',
        'RUNNER_GIT_SHA = "PASTE_FINAL_GIT_SHA"',
        'PARENT_EXPERIMENT_GIT_SHA = "PASTE_PARENT_EXPERIMENT_GIT_SHA"',
        'PARENT_DEPLOYMENT_GATE_SHA256 = "PASTE_PARENT_DEPLOYMENT_GATE_SHA256"',
        'PARENT_CHECKPOINT_SHA256 = "PASTE_PARENT_CHECKPOINT_SHA256"',
        'PARENT_ONNX_SHA256 = "PASTE_PARENT_ONNX_SHA256"',
        'L4_HANDOFF_DIRECTORY = "PASTE_L4_HANDOFF_DIRECTORY"',
    )
    for assignment in assignments:
        assert first.count(assignment) == 1
    assert code.count("PASTE_") == 7
    assert (
        'PARENT_WORKSPACE = (\n    Path("/content/drive/MyDrive/pcb-defect-paired/workspaces")\n'
        "    / PARENT_EXPERIMENT_GIT_SHA[:12]\n)"
    ) in first
    assert "PARENT_WORKSPACE = DRIVE_ROOT / 'workspaces' / RUNNER_GIT_SHA[:12]" not in code
    assert "train-all" not in code
    assert "pcb_defect.experiment" not in code
    assert "input_lock" not in code
    assert "gate_report" not in code
    assert "os.environ['YOLO_AUTOINSTALL'] = 'false'" in first
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


def test_l4_notebook_verifies_before_locked_install_benchmark_package_and_success() -> None:
    code = _code(_notebook("notebooks/deployment_benchmark_l4.ipynb"))

    _assert_in_order(
        code,
        "drive.mount('/content/drive')",
        "if sha256_file(SOURCE_BUNDLE) != SOURCE_BUNDLE_SHA256:",
        "subprocess.run(['git', 'clone', str(SOURCE_BUNDLE), str(REPO)]",
        "subprocess.run(['git', 'checkout', '--detach', RUNNER_GIT_SHA]",
        "observed_runner = subprocess.run(['git', 'rev-parse', 'HEAD']",
        "status = subprocess.run(['git', 'status', '--porcelain']",
        "sys.path.insert(0, str(REPO / 'src'))",
        "parent = L4ParentIdentity.parse(",
        "verify_l4_parent_inputs(PARENT_WORKSPACE, PARENT_DATASET, parent)",
        "'pip', 'install', '--quiet', 'uv==0.11.18'",
        "subprocess.run([UV, 'sync', '--locked', '--no-editable'",
        "def run_project_json(",
        "VERIFY_INPUTS_SCRIPT = r'''",
        "verified = verify_l4_inputs(\n"
        "    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), identity\n"
        ")",
        "input_verification = run_project_json(",
        "LOCKED_RUNTIME_STATE = runtime_contract_state(",
        "benchmark_report_path = (",
        "VERIFY_BENCHMARK_SCRIPT = r'''",
        "benchmark_verification = run_project_json(",
        "if benchmark_verification == {'complete': True}:",
        "benchmark_log = (",
        "BENCHMARK_RUNTIME_BEFORE = runtime_contract_state(",
        "benchmark_command = [",
        "run_streaming_command(\n        benchmark_command,",
        "BENCHMARK_RUNTIME_AFTER = runtime_contract_state(",
        "benchmark_verification = run_project_json(",
        "package_command = [",
        "run_streaming_command(\n    package_command,",
        "VERIFY_PACKAGE_SCRIPT = r'''",
        "package_verification = run_project_json(",
        "print('L4 HANDOFF COMPLETE', package, package_sha256)",
    )
    assert (
        "benchmark_log = (\n        PARENT_WORKSPACE / 'l4_logs' / RUNNER_GIT_SHA[:12] "
        "/ 'benchmark_command.log'\n    )"
    ) in code
    assert "def run_streaming_command(" in code
    assert "from pcb_defect.notebook_runtime import run_streaming_command" not in code
    assert "benchmark_log.write_text" not in code
    assert ".unlink(" not in code
    assert "shutil.rmtree(" not in code


def test_l4_notebook_never_calls_project_code_in_the_host_kernel_after_sync() -> None:
    notebook = _notebook("notebooks/deployment_benchmark_l4.ipynb")
    after_sync: list[ast.AST] = []
    sync_seen = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for statement in ast.parse(source).body:
            segment = ast.get_source_segment(source, statement) or ""
            if sync_seen:
                after_sync.extend(ast.walk(statement))
            if "[UV, 'sync', '--locked', '--no-editable'" in segment:
                sync_seen = True

    assert sync_seen
    project_imports = [
        node
        for node in after_sync
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("pcb_defect")
    ]
    forbidden_calls = []
    for node in after_sync:
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {
            "verify_l4_inputs",
            "benchmark_is_complete",
            "l4_package_name",
            "verify_verifiable_zip",
        }:
            forbidden_calls.append(node.func.id)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "parse"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "L4RunIdentity"
        ):
            forbidden_calls.append("L4RunIdentity.parse")

    assert project_imports == []
    assert forbidden_calls == []


def test_l4_notebook_project_json_bridge_ignores_incompatible_host_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = _notebook("notebooks/deployment_benchmark_l4.ipynb")
    helper_source = _code_cell_containing(notebook, "def run_project_json(")
    helper = next(
        node
        for node in ast.parse(helper_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_project_json"
    )
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], *, cwd: Path, **_kwargs: object) -> SimpleNamespace:
        calls.append((command, cwd))
        return SimpleNamespace(
            returncode=0,
            stdout='host-version="incompatible"\n{"verified": true}\n',
            stderr="",
        )

    real_import = builtins.__import__

    def incompatible_host_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("pcb_defect"):
            raise AssertionError("host pcb_defect must not be imported after sync")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", incompatible_host_import)
    namespace = {
        "Path": Path,
        "json": json,
        "subprocess": SimpleNamespace(run=fake_run),
        "sys": sys,
        "VENV_PYTHON": Path("/locked/.venv/bin/python"),
        "REPO": Path("/runner"),
    }
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    exec(compile(module, "notebook-project-json-bridge", "exec"), namespace)

    observed = namespace["run_project_json"]("mismatch probe", "print('{}')", "arg")

    assert observed == {"verified": True}
    assert calls == [
        (
            [str(Path("/locked/.venv/bin/python")), "-c", "print('{}')", "arg"],
            Path("/runner"),
        )
    ]


def test_l4_notebook_full_retry_preserves_packaged_benchmark_log(tmp_path: Path) -> None:
    cell = _code_cell_containing(
        _notebook("notebooks/deployment_benchmark_l4.ipynb"), "benchmark_command = ["
    )
    repo = tmp_path / "runner"
    repo.mkdir()
    workspace = tmp_path / "parent"
    dataset_root = tmp_path / "dataset" / "pcb"
    package_root = tmp_path / "packages"
    report = workspace / "benchmark_l4" / ("a" * 12) / "benchmark_l4.json"
    benchmark_log = workspace / "l4_logs" / ("a" * 12) / "benchmark_command.log"
    package = package_root / "result.zip"
    benchmark_attempts = 0
    package_attempts = 0
    packaged_log: bytes | None = None

    def run_project_json(label: str, _script: str, *_arguments: object) -> dict[str, object]:
        if label == "L4 benchmark verification":
            return {"complete": report.is_file()}
        if label == "L4 package verification":
            return {
                "package": str(package),
                "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
            }
        raise AssertionError(f"unexpected bridge label: {label}")

    def run_streaming_command(
        _command: list[str], *, cwd: Path, log_path: Path, label: str
    ) -> None:
        nonlocal benchmark_attempts, package_attempts, packaged_log
        assert cwd == repo
        if label == "L4 benchmark":
            benchmark_attempts += 1
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as handle:
                handle.write(f"benchmark-attempt-{benchmark_attempts}\n".encode())
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("{}\n", encoding="utf-8", newline="\n")
            return
        if label == "L4 package":
            package_attempts += 1
            package_root.mkdir(parents=True, exist_ok=True)
            if package.exists():
                assert benchmark_log.read_bytes() == packaged_log
            else:
                packaged_log = benchmark_log.read_bytes()
                package.write_bytes(b"immutable-package")
            return
        raise AssertionError(f"unexpected command label: {label}")

    namespace = {
        "Path": Path,
        "json": json,
        "hashlib": hashlib,
        "VENV_PYTHON": tmp_path / ".venv" / "bin" / "python",
        "REPO": repo,
        "PARENT_WORKSPACE": workspace,
        "PARENT_DATASET": dataset_root,
        "DRIVE_ROOT": tmp_path,
        "RUNNER_GIT_SHA": "a" * 40,
        "PARENT_EXPERIMENT_GIT_SHA": "b" * 40,
        "PARENT_DEPLOYMENT_GATE_SHA256": "c" * 64,
        "PARENT_CHECKPOINT_SHA256": "d" * 64,
        "PARENT_ONNX_SHA256": "e" * 64,
        "LOCKED_RUNTIME_STATE": {"runtime": "locked"},
        "runtime_contract_state": lambda _label: {"runtime": "locked"},
        "run_project_json": run_project_json,
        "run_streaming_command": run_streaming_command,
    }

    exec(compile(cell, "l4-notebook-orchestration", "exec"), namespace)
    first_log = benchmark_log.read_bytes()
    first_package = package.read_bytes()
    exec(compile(cell, "l4-notebook-orchestration", "exec"), namespace)

    assert benchmark_attempts == 1
    assert package_attempts == 2
    assert benchmark_log.read_bytes() == first_log
    assert package.read_bytes() == first_package


def test_l4_notebook_benchmark_command_is_bound_only_to_immutable_values() -> None:
    cell = _code_cell_containing(
        _notebook("notebooks/deployment_benchmark_l4.ipynb"), "benchmark_command = ["
    )
    tree = ast.parse(cell)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "benchmark_command"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    assert isinstance(assignment.value, ast.List)
    assert [ast.unparse(item) for item in assignment.value.elts] == [
        "str(VENV_PYTHON)",
        "'-m'",
        "'pcb_defect.benchmark'",
        "'--repo'",
        "str(REPO)",
        "'--workspace'",
        "str(PARENT_WORKSPACE)",
        "'--dataset'",
        "str(PARENT_DATASET)",
        "'--expected-runner-git-sha'",
        "RUNNER_GIT_SHA",
        "'--expected-experiment-git-sha'",
        "PARENT_EXPERIMENT_GIT_SHA",
        "'--expected-deployment-gate-sha256'",
        "PARENT_DEPLOYMENT_GATE_SHA256",
        "'--expected-checkpoint-sha256'",
        "PARENT_CHECKPOINT_SHA256",
        "'--expected-onnx-sha256'",
        "PARENT_ONNX_SHA256",
        "'--warmup'",
        "'30'",
        "'--cycles'",
        "'4'",
    ]


def test_claim_evidence_paths_exist_and_only_supported_claims_are_verified() -> None:
    registry = yaml.safe_load((ROOT / "reports" / "claims.yaml").read_text(encoding="utf-8"))
    claims = registry["claims"]

    assert registry["schema_version"] == "1.0"
    assert {name for name, claim in claims.items() if claim["status"] == "verified"} == {
        "base_initialization",
        "paired_leakage_effect",
        "paired_protocol",
    }
    assert {name for name, claim in claims.items() if claim["status"] == "verified_candidate"} == {
        "onnx_deployment"
    }
    assert claims["tensorrt_performance"]["status"] == "verified_metadata"
    assert claims["backend_prediction_parity"]["status"] == "failed_gate"
    assert claims["hosted_demo"]["status"] == "out_of_scope"
    for claim in claims.values():
        for relative in claim["evidence"]:
            assert (ROOT / relative).is_file(), relative
        if claim["status"] in {
            "pending_colab",
            "pending_l4",
            "blocked",
            "verified_candidate",
            "verified_private",
            "verified_metadata",
            "failed_gate",
        }:
            assert claim["limitations"]


def test_promoted_claims_point_only_to_current_a100_evidence() -> None:
    claims = yaml.safe_load((ROOT / "reports" / "claims.yaml").read_text(encoding="utf-8"))[
        "claims"
    ]
    paired = claims["paired_leakage_effect"]
    onnx = claims["onnx_deployment"]

    assert paired["status"] == "verified"
    assert set(paired["evidence"]) >= {
        "reports/paired_a100/input_lock.json",
        "reports/paired_a100/deployment_selection.json",
        "reports/paired_a100/final_metrics.json",
        "reports/paired_a100/finalization_record.json",
        "reports/paired_a100/result_package_receipt.json",
    }
    assert onnx["status"] == "verified_candidate"
    assert set(onnx["evidence"]) == {
        "reports/paired_a100/input_lock.json",
        "reports/paired_a100/deployment_gate.public.json",
        "reports/paired_a100/result_package_receipt.json",
    }
    assert "reports/export_fidelity.json" not in onnx["evidence"]
    assert "reports/onnx_parity.json" not in onnx["evidence"]


def test_promoted_l4_evidence_is_hash_bound_complete_and_reports_failed_parity() -> None:
    claims = yaml.safe_load((ROOT / "reports" / "claims.yaml").read_text(encoding="utf-8"))[
        "claims"
    ]
    l4 = _read_json(ROOT / "reports" / "benchmark_l4.json")
    raw = _read_json(ROOT / "reports" / "benchmark_l4_raw.json")
    parity = _read_json(ROOT / "reports" / "backend_parity_l4.json")
    summary = (ROOT / "reports" / "benchmark_l4.md").read_text(encoding="utf-8")

    assert claims["tensorrt_performance"]["status"] == "verified_metadata"
    assert claims["tensorrt_performance"]["evidence"] == [
        "reports/benchmark_l4.json",
        "reports/benchmark_l4_raw.json",
    ]
    assert claims["backend_prediction_parity"]["status"] == "failed_gate"
    assert claims["backend_prediction_parity"]["evidence"] == [
        "reports/backend_parity_l4.json",
        "reports/benchmark_l4.json",
    ]
    assert l4["schema_version"] == "2.0"
    assert l4["status"] == "complete"
    assert l4["evidence_visibility"] == "public_metadata_from_private_unreleased_package"
    assert l4["package"] == {
        "bytes": 24_150_052,
        "filename": "paired-results-l4-9e3a1ed5827a-runner-fe9005d77920.zip",
        "sha256": "482c3bc35d8069bc3301a34f483ed599206a488626e48f34de4c8c9b7619572b",
    }
    assert l4["raw_report_sha256"] == (
        "6080de5237755444ed516e46fd903e20016b3fc562bbdf0c33dfcf90f4e718ee"
    )
    assert l4["provenance"] == {
        "dataset_sha256": "8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4",
        "deployment_gate_sha256": (
            "466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803"
        ),
        "experiment_git_sha": "9e3a1ed5827ac3759cbb15632f041e3e5c183b51",
        "manifest_sha256": "5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5",
        "runner_git_sha": "fe9005d7792036460029a376bbd9f97d7159ed41",
    }
    assert l4["protocol"]["split"] == "calibration"
    assert l4["protocol"]["images"] == 60
    assert l4["protocol"]["timing_schedule"] == "interleaved-rotating-backend-order"
    assert l4["protocol"]["sessions"] == 1
    assert l4["fidelity"]["passed"] is True
    assert l4["fidelity"]["tensorrt_minus_source"] == -0.014537137094089408
    assert l4["timings"]["onnxruntime_cuda_fp32"]["p50_ms"] == 20.277195000005577
    assert l4["timings"]["pytorch_fp32"]["p95_ms"] == 62.36269444993923
    assert l4["timings"]["tensorrt_fp16"]["fps_from_p50"] == 19.561083709081387

    assert raw["package"] == l4["package"]
    assert raw["provenance"] == l4["provenance"]
    assert raw["raw_report_sha256"] == l4["raw_report_sha256"]
    assert raw["statistical_scope"] == {
        "between_session_uncertainty_estimated": False,
        "descriptive_only": True,
        "sessions": 1,
    }
    for timing in raw["timings"].values():
        assert timing["n_runs"] == 240
        assert len(timing["raw_ms"]) == 240

    assert parity["package"] == l4["package"]
    assert parity["provenance"] == l4["provenance"]
    assert parity["raw_report_sha256"] == l4["raw_report_sha256"]
    assert parity["passed"] is False
    assert parity["thresholds"] == {
        "allowed_max_conf_delta": 0.15,
        "confidence": 0.25,
        "match_iou": 0.5,
        "required_min_iou": 0.9,
    }
    expected_comparisons = {
        "onnxruntime_cuda_fp32": (95, 62, 57, 38, 5, 40, 0.8410022500497364, 0.1983642280101776),
        "tensorrt_fp16": (95, 61, 56, 39, 5, 40, 0.8451429620055757, 0.1961173713207245),
    }
    for backend, expected in expected_comparisons.items():
        comparison = parity["comparisons"][backend]
        observed = (
            comparison["reference_detections"],
            comparison["candidate_detections"],
            comparison["matched_detections"],
            comparison["unmatched_reference_detections"],
            comparison["unmatched_candidate_detections"],
            comparison["n_failed_images"],
            comparison["min_iou"],
            comparison["max_conf_delta"],
        )
        assert observed == expected
        assert comparison["passed"] is False
        assert list(comparison["per_image"]) == [f"image_{index:03d}" for index in range(1, 61)]
    serialized_parity = json.dumps(parity, sort_keys=True)
    assert "/content/" not in serialized_parity
    assert "xyxy" not in serialized_parity
    for boundary in (
        "public metadata",
        "private and unreleased",
        "calibration-only",
        "strict per-box prediction-parity gate failed",
        "not a production SLA",
        "non-portable",
        "No public model",
    ):
        assert boundary in summary


def test_a100_report_defers_private_l4_evidence_to_its_metadata_summary() -> None:
    a100_report = (ROOT / "reports" / "paired_a100" / "README.md").read_text(encoding="utf-8")

    assert (
        "No L4 PyTorch/ONNX Runtime CUDA/TensorRT FP16 benchmark has been completed."
        not in a100_report
    )
    assert "A100 report itself does not include an L4 benchmark" in a100_report
    assert "[`benchmark_l4.md`](../benchmark_l4.md)" in a100_report
    assert "[`benchmark_l4.json`](../benchmark_l4.json)" in a100_report


def test_readme_protocol_numbers_match_machine_artifacts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    protocol_config = yaml.safe_load(
        (ROOT / "configs" / "paired_protocol.yaml").read_text(encoding="utf-8")
    )
    manifest_path = ROOT / "reports" / "protocol" / "paired_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"] == protocol_config["frozen_hashes"]["manifest_sha256"]
    assert manifest["dataset"]["sha256"] == protocol_config["frozen_hashes"]["dataset_sha256"]
    assert protocol_config["frozen_hashes"]["manifest_sha256"] in readme
    assert protocol_config["frozen_hashes"]["dataset_sha256"] in readme
    assert manifest["counts"]["final_test"]["images"] == 30
    assert manifest["counts"]["grouped_train"]["images"] == 513
    assert manifest["counts"]["leaky_train"]["images"] == 513
    file_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sidecar = (manifest_path.parent / "paired_split_manifest.sha256").read_text(encoding="ascii")
    assert sidecar == f"{file_digest}  paired_split_manifest.json\n"


def test_portfolio_documents_match_promoted_a100_metrics() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    metrics = _read_json(PAIRED_A100 / "final_metrics.json")
    grouped = metrics["aggregate"]["by_arm"]["grouped"]["map50"]
    leaky = metrics["aggregate"]["by_arm"]["leaky_control"]["map50"]
    grouped_map50_95 = metrics["aggregate"]["by_arm"]["grouped"]["map50_95"]
    leaky_map50_95 = metrics["aggregate"]["by_arm"]["leaky_control"]["map50_95"]
    difference_pp = (leaky["mean"] - grouped["mean"]) * 100

    for document in (readme, model_card):
        assert f"{grouped['mean']:.4f}" in document
        assert f"{grouped['std']:.4f}" in document
        assert f"{leaky['mean']:.4f}" in document
        assert f"{leaky['std']:.4f}" in document
        assert f"{difference_pp:.1f}" in document
        assert "5,544,453 bytes" not in document
    assert (
        f"{grouped_map50_95['mean'] * 100:.2f}% ± {grouped_map50_95['std'] * 100:.2f}%"
    ) in readme
    assert (f"{leaky_map50_95['mean'] * 100:.2f}% ± {leaky_map50_95['std'] * 100:.2f}%") in readme
    assert "0.6087" not in readme
    assert "0.8633" not in readme
    assert "60/60" in readme
    assert "Same-ONNX wrapper parity" in readme
    assert "非 PyTorch reference" in readme


def test_portfolio_documents_bound_paired_inference_to_single_board() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    paired_section = readme.split("### 1. 成對板級洩漏實驗結果", maxsplit=1)[1].split(
        "### 2. NVIDIA L4", maxsplit=1
    )[0]
    model_card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    limitations = (ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    paired_report = (ROOT / "reports" / "paired_a100" / "README.md").read_text(encoding="utf-8")
    paired_claim = yaml.safe_load((ROOT / "reports" / "claims.yaml").read_text(encoding="utf-8"))[
        "claims"
    ]["paired_leakage_effect"]

    for forbidden in (
        "真實跨板泛化表現",
        "反映實際產線部署效能",
        "證明板級洩漏顯著性",
        "確認洩漏效應具備嚴格統計顯著性",
        "統計顯著",
    ):
        assert forbidden not in readme

    for required in (
        "30 張",
        "Board 08",
        "Resampling unit 是 image，不是 board",
        "不估計 between-board uncertainty",
        "production generalization",
    ):
        assert required in paired_section

    assert "one board" in paired_claim["limitations"][0]
    assert "between-board variance" in paired_claim["limitations"][0]
    assert "not a universal" in paired_claim["limitations"][1]
    assert "single PCB template board" in model_card
    assert "Image-bootstrap intervals do not estimate board-level uncertainty" in model_card
    assert "one board and 30 images" in limitations
    assert "image, not board, is the resampling unit" in paired_report


def test_portfolio_documents_bound_l4_metrics_and_failed_parity_to_calibration() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    l4 = _read_json(ROOT / "reports" / "benchmark_l4.json")
    parity = _read_json(ROOT / "reports" / "backend_parity_l4.json")

    for document in (readme, model_card):
        assert "private" in document
        assert "calibration" in document
        assert "production" in document
        assert "public model" in document or "public checkpoint" in document
        assert "strict per-box" in document
    assert str(l4["timings"]["tensorrt_fp16"]["p50_ms"]) in readme
    assert str(l4["timings"]["tensorrt_fp16"]["p95_ms"]) in model_card
    assert str(l4["fidelity"]["tensorrt_minus_source"]) in model_card
    assert parity["passed"] is False
    assert "strict per-box prediction-parity gate failed" in model_card


def test_release_checklist_marks_returned_a100_and_private_l4_evidence_complete() -> None:
    checklist = (ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")

    for text in (
        "A100 clean-runtime, data/hash, tiny-train, resume, and speed gates pass.",
        "Six runs complete with matching run records and checkpoint hashes.",
        "Deployment checkpoint is selected from grouped validation before final-test access.",
        "One-shot common final evaluation completes and reports three-seed mean/std "
        "and paired image",
        "Calibration-only ONNX fidelity and standalone parity gates pass.",
        "Final result ZIP and sidecar SHA-256 are returned from Drive.",
    ):
        assert f"- [x] {text}" in checklist
    assert "- [x] L4 PyTorch/ORT CUDA/TensorRT FP16" in checklist
    assert "reports/benchmark_l4.json" in checklist
    assert "reports/benchmark_l4_raw.json" in checklist
    assert "reports/backend_parity_l4.json" in checklist
    assert "strict per-box prediction-parity gate failed" in checklist
    assert "metadata-only portfolio release candidate" in checklist
    assert (
        "- [x] Official GitHub namespace is independently verified as "
        "`kuotunyu/pcb-defect-detection`." in checklist
    )
    assert "- [x] The annotated `v0.1.0` tag resolves to" in checklist
    assert "`56c086206eab9be1a9c6a4e36410fd13ed42f5ec`" in checklist
    assert "https://github.com/kuotunyu/pcb-defect-detection/releases/tag/v0.1.0" in checklist
    assert "- [x] Zenodo version DOI" in checklist
    assert "10.5281/zenodo.21877497" in checklist
    assert "10.5281/zenodo.21877496" in checklist
    assert "A Zenodo deposit remains a separate external action" not in checklist
    assert "Owner-authorized push, tag, GitHub Release" not in checklist
    assert "Official push/review is completed" not in checklist
    assert "Hugging Face publication and hosted inference are intentional non-goals" in checklist


def test_public_metadata_matches_authoritative_release_state() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    base_model = yaml.safe_load((ROOT / "configs" / "base_model.yaml").read_text(encoding="utf-8"))
    claims = yaml.safe_load((ROOT / "reports" / "claims.yaml").read_text(encoding="utf-8"))[
        "claims"
    ]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    app_readme = (ROOT / "app" / "README.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    limitations = (ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    license_boundary = (ROOT / "docs" / "license-boundary.md").read_text(encoding="utf-8")
    paired_readme = (ROOT / "reports" / "paired_a100" / "README.md").read_text(encoding="utf-8")
    l4 = _read_json(ROOT / "reports" / "benchmark_l4.json")
    contract = _read_json(ROOT / "app" / "model_contract.json")

    python_floor = project["requires-python"].removeprefix(">=")
    torch_requirement = next(
        requirement
        for requirement in project["optional-dependencies"]["train"]
        if requirement.startswith("torch>=")
    )
    torch_floor = torch_requirement.removeprefix("torch>=")
    license_expression = project["license"]
    license_badge = license_expression.replace("-", "--")
    model_filename = base_model["filename"]
    model_name = Path(model_filename).stem
    model_label = f"YOLO{model_name.removeprefix('yolo')}"

    assert project["requires-python"].startswith(">=")
    assert base_model["source"].endswith(f"/{model_filename}")
    assert base_model["revision"] in claims["base_initialization"]["statement"]
    assert model_label in claims["base_initialization"]["statement"]
    assert claims["onnx_deployment"]["status"] == "verified_candidate"
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (ROOT / "LICENSE").read_text(encoding="utf-8")

    for required in (
        f"Python-{python_floor}%2B",
        f"PyTorch-{torch_floor}%2B",
        model_label,
        f"License-{license_badge}",
        license_expression,
    ):
        assert required in readme
    public_source_paths = (
        "src/pcb_defect/data_prep/paired.py",
        "src/pcb_defect/experiment.py",
        "src/pcb_defect/final_evaluation.py",
    )
    for relative in public_source_paths:
        assert (ROOT / relative).is_file()
        assert relative in readme
    for stale in ("Python-3.10%2B", "PyTorch-2.0%2B", "YOLOv8", "License-MIT"):
        assert stale not in readme

    backend_rows = {
        "pytorch_fp32": ("PyTorch", "FP32"),
        "onnxruntime_cuda_fp32": ("ONNX Runtime CUDA", "FP32"),
        "tensorrt_fp16": ("TensorRT", "FP16"),
    }
    for key, (backend_name, precision) in backend_rows.items():
        timing = l4["timings"][key]
        expected_row = (
            f"| {backend_name} | {precision} | {timing['p50_ms']:.2f} | "
            f"{timing['p95_ms']:.2f} | {timing['fps_from_p50']:.2f} |"
        )
        assert expected_row in readme
    assert "**ONNX Runtime CUDA FP32 是本次最快後端**" in readme
    assert "Opset 17" not in readme
    assert "60 張測試影像" not in readme
    for stale_command in (
        "python -m pcb_defect.experiment preflight",
        "python -m pcb_defect.experiment gates",
        "python -m pcb_defect.experiment train-all",
        "python -m pcb_defect.final_evaluation",
    ):
        assert stale_command not in readme
    assert "--extra train --group eval" in readme
    assert "python -m pcb_defect.experiment --help" in readme

    assert contract["status"] == "blocked"
    assert "Aggregate fidelity gate passed" in contract["reason"]
    assert "strict L4 backend prediction-parity gate failed" in contract["reason"]
    assert "Deployment gate passed" not in contract["reason"]
    assert "metadata-only portfolio release candidate" in contract["reason"]
    assert contract["reason"] in app_readme
    app_metadata = yaml.safe_load(app_readme.split("---", 2)[1])
    assert app_metadata["license"] == "agpl-3.0"
    app_description = app_metadata["short_description"].lower()
    assert "metadata-only" in app_description
    assert "intentionally" in app_description
    for stale in (
        "blocked until the paired deployment gate passes",
        "until the newly trained grouped checkpoint passes",
    ):
        assert stale not in app_readme
    for field in (
        "onnx_sha256",
        "source_checkpoint_sha256",
        "deployment_gate_sha256",
        "hf_repo_id",
        "hf_revision",
    ):
        assert contract[field] is None

    assert "Aggregate fidelity passed" in license_boundary
    assert "strict backend prediction parity failed" in license_boundary
    assert "- [x] The annotated `v0.1.0` tag resolves to" in checklist
    assert "`56c086206eab9be1a9c6a4e36410fd13ed42f5ec`" in checklist
    assert "https://github.com/kuotunyu/pcb-defect-detection/releases/tag/v0.1.0" in checklist
    assert "- [x] Zenodo version DOI" in checklist
    assert "10.5281/zenodo.21877497" in checklist
    assert "10.5281/zenodo.21877496" in checklist
    assert "A Zenodo deposit remains a separate external action" not in checklist
    assert "Owner-authorized push, tag, GitHub Release" not in checklist
    assert "Official push/review is completed" not in checklist
    assert "metadata-only portfolio release candidate" in model_card
    assert "identity review" not in model_card
    assert claims["hosted_demo"]["status"] == "out_of_scope"
    assert "metadata-only portfolio release candidate" in claims["hosted_demo"]["statement"]
    assert "out of scope" in claims["hosted_demo"]["limitations"][0]
    assert "legacy_split_sensitivity" not in claims
    for document in (readme, limitations, model_card, checklist, paired_readme):
        assert "12.1" not in document
    assert "v0.1.0 source-and-metadata evidence is published on GitHub and Zenodo" in paired_readme
    assert "External publication is not asserted" not in paired_readme
    assert "No public checkpoint, ONNX export, model-Hub" in model_card
    assert "revision, or hosted demo is claimed" in model_card
    assert "External publication, a public checkpoint" not in model_card
    assert "Current release candidate has clean single-author reachable history" in limitations
    assert "future L4 benchmark" not in limitations
    assert "Git history contains legacy identity" not in limitations


def test_project_metadata_points_to_official_single_author_portfolio() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["authors"] == [{"name": "kuotunyu"}]
    assert project["urls"] == {
        "Repository": "https://github.com/kuotunyu/pcb-defect-detection",
        "Issues": "https://github.com/kuotunyu/pcb-defect-detection/issues",
    }
    assert {"computer vision", "object detection", "model evaluation"} <= set(project["keywords"])


def test_readme_presents_the_recorded_evidence_workstation() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/assets/ui-workstation-desktop.png" in readme
    assert "PCB 人工複核工作站" in readme
    assert "Recorded evidence" in readme
    assert "不宣稱提供 hosted inference" in readme


def test_readme_primary_diagrams_form_an_evidence_narrative() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = (
        "## 系統全貌與證據邊界",
        "## Evidence-first 系統架構",
        "## Fail-closed 工作站啟動時序",
        "## Deep Dive：實驗與部署 Pipeline",
    )

    assert all(heading in readme for heading in headings)
    assert [readme.index(heading) for heading in headings] == sorted(
        readme.index(heading) for heading in headings
    )
    assert "sequenceDiagram" in readme
    assert readme.count("```mermaid") == 5


def test_readme_diagrams_are_accessible_and_github_native() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.count("accTitle:") == 5
    assert readme.count("accDescr:") == 5
    assert "mermaid.live" not in readme
    assert "kroki" not in readme.lower()
    assert "plantuml" not in readme.lower()


def test_readme_preserves_pipeline_deep_dives() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "<summary>展開：PCB 板級防洩漏與成對實驗流程</summary>" in readme
    assert "<summary>展開：ONNX fidelity 與 NVIDIA L4 多後端部署管線</summary>" in readme
    assert "Leaky - grouped: +21.3 pp" in readme
    assert "p50: 20.28 ms · 49.32 FPS" in readme
    assert "strict per-box parity" in readme


def test_readme_context_diagram_keeps_public_and_private_assets_separate() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    context = readme.split("## 系統全貌與證據邊界", maxsplit=1)[1].split(
        "## 30 秒證據索引", maxsplit=1
    )[0]

    for required in (
        "Public Portfolio",
        "Private Evidence Production",
        "No hosted inference",
        "No public model artifact",
        "GitHub Release + Zenodo DOI",
    ):
        assert required in context


def test_public_ui_screenshots_exist() -> None:
    from PIL import Image

    expected_dimensions = {
        "ui-workstation-desktop.png": (1440, 900),
        "ui-workstation-mobile.png": (390, 844),
    }
    for name in ("ui-workstation-desktop.png", "ui-workstation-mobile.png"):
        path = ROOT / "docs" / "assets" / name
        assert path.is_file()
        assert path.stat().st_size > 20_000
        with Image.open(path) as screenshot:
            assert screenshot.size == expected_dimensions[name]


def test_candidate_tree_contains_no_dataset_or_model_binaries() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    pixel_files = {
        path
        for path in tracked
        if Path(path).suffix.lower() in {".jpg", ".jpeg", ".gif"}
        and not path.startswith("tests/fixtures/")
    }
    model_or_package_files = {
        path
        for path in tracked
        if Path(path).suffix.lower() in {".pt", ".onnx", ".engine", ".plan", ".trt", ".zip"}
    }

    assert pixel_files == set()
    assert model_or_package_files == set()


def test_public_tree_excludes_internal_planning_artifacts() -> None:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines()
    )

    assert "plan.md" not in tracked
    assert not any(path.startswith("docs/superpowers/") for path in tracked)


def test_public_tree_excludes_superseded_prototype_artifacts() -> None:
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines()
    )

    assert tracked.isdisjoint(
        {
            "assets/figures/grouped_BoxPR_curve.png",
            "assets/figures/grouped_confusion_matrix_normalized.png",
            "assets/figures/random_BoxPR_curve.png",
            "assets/figures/random_confusion_matrix_normalized.png",
            "reports/bbox_size_at_640.png",
            "reports/bbox_size_relative.png",
            "reports/benchmark.md",
            "reports/benchmark_cpu.json",
            "reports/benchmark_gpu.json",
            "reports/class_balance.png",
            "reports/export_fidelity.json",
            "reports/leakage_comparison.md",
            "reports/legacy-evidence.md",
            "reports/onnx_parity.json",
            "reports/sahi_ablation.json",
            "reports/sahi_ablation.md",
            "reports/stats.md",
            "reports/test_metrics.json",
            "src/pcb_defect/stats.py",
        }
    )


def test_current_tracked_text_has_no_personal_account_or_local_path() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    excluded = {"tests/test_evidence.py", "tests/test_release_contract.py"}
    forbidden = ("steven0226", "tun0000", "C:/Users/", "C:\\Users\\")
    findings = []
    for relative in tracked:
        if relative in excluded:
            continue
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for token in forbidden:
            if token in text:
                findings.append((relative, token))

    assert findings == []


def test_notebooks_are_thin_unexecuted_and_require_immutable_handoff_values() -> None:
    for relative in (
        "notebooks/paired_experiment_a100.ipynb",
        "notebooks/deployment_benchmark_l4.ipynb",
        "notebooks/deployment_parity_probe_a100.ipynb",
    ):
        notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        assert all(not cell.get("outputs") for cell in notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                compile("".join(cell["source"]), f"{relative}:{cell}", "exec")
        assert "PASTE_FINAL_BUNDLE_SHA256" in code
        assert "PASTE_FINAL_GIT_SHA" in code
        assert "uv==0.11.18" in code
        assert "--locked" in code
        assert "--no-editable" in code
        assert "--reinstall-package', 'pcb-defect'" in code
        assert "VENV_PYTHON" in code
        assert "MPLBACKEND'] = 'Agg'" in code
        assert "capture_output=True" in code
        assert "[UV, 'run'" not in code

    a100 = json.loads(
        (ROOT / "notebooks" / "paired_experiment_a100.ipynb").read_text(encoding="utf-8")
    )
    a100_code = "\n".join(
        "".join(cell["source"]) for cell in a100["cells"] if cell["cell_type"] == "code"
    )
    assert "IMPORT PROBE FAILED" in a100_code
    assert "capture_output=True" in a100_code
    assert "GPU GATE COMMAND FAILED" in a100_code
    assert "gate_report.json" in a100_code
    assert "os.environ['MPLBACKEND'] = 'Agg'" in a100_code
    assert "deployment_gate.json" in a100_code
    assert "model_contract.candidate.json" in a100_code
    assert "DEPLOYMENT EVIDENCE" in a100_code
    assert "run_logged" in a100_code

    probe = json.loads(
        (ROOT / "notebooks" / "deployment_parity_probe_a100.ipynb").read_text(encoding="utf-8")
    )
    probe_code = "\n".join(
        "".join(cell["source"]) for cell in probe["cells"] if cell["cell_type"] == "code"
    )
    assert "PASTE_PARENT_EXPERIMENT_GIT_SHA" in probe_code
    assert "PASTE_PARENT_DEPLOYMENT_GATE_SHA256" in probe_code
    assert "PASTE_PARENT_ONNX_SHA256" in probe_code
    assert "probe_command.log" in probe_code
    assert "PARITY PROBE PASS" in probe_code
    assert "train-all" not in probe_code
    assert "experiment train" not in probe_code


def test_a100_train_all_streams_combined_output_to_an_append_only_drive_log() -> None:
    """The long six-run stage must be live, durable, and safe to resume."""
    notebook = json.loads(
        (ROOT / "notebooks" / "paired_experiment_a100.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert "from pcb_defect.notebook_runtime import run_streaming_command" in code
    assert "train_all_command.log" in code
    assert "run_streaming_command(" in code
    direct_train_all = (
        "subprocess.run([str(VENV_PYTHON), '-m', 'pcb_defect.experiment', 'train-all'"
    )
    assert direct_train_all not in code


def test_probe_notebook_requires_exclusive_log_and_complete_report_before_pass() -> None:
    """The handoff probe may print PASS only after validating its actual report schema."""
    notebook = json.loads(
        (ROOT / "notebooks" / "deployment_parity_probe_a100.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    pass_offset = code.index("print('PARITY PROBE PASS'")
    completion_gate = code[:pass_offset]

    assert "PROBE_REPORT = PROBE_DIRECTORY / 'parity_probe.json'" in code
    runtime_import = (
        "from pcb_defect.notebook_runtime import run_captured_command, verify_probe_result"
    )
    assert runtime_import in code
    assert "run_captured_command(" in completion_gate
    assert "verify_probe_result(" in completion_gate
    assert completion_gate.index("verify_probe_result(") < pass_offset


def test_release_python_and_ci_install_contract_are_consistent() -> None:
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11"
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv sync --locked --no-editable" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9" in workflow


def test_non_editable_install_cache_tracks_local_wheel_inputs() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cache_patterns = {
        entry["file"] for entry in project["tool"]["uv"]["cache-keys"] if "file" in entry
    }

    assert {"pyproject.toml", "README.md", "LICENSE", "src/**/*.py"} <= cache_patterns
    source_files = subprocess.run(
        ["git", "ls-files", "src"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    assert source_files
    assert all(Path(relative).suffix == ".py" for relative in source_files)


def test_source_distribution_excludes_binary_test_fixtures(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    archives = list(tmp_path.glob("*.tar.gz"))
    assert len(archives) == 1

    with tarfile.open(archives[0], mode="r:gz") as archive:
        members = archive.getnames()

    forbidden_suffixes = {".engine", ".jpg", ".onnx", ".plan", ".pt", ".zip"}
    assert not [name for name in members if Path(name).suffix.lower() in forbidden_suffixes]


def test_linux_onnxruntime_contract_targets_colab_cuda_12() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    evaluation_dependencies = project["dependency-groups"]["eval"]

    assert "onnxruntime==1.26.0; sys_platform == 'win32'" in evaluation_dependencies
    assert "onnxruntime-gpu==1.26.0; sys_platform == 'linux'" in evaluation_dependencies


def test_l4_notebook_installs_and_probes_locked_tensorrt_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["dependency-groups"]["l4"] == [
        "tensorrt-cu12==10.13.3.9; sys_platform == 'linux'"
    ]

    code = _code(_notebook("notebooks/deployment_benchmark_l4.ipynb"))
    sync = (
        "[UV, 'sync', '--locked', '--no-editable', '--extra', 'train', '--group', 'eval', "
        "'--group', 'l4'"
    )
    probe = "LOCKED_TENSORRT_STATE = run_project_json('LOCKED TENSORRT CONTRACT'"
    _assert_in_order(code, sync, probe, "LOCKED_RUNTIME_STATE = runtime_contract_state(")
    assert "import tensorrt as trt" in code
    assert "trt.__version__ == '10.13.3.9'" in code
    assert "bool(trt.Builder(trt.Logger()))" in code


def test_gpu_notebook_runtime_contract_helpers_are_strictly_bound() -> None:
    for relative in GPU_NOTEBOOKS:
        helper = _code_cell_containing(_notebook(relative), "def runtime_contract_state(")
        tree = ast.parse(helper)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "runtime_contract_state"
        )
        command = next(
            node
            for node in function.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "command" for target in node.targets
            )
        )
        assert isinstance(command.value, ast.List)
        assert [ast.unparse(item) for item in command.value.elts] == [
            "str(VENV_PYTHON)",
            "'-m'",
            "'pcb_defect.runtime_contract'",
            "'--require-cuda-provider'",
        ]
        _assert_in_order(
            helper,
            "if not VENV_PYTHON.is_file():",
            "raise RuntimeError(f'Locked environment Python is missing: {VENV_PYTHON}')",
            "def runtime_contract_state(label: str) -> dict[str, object]:",
            "command = [",
            "result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)",
            "if result.returncode:",
            "raise RuntimeError(f'{label} FAILED')",
            "lines = [line for line in result.stdout.splitlines() if line.strip()]",
            "if not lines:",
            "raise RuntimeError(f'{label} returned no runtime state')",
            "return json.loads(lines[-1])",
            "except json.JSONDecodeError as exc:",
            "raise RuntimeError(f'{label} returned invalid runtime JSON') from exc",
            "LOCKED_RUNTIME_STATE = runtime_contract_state(",
        )
