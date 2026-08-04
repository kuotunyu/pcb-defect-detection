from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import pcb_defect.l4_handoff as l4_handoff
from pcb_defect.handoff import HandoffError
from pcb_defect.l4_contract import L4ContractError, L4ParentIdentity
from pcb_defect.l4_handoff import create_l4_handoff, main

ROOT = Path(__file__).resolve().parent.parent


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _commit(repo: Path, *, amend: bool = False) -> None:
    _git(repo, "add", "-A")
    arguments = [
        "-c",
        "user.name=Release",
        "-c",
        "user.email=release@example.invalid",
        "commit",
    ]
    if amend:
        arguments.extend(["--amend", "--no-edit"])
    else:
        arguments.extend(["-m", "source"])
    _git(repo, *arguments)


def _parent_identity() -> L4ParentIdentity:
    return L4ParentIdentity.parse(
        experiment_git_sha="9e3a1ed5827ac3759cbb15632f041e3e5c183b51",
        deployment_gate_sha256=("466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803"),
        checkpoint_sha256="44646b130b8b42282b752f77659cabfc1c484dc3aaa9a2dc8f710da8468f511a",
        onnx_sha256="b62590a14e2e88a414eb06389058d13d69ff1ea3998232996877088951fe3bb8",
    )


def _complete_source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "reports" / "protocol").mkdir(parents=True)
    (repo / "configs").mkdir()
    shutil.copyfile(
        ROOT / "notebooks" / "deployment_benchmark_l4.ipynb",
        repo / "notebooks" / "deployment_benchmark_l4.ipynb",
    )
    (repo / "reports" / "protocol" / "paired_split_manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "1.0",
                "dataset": {"sha256": "d" * 64},
                "manifest_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )
    (repo / "configs" / "base_model.yaml").write_text(
        "source: model.pt\nrevision: v1\nsha256: " + "f" * 64 + "\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-b", "main")
    _commit(repo)
    return repo


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_cli_arguments(repo: Path, output_root: Path) -> list[str]:
    parent = _parent_identity()
    return [
        "--repo",
        str(repo),
        "--output-root",
        str(output_root),
        "--parent-experiment-git-sha",
        parent.experiment_git_sha,
        "--parent-deployment-gate-sha256",
        parent.deployment_gate_sha256,
        "--parent-checkpoint-sha256",
        parent.checkpoint_sha256,
        "--parent-onnx-sha256",
        parent.onnx_sha256,
    ]


def _verify_published_handoff(output: Path) -> None:
    manifest = json.loads((output / "handoff_manifest.json").read_text(encoding="utf-8"))
    runner_sha = manifest["snapshot_git_sha"]
    l4_handoff._verify_l4_handoff(
        output,
        _parent_identity(),
        expected_runner_sha=runner_sha,
        expected_bundle_sha256=_sha256(output / "pcb-defect-source.bundle"),
        expected_drive_directory=(
            "/content/drive/MyDrive/pcb-defect-paired/handoff-l4/" + runner_sha[:12]
        ),
    )


def _append_code_and_rehash(output: Path, source: str) -> None:
    notebook_path = output / "deployment_benchmark_l4.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
    )
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest_path = output / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["l4_notebook_sha256"] = _sha256(notebook_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _synthetic_ast_node(class_name: str, name: object) -> ast.AST:
    node_type = type(class_name, (ast.AST,), {"_fields": ("name",)})
    node = node_type()
    node.name = name
    return node


def test_l4_handoff_contains_only_l4_stage_files(tmp_path: Path) -> None:
    repo = _complete_source_repo(tmp_path)

    output = create_l4_handoff(repo, tmp_path / "dist", _parent_identity())

    manifest = json.loads((output / "handoff_manifest.json").read_text(encoding="utf-8"))
    assert output.name == f"colab-handoff-l4-{manifest['snapshot_git_sha'][:12]}"
    assert {path.name for path in output.iterdir()} == {
        "deployment_benchmark_l4.ipynb",
        "handoff_manifest.json",
        "pcb-defect-source.bundle",
    }
    assert manifest["stage"] == "l4-benchmark"
    assert manifest["parent_experiment_git_sha"] == _parent_identity().experiment_git_sha
    assert "a100_notebook" not in manifest
    assert "probe_notebook" not in manifest
    assert manifest["bundle_sha256"] == _sha256(output / manifest["bundle"])
    assert manifest["l4_notebook_sha256"] == _sha256(output / manifest["l4_notebook"])
    assert manifest["l4_template_sha256"] == _sha256(
        repo / "notebooks" / "deployment_benchmark_l4.ipynb"
    )


def test_l4_handoff_bundle_is_one_clean_exact_snapshot(tmp_path: Path) -> None:
    repo = _complete_source_repo(tmp_path)
    source_sha = _git(repo, "rev-parse", "HEAD")
    output = create_l4_handoff(repo, tmp_path / "dist", _parent_identity())
    bundle = output / "pcb-defect-source.bundle"
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=repo, check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(bundle), str(clone)], check=True)
    manifest = json.loads((output / "handoff_manifest.json").read_text(encoding="utf-8"))
    _git(clone, "checkout", "--detach", manifest["snapshot_git_sha"])

    assert _git(clone, "rev-list", "--count", "HEAD") == "1"
    assert _git(clone, "rev-parse", "HEAD") == manifest["snapshot_git_sha"]
    assert _git(clone, "status", "--porcelain") == ""
    provenance = json.loads((clone / ".source_provenance.json").read_text(encoding="utf-8"))
    assert provenance == {
        "base_model_contract_sha256": _sha256(repo / "configs" / "base_model.yaml"),
        "base_model_revision": "v1",
        "base_model_sha256": "f" * 64,
        "base_model_source": "model.pt",
        "dataset_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "parent_checkpoint_sha256": _parent_identity().checkpoint_sha256,
        "parent_deployment_gate_sha256": _parent_identity().deployment_gate_sha256,
        "parent_experiment_git_sha": _parent_identity().experiment_git_sha,
        "parent_onnx_sha256": _parent_identity().onnx_sha256,
        "protocol_version": "1.0",
        "source_git_sha": source_sha,
        "stage": "l4-benchmark",
    }


def test_l4_handoff_renders_unexecuted_compilable_notebook(tmp_path: Path) -> None:
    output = create_l4_handoff(
        _complete_source_repo(tmp_path), tmp_path / "dist", _parent_identity()
    )

    rendered = (output / "deployment_benchmark_l4.ipynb").read_text(encoding="utf-8")
    notebook = json.loads(rendered)
    assert "PASTE_" not in rendered
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


def test_l4_handoff_renders_each_identity_into_its_declared_role(tmp_path: Path) -> None:
    output = create_l4_handoff(
        _complete_source_repo(tmp_path), tmp_path / "dist", _parent_identity()
    )
    manifest = json.loads((output / "handoff_manifest.json").read_text(encoding="utf-8"))
    rendered = (output / "deployment_benchmark_l4.ipynb").read_text(encoding="utf-8")

    expected_values = (
        manifest["bundle_sha256"],
        manifest["snapshot_git_sha"],
        _parent_identity().experiment_git_sha,
        _parent_identity().deployment_gate_sha256,
        _parent_identity().checkpoint_sha256,
        _parent_identity().onnx_sha256,
        manifest["drive_handoff_directory"],
    )
    assert manifest["snapshot_git_sha"] != _parent_identity().experiment_git_sha
    assert manifest["drive_handoff_directory"] == (
        "/content/drive/MyDrive/pcb-defect-paired/handoff-l4/" + manifest["snapshot_git_sha"][:12]
    )
    for value in expected_values:
        assert rendered.count(value) == 1

    first_code = next(cell for cell in json.loads(rendered)["cells"] if cell["cell_type"] == "code")
    expected_assignments = {
        "SOURCE_BUNDLE_SHA256": manifest["bundle_sha256"],
        "RUNNER_GIT_SHA": manifest["snapshot_git_sha"],
        "PARENT_EXPERIMENT_GIT_SHA": _parent_identity().experiment_git_sha,
        "PARENT_DEPLOYMENT_GATE_SHA256": _parent_identity().deployment_gate_sha256,
        "PARENT_CHECKPOINT_SHA256": _parent_identity().checkpoint_sha256,
        "PARENT_ONNX_SHA256": _parent_identity().onnx_sha256,
        "L4_HANDOFF_DIRECTORY": manifest["drive_handoff_directory"],
    }
    observed_assignments = {
        target.id: statement.value.value
        for statement in ast.parse("".join(first_code["source"])).body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance((target := statement.targets[0]), ast.Name)
        and target.id in expected_assignments
        and isinstance(statement.value, ast.Constant)
    }
    assert observed_assignments == expected_assignments


def test_l4_handoff_verifier_rejects_checkpoint_onnx_role_swap_after_rehash(
    tmp_path: Path,
) -> None:
    output = create_l4_handoff(
        _complete_source_repo(tmp_path), tmp_path / "dist", _parent_identity()
    )
    notebook_path = output / "deployment_benchmark_l4.ipynb"
    notebook_source = notebook_path.read_text(encoding="utf-8")
    checkpoint = _parent_identity().checkpoint_sha256
    onnx = _parent_identity().onnx_sha256
    notebook_path.write_text(
        notebook_source.replace(checkpoint, "SWAP-TEMP", 1)
        .replace(onnx, checkpoint, 1)
        .replace("SWAP-TEMP", onnx, 1),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = output / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["l4_notebook_sha256"] = _sha256(notebook_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(HandoffError, match="assignment"):
        _verify_published_handoff(output)


def test_l4_handoff_verifier_rejects_named_expression_identity_rebinding(
    tmp_path: Path,
) -> None:
    output = create_l4_handoff(
        _complete_source_repo(tmp_path), tmp_path / "dist", _parent_identity()
    )
    _append_code_and_rehash(
        output,
        "if (RUNNER_GIT_SHA := 'attacker-rebound-runner'):\n    pass\n",
    )

    with pytest.raises(HandoffError, match="assignment|binding"):
        _verify_published_handoff(output)


def test_l4_handoff_verifier_rejects_unpacked_identity_rebinding(tmp_path: Path) -> None:
    output = create_l4_handoff(
        _complete_source_repo(tmp_path), tmp_path / "dist", _parent_identity()
    )
    _append_code_and_rehash(
        output,
        "(RUNNER_GIT_SHA, harmless) = ('attacker-rebound-runner', None)\n",
    )

    with pytest.raises(HandoffError, match="assignment|binding"):
        _verify_published_handoff(output)


@pytest.mark.parametrize(
    "source",
    [
        "for RUNNER_GIT_SHA in ():\n    pass\n",
        "[RUNNER_GIT_SHA for RUNNER_GIT_SHA in ()]\n",
        "with nullcontext() as RUNNER_GIT_SHA:\n    pass\n",
        "try:\n    pass\nexcept Exception as RUNNER_GIT_SHA:\n    pass\n",
        "def helper(RUNNER_GIT_SHA):\n    return RUNNER_GIT_SHA\n",
        "class RUNNER_GIT_SHA:\n    pass\n",
        "import json as RUNNER_GIT_SHA\n",
        "global RUNNER_GIT_SHA\n",
        "del RUNNER_GIT_SHA\n",
    ],
    ids=(
        "for-target",
        "comprehension-target",
        "with-as-target",
        "except-target",
        "function-argument",
        "class-name",
        "import-alias",
        "global-declaration",
        "delete-target",
    ),
)
def test_l4_handoff_verifier_rejects_other_identity_binding_forms(
    tmp_path: Path, source: str
) -> None:
    output = create_l4_handoff(
        _complete_source_repo(tmp_path), tmp_path / "dist", _parent_identity()
    )
    _append_code_and_rehash(output, source)

    with pytest.raises(HandoffError, match="assignment|binding"):
        _verify_published_handoff(output)


@pytest.mark.parametrize("class_name", ["TypeVar", "TypeVarTuple", "ParamSpec"])
def test_l4_binding_collector_records_pep695_type_parameter_names(class_name: str) -> None:
    node = _synthetic_ast_node(class_name, "RUNNER_GIT_SHA")
    collector = l4_handoff._ImmutableBindingCollector()

    collector.visit(node)

    assert collector.bindings["RUNNER_GIT_SHA"] == [node]


@pytest.mark.parametrize("class_name", ["TypeVar", "TypeVarTuple", "ParamSpec"])
@pytest.mark.parametrize("malformed_name", [None, "", 7])
def test_l4_binding_collector_rejects_malformed_pep695_type_parameter_names(
    class_name: str, malformed_name: object
) -> None:
    node = _synthetic_ast_node(class_name, malformed_name)
    collector = l4_handoff._ImmutableBindingCollector()

    with pytest.raises(HandoffError, match="type parameter"):
        collector.visit(node)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 parsing requires Python 3.12 or newer"
)
@pytest.mark.parametrize(
    "source",
    [
        "def helper[RUNNER_GIT_SHA]():\n    pass\n",
        "class Helper[RUNNER_GIT_SHA]:\n    pass\n",
        "def helper[*RUNNER_GIT_SHA]():\n    pass\n",
        "def helper[**RUNNER_GIT_SHA]():\n    pass\n",
    ],
    ids=("function-typevar", "class-typevar", "typevartuple", "paramspec"),
)
def test_l4_handoff_verifier_rejects_pep695_identity_shadowing(tmp_path: Path, source: str) -> None:
    output = create_l4_handoff(
        _complete_source_repo(tmp_path), tmp_path / "dist", _parent_identity()
    )
    _append_code_and_rehash(output, source)

    with pytest.raises(HandoffError, match="binding"):
        _verify_published_handoff(output)


def test_l4_handoff_verifier_rejects_wrong_manifest_drive_directory(tmp_path: Path) -> None:
    output = create_l4_handoff(
        _complete_source_repo(tmp_path), tmp_path / "dist", _parent_identity()
    )
    manifest_path = output / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["drive_handoff_directory"] = (
        "/content/drive/MyDrive/pcb-defect-paired/handoff-l4/wrong-runner"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(HandoffError, match="Drive handoff directory"):
        _verify_published_handoff(output)


def test_l4_handoff_rejects_dirty_source_and_removes_owned_staging(tmp_path: Path) -> None:
    repo = _complete_source_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    output_root = tmp_path / "dist"

    with pytest.raises(HandoffError, match="source repository must be clean"):
        create_l4_handoff(repo, output_root, _parent_identity())

    assert list(output_root.iterdir()) == []


def test_l4_handoff_does_not_overwrite_existing_final_directory(tmp_path: Path) -> None:
    repo = _complete_source_repo(tmp_path)
    output_root = tmp_path / "dist"
    output = create_l4_handoff(repo, output_root, _parent_identity())
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(HandoffError, match="refusing to overwrite handoff directory"):
        create_l4_handoff(repo, output_root, _parent_identity())

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert {path.name for path in output_root.iterdir()} == {output.name}


def test_l4_handoff_rejects_late_empty_destination_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _complete_source_repo(tmp_path)
    output_root = tmp_path / "dist"
    original_verify = l4_handoff._verify_l4_handoff
    late_destinations: list[Path] = []

    def inject_late_destination(content: Path, *args: object, **kwargs: object) -> None:
        original_verify(content, *args, **kwargs)
        manifest = json.loads((content / "handoff_manifest.json").read_text(encoding="utf-8"))
        final = output_root / f"colab-handoff-l4-{manifest['snapshot_git_sha'][:12]}"
        final.mkdir()
        late_destinations.append(final)

    monkeypatch.setattr(l4_handoff, "_verify_l4_handoff", inject_late_destination)

    with pytest.raises(HandoffError, match="refusing to overwrite handoff directory"):
        create_l4_handoff(repo, output_root, _parent_identity())

    assert len(late_destinations) == 1
    assert late_destinations[0].is_dir()
    assert list(late_destinations[0].iterdir()) == []
    assert {path.name for path in output_root.iterdir()} == {late_destinations[0].name}


def test_l4_handoff_preserves_late_destination_marker_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _complete_source_repo(tmp_path)
    output_root = tmp_path / "dist"
    marker_bytes = b"pre-existing owner marker\x00\xff"
    original_verify = l4_handoff._verify_l4_handoff
    markers: list[Path] = []

    def inject_late_marker(content: Path, *args: object, **kwargs: object) -> None:
        original_verify(content, *args, **kwargs)
        manifest = json.loads((content / "handoff_manifest.json").read_text(encoding="utf-8"))
        final = output_root / f"colab-handoff-l4-{manifest['snapshot_git_sha'][:12]}"
        final.mkdir()
        marker = final / "owner.marker"
        marker.write_bytes(marker_bytes)
        markers.append(marker)

    monkeypatch.setattr(l4_handoff, "_verify_l4_handoff", inject_late_marker)

    with pytest.raises(HandoffError, match="refusing to overwrite handoff directory"):
        create_l4_handoff(repo, output_root, _parent_identity())

    assert len(markers) == 1
    assert markers[0].read_bytes() == marker_bytes
    assert {path.name for path in output_root.iterdir()} == {markers[0].parent.name}


def test_l4_handoff_treats_late_dangling_symlink_as_occupied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _complete_source_repo(tmp_path)
    output_root = tmp_path / "dist"
    original_verify = l4_handoff._verify_l4_handoff
    links: list[tuple[Path, Path]] = []

    def inject_late_symlink(content: Path, *args: object, **kwargs: object) -> None:
        original_verify(content, *args, **kwargs)
        manifest = json.loads((content / "handoff_manifest.json").read_text(encoding="utf-8"))
        final = output_root / f"colab-handoff-l4-{manifest['snapshot_git_sha'][:12]}"
        missing_target = output_root / "missing-owner-target"
        try:
            final.symlink_to(missing_target, target_is_directory=True)
        except OSError as exc:
            if os.name == "nt" and exc.winerror == 1314:
                pytest.skip("Windows symlink privilege is unavailable")
            raise
        links.append((final, missing_target))

    monkeypatch.setattr(l4_handoff, "_verify_l4_handoff", inject_late_symlink)

    with pytest.raises(HandoffError, match="refusing to overwrite handoff directory"):
        create_l4_handoff(repo, output_root, _parent_identity())

    assert len(links) == 1
    link, target = links[0]
    assert link.is_symlink()
    assert link.readlink() == target
    assert {path.name for path in output_root.iterdir()} == {link.name}


def test_l4_handoff_render_failure_publishes_nothing(tmp_path: Path) -> None:
    repo = _complete_source_repo(tmp_path)
    template = repo / "notebooks" / "deployment_benchmark_l4.ipynb"
    source = template.read_text(encoding="utf-8")
    template.write_text(
        source.replace(
            "PASTE_FINAL_BUNDLE_SHA256",
            "PASTE_FINAL_BUNDLE_SHA256 PASTE_FINAL_BUNDLE_SHA256",
            1,
        ),
        encoding="utf-8",
    )
    _commit(repo, amend=True)
    output_root = tmp_path / "dist"

    with pytest.raises(HandoffError, match="exactly one PASTE_FINAL_BUNDLE_SHA256"):
        create_l4_handoff(repo, output_root, _parent_identity())

    assert list(output_root.iterdir()) == []


def test_l4_handoff_cli_rejects_missing_parent_value_before_bundle_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        l4_handoff,
        "create_clean_bundle",
        lambda *_args, **_kwargs: pytest.fail("bundle creation must not run"),
    )
    arguments = _valid_cli_arguments(tmp_path / "repo", tmp_path / "dist")
    missing_checkpoint = arguments.index("--parent-checkpoint-sha256")
    del arguments[missing_checkpoint : missing_checkpoint + 2]

    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2


def test_l4_handoff_cli_rejects_malformed_parent_before_bundle_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        l4_handoff,
        "create_clean_bundle",
        lambda *_args, **_kwargs: pytest.fail("bundle creation must not run"),
    )
    arguments = _valid_cli_arguments(tmp_path / "repo", tmp_path / "dist")
    arguments[arguments.index("--parent-onnx-sha256") + 1] = "A" * 64

    with pytest.raises(L4ContractError, match="ONNX SHA-256"):
        main(arguments)


def test_l4_handoff_cli_rejects_mixed_stage_flags_before_bundle_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        l4_handoff,
        "create_clean_bundle",
        lambda *_args, **_kwargs: pytest.fail("bundle creation must not run"),
    )
    arguments = _valid_cli_arguments(tmp_path / "repo", tmp_path / "dist")
    arguments.extend(["--probe-parent-git-sha", "a" * 40])

    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2


def test_l4_handoff_cli_prints_only_the_created_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = tmp_path / "dist" / ("colab-handoff-l4-" + "a" * 12)
    captured: list[tuple[Path, Path, L4ParentIdentity]] = []

    def fake_create(repo: Path, output_root: Path, parent: L4ParentIdentity) -> Path:
        captured.append((repo, output_root, parent))
        return expected

    monkeypatch.setattr(l4_handoff, "create_l4_handoff", fake_create)

    assert main(_valid_cli_arguments(tmp_path / "repo", tmp_path / "dist")) == 0
    assert captured == [(tmp_path / "repo", tmp_path / "dist", _parent_identity())]
    assert capsys.readouterr().out == f"l4_handoff_dir={expected}\n"
