from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pcb_defect.handoff import (
    HandoffError,
    create_clean_bundle,
    main,
    project_handoff_metadata,
    render_notebook,
)

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


def _minimal_handoff_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "reports" / "protocol").mkdir(parents=True)
    (repo / "configs").mkdir()
    for name in (
        "paired_experiment_a100.ipynb",
        "deployment_benchmark_l4.ipynb",
        "deployment_parity_probe_a100.ipynb",
    ):
        shutil.copyfile(ROOT / "notebooks" / name, repo / "notebooks" / name)
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
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=Release",
        "-c",
        "user.email=release@example.invalid",
        "commit",
        "-m",
        "source",
    )
    return repo


def test_handoff_bundle_contains_one_clean_snapshot_commit(tmp_path: Path) -> None:
    source = tmp_path / "資料" / "source"
    source.mkdir(parents=True)
    _git(source, "init", "-b", "main")
    (source / "legacy.jpg").write_bytes(b"legacy pixel")
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Legacy",
        "-c",
        "user.email=legacy@example.invalid",
        "commit",
        "-m",
        "legacy",
    )
    (source / "legacy.jpg").unlink()
    (source / "current.py").write_text("print('safe')\n", encoding="utf-8")
    _git(source, "add", "-A")
    _git(
        source,
        "-c",
        "user.name=Release",
        "-c",
        "user.email=release@example.invalid",
        "commit",
        "-m",
        "current",
    )
    source_sha = _git(source, "rev-parse", "HEAD")

    result = create_clean_bundle(source, tmp_path / "資料" / "handoff", {"purpose": "test"})

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(result["bundle_path"]), str(clone)], check=True)
    assert int(_git(clone, "rev-list", "--count", "HEAD")) == 1
    assert not (clone / "legacy.jpg").exists()
    assert (clone / "current.py").is_file()
    provenance = json.loads((clone / ".source_provenance.json").read_text(encoding="utf-8"))
    assert provenance == {"purpose": "test", "source_git_sha": source_sha}
    assert result["source_git_sha"] == source_sha
    assert len(result["snapshot_git_sha"]) == 40
    assert len(result["bundle_sha256"]) == 64


def test_rendered_handoff_notebook_contains_exact_immutable_values(tmp_path: Path) -> None:
    template = tmp_path / "template.ipynb"
    destination = tmp_path / "ready.ipynb"
    template.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "SOURCE_BUNDLE_SHA256 = 'PASTE_FINAL_BUNDLE_SHA256'\n",
                            "EXPECTED_GIT_SHA = 'PASTE_FINAL_GIT_SHA'\n",
                        ],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    digest = render_notebook(
        template,
        destination,
        replacements={
            "PASTE_FINAL_BUNDLE_SHA256": "a" * 64,
            "PASTE_FINAL_GIT_SHA": "b" * 40,
        },
    )

    rendered = destination.read_text(encoding="utf-8")
    assert "PASTE_FINAL" not in rendered
    assert "a" * 64 in rendered
    assert "b" * 40 in rendered
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert json.loads(rendered)["cells"][0]["outputs"] == []


def test_renderer_replaces_each_of_five_immutable_placeholders_once(tmp_path: Path) -> None:
    template = tmp_path / "probe.ipynb"
    destination = tmp_path / "ready-probe.ipynb"
    placeholders = {
        "PASTE_FINAL_BUNDLE_SHA256": "a" * 64,
        "PASTE_FINAL_GIT_SHA": "b" * 40,
        "PASTE_PARENT_EXPERIMENT_GIT_SHA": "c" * 40,
        "PASTE_PARENT_DEPLOYMENT_GATE_SHA256": "d" * 64,
        "PASTE_PARENT_ONNX_SHA256": "e" * 64,
    }
    template.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [f"VALUE = '{key}'\\n" for key in placeholders],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    render_notebook(template, destination, replacements=placeholders)

    rendered = destination.read_text(encoding="utf-8")
    for placeholder, replacement in placeholders.items():
        assert placeholder not in rendered
        assert rendered.count(replacement) == 1


@pytest.mark.parametrize(
    ("relative_template", "replacements"),
    [
        (
            "notebooks/paired_experiment_a100.ipynb",
            {
                "PASTE_FINAL_BUNDLE_SHA256": "a" * 64,
                "PASTE_FINAL_GIT_SHA": "b" * 40,
            },
        ),
        (
            "notebooks/deployment_benchmark_l4.ipynb",
            {
                "PASTE_FINAL_BUNDLE_SHA256": "a" * 64,
                "PASTE_FINAL_GIT_SHA": "b" * 40,
                "PASTE_PARENT_EXPERIMENT_GIT_SHA": "c" * 40,
                "PASTE_PARENT_DEPLOYMENT_GATE_SHA256": "d" * 64,
                "PASTE_PARENT_CHECKPOINT_SHA256": "e" * 64,
                "PASTE_PARENT_ONNX_SHA256": "f" * 64,
                "PASTE_L4_HANDOFF_DIRECTORY": "/content/drive/MyDrive/test",
            },
        ),
        (
            "notebooks/deployment_parity_probe_a100.ipynb",
            {
                "PASTE_FINAL_BUNDLE_SHA256": "a" * 64,
                "PASTE_FINAL_GIT_SHA": "b" * 40,
                "PASTE_PARENT_EXPERIMENT_GIT_SHA": "c" * 40,
                "PASTE_PARENT_DEPLOYMENT_GATE_SHA256": "d" * 64,
                "PASTE_PARENT_ONNX_SHA256": "e" * 64,
            },
        ),
    ],
)
def test_rendered_handoff_templates_contain_no_placeholder_sentinel(
    tmp_path: Path, relative_template: str, replacements: dict[str, str]
) -> None:
    """Rendered handoff notebooks must be JSON-valid runnable cells with no PASTE_ sentinel."""
    destination = tmp_path / Path(relative_template).name

    render_notebook(ROOT / relative_template, destination, replacements)

    rendered = destination.read_text(encoding="utf-8")
    notebook = json.loads(rendered)
    assert "PASTE_" not in rendered
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"{relative_template}:{cell}", "exec")


def test_handoff_rejects_partial_probe_parent_arguments(tmp_path: Path) -> None:
    with pytest.raises(HandoffError, match="probe arguments must be supplied together"):
        main(
            [
                "--repo",
                str(tmp_path / "repo"),
                "--output",
                str(tmp_path / "handoff"),
                "--probe-parent-git-sha",
                "c" * 40,
            ]
        )


def test_handoff_rejects_empty_probe_hashes_when_all_flags_are_present(tmp_path: Path) -> None:
    with pytest.raises(
        HandoffError, match="probe parent Git SHA must be 40 lowercase hexadecimal characters"
    ):
        main(
            [
                "--repo",
                str(tmp_path / "repo"),
                "--output",
                str(tmp_path / "handoff"),
                "--probe-parent-git-sha",
                "",
                "--probe-parent-deployment-gate-sha256",
                "",
                "--probe-parent-onnx-sha256",
                "",
            ]
        )


def test_project_handoff_metadata_exposes_shared_protocol_and_model_identity(
    tmp_path: Path,
) -> None:
    repo = _minimal_handoff_repo(tmp_path)

    metadata = project_handoff_metadata(repo)

    assert metadata == {
        "protocol_version": "1.0",
        "dataset_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "base_model_contract_sha256": hashlib.sha256(
            (repo / "configs" / "base_model.yaml").read_bytes()
        ).hexdigest(),
        "base_model_source": "model.pt",
        "base_model_revision": "v1",
        "base_model_sha256": "f" * 64,
    }


def test_existing_handoff_cli_preserves_paired_outputs(tmp_path: Path) -> None:
    repo = _minimal_handoff_repo(tmp_path)
    output = tmp_path / "paired-handoff"

    assert main(["--repo", str(repo), "--output", str(output)]) == 0

    assert {path.name for path in output.iterdir()} == {
        "handoff_manifest.json",
        "paired_experiment_a100.ipynb",
        "deployment_benchmark_l4.ipynb",
        "pcb-defect-source.bundle",
    }


def test_existing_handoff_cli_emits_sentinel_free_fail_closed_l4_migration(
    tmp_path: Path,
) -> None:
    repo = _minimal_handoff_repo(tmp_path)
    output = tmp_path / "paired-handoff"

    assert main(["--repo", str(repo), "--output", str(output)]) == 0

    rendered = (output / "deployment_benchmark_l4.ipynb").read_text(encoding="utf-8")
    notebook = json.loads(rendered)
    assert "PASTE_" not in rendered
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), "legacy-l4-migration", "exec")
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    with pytest.raises(RuntimeError, match=r"python -m pcb_defect\.l4_handoff"):
        exec(compile("".join(first_code["source"]), "legacy-l4-migration", "exec"), {})


def test_existing_handoff_cli_adds_probe_only_with_complete_probe_arguments(
    tmp_path: Path,
) -> None:
    repo = _minimal_handoff_repo(tmp_path)
    output = tmp_path / "paired-probe-handoff"

    assert (
        main(
            [
                "--repo",
                str(repo),
                "--output",
                str(output),
                "--probe-parent-git-sha",
                "a" * 40,
                "--probe-parent-deployment-gate-sha256",
                "b" * 64,
                "--probe-parent-onnx-sha256",
                "c" * 64,
            ]
        )
        == 0
    )

    assert {path.name for path in output.iterdir()} == {
        "handoff_manifest.json",
        "paired_experiment_a100.ipynb",
        "deployment_benchmark_l4.ipynb",
        "deployment_parity_probe_a100.ipynb",
        "pcb-defect-source.bundle",
    }
