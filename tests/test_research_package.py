from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from pcb_defect.research_package import (
    ResearchPackageError,
    create_research_package,
    verify_research_package,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _research_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    files = {
        "README.md": "# research package\n",
        "LICENSE": "AGPL fixture\n",
        "CITATION.cff": "cff-version: 1.2.0\ntitle: fixture\n",
        ".zenodo.json": '{"title":"fixture"}\n',
        ".env.example": "OPTIONAL_VALUE=\n",
        "configs/paired_protocol.yaml": "schema: fixture\n",
        "docs/license-boundary.md": "No redistributed data or model binaries.\n",
        "reports/training_recipe.json": '{"schema_version":"1.0"}\n',
        "reports/evidence.json": '{"value":1}\n',
        "src/pcb_defect/example.py": "VALUE = 1\n",
        "tests/test_example.py": "def test_example():\n    assert True\n",
        "tests/fixtures/excluded.jpg": "not-real-pixels",
    }
    for relative, contents in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "kuotunyu")
    _git(repo, "config", "user.email", "61350295+kuotunyu@users.noreply.github.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_research_package_is_deterministic_verified_and_license_safe(tmp_path: Path) -> None:
    repo = _research_repo(tmp_path)
    first = create_research_package(repo, tmp_path / "first.zip")
    second = create_research_package(repo, tmp_path / "second.zip")

    assert first.read_bytes() == second.read_bytes()
    assert first.with_suffix(".zip.sha256").read_text(encoding="ascii").split()[0] == (
        hashlib.sha256(first.read_bytes()).hexdigest()
    )
    manifest = verify_research_package(first)
    assert manifest["schema_version"] == "1.0"
    assert manifest["source_git_sha"] == _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert manifest["excluded_tracked_files"] == [
        {"path": "tests/fixtures/excluded.jpg", "reason": "redistribution-boundary"}
    ]
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "pcb-defect-detection/RESEARCH_PACKAGE_MANIFEST.json" in names
        assert "pcb-defect-detection/.env.example" in names
        assert "pcb-defect-detection/tests/fixtures/excluded.jpg" not in names
        for row in manifest["files"]:
            payload = archive.read(f"pcb-defect-detection/{row['path']}")
            assert len(payload) == row["bytes"]
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]


def test_research_package_rejects_dirty_tracked_source(tmp_path: Path) -> None:
    repo = _research_repo(tmp_path)
    (repo / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ResearchPackageError, match="clean tracked worktree"):
        create_research_package(repo, tmp_path / "research.zip")


@pytest.mark.parametrize("secret_name", [".env", ".env.production", "private-key.pem"])
def test_research_package_rejects_tracked_secret_shaped_paths(
    tmp_path: Path, secret_name: str
) -> None:
    repo = _research_repo(tmp_path)
    secret = repo / secret_name
    secret.write_text("placeholder\n", encoding="utf-8")
    _git(repo, "add", secret_name)
    _git(repo, "commit", "-m", "unsafe fixture")

    with pytest.raises(ResearchPackageError, match="secret-shaped tracked path"):
        create_research_package(repo, tmp_path / "research.zip")


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/model.pth",
        "artifacts/model.ckpt",
        "artifacts/model.safetensors",
        "artifacts/model.bin",
        "artifacts/array.npy",
        "artifacts/image.tif",
        "artifacts/engine.plan",
    ],
)
def test_research_package_excludes_every_unapproved_binary_suffix(
    tmp_path: Path, relative: str
) -> None:
    repo = _research_repo(tmp_path)
    artifact = repo / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"\x00\xffprivate-binary")
    _git(repo, "add", relative)
    _git(repo, "commit", "-m", "binary fixture")

    package = create_research_package(repo, tmp_path / "research.zip")
    manifest = verify_research_package(package)
    assert {"path": relative, "reason": "redistribution-boundary"} in manifest[
        "excluded_tracked_files"
    ]
    with zipfile.ZipFile(package) as archive:
        assert f"pcb-defect-detection/{relative}" not in archive.namelist()


def test_research_package_rejects_non_utf8_payload_with_approved_text_suffix(
    tmp_path: Path,
) -> None:
    repo = _research_repo(tmp_path)
    poison = repo / "reports" / "poison.json"
    poison.write_bytes(b"\xff\xfe\x00\x01")
    _git(repo, "add", "reports/poison.json")
    _git(repo, "commit", "-m", "invalid text fixture")

    with pytest.raises(ResearchPackageError, match="UTF-8 text metadata"):
        create_research_package(repo, tmp_path / "research.zip")


def test_research_package_verifier_rejects_member_tampering(tmp_path: Path) -> None:
    repo = _research_repo(tmp_path)
    package = create_research_package(repo, tmp_path / "research.zip")
    with zipfile.ZipFile(package) as source:
        members = {info.filename: source.read(info) for info in source.infolist()}
    members["pcb-defect-detection/README.md"] = b"# tampered package\n"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as output:
        for name, payload in members.items():
            output.writestr(name, payload)

    with pytest.raises(ResearchPackageError, match="SHA-256 mismatch"):
        verify_research_package(package)


@pytest.mark.parametrize(
    ("relative", "payload"),
    [
        ("artifacts/model.safetensors", b"\x00\xffprivate-binary"),
        ("reports/poison.json", b"\xff\xfe\x00\x01"),
    ],
)
def test_research_package_verifier_rejects_self_consistent_non_text_member(
    tmp_path: Path, relative: str, payload: bytes
) -> None:
    repo = _research_repo(tmp_path)
    package = create_research_package(repo, tmp_path / "research.zip")
    manifest_name = "pcb-defect-detection/RESEARCH_PACKAGE_MANIFEST.json"
    readme_name = "pcb-defect-detection/README.md"
    with zipfile.ZipFile(package) as source:
        members = {info.filename: source.read(info) for info in source.infolist()}
    manifest = json.loads(members[manifest_name])
    manifest["files"] = [row for row in manifest["files"] if row["path"] != "README.md"]
    manifest["files"].append(
        {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    )
    manifest["files"].sort(key=lambda row: row["path"])
    del members[readme_name]
    members[f"pcb-defect-detection/{relative}"] = payload
    members[manifest_name] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as output:
        for name in sorted(members):
            output.writestr(name, members[name])

    with pytest.raises(ResearchPackageError, match="UTF-8 text metadata"):
        verify_research_package(package)
