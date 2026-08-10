"""Build a deterministic, license-safe source-and-evidence archive for Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT_PREFIX = "pcb-defect-detection"
MANIFEST_NAME = "RESEARCH_PACKAGE_MANIFEST.json"
MAX_FILES = 2_000
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REDISTRIBUTION_BOUNDARY_SUFFIXES = {
    ".bmp",
    ".bundle",
    ".cache",
    ".engine",
    ".gif",
    ".jpeg",
    ".jpg",
    ".log",
    ".onnx",
    ".png",
    ".pt",
    ".tar",
    ".webp",
    ".zip",
}
_SECRET_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
_REQUIRED_TRACKED_FILES = {
    ".zenodo.json",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "configs/paired_protocol.yaml",
    "docs/license-boundary.md",
    "reports/training_recipe.json",
}


class ResearchPackageError(RuntimeError):
    """Raised when the source tree or research archive violates the package contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    package = create_research_package(args.repo.resolve(), args.output.resolve())
    print(f"RESEARCH PACKAGE COMPLETE: {package}")
    return 0


def create_research_package(repo: Path, output: Path) -> Path:
    """Create and verify one deterministic archive from a clean tracked source tree."""
    repo = repo.resolve()
    if not (repo / ".git").exists():
        raise ResearchPackageError("research package source must be a Git worktree")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ResearchPackageError("research package requires a clean tracked worktree")
    source_git_sha = _git(repo, "rev-parse", "HEAD")
    if _GIT_SHA_RE.fullmatch(source_git_sha) is None:
        raise ResearchPackageError("source Git SHA is malformed")
    tracked = _tracked_paths(repo)
    missing = sorted(_REQUIRED_TRACKED_FILES - {path.as_posix() for path in tracked})
    if missing:
        raise ResearchPackageError(f"required research-package files are untracked: {missing}")

    included: list[tuple[str, bytes]] = []
    excluded: list[dict[str, str]] = []
    total_bytes = 0
    for relative in tracked:
        relative_text = relative.as_posix()
        if _is_secret_shaped(relative):
            raise ResearchPackageError(f"secret-shaped tracked path is forbidden: {relative_text}")
        if relative.suffix.casefold() in _REDISTRIBUTION_BOUNDARY_SUFFIXES:
            excluded.append({"path": relative_text, "reason": "redistribution-boundary"})
            continue
        source = repo / relative
        if source.is_symlink() or not source.is_file():
            raise ResearchPackageError(f"tracked path must be a regular file: {relative_text}")
        resolved = source.resolve()
        try:
            resolved.relative_to(repo)
        except ValueError as exc:
            raise ResearchPackageError(
                f"tracked path escapes source root: {relative_text}"
            ) from exc
        payload = source.read_bytes()
        if len(payload) > MAX_MEMBER_BYTES:
            raise ResearchPackageError(
                f"research-package member exceeds size limit: {relative_text}"
            )
        total_bytes += len(payload)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ResearchPackageError("research package exceeds total size limit")
        included.append((relative_text, payload))
    if len(included) > MAX_FILES:
        raise ResearchPackageError("research package exceeds file-count limit")

    rows = [
        {"path": path, "bytes": len(payload), "sha256": _sha256_bytes(payload)}
        for path, payload in included
    ]
    manifest = {
        "schema_version": "1.0",
        "archive_scope": "license-safe-source-and-metadata-evidence",
        "source_git_sha": source_git_sha,
        "root_prefix": ROOT_PREFIX,
        "files": rows,
        "excluded_tracked_files": excluded,
        "redistribution_policy": {
            "dataset_pixels": "excluded",
            "checkpoints": "excluded",
            "onnx_exports": "excluded",
            "tensorrt_engines": "excluded",
            "result_packages": "excluded",
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    entries = {
        **{f"{ROOT_PREFIX}/{path}": payload for path, payload in included},
        f"{ROOT_PREFIX}/{MANIFEST_NAME}": manifest_bytes,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise ResearchPackageError("refusing to overwrite an existing research package")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(entries):
                archive.writestr(_zip_info(name), entries[name])
        verify_research_package(temporary)
        with temporary.open("rb") as source, output.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        temporary.unlink()
        temporary = None
        package_sha256 = _sha256_file(output)
        with sidecar.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(f"{package_sha256}  {output.name}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def verify_research_package(package: Path) -> dict[str, Any]:
    """Fail closed unless every archive member matches the embedded manifest."""
    try:
        with zipfile.ZipFile(package) as archive:
            members = archive.infolist()
            if len(members) > MAX_FILES + 1:
                raise ResearchPackageError("research archive exceeds file-count limit")
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or names != sorted(names):
                raise ResearchPackageError("research archive members must be unique and sorted")
            manifest_name = f"{ROOT_PREFIX}/{MANIFEST_NAME}"
            if names.count(manifest_name) != 1:
                raise ResearchPackageError("research archive manifest is missing or duplicated")
            manifest = json.loads(_read_member(archive, archive.getinfo(manifest_name)).decode())
            _validate_manifest(manifest)
            expected_names = {
                manifest_name,
                *(f"{ROOT_PREFIX}/{row['path']}" for row in manifest["files"]),
            }
            if set(names) != expected_names:
                raise ResearchPackageError("research archive members do not match manifest")
            for row in manifest["files"]:
                name = f"{ROOT_PREFIX}/{row['path']}"
                payload = _read_member(archive, archive.getinfo(name))
                if len(payload) != row["bytes"]:
                    raise ResearchPackageError(f"byte length mismatch: {row['path']}")
                if _sha256_bytes(payload) != row["sha256"]:
                    raise ResearchPackageError(f"SHA-256 mismatch: {row['path']}")
            return manifest
    except ResearchPackageError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise ResearchPackageError("research package archive is invalid") from exc


def _validate_manifest(manifest: object) -> None:
    fields = {
        "schema_version",
        "archive_scope",
        "source_git_sha",
        "root_prefix",
        "files",
        "excluded_tracked_files",
        "redistribution_policy",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ResearchPackageError("research package manifest schema is invalid")
    if (
        manifest["schema_version"] != "1.0"
        or manifest["archive_scope"] != "license-safe-source-and-metadata-evidence"
        or manifest["root_prefix"] != ROOT_PREFIX
        or not isinstance(manifest["source_git_sha"], str)
        or _GIT_SHA_RE.fullmatch(manifest["source_git_sha"]) is None
        or not isinstance(manifest["files"], list)
        or not isinstance(manifest["excluded_tracked_files"], list)
    ):
        raise ResearchPackageError("research package manifest values are invalid")
    paths: list[str] = []
    total_bytes = 0
    for row in manifest["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ResearchPackageError("research package file row is invalid")
        path = row["path"]
        size = row["bytes"]
        digest = row["sha256"]
        if not _safe_relative_path(path):
            raise ResearchPackageError("research package manifest path is unsafe")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_MEMBER_BYTES:
            raise ResearchPackageError("research package manifest size is invalid")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ResearchPackageError("research package manifest SHA-256 is invalid")
        paths.append(path)
        total_bytes += size
    if paths != sorted(paths) or len(paths) != len(set(paths)) or total_bytes > MAX_TOTAL_BYTES:
        raise ResearchPackageError("research package file inventory is invalid")
    for row in manifest["excluded_tracked_files"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "reason"}
            or not _safe_relative_path(row["path"])
            or row["reason"] != "redistribution-boundary"
        ):
            raise ResearchPackageError("research package exclusion row is invalid")
    expected_policy = {
        "dataset_pixels": "excluded",
        "checkpoints": "excluded",
        "onnx_exports": "excluded",
        "tensorrt_engines": "excluded",
        "result_packages": "excluded",
    }
    if manifest["redistribution_policy"] != expected_policy:
        raise ResearchPackageError("research package redistribution policy is invalid")


def _tracked_paths(repo: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    paths = [Path(value.decode("utf-8")) for value in output.split(b"\0") if value]
    return sorted(paths, key=lambda path: path.as_posix())


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise ResearchPackageError("cannot inspect source Git worktree") from exc
    return result.stdout.strip()


def _is_secret_shaped(path: Path) -> bool:
    name = path.name.casefold()
    if name == ".env.example":
        return False
    return (
        name == ".env"
        or name.startswith(".env.")
        or path.suffix.casefold() in _SECRET_SUFFIXES
        or name in {"id_ed25519", "id_rsa"}
    )


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _read_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> bytes:
    if member.is_dir() or member.file_size > MAX_MEMBER_BYTES:
        raise ResearchPackageError("research archive member exceeds size limit")
    with archive.open(member) as handle:
        payload = handle.read(MAX_MEMBER_BYTES + 1)
    if len(payload) != member.file_size or len(payload) > MAX_MEMBER_BYTES:
        raise ResearchPackageError("research archive member byte length mismatch")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
