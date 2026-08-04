"""Build one deterministic, content-addressed Colab result package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from pcb_defect.experiment import ARMS, InputLock, run_is_complete
from pcb_defect.final_evaluation import final_evaluation_is_complete


class PackageError(RuntimeError):
    """The evidence package is incomplete, unsafe, or would overwrite an existing file."""


# Private L4 evidence has ten small metadata/log files plus model artifacts. 256 MiB per
# member and 1 GiB per package leave ample room for those artifacts while bounding ZIP abuse.
MAX_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 1 * 1024 * 1024 * 1024
# Ten L4 evidence members are expected; 4,096 permits future audited metadata while
# bounding central-directory parsing and preventing zero-byte-member exhaustion.
MAX_ARCHIVE_MEMBERS = 4_096
MAX_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
_IO_CHUNK_BYTES = 1024 * 1024
_MANIFEST_NAME = "package_manifest.json"
_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
_EOCD = struct.Struct("<4s4H2LH")
_CENTRAL_DIRECTORY = struct.Struct("<4s6H3L5H2L")


def create_verifiable_zip(
    root: Path, relative_files: list[Path], destination: Path
) -> dict[str, Any]:
    """Write deterministic ZIP bytes, an embedded file manifest, and an external package hash."""
    root = root.resolve()
    destination = _logical_leaf_path(destination)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    _reject_leaf_symlinks(destination, sidecar)
    if any(_path_exists(path) for path in (destination, sidecar)):
        raise PackageError(f"refusing to overwrite an existing result package: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    embedded, resolved_files = _package_manifest(root, relative_files)
    temporary = _unique_staging_path(destination)
    sidecar_temporary = _unique_staging_path(sidecar)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for relative_text, resolved in resolved_files:
                _write_deterministic_file(archive, relative_text, resolved)
            _write_deterministic_bytes(
                archive,
                _MANIFEST_NAME,
                (json.dumps(embedded, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        package_hash = _sha256_file(temporary)
        sidecar_temporary.write_bytes(f"{package_hash}  {destination.name}\n".encode("ascii"))
        _publish_no_replace(temporary, destination)
        _publish_no_replace(sidecar_temporary, sidecar)
    except FileExistsError as exc:
        raise PackageError(
            f"refusing to overwrite an existing result package: {destination}"
        ) from exc
    finally:
        _remove_own_staging(temporary)
        _remove_own_staging(sidecar_temporary)
    return {**embedded, "package_sha256": package_hash}


def verify_verifiable_zip(package: Path) -> dict[str, Any]:
    """Fail closed unless a ZIP, sidecar, manifest, and every member agree."""
    package = _logical_leaf_path(package)
    sidecar = package.with_suffix(package.suffix + ".sha256")
    _reject_leaf_symlinks(package, sidecar)
    if package.exists() != sidecar.exists():
        raise PackageError("result package and SHA-256 sidecar must exist together")
    if not package.is_file():
        raise PackageError("result package is missing")
    package_sha256 = _sha256_file(package)
    expected_sidecar = f"{package_sha256}  {package.name}\n".encode("ascii")
    try:
        sidecar_contents = sidecar.read_bytes()
    except OSError as exc:
        raise PackageError("result package sidecar bytes, name, or hash are invalid") from exc
    if sidecar_contents != expected_sidecar:
        raise PackageError("result package sidecar bytes, name, or hash are invalid")

    try:
        declared_members = _preflight_zip_member_count(package)
        with zipfile.ZipFile(package) as archive:
            members = archive.infolist()
            if len(members) != declared_members or len(members) > MAX_ARCHIVE_MEMBERS:
                raise PackageError("archive member count does not match bounded central directory")
            _verify_zip_resource_bounds(members)
            manifest_members = [member for member in members if member.filename == _MANIFEST_NAME]
            if len(manifest_members) != 1:
                raise PackageError("package manifest is invalid")
            try:
                manifest = json.loads(
                    _read_zip_member(archive, manifest_members[0], MAX_MANIFEST_BYTES).decode(
                        "utf-8"
                    )
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise PackageError("package manifest is invalid") from exc
            files = _validated_package_manifest(manifest)
            actual_members = [member for member in members if member.filename != _MANIFEST_NAME]
            actual_names = [member.filename for member in actual_members]
            expected_names = [row["path"] for row in files]
            if (
                len({_canonical_path_key(name) for name in actual_names}) != len(actual_names)
                or len({_canonical_path_key(name) for name in expected_names})
                != len(expected_names)
                or {_canonical_path_key(name) for name in actual_names}
                != {_canonical_path_key(name) for name in expected_names}
            ):
                raise PackageError("archive members do not match package manifest")
            actual_by_key = {
                _canonical_path_key(member.filename): member for member in actual_members
            }
            for member in actual_members:
                if not _is_portable_package_path(member.filename):
                    raise PackageError("archive members do not match package manifest")
            for row in files:
                member = actual_by_key[_canonical_path_key(row["path"])]
                if member.file_size != row["bytes"]:
                    raise PackageError("archive member byte length mismatch")
                if _sha256_zip_member(archive, member) != row["sha256"]:
                    raise PackageError("archive member SHA-256 mismatch")
    except PackageError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise PackageError("result package archive is invalid") from exc
    return {**manifest, "package_sha256": package_sha256}


def _validated_package_manifest(manifest: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "files"}
        or manifest["schema_version"] != "1.0"
        or not isinstance(manifest["files"], list)
    ):
        raise PackageError("package manifest is invalid")
    files = manifest["files"]
    keys = set()
    for row in files:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "bytes"}
            or not isinstance(row["path"], str)
            or not _is_portable_package_path(row["path"])
            or not isinstance(row["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or _canonical_path_key(row["path"]) == _canonical_path_key(_MANIFEST_NAME)
        ):
            raise PackageError("package manifest is invalid")
        key = _canonical_path_key(row["path"])
        if key in keys:
            raise PackageError("package manifest is invalid")
        keys.add(key)
    return files


def _is_portable_package_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and "\\" not in value
        and ":" not in value
        and not path.is_absolute()
        and path.as_posix() == value
        and "." not in path.parts
        and ".." not in path.parts
        and all(not _is_windows_reserved_component(part) for part in path.parts)
        and all(not part.endswith((".", " ")) for part in path.parts)
        and all(not unicodedata.category(char).startswith("C") for char in value)
    )


def _is_windows_reserved_component(component: str) -> bool:
    base = component.split(".", 1)[0].casefold()
    return base in {"con", "prn", "aux", "nul", "clock$"} or (
        len(base) == 4 and base[:3] in {"com", "lpt"} and base[3] in "123456789"
    )


def _canonical_path_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _verify_zip_resource_bounds(members: list[zipfile.ZipInfo]) -> None:
    total = 0
    for member in members:
        if member.extract_version >= 45 or _contains_zip64_extra(member.extra):
            raise PackageError("ZIP64 members are not supported")
        if member.compress_type != zipfile.ZIP_STORED or member.compress_size != member.file_size:
            raise PackageError("ZIP member compression must be stored")
        mode = member.external_attr >> 16
        if stat.S_IFMT(mode) != stat.S_IFREG:
            raise PackageError("ZIP members must be regular files")
        if member.file_size > MAX_MEMBER_BYTES:
            raise PackageError("archive member exceeds size limit")
        total += member.file_size
        if total > MAX_TOTAL_BYTES:
            raise PackageError("archive exceeds total size limit")


def _preflight_zip_member_count(package: Path) -> int:
    try:
        file_size = package.stat().st_size
        if file_size < _EOCD.size:
            raise PackageError("ZIP end record is missing or truncated")
        tail_size = min(file_size, _EOCD.size + 0xFFFF)
        with package.open("rb") as handle:
            handle.seek(file_size - tail_size)
            tail = handle.read(tail_size)
            candidates = _eocd_candidates(handle, tail, file_size - tail_size, file_size)
    except OSError as exc:
        raise PackageError("result package archive is invalid") from exc
    if len(candidates) != 1:
        raise PackageError("ZIP end record is missing, ambiguous, or truncated")
    member_count = candidates[0]
    if member_count > MAX_ARCHIVE_MEMBERS:
        raise PackageError("archive exceeds member count limit before metadata load")
    return member_count


def _eocd_candidates(handle: Any, tail: bytes, tail_start: int, file_size: int) -> list[int]:
    candidates = []
    offset = tail.find(_EOCD_SIGNATURE)
    while offset >= 0:
        if offset + _EOCD.size <= len(tail):
            record = _EOCD.unpack_from(tail, offset)
            (
                _,
                disk,
                central_disk,
                disk_count,
                total_count,
                central_size,
                central_offset,
                comment,
            ) = record
            absolute = tail_start + offset
            if absolute + _EOCD.size + comment == file_size:
                if (
                    total_count == 0xFFFF
                    or central_size == 0xFFFFFFFF
                    or central_offset == 0xFFFFFFFF
                    or _has_zip64_locator(handle, absolute)
                ):
                    raise PackageError("ZIP64 archives are not supported")
                if total_count > MAX_ARCHIVE_MEMBERS:
                    raise PackageError("archive exceeds member count limit before metadata load")
                if (
                    disk != 0
                    or central_disk != 0
                    or disk_count != total_count
                    or central_size > MAX_CENTRAL_DIRECTORY_BYTES
                    or central_offset + central_size != absolute
                    or not _central_directory_matches(
                        handle, central_offset, central_size, total_count
                    )
                ):
                    raise PackageError("ZIP end record is missing, ambiguous, or truncated")
                candidates.append(total_count)
        offset = tail.find(_EOCD_SIGNATURE, offset + 1)
    return candidates


def _central_directory_matches(handle: Any, offset: int, size: int, declared_count: int) -> bool:
    if declared_count == 0:
        return size == 0
    try:
        handle.seek(offset)
        consumed = 0
        for _ in range(declared_count):
            header = handle.read(_CENTRAL_DIRECTORY.size)
            if len(header) != _CENTRAL_DIRECTORY.size:
                return False
            fields = _CENTRAL_DIRECTORY.unpack(header)
            if fields[0] != _CENTRAL_DIRECTORY_SIGNATURE:
                return False
            name_length, extra_length, comment_length = fields[10:13]
            entry_size = _CENTRAL_DIRECTORY.size + name_length + extra_length + comment_length
            consumed += entry_size
            variable = handle.read(name_length + extra_length + comment_length)
            if consumed > size or len(variable) != name_length + extra_length + comment_length:
                return False
            extra = variable[name_length : name_length + extra_length]
            if fields[2] >= 45 or _contains_zip64_extra(extra):
                raise PackageError("ZIP64 members are not supported")
        return consumed == size
    except OSError:
        return False


def _has_zip64_locator(handle: Any, eocd_offset: int) -> bool:
    if eocd_offset < 20:
        return False
    handle.seek(eocd_offset - 20)
    return handle.read(4) == _ZIP64_LOCATOR_SIGNATURE


def _contains_zip64_extra(extra: bytes) -> bool:
    offset = 0
    while offset < len(extra):
        if len(extra) - offset < 4:
            raise PackageError("ZIP extra field is malformed")
        header, size = struct.unpack_from("<HH", extra, offset)
        offset += 4
        if size > len(extra) - offset:
            raise PackageError("ZIP extra field is malformed")
        if header == 0x0001:
            return True
        offset += size
    return False


def _read_zip_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, limit: int) -> bytes:
    if member.file_size > limit:
        raise PackageError("package manifest exceeds size limit")
    chunks = []
    size = 0
    with archive.open(member) as stream:
        for chunk in iter(lambda: stream.read(_IO_CHUNK_BYTES), b""):
            size += len(chunk)
            if size > limit:
                raise PackageError("package manifest exceeds size limit")
            chunks.append(chunk)
    if size != member.file_size:
        raise PackageError("package manifest is invalid")
    return b"".join(chunks)


def _sha256_zip_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member) as stream:
        for chunk in iter(lambda: stream.read(_IO_CHUNK_BYTES), b""):
            size += len(chunk)
            if size > member.file_size:
                raise PackageError("archive member byte length mismatch")
            digest.update(chunk)
    if size != member.file_size:
        raise PackageError("archive member byte length mismatch")
    return digest.hexdigest()


def _package_manifest(
    root: Path, relative_files: list[Path]
) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    normalized: dict[str, Path] = {}
    canonical_keys = set()
    for relative_path in relative_files:
        relative_text = relative_path.as_posix()
        if not _is_portable_package_path(relative_text):
            raise PackageError(f"package path must be portable and relative: {relative_path}")
        key = _canonical_path_key(relative_text)
        if key in canonical_keys:
            raise PackageError(f"package paths collide after normalization: {relative_path}")
        canonical_keys.add(key)
        normalized[relative_text] = relative_path
    files = []
    resolved_files = []
    for relative_text, relative_path in sorted(normalized.items()):
        resolved = (root / relative_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PackageError(f"package path escapes root: {relative_path}") from exc
        if not resolved.is_file():
            raise PackageError(f"required package file is missing: {relative_path}")
        size = resolved.stat().st_size
        if size > MAX_MEMBER_BYTES:
            raise PackageError(f"package member exceeds size limit: {relative_path}")
        files.append({"path": relative_text, "sha256": _sha256_file(resolved), "bytes": size})
        resolved_files.append((relative_text, resolved))
    if sum(row["bytes"] for row in files) > MAX_TOTAL_BYTES:
        raise PackageError("package exceeds total size limit")
    return {"schema_version": "1.0", "files": files}, resolved_files


def _unique_staging_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _publish_no_replace(source: Path, destination: Path) -> None:
    os.link(source, destination)


def _remove_own_staging(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _logical_leaf_path(path: Path) -> Path:
    return path.parent.resolve() / path.name


def _reject_leaf_symlinks(*paths: Path) -> None:
    for path in paths:
        try:
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise PackageError("result package and sidecar leaf paths must not be symlinks")
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    files = _collect_result_files(workspace)
    manifest = create_verifiable_zip(workspace, files, args.output)
    print(f"RESULT PACKAGE: {args.output.resolve()}")
    print(f"sha256={manifest['package_sha256']}")
    return 0


def _collect_result_files(workspace: Path) -> list[Path]:
    workspace = workspace.resolve()
    deployment_gate = workspace / "deployment" / "deployment_gate.json"
    if not final_evaluation_is_complete(workspace / "final"):
        raise PackageError("final evaluation is incomplete or hash-mismatched")
    if not deployment_gate.is_file():
        raise PackageError("deployment gate is missing")
    try:
        lock = InputLock(**json.loads((workspace / "inputs" / "input_lock.json").read_text()))
        gate = json.loads(deployment_gate.read_text(encoding="utf-8"))
        gate_report = json.loads((workspace / "gates" / "gate_report.json").read_text())
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise PackageError("input lock, GPU gate, or deployment gate is malformed") from exc
    if gate_report.get("passed") is not True or gate_report.get("input_lock") != lock.as_dict():
        raise PackageError("GPU gates are incomplete or do not match the input lock")
    if gate.get("passed") is not True:
        raise PackageError("deployment gate is incomplete or failed")
    try:
        onnx = workspace / "deployment" / "best.onnx"
        if not onnx.is_file() or _sha256_file(onnx) != gate["artifacts"]["onnx_sha256"]:
            raise PackageError("deployment ONNX is missing or hash-mismatched")
        source_weights = (workspace / gate["artifacts"]["source_checkpoint"]).resolve()
        source_weights.relative_to(workspace)
        if _sha256_file(source_weights) != gate["artifacts"]["source_checkpoint_sha256"]:
            raise PackageError("deployment source checkpoint is hash-mismatched")
        contract = json.loads(
            (workspace / "deployment" / "model_contract.candidate.json").read_text()
        )
        if (
            contract.get("status") != "passed"
            or contract.get("onnx_sha256") != gate["artifacts"]["onnx_sha256"]
            or contract.get("source_checkpoint_sha256")
            != gate["artifacts"]["source_checkpoint_sha256"]
            or contract.get("deployment_gate_sha256") != _sha256_file(deployment_gate)
        ):
            raise PackageError("deployment model contract does not match the gate")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PackageError("deployment source or model contract is malformed") from exc
    files = [
        Path("inputs/input_lock.json"),
        Path("gates/gate_report.json"),
        Path("final/FINAL_TEST_STARTED.json"),
        Path("final/deployment_selection.json"),
        Path("final/final_metrics.json"),
        Path("final/finalization_record.json"),
        Path("deployment/calibration.yaml"),
        Path("deployment/deployment_gate.json"),
        Path("deployment/model_contract.candidate.json"),
        Path("deployment/best.onnx"),
    ]
    for arm in ARMS:
        for seed in (42, 43, 44):
            prefix = Path("runs") / arm / f"seed{seed}"
            if not run_is_complete(workspace / prefix, lock):
                raise PackageError(f"run is incomplete or hash-mismatched: {arm} seed={seed}")
            files.extend(
                [
                    prefix / "run_record.json",
                    prefix / "inputs/train_paired.yaml",
                    prefix / "inputs/base_model.yaml",
                    prefix / "inputs/paired_split_manifest.json",
                    prefix / "metrics/validation.json",
                    prefix / "weights/best.pt",
                    prefix / "weights/last.pt",
                ]
            )
    return files


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    return info


def _write_deterministic_file(archive: zipfile.ZipFile, name: str, source: Path) -> None:
    with source.open("rb") as input_handle, archive.open(_zip_info(name), "w") as output_handle:
        for chunk in iter(lambda: input_handle.read(_IO_CHUNK_BYTES), b""):
            output_handle.write(chunk)


def _write_deterministic_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    with archive.open(_zip_info(name), "w") as output_handle:
        output_handle.write(data)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
