from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import warnings
import zipfile
from pathlib import Path

import pytest

import pcb_defect.result_package as result_package
from pcb_defect.experiment import InputLock
from pcb_defect.result_package import (
    PackageError,
    _collect_result_files,
    create_verifiable_zip,
    verify_verifiable_zip,
)


def test_result_package_contains_manifest_and_hash_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "final").mkdir(parents=True)
    (root / "final" / "metrics.json").write_text("{}\n", encoding="utf-8")
    (root / "input_lock.json").write_text('{"locked": true}\n', encoding="utf-8")
    destination = tmp_path / "package" / "results.zip"

    manifest = create_verifiable_zip(
        root,
        [Path("final/metrics.json"), Path("input_lock.json")],
        destination,
    )

    assert manifest["schema_version"] == "1.0"
    assert [row["path"] for row in manifest["files"]] == [
        "final/metrics.json",
        "input_lock.json",
    ]
    assert len(manifest["package_sha256"]) == 64
    assert destination.with_suffix(".zip.sha256").read_text(encoding="ascii") == (
        f"{manifest['package_sha256']}  results.zip\n"
    )
    assert not destination.with_name(".results.zip.tmp").exists()
    assert not destination.with_name(".results.zip.sha256.tmp").exists()
    with zipfile.ZipFile(destination) as archive:
        embedded = json.loads(archive.read("package_manifest.json"))
        assert embedded["files"] == manifest["files"]


def test_verify_verifiable_zip_rejects_missing_or_mutated_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    package = tmp_path / "result.zip"
    create_verifiable_zip(root, [Path("report.json")], package)
    sidecar = package.with_suffix(".zip.sha256")
    sidecar.write_text(f"{'0' * 64}  {package.name}\n", encoding="ascii")

    with pytest.raises(PackageError, match="sidecar bytes, name, or hash are invalid"):
        verify_verifiable_zip(package)


@pytest.mark.parametrize("suffix", [b"\r\n", b"\n\n", b"\x80"])
def test_verify_verifiable_zip_requires_byte_exact_ascii_sidecar(
    tmp_path: Path, suffix: bytes
) -> None:
    package = _valid_package(tmp_path)
    sidecar = package.with_suffix(".zip.sha256")
    sidecar.write_bytes(sidecar.read_bytes().rstrip(b"\n") + suffix)

    with pytest.raises(PackageError, match="sidecar bytes, name, or hash are invalid"):
        verify_verifiable_zip(package)


@pytest.mark.parametrize("member_name", ["AUX.txt", "report. ", "bad\x01name.txt"])
def test_verify_verifiable_zip_rejects_nonportable_windows_member_names(
    tmp_path: Path, member_name: str
) -> None:
    package = _rewritten_single_member_package(tmp_path, member_name)

    with pytest.raises(PackageError, match="package manifest is invalid"):
        verify_verifiable_zip(package)


def test_verify_verifiable_zip_rejects_unicode_and_casefold_member_collisions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "collision.zip"
    members = {"café.txt": b"one", "cafe\u0301.txt": b"two"}
    _write_package(package, members)

    with pytest.raises(PackageError, match="package manifest is invalid"):
        verify_verifiable_zip(package)


def test_verify_verifiable_zip_rejects_compressed_or_non_regular_members(tmp_path: Path) -> None:
    compressed = _valid_package(tmp_path / "compressed")
    _rewrite_zip(compressed, compression=zipfile.ZIP_DEFLATED)
    symlink = _valid_package(tmp_path / "symlink")
    _rewrite_zip(symlink, external_attr=stat.S_IFLNK | 0o777)

    with pytest.raises(PackageError, match="compression"):
        verify_verifiable_zip(compressed)
    with pytest.raises(PackageError, match="regular"):
        verify_verifiable_zip(symlink)


def test_verify_verifiable_zip_rejects_member_and_total_size_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _valid_package(tmp_path)
    monkeypatch.setattr(result_package, "MAX_MEMBER_BYTES", 1)

    with pytest.raises(PackageError, match="size limit"):
        verify_verifiable_zip(package)

    monkeypatch.setattr(result_package, "MAX_MEMBER_BYTES", 1024)
    monkeypatch.setattr(result_package, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(PackageError, match="total size limit"):
        verify_verifiable_zip(package)


def test_create_verifiable_zip_ignores_stale_legacy_temporary_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    package = tmp_path / "result.zip"
    package.with_name(".result.zip.tmp").write_bytes(b"other invocation")
    package.with_name(".result.zip.sha256.tmp").write_bytes(b"other invocation")

    create_verifiable_zip(root, [Path("report.json")], package)

    assert verify_verifiable_zip(package)["files"][0]["path"] == "report.json"


def test_create_verifiable_zip_never_overwrites_a_late_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    package = tmp_path / "result.zip"
    real_link = os.link

    def late_publish(source: str, destination: str, *args: object, **kwargs: object) -> None:
        if Path(destination) == package:
            package.write_bytes(b"late writer bytes")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(result_package.os, "link", late_publish)
    with pytest.raises(PackageError, match="refusing to overwrite"):
        create_verifiable_zip(root, [Path("report.json")], package)

    assert package.read_bytes() == b"late writer bytes"


def test_verify_verifiable_zip_rejects_member_limit_before_zipfile_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "many-members.zip"
    _write_many_members_package(package, 20)
    monkeypatch.setattr(result_package, "MAX_ARCHIVE_MEMBERS", 10, raising=False)
    monkeypatch.setattr(
        result_package,
        "_central_directory_matches",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not read central directory")),
    )
    monkeypatch.setattr(
        result_package.zipfile,
        "ZipFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load metadata")),
    )

    with pytest.raises(PackageError, match="member count limit"):
        verify_verifiable_zip(package)


def test_create_verifiable_zip_rejects_broken_destination_leaf_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    package = tmp_path / "result.zip"
    external = tmp_path / "external" / "target.zip"
    _symlink_or_skip(package, external)

    with pytest.raises(PackageError, match="leaf paths must not be symlinks"):
        create_verifiable_zip(root, [Path("report.json")], package)

    assert package.is_symlink()
    assert not external.exists()


def test_verify_verifiable_zip_rejects_package_and_sidecar_leaf_symlinks(tmp_path: Path) -> None:
    target = _valid_package(tmp_path / "target")
    package = tmp_path / "link.zip"
    sidecar = package.with_suffix(".zip.sha256")
    _symlink_or_skip(package, target)
    _symlink_or_skip(sidecar, target.with_suffix(".zip.sha256"))

    with pytest.raises(PackageError, match="leaf paths must not be symlinks"):
        verify_verifiable_zip(package)


def test_verify_verifiable_zip_uses_real_eocd_when_comment_contains_false_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _valid_package(tmp_path)
    payload = bytearray(package.read_bytes())
    eocd_offset = payload.rfind(b"PK\x05\x06")
    fields = list(struct.unpack_from("<4s4H2LH", payload, eocd_offset))
    false_eocd = struct.pack("<4s4H2LH", *fields[:-1], 0)
    struct.pack_into("<H", payload, eocd_offset + 20, len(false_eocd))
    payload.extend(false_eocd)
    package.write_bytes(payload)
    _write_sidecar(package)
    monkeypatch.setattr(
        result_package.zipfile,
        "ZipFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load metadata")),
    )

    with pytest.raises(PackageError, match="end record"):
        verify_verifiable_zip(package)


def test_verify_verifiable_zip_rejects_truncated_or_zip64_eocd(tmp_path: Path) -> None:
    truncated = _valid_package(tmp_path / "truncated")
    truncated.write_bytes(truncated.read_bytes()[:-1])
    _write_sidecar(truncated)
    zip64 = _valid_package(tmp_path / "zip64")
    payload = bytearray(zip64.read_bytes())
    eocd_offset = payload.rfind(b"PK\x05\x06")
    struct.pack_into("<H", payload, eocd_offset + 10, 0xFFFF)
    zip64.write_bytes(payload)
    _write_sidecar(zip64)

    with pytest.raises(PackageError, match="end record"):
        verify_verifiable_zip(truncated)
    with pytest.raises(PackageError, match="ZIP64 archives are not supported"):
        verify_verifiable_zip(zip64)


def test_verify_verifiable_zip_rejects_zip64_member_without_zip64_eocd(tmp_path: Path) -> None:
    package = tmp_path / "zip64-member.zip"
    _write_zip64_member_package(package)

    with pytest.raises(PackageError, match="ZIP64 members are not supported"):
        verify_verifiable_zip(package)


def test_zip64_extra_parser_rejects_truncated_extra_field() -> None:
    with pytest.raises(PackageError, match="ZIP extra field is malformed"):
        result_package._contains_zip64_extra(b"\x01\x00\x04")


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), OSError("io failure")])
def test_symlink_helper_reraises_unexpected_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    monkeypatch.setattr(Path, "symlink_to", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    monkeypatch.setattr(
        pytest,
        "skip",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not skip")),
    )

    with pytest.raises(type(error)):
        _symlink_or_skip(tmp_path / "link", tmp_path / "target")


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("package_only", "must exist together"),
        ("sidecar_only", "must exist together"),
        ("manifest_json", "package manifest is invalid"),
        ("unsafe_path", "package manifest is invalid"),
        ("nonportable_path", "package manifest is invalid"),
        ("unlisted_member", "archive members do not match"),
        ("member_bytes", "archive member byte length mismatch"),
    ],
)
def test_verify_verifiable_zip_rejects_corruption(
    tmp_path: Path, corruption: str, message: str
) -> None:
    package = _corrupted_package(tmp_path, corruption)

    with pytest.raises(PackageError, match=message):
        verify_verifiable_zip(package)


def _corrupted_package(tmp_path: Path, corruption: str) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    package = tmp_path / "result.zip"
    create_verifiable_zip(root, [Path("report.json")], package)
    sidecar = package.with_suffix(".zip.sha256")
    if corruption == "package_only":
        sidecar.unlink()
        return package
    if corruption == "sidecar_only":
        package.unlink()
        return package

    with zipfile.ZipFile(package) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if corruption == "manifest_json":
        members["package_manifest.json"] = b"not-json"
    elif corruption == "unsafe_path":
        manifest = json.loads(members["package_manifest.json"])
        manifest["files"][0]["path"] = "C:report.json"
        members["C:report.json"] = members.pop("report.json")
        members["package_manifest.json"] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    elif corruption == "nonportable_path":
        manifest = json.loads(members["package_manifest.json"])
        manifest["files"][0]["path"] = "nested//report.json"
        members["nested//report.json"] = members.pop("report.json")
        members["package_manifest.json"] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    elif corruption == "unlisted_member":
        members["unexpected.txt"] = b"unexpected"
    elif corruption == "member_bytes":
        members["report.json"] = b'{"changed": true}\n'
    else:
        raise AssertionError(f"unknown corruption: {corruption}")
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, contents in members.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, contents)
    sidecar.write_bytes(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n".encode("ascii")
    )
    return package


def _valid_package(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "root"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    package = tmp_path / "result.zip"
    create_verifiable_zip(root, [Path("report.json")], package)
    return package


def _rewritten_single_member_package(tmp_path: Path, member_name: str) -> Path:
    package = tmp_path / "result.zip"
    _write_package(package, {member_name: b"{}\n"})
    return package


def _rewrite_zip(
    package: Path, *, compression: int = zipfile.ZIP_STORED, external_attr: int | None = None
) -> None:
    with zipfile.ZipFile(package) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(package, "w", compression=compression) as archive:
        for name, contents in members.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = compression
            if external_attr is not None and name != "package_manifest.json":
                info.external_attr = external_attr << 16
            archive.writestr(info, contents)
    _write_sidecar(package)


def _write_package(package: Path, members: dict[str, bytes]) -> None:
    files = [
        {"path": name, "sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}
        for name, contents in sorted(members.items())
    ]
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, contents in members.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, contents)
        manifest = {"schema_version": "1.0", "files": files}
        info = zipfile.ZipInfo("package_manifest.json")
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(info, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    _write_sidecar(package)


def _write_sidecar(package: Path) -> None:
    package.with_suffix(".zip.sha256").write_bytes(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n".encode("ascii")
    )


def _write_many_members_package(package: Path, count: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
            for _ in range(count):
                info = zipfile.ZipInfo("duplicate-empty")
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, b"")
            info = zipfile.ZipInfo("package_manifest.json")
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, b'{"schema_version":"1.0","files":[]}')
    _write_sidecar(package)


def _write_zip64_member_package(package: Path) -> None:
    report = b"{}\n"
    manifest = {
        "schema_version": "1.0",
        "files": [
            {"path": "report.json", "sha256": hashlib.sha256(report).hexdigest(), "bytes": 3}
        ],
    }
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        info = zipfile.ZipInfo("report.json")
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        with archive.open(info, "w", force_zip64=True) as output:
            output.write(report)
        info = zipfile.ZipInfo("package_manifest.json")
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(info, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    _write_sidecar(package)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 1314:
            pytest.skip(f"symlink privilege unavailable: {exc}")
        raise


def test_collector_verifies_all_six_runs_and_includes_last_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path.resolve()
    lock = InputLock(
        git_sha="a" * 40,
        config_sha256="b" * 64,
        dataset_sha256="c" * 64,
        manifest_sha256="d" * 64,
        base_model_contract_sha256="e" * 64,
        base_model_sha256="f" * 64,
    )
    (workspace / "inputs").mkdir()
    (workspace / "inputs" / "input_lock.json").write_text(
        json.dumps(lock.as_dict()), encoding="utf-8"
    )
    (workspace / "gates").mkdir()
    (workspace / "gates" / "gate_report.json").write_text(
        json.dumps({"passed": True, "input_lock": lock.as_dict()}), encoding="utf-8"
    )
    deployment = workspace / "deployment"
    deployment.mkdir()
    onnx = deployment / "best.onnx"
    onnx.write_bytes(b"onnx")
    source = workspace / "runs" / "grouped" / "seed42" / "weights" / "best.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"weights")

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    gate = {
        "passed": True,
        "artifacts": {
            "onnx_sha256": sha(onnx),
            "source_checkpoint": source.relative_to(workspace).as_posix(),
            "source_checkpoint_sha256": sha(source),
        },
    }
    gate_path = deployment / "deployment_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    (deployment / "model_contract.candidate.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "onnx_sha256": sha(onnx),
                "source_checkpoint_sha256": sha(source),
                "deployment_gate_sha256": sha(gate_path),
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(result_package, "final_evaluation_is_complete", lambda _path: True)
    monkeypatch.setattr(
        result_package,
        "run_is_complete",
        lambda run_dir, observed_lock: calls.append((run_dir, observed_lock)) or True,
    )

    files = _collect_result_files(workspace)

    assert len(calls) == 6
    assert all(observed_lock == lock for _, observed_lock in calls)
    assert {
        Path("runs") / arm / f"seed{seed}" / "weights" / "last.pt"
        for arm in ("grouped", "leaky_control")
        for seed in (42, 43, 44)
    }.issubset(files)
    assert not any(path.as_posix().startswith("benchmark_l4/") for path in files)
