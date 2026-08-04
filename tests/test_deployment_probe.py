from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from pcb_defect.deployment import parity_passes
from pcb_defect.viz import Box


def _runtime_state() -> dict[str, object]:
    return {
        "platform": "linux",
        "expected_distribution": "onnxruntime-gpu",
        "distribution_versions": {
            "onnxruntime": None,
            "onnxruntime-gpu": "1.26.0",
        },
        "module_version": "1.26.0",
        "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda_required": False,
    }


@pytest.fixture(autouse=True)
def _stable_runtime_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import pcb_defect.deployment_probe as deployment_probe

    monkeypatch.setattr(
        deployment_probe,
        "onnxruntime_state",
        lambda require_cuda_provider=False: _runtime_state(),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_parent_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Build an immutable, failed deployment fixture with 60 calibration images."""
    parent = tmp_path / "parent-workspace"
    deployment = parent / "deployment"
    images = tmp_path / "frozen-dataset" / "images"
    images.mkdir(parents=True)
    deployment.mkdir(parents=True)

    lock = {
        "git_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "base_model_contract_sha256": "e" * 64,
        "base_model_sha256": "f" * 64,
    }
    (parent / "inputs").mkdir()
    (parent / "inputs" / "input_lock.json").write_text(json.dumps(lock), encoding="utf-8")

    calibration_paths = []
    for index in range(60):
        image = images / f"calibration-{index:02d}.jpg"
        image.write_bytes(f"image-{index}".encode())
        calibration_paths.append(str(image.resolve()))
    calibration_list = parent / "runtime_data" / "grouped" / "calibration.txt"
    calibration_list.parent.mkdir(parents=True)
    calibration_list.write_text("\n".join(calibration_paths) + "\n", encoding="utf-8")
    (deployment / "calibration.yaml").write_text(
        yaml.safe_dump({"val": calibration_list.relative_to(parent).as_posix()}),
        encoding="utf-8",
    )

    onnx_path = deployment / "best.onnx"
    onnx_path.write_bytes(b"frozen-onnx")
    source_checkpoint = parent / "runs" / "grouped" / "seed42" / "weights" / "best.pt"
    source_checkpoint.parent.mkdir(parents=True)
    source_checkpoint.write_bytes(b"frozen-checkpoint")
    gate = {
        "passed": False,
        "artifacts": {
            "onnx": "best.onnx",
            "onnx_sha256": _sha256(onnx_path),
            "source_checkpoint": source_checkpoint.relative_to(parent).as_posix(),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
        },
    }
    gate_path = deployment / "deployment_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    return parent, {
        "git_sha": lock["git_sha"],
        "gate_sha256": _sha256(gate_path),
        "onnx_sha256": _sha256(onnx_path),
    }


def _parent_hashes(parent: Path) -> dict[Path, str]:
    return {path.relative_to(parent): _sha256(path) for path in parent.rglob("*") if path.is_file()}


def _passing_parity() -> dict[str, object]:
    return {
        "reference_backend": "ultralytics-onnx",
        "candidate_backend": "standalone-onnxruntime",
        "onnx_sha256": "filled-in-by-test",
        "n_images": 60,
        "required_images": 60,
        "n_failed": 0,
        "min_iou": 0.95,
        "required_min_iou": 0.90,
        "max_conf_delta": 0.10,
        "allowed_max_conf_delta": 0.15,
    }


def _write_probe_config(repo: Path, **overrides: float | int) -> None:
    config: dict[str, float | int] = {
        "calibration_images": 60,
        "parity_confidence": 0.25,
        "parity_match_iou": 0.5,
        "parity_min_iou": 0.90,
        "parity_max_confidence_delta": 0.15,
    }
    config.update(overrides)
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "deployment_gate.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _passing_parity_for(expected_onnx_sha256: str) -> dict[str, object]:
    parity = _passing_parity()
    parity["onnx_sha256"] = expected_onnx_sha256
    return parity


def test_probe_rejects_runtime_mutation_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime change during parity must prevent report publication."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    output = tmp_path / "runtime-mutated.json"
    _write_probe_config(repo)
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))
    monkeypatch.setattr(
        deployment_probe,
        "_standalone_parity",
        lambda _onnx, _images, _config: _passing_parity_for(expected["onnx_sha256"]),
    )
    changed = {**_runtime_state(), "module_version": "1.28.0"}
    states = iter((_runtime_state(), changed))
    monkeypatch.setattr(
        deployment_probe,
        "onnxruntime_state",
        lambda require_cuda_provider=False: next(states),
    )

    with pytest.raises(deployment_probe.ProbeError, match="changed during parity"):
        deployment_probe.run_probe(
            repo,
            parent,
            output,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )

    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()


def test_probe_rejects_parent_gate_or_onnx_hash_mismatch(tmp_path: Path) -> None:
    """Changing an expected immutable parent hash must block inference setup."""
    import importlib
    import importlib.util

    spec = importlib.util.find_spec("pcb_defect.deployment_probe")
    assert spec is not None
    deployment_probe = importlib.import_module("pcb_defect.deployment_probe")

    parent, expected = make_parent_workspace(tmp_path)
    verify = getattr(deployment_probe, "verify_probe_inputs", None)
    assert verify is not None

    with pytest.raises(deployment_probe.ProbeError, match="deployment-gate SHA-256"):
        verify(
            parent,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256="0" * 64,
            expected_onnx_sha256=expected["onnx_sha256"],
        )

    with pytest.raises(deployment_probe.ProbeError, match="ONNX SHA-256"):
        verify(
            parent,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256="0" * 64,
        )

    verified = verify(
        parent,
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )
    assert verified.parent_workspace == parent.resolve()
    assert verified.failed_gate_path == (parent / "deployment" / "deployment_gate.json").resolve()
    assert verified.onnx_path == (parent / "deployment" / "best.onnx").resolve()
    assert len(verified.calibration_paths) == 60
    assert verified.failed_gate["passed"] is False


def test_probe_writes_external_provenance_without_mutating_parent_or_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe result is external, content-addressed, and never changes failed evidence."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo)
    output = tmp_path / "probes" / "parity_probe.json"
    parent_hashes_before = _parent_hashes(parent)
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))
    monkeypatch.setattr(
        deployment_probe,
        "_standalone_parity",
        lambda _onnx, _images, _config: _passing_parity_for(expected["onnx_sha256"]),
    )

    report = deployment_probe.run_probe(
        repo,
        parent,
        output,
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )

    parent_hashes_after = _parent_hashes(parent)
    assert report["schema_version"] == "1.0"
    assert report["status"] == "complete"
    assert report["passed"] is True
    assert report["parent"]["experiment_git_sha"] == expected["git_sha"]
    assert report["parent"]["deployment_gate_sha256"] == expected["gate_sha256"]
    assert report["parent"]["onnx_sha256"] == expected["onnx_sha256"]
    assert parent_hashes_after == parent_hashes_before
    assert output.with_suffix(".json.sha256").is_file()
    assert parity_passes(report["parity"])
    assert report["runtime_contract"] == {
        "before": _runtime_state(),
        "after": _runtime_state(),
    }

    with pytest.raises(deployment_probe.ProbeError, match="refusing to overwrite"):
        deployment_probe.run_probe(
            repo,
            parent,
            output,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )


def test_probe_keeps_public_partial_state_if_sidecar_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed sidecar promotion leaves public evidence untouched for fail-closed inspection."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo)
    output = tmp_path / "probes" / "parity_probe.json"
    sidecar = output.with_suffix(".json.sha256")
    original_open = deployment_probe.os.open
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))
    monkeypatch.setattr(
        deployment_probe,
        "_standalone_parity",
        lambda _onnx, _images, _config: _passing_parity_for(expected["onnx_sha256"]),
    )

    def fail_sidecar_open(path: str, flags: int, *args: object) -> int:
        if Path(path) == sidecar:
            raise OSError("sidecar storage failure")
        return original_open(path, flags, *args)

    monkeypatch.setattr(deployment_probe.os, "open", fail_sidecar_open)
    with pytest.raises(
        deployment_probe.ProbeError, match="cannot publish probe report without overwrite"
    ):
        deployment_probe.run_probe(
            repo,
            parent,
            output,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )

    assert output.is_file()
    assert not sidecar.exists()
    assert output.with_name(".parity_probe.json.probe-reservation").is_file()


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("calibration_images", 59),
        ("parity_confidence", 0.24),
        ("parity_match_iou", 0.49),
        ("parity_min_iou", 0.89),
        ("parity_max_confidence_delta", 0.16),
    ],
)
def test_probe_rejects_altered_frozen_runtime_thresholds_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str, value: float | int
) -> None:
    """A changed local gate config cannot silently relax the frozen probe contract."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo, **{setting: value})
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))
    monkeypatch.setattr(
        deployment_probe,
        "_standalone_parity",
        lambda _onnx, _images, _config: _passing_parity_for(expected["onnx_sha256"]),
    )

    with pytest.raises(deployment_probe.ProbeError, match="frozen deployment-gate configuration"):
        deployment_probe.run_probe(
            repo,
            parent,
            tmp_path / f"{setting}.json",
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )


def test_probe_wraps_calibration_list_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreadable frozen calibration evidence is a probe failure, not a raw filesystem error."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, _expected = make_parent_workspace(tmp_path)
    calibration_list = parent / "runtime_data" / "grouped" / "calibration.txt"
    original_read_text = Path.read_text

    def fail_calibration_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == calibration_list:
            raise OSError("calibration list unavailable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_calibration_read)
    with pytest.raises(deployment_probe.ProbeError, match="cannot read calibration list"):
        deployment_probe._load_calibration_paths(parent, parent / "deployment" / "calibration.yaml")


def test_probe_wraps_output_directory_creation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable external destination is reported as a probe failure."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, _expected = make_parent_workspace(tmp_path)
    output = tmp_path / "unwritable" / "parity_probe.json"
    original_mkdir = Path.mkdir

    def fail_output_directory(path: Path, *args: object, **kwargs: object) -> None:
        if path == output.parent:
            raise OSError("output directory unavailable")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_output_directory)
    with pytest.raises(deployment_probe.ProbeError, match="cannot create probe output directory"):
        deployment_probe._prepare_output(output, parent)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_backend", "wrong-reference"),
        ("candidate_backend", "wrong-candidate"),
        ("reference_backend", None),
    ],
)
def test_probe_marks_altered_or_missing_parity_backend_identity_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """A metrics-shaped result cannot pass when either runtime identity is not frozen."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo)
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))

    def altered_parity(
        _onnx: Path, _images: list[Path], _config: dict[str, object]
    ) -> dict[str, object]:
        parity = _passing_parity_for(expected["onnx_sha256"])
        if value is None:
            parity.pop(field)
        else:
            parity[field] = value
        return parity

    monkeypatch.setattr(deployment_probe, "_standalone_parity", altered_parity)
    report = deployment_probe.run_probe(
        repo,
        parent,
        tmp_path / f"{field}-{value}.json",
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )

    assert report["passed"] is False


def test_probe_marks_parity_onnx_hash_mismatch_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parity metrics are rejected when they name any ONNX other than the verified parent."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo)
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))
    monkeypatch.setattr(
        deployment_probe,
        "_standalone_parity",
        lambda _onnx, _images, _config: _passing_parity_for("0" * 64),
    )

    report = deployment_probe.run_probe(
        repo,
        parent,
        tmp_path / "wrong-parity-onnx.json",
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )

    assert report["passed"] is False


@pytest.mark.parametrize("mutate_parent", [False, True])
def test_probe_rejects_onnx_mutation_during_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate_parent: bool
) -> None:
    """Changing either staged or parent ONNX during parity cannot produce a report."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo)
    output = tmp_path / f"mutated-{mutate_parent}.json"
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))

    def mutate_onnx(
        staged: Path, _images: list[Path], _config: dict[str, object]
    ) -> dict[str, object]:
        target = parent / "deployment" / "best.onnx" if mutate_parent else staged
        target.write_bytes(b"mutated-after-preflight")
        return _passing_parity_for(expected["onnx_sha256"])

    monkeypatch.setattr(deployment_probe, "_standalone_parity", mutate_onnx)
    with pytest.raises(deployment_probe.ProbeError, match="ONNX changed during parity inference"):
        deployment_probe.run_probe(
            repo,
            parent,
            output,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )

    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()


def test_probe_uses_staged_onnx_outside_parent_for_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both parity backends receive a hash-verified copy outside immutable evidence."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo)
    output = tmp_path / "staged.json"
    observed: list[Path] = []
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))

    def observe_staged(
        staged: Path, _images: list[Path], _config: dict[str, object]
    ) -> dict[str, object]:
        observed.append(staged)
        assert staged.read_bytes() == b"frozen-onnx"
        assert not staged.is_relative_to(parent)
        return _passing_parity_for(expected["onnx_sha256"])

    monkeypatch.setattr(deployment_probe, "_standalone_parity", observe_staged)
    report = deployment_probe.run_probe(
        repo,
        parent,
        output,
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )

    assert report["passed"] is True
    assert len(observed) == 1
    assert observed[0].is_file()
    assert observed[0].parent.is_dir()


def test_probe_refuses_late_destination_without_overwriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file created after reservation wins; probe publication fails closed and preserves it."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo)
    output = tmp_path / "late" / "parity.json"
    late_contents = b"created-after-preflight"
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))

    def create_late_destination(
        _onnx: Path, _images: list[Path], _config: dict[str, object]
    ) -> dict[str, object]:
        output.write_bytes(late_contents)
        return _passing_parity_for(expected["onnx_sha256"])

    monkeypatch.setattr(deployment_probe, "_standalone_parity", create_late_destination)
    with pytest.raises(
        deployment_probe.ProbeError, match="cannot publish probe report without overwrite"
    ):
        deployment_probe.run_probe(
            repo,
            parent,
            output,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )

    assert output.read_bytes() == late_contents


def test_probe_reservation_excludes_competing_probe_without_timing(tmp_path: Path) -> None:
    """The deterministic reservation file lets only one invocation own a destination."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, _expected = make_parent_workspace(tmp_path)
    output = tmp_path / "exclusive" / "parity.json"
    deployment_probe._prepare_output(output, parent)
    with pytest.raises(deployment_probe.ProbeError, match="reservation"):
        deployment_probe._prepare_output(output, parent)


def test_probe_hashes_and_parses_failed_gate_from_one_byte_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate digest and parsed object are bound to the same bytes, eliminating a read race."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    gate_path = parent / "deployment" / "deployment_gate.json"
    original_read_bytes = Path.read_bytes
    reads = 0

    def count_gate_reads(path: Path) -> bytes:
        nonlocal reads
        if path == gate_path:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count_gate_reads)
    verified = deployment_probe.verify_probe_inputs(
        parent,
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )

    assert verified.failed_gate["passed"] is False
    assert reads == 1


def test_probe_publishes_with_exclusive_create_when_hardlinks_are_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive-compatible exclusive files publish a complete report without using hardlinks."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    output = tmp_path / "portable" / "parity.json"
    _write_probe_config(repo)
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))
    monkeypatch.setattr(
        deployment_probe,
        "_standalone_parity",
        lambda _onnx, _images, _config: _passing_parity_for(expected["onnx_sha256"]),
    )
    monkeypatch.setattr(
        deployment_probe.os, "link", lambda *_args: (_ for _ in ()).throw(OSError())
    )

    report = deployment_probe.run_probe(
        repo,
        parent,
        output,
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )

    assert report["passed"] is True
    assert output.is_file()
    assert output.with_suffix(".json.sha256").is_file()


def test_probe_does_not_delete_other_writer_replacement_after_sidecar_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement survives because public paths are never cleaned after publication failure."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    output = tmp_path / "contested" / "parity.json"
    sidecar = output.with_suffix(".json.sha256")
    replacement = b"other-writer"
    _write_probe_config(repo)
    original_open = deployment_probe.os.open
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))
    monkeypatch.setattr(
        deployment_probe,
        "_standalone_parity",
        lambda _onnx, _images, _config: _passing_parity_for(expected["onnx_sha256"]),
    )

    def replace_before_sidecar(path: str, flags: int, *args: object) -> int:
        if Path(path) == sidecar:
            output.unlink()
            output.write_bytes(replacement)
            raise OSError("sidecar storage failure")
        return original_open(path, flags, *args)

    monkeypatch.setattr(deployment_probe.os, "open", replace_before_sidecar)
    with pytest.raises(deployment_probe.ProbeError, match="cannot publish probe report"):
        deployment_probe.run_probe(
            repo,
            parent,
            output,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )

    assert output.read_bytes() == replacement
    assert not sidecar.exists()


def test_probe_reports_reservation_setup_failure_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reservation failure aborts before inference or external report publication."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, _expected = make_parent_workspace(tmp_path)
    output = tmp_path / "reserve-error" / "parity.json"
    reservation = output.with_name(".parity.json.probe-reservation")
    original_open = deployment_probe.os.open

    def fail_reservation(path: str, flags: int, *args: object) -> int:
        if Path(path) == reservation:
            raise OSError("reservation unavailable")
        return original_open(path, flags, *args)

    monkeypatch.setattr(deployment_probe.os, "open", fail_reservation)
    with pytest.raises(deployment_probe.ProbeError, match="cannot exclusively reserve"):
        deployment_probe._prepare_output(output, parent)

    assert not output.exists()
    assert not reservation.exists()


@pytest.mark.parametrize(
    ("field", "value", "reject_serialization"),
    [
        ("n_images", 60.9, False),
        ("required_images", True, False),
        ("n_failed", 0.0, False),
        ("min_iou", math.nan, True),
        ("max_conf_delta", math.inf, True),
        ("required_min_iou", -math.inf, True),
        ("allowed_max_conf_delta", 1.1, False),
    ],
)
def test_probe_rejects_malformed_parity_aggregate_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    reject_serialization: bool,
) -> None:
    """Only finite, range-valid metrics and exact non-boolean count integers may pass."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    _write_probe_config(repo)
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))

    def malformed_parity(
        _onnx: Path, _images: list[Path], _config: dict[str, object]
    ) -> dict[str, object]:
        parity = _passing_parity_for(expected["onnx_sha256"])
        parity[field] = value
        return parity

    monkeypatch.setattr(deployment_probe, "_standalone_parity", malformed_parity)
    output = tmp_path / f"malformed-{field}.json"
    if reject_serialization:
        with pytest.raises(deployment_probe.ProbeError, match="cannot serialize probe report"):
            deployment_probe.run_probe(
                repo,
                parent,
                output,
                expected_parent_git_sha=expected["git_sha"],
                expected_gate_sha256=expected["gate_sha256"],
                expected_onnx_sha256=expected["onnx_sha256"],
            )
        assert not output.exists()
        return

    report = deployment_probe.run_probe(
        repo,
        parent,
        output,
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )
    assert report["passed"] is False


def test_probe_routes_staged_onnx_through_real_parity_backend_constructors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real parity plumbing constructs both fake backends from the external staged ONNX."""
    import pcb_defect.deployment as deployment
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    for image_path in (tmp_path / "frozen-dataset" / "images").glob("*.jpg"):
        Image.new("RGB", (8, 8), color="white").save(image_path)
    repo = tmp_path / "probe-repo"
    output = tmp_path / "real-parity.json"
    _write_probe_config(repo)
    observed: dict[str, Path] = {}
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))

    class FakeReference:
        def __init__(self, path: str) -> None:
            observed["reference"] = Path(path)

        def predict(self, _path: str, **_kwargs: object) -> list[SimpleNamespace]:
            values = SimpleNamespace(tolist=lambda: [[1.0, 1.0, 4.0, 4.0]])
            boxes = SimpleNamespace(
                xyxy=values,
                cls=SimpleNamespace(tolist=lambda: [0]),
                conf=SimpleNamespace(tolist=lambda: [0.9]),
            )
            return [SimpleNamespace(boxes=boxes)]

    class FakeStandalone:
        def __init__(self, path: Path) -> None:
            observed["candidate"] = path

        def predict(self, _image: Image.Image, conf: float) -> list[Box]:
            assert conf == 0.25
            return [Box(0, (1.0, 1.0, 4.0, 4.0), 0.9)]

    def build_models(staged: Path) -> tuple[FakeReference, FakeStandalone]:
        return FakeReference(str(staged)), FakeStandalone(staged)

    monkeypatch.setattr(deployment, "_build_runtime_parity_models", build_models)
    report = deployment_probe.run_probe(
        repo,
        parent,
        output,
        expected_parent_git_sha=expected["git_sha"],
        expected_gate_sha256=expected["gate_sha256"],
        expected_onnx_sha256=expected["onnx_sha256"],
    )

    assert report["passed"] is True
    assert observed["reference"] == observed["candidate"]
    assert not observed["reference"].is_relative_to(parent)


def test_probe_rejects_report_replaced_before_sidecar_publication_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report swapped while the sidecar opens is detected after both public writes finish."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    output = tmp_path / "replaced" / "parity.json"
    sidecar = output.with_suffix(".json.sha256")
    replacement = b"replacement-writer-report"
    _write_probe_config(repo)
    original_open = deployment_probe.os.open
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))
    monkeypatch.setattr(
        deployment_probe,
        "_standalone_parity",
        lambda _onnx, _images, _config: _passing_parity_for(expected["onnx_sha256"]),
    )

    def replace_report_then_open_sidecar(path: str, flags: int, *args: object) -> int:
        if Path(path) == sidecar:
            output.unlink()
            output.write_bytes(replacement)
        return original_open(path, flags, *args)

    monkeypatch.setattr(deployment_probe.os, "open", replace_report_then_open_sidecar)
    with pytest.raises(deployment_probe.ProbeError, match="published probe report pair changed"):
        deployment_probe.run_probe(
            repo,
            parent,
            output,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )

    assert output.read_bytes() == replacement
    assert sidecar.is_file()


def test_probe_rejects_nonfinite_report_content_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No report or sidecar is published when nested parity data contains non-finite JSON values."""
    import pcb_defect.deployment_probe as deployment_probe

    parent, expected = make_parent_workspace(tmp_path)
    repo = tmp_path / "probe-repo"
    output = tmp_path / "nonfinite" / "parity.json"
    _write_probe_config(repo)
    monkeypatch.setattr(deployment_probe, "_git_provenance", lambda _repo: ("b" * 40, False))

    def nonfinite_parity(
        _onnx: Path, _images: list[Path], _config: dict[str, object]
    ) -> dict[str, object]:
        parity = _passing_parity_for(expected["onnx_sha256"])
        parity["per_image"] = {"image": {"debug_value": math.nan}}
        return parity

    monkeypatch.setattr(deployment_probe, "_standalone_parity", nonfinite_parity)
    with pytest.raises(deployment_probe.ProbeError, match="cannot serialize probe report"):
        deployment_probe.run_probe(
            repo,
            parent,
            output,
            expected_parent_git_sha=expected["git_sha"],
            expected_gate_sha256=expected["gate_sha256"],
            expected_onnx_sha256=expected["onnx_sha256"],
        )

    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()
