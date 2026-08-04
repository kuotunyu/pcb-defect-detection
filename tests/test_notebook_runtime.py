from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


class _FakeProcess:
    def __init__(self, lines: list[str], returncode: int) -> None:
        self.stdout = iter(lines)
        self.returncode = returncode
        self.wait_calls = 0

    def wait(self) -> int:
        self.wait_calls += 1
        return self.returncode


class _RecordingHandle:
    def __init__(self, handle: Any, snapshots: list[str], path: Path) -> None:
        self.handle = handle
        self.snapshots = snapshots
        self.path = path

    def __enter__(self) -> _RecordingHandle:
        self.handle.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.handle.__exit__(*args)

    def write(self, value: str) -> int:
        return self.handle.write(value)

    def read(self) -> str:
        return self.handle.read()

    def flush(self) -> None:
        self.handle.flush()
        self.snapshots.append(self.path.read_text(encoding="utf-8"))


def test_streaming_command_appends_flushes_each_line_and_prints_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pcb_defect.notebook_runtime import run_streaming_command

    log_path = tmp_path / "drive" / "train.log"
    log_path.parent.mkdir()
    log_path.write_text("previous attempt\n", encoding="utf-8")
    process = _FakeProcess(["first line\n", "second line\n"], 0)
    popen_calls: list[tuple[list[str], dict[str, object]]] = []
    flush_snapshots: list[str] = []
    original_open = Path.open

    def recording_open(path: Path, *args: object, **kwargs: object) -> Any:
        handle = original_open(path, *args, **kwargs)
        if path == log_path:
            return _RecordingHandle(handle, flush_snapshots, log_path)
        return handle

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        popen_calls.append((command, kwargs))
        return process

    monkeypatch.setattr(Path, "open", recording_open)
    run_streaming_command(
        ["train", "--all"],
        cwd=tmp_path,
        log_path=log_path,
        label="train-all",
        popen=fake_popen,
    )

    assert popen_calls == [
        (
            ["train", "--all"],
            {
                "cwd": tmp_path,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
            },
        )
    ]
    assert capsys.readouterr().out == "first line\nsecond line\n"
    assert process.wait_calls == 1
    log = log_path.read_text(encoding="utf-8")
    assert log.startswith("previous attempt\n\n===== train-all attempt started ")
    assert "first line\nsecond line\n" in log
    assert "===== train-all attempt finished returncode=0 =====" in log
    assert len(flush_snapshots) >= 4
    assert "first line\n" in flush_snapshots[1]
    assert "second line\n" not in flush_snapshots[1]
    source = inspect.getsource(run_streaming_command)
    assert ".communicate(" not in source
    assert "capture_output" not in source


def test_streaming_command_persists_failure_before_raising(tmp_path: Path) -> None:
    from pcb_defect.notebook_runtime import NotebookRuntimeError, run_streaming_command

    log_path = tmp_path / "train.log"
    process = _FakeProcess(["failed output\n"], 9)

    with pytest.raises(NotebookRuntimeError, match="train-all.*train.log"):
        run_streaming_command(
            ["train", "--all"],
            cwd=tmp_path,
            log_path=log_path,
            label="train-all",
            popen=lambda *_args, **_kwargs: process,
            emit=lambda _line: None,
        )

    assert process.wait_calls == 1
    assert "failed output\n" in log_path.read_text(encoding="utf-8")
    assert "returncode=9" in log_path.read_text(encoding="utf-8")


def test_captured_command_reserves_log_before_running_and_persists_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from pcb_defect.notebook_runtime import NotebookRuntimeError, run_captured_command

    log_path = tmp_path / "probe.log"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="out\n", stderr="err\n")

    result = run_captured_command(
        ["probe"], cwd=tmp_path, log_path=log_path, label="probe", runner=runner
    )

    assert result.returncode == 0
    assert calls == [(["probe"], {"cwd": tmp_path, "text": True, "capture_output": True})]
    assert log_path.read_text(encoding="utf-8") == "out\n\n--- STDERR ---\nerr\n"
    assert capsys.readouterr().out == ""

    with pytest.raises(NotebookRuntimeError, match="already exists"):
        run_captured_command(
            ["probe-again"], cwd=tmp_path, log_path=log_path, label="probe", runner=runner
        )
    assert len(calls) == 1


def test_captured_command_persists_failure_before_raising(tmp_path: Path) -> None:
    from pcb_defect.notebook_runtime import NotebookRuntimeError, run_captured_command

    log_path = tmp_path / "failed-probe.log"

    with pytest.raises(NotebookRuntimeError, match="returncode=7.*failed-probe.log"):
        run_captured_command(
            ["probe"],
            cwd=tmp_path,
            log_path=log_path,
            label="probe",
            runner=lambda command, **_kwargs: subprocess.CompletedProcess(
                command, 7, stdout="partial\n", stderr="failure\n"
            ),
        )

    assert log_path.read_text(encoding="utf-8") == "partial\n\n--- STDERR ---\nfailure\n"


def test_captured_command_prints_failure_transcript_before_raising(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from pcb_defect.notebook_runtime import NotebookRuntimeError, run_captured_command

    log_path = tmp_path / "failed-probe.log"

    with pytest.raises(NotebookRuntimeError, match="returncode=7.*failed-probe.log"):
        run_captured_command(
            ["probe"],
            cwd=tmp_path,
            log_path=log_path,
            label="probe",
            runner=lambda command, **_kwargs: subprocess.CompletedProcess(
                command, 7, stdout="partial\n", stderr="failure\n"
            ),
        )

    assert capsys.readouterr().out == "partial\n\n--- STDERR ---\nfailure\n"


def _valid_report() -> dict[str, object]:
    return {
        "status": "complete",
        "passed": True,
        "parent": {
            "experiment_git_sha": "a" * 40,
            "deployment_gate_sha256": "b" * 64,
            "onnx_sha256": "c" * 64,
            "parity_onnx_sha256": "c" * 64,
        },
        "parity": {
            "reference_backend": "ultralytics-onnx",
            "candidate_backend": "standalone-onnxruntime",
            "onnx_sha256": "c" * 64,
            "n_images": 60,
            "required_images": 60,
            "n_failed": 0,
            "per_image": {f"image-{index:02d}": {} for index in range(60)},
        },
    }


def _write_report_pair(tmp_path: Path, report: dict[str, object]) -> Path:
    report_path = tmp_path / "parity_probe.json"
    report_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
    report_path.write_bytes(report_bytes)
    report_path.with_suffix(".json.sha256").write_bytes(
        f"{hashlib.sha256(report_bytes).hexdigest()}  parity_probe.json\n".encode("ascii")
    )
    return report_path


def _verify(report_path: Path) -> dict[str, object]:
    from pcb_defect.notebook_runtime import verify_probe_result

    return verify_probe_result(
        report_path,
        expected_parent_git_sha="a" * 40,
        expected_gate_sha256="b" * 64,
        expected_onnx_sha256="c" * 64,
    )


def test_probe_result_verifier_accepts_the_complete_deployment_probe_schema(tmp_path: Path) -> None:
    report = _valid_report()

    assert _verify(_write_report_pair(tmp_path, report)) == report


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "partial"),
        (("passed",), False),
        (("parent", "experiment_git_sha"), "0" * 40),
        (("parent", "deployment_gate_sha256"), "0" * 64),
        (("parent", "onnx_sha256"), "0" * 64),
        (("parent", "parity_onnx_sha256"), "0" * 64),
        (("parity", "reference_backend"), "other-reference"),
        (("parity", "candidate_backend"), "other-candidate"),
        (("parity", "onnx_sha256"), "0" * 64),
        (("parity", "n_images"), 60.0),
        (("parity", "required_images"), 60.0),
        (("parity", "n_failed"), 0.0),
        (("parity", "n_failed"), False),
        (("parity", "n_images"), 59),
        (("parity", "required_images"), 59),
        (("parity", "n_failed"), 1),
        (("parity", "per_image"), {"only-one": {}}),
    ],
)
def test_probe_result_verifier_rejects_semantic_mutations(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    from pcb_defect.notebook_runtime import NotebookRuntimeError

    report = _valid_report()
    target: dict[str, object] = report
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = value

    with pytest.raises(NotebookRuntimeError):
        _verify(_write_report_pair(tmp_path, report))


def test_probe_result_verifier_rejects_mutated_sidecar_bytes(tmp_path: Path) -> None:
    from pcb_defect.notebook_runtime import NotebookRuntimeError

    report_path = _write_report_pair(tmp_path, _valid_report())
    report_path.with_suffix(".json.sha256").write_bytes(b"wrong sidecar\n")

    with pytest.raises(NotebookRuntimeError, match="sidecar"):
        _verify(report_path)


def test_probe_result_verifier_rejects_invalid_json_with_a_valid_sidecar(tmp_path: Path) -> None:
    from pcb_defect.notebook_runtime import NotebookRuntimeError

    report_path = tmp_path / "parity_probe.json"
    report_bytes = b"{not valid JSON}\n"
    report_path.write_bytes(report_bytes)
    report_path.with_suffix(".json.sha256").write_bytes(
        f"{hashlib.sha256(report_bytes).hexdigest()}  parity_probe.json\n".encode("ascii")
    )

    with pytest.raises(NotebookRuntimeError, match="valid JSON"):
        _verify(report_path)
