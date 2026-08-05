from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from pcb_defect.l4_contract import (
    L4ContractError,
    L4RunIdentity,
    VerifiedL4Inputs,
    VerifiedL4ParentInputs,
)
from pcb_defect.runtime_contract import RuntimeContractError


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _FakeImage:
    def __enter__(self) -> _FakeImage:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def convert(self, _: str) -> _FakeImage:
        return self

    def copy(self) -> object:
        return object()


class _FakeModel:
    def __init__(self, _: object, **__: object) -> None:
        self.session = SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def predict(self, *_: object, **__: object) -> list[object]:
        return []

    def val(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(box=SimpleNamespace(map=0.9))


class _FakeCuda:
    def __init__(
        self, calls: list[str], *, available: bool = True, name: str = "NVIDIA L4"
    ) -> None:
        self.calls = calls
        self.available = available
        self.name = name

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, _: int) -> str:
        return self.name

    def synchronize(self) -> None:
        self.calls.append("sync")


def _install_model_adjacent_export_fake(
    monkeypatch: pytest.MonkeyPatch, captured_weights: list[Path]
) -> None:
    ultralytics = ModuleType("ultralytics")

    class ModelAdjacentExport:
        def __init__(self, weights: str) -> None:
            self.weights = Path(weights)
            captured_weights.append(self.weights)

        def export(self, **_: object) -> str:
            self.weights.with_suffix(".onnx").write_bytes(b"scratch-onnx")
            engine = self.weights.with_suffix(".engine")
            engine.write_bytes(b"scratch-engine")
            return str(engine)

    ultralytics.YOLO = ModelAdjacentExport
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)


def _directory_inventory(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def _fake_l4_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    available_providers: list[str] | None = None,
) -> tuple[Path, Path, Path, L4RunIdentity, dict[str, object]]:
    import pcb_defect.benchmark as benchmark_module

    repo = tmp_path / "repo"
    workspace = tmp_path / ("b" * 12)
    dataset_root = tmp_path / "dataset" / "pcb"
    repo.mkdir()
    workspace.mkdir()
    checkpoint = workspace / "best.pt"
    onnx = workspace / "deployment" / "best.onnx"
    calibration = dataset_root / "images" / "calibration.jpg"
    onnx.parent.mkdir(parents=True)
    calibration.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    onnx.write_bytes(b"onnx")
    calibration.write_bytes(b"calibration")
    identity = L4RunIdentity.parse(
        runner_git_sha="a" * 40,
        experiment_git_sha="b" * 40,
        deployment_gate_sha256="c" * 64,
        checkpoint_sha256=_sha256(checkpoint.read_bytes()),
        onnx_sha256=_sha256(onnx.read_bytes()),
    )
    gate: dict[str, Any] = {
        "dataset_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "fidelity": {
            "pt": {"map50_95": 0.9},
            "onnx": {"map50_95": 0.9},
            "threshold": 0.01,
        },
        "artifacts": {
            "source_checkpoint_sha256": identity.parent.checkpoint_sha256,
            "onnx_sha256": identity.parent.onnx_sha256,
        },
    }
    verified = VerifiedL4Inputs(
        identity.runner_git_sha,
        VerifiedL4ParentInputs(
            identity.parent.experiment_git_sha,
            workspace / "deployment" / "deployment_gate.json",
            gate,
            workspace / "runs" / "grouped" / "seed42" / "inputs" / "paired_split_manifest.json",
            checkpoint,
            onnx,
            workspace / "deployment" / "calibration.yaml",
            (calibration,),
        ),
    )

    def verify(
        _: Path, __: Path, observed_dataset_root: Path, ___: L4RunIdentity
    ) -> VerifiedL4Inputs:
        if observed_dataset_root != dataset_root.resolve():
            raise L4ContractError("dataset root mismatch")
        if calibration.read_bytes() != b"calibration":
            raise L4ContractError("calibration image SHA-256 mismatch")
        return verified

    calls: list[str] = []
    cuda = _FakeCuda(calls)
    torch = SimpleNamespace(cuda=cuda)
    gpu = benchmark_module._GpuRuntime(torch, _FakeModel, _FakeModel, "10.0")
    fake_runtime: dict[str, object] = {
        "available_providers": available_providers
        or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda_required": True,
    }
    fake_environment: dict[str, object] = {
        "python": "test",
        "platform": "test",
        "packages": {"torch": "test"},
    }
    fake_hardware: dict[str, object] = {
        "gpu": "NVIDIA L4",
        "driver": "test",
        "torch_cuda": "12.6",
        "cudnn": 9000,
        "tensorrt": "10.0",
    }
    clock = iter(value / 1000 for value in range(1000))
    monkeypatch.setattr(benchmark_module, "verify_l4_inputs", verify)
    monkeypatch.setattr(benchmark_module, "_load_gpu_runtime", lambda: gpu)
    monkeypatch.setattr(
        benchmark_module,
        "_export_engine",
        lambda _weights, output: output / "best_fp16.engine",
    )
    monkeypatch.setattr(benchmark_module, "_environment", lambda: fake_environment)
    monkeypatch.setattr(benchmark_module, "_hardware", lambda *_: fake_hardware)

    def runtime_state(**_: object) -> dict[str, object]:
        if (workspace / "runtime_state_changed").exists():
            return {**fake_runtime, "changed": True}
        return fake_runtime

    monkeypatch.setattr(benchmark_module, "onnxruntime_state", runtime_state)
    monkeypatch.setattr(benchmark_module.Image, "open", lambda _: _FakeImage())
    monkeypatch.setattr(benchmark_module.time, "perf_counter", lambda: next(clock))

    def export_engine(_weights: Path, output: Path) -> Path:
        engine = output / "best_fp16.engine"
        engine.write_bytes(b"engine")
        return engine

    monkeypatch.setattr(benchmark_module, "_export_engine", export_engine)
    return repo, workspace, dataset_root, identity, fake_runtime


def test_export_engine_uses_runner_owned_scratch_and_preserves_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pcb_defect.benchmark as benchmark_module

    parent = tmp_path / "immutable-parent" / "weights"
    parent.mkdir(parents=True)
    checkpoint = parent / "best.pt"
    checkpoint.write_bytes(b"parent-checkpoint")
    (parent / "owner.marker").write_bytes(b"parent-owner")
    output = tmp_path / "benchmark_l4" / ("a" * 12)
    output.mkdir(parents=True)
    captured_weights: list[Path] = []
    _install_model_adjacent_export_fake(monkeypatch, captured_weights)
    before = _directory_inventory(parent)

    engine = benchmark_module._export_engine(checkpoint, output)

    assert engine == output / "best_fp16.engine"
    assert engine.read_bytes() == b"scratch-engine"
    assert len(captured_weights) == 1
    assert captured_weights[0] != checkpoint
    assert captured_weights[0].is_relative_to(output)
    assert _directory_inventory(parent) == before
    assert {path.name for path in output.iterdir()} == {"best_fp16.engine"}


def test_export_engine_never_overwrites_existing_final_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pcb_defect.benchmark as benchmark_module

    checkpoint = tmp_path / "immutable-parent" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"parent-checkpoint")
    output = tmp_path / "benchmark_l4" / ("a" * 12)
    output.mkdir(parents=True)
    destination = output / "best_fp16.engine"
    destination.write_bytes(b"other-invocation")
    _install_model_adjacent_export_fake(monkeypatch, [])

    with pytest.raises(BenchmarkError, match="refusing to overwrite"):
        benchmark_module._export_engine(checkpoint, output)

    assert destination.read_bytes() == b"other-invocation"


def test_benchmark_rejects_partial_output_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity, _ = _fake_l4_benchmark(tmp_path, monkeypatch)
    output = workspace / "benchmark_l4" / identity.runner_git_sha[:12]
    output.mkdir(parents=True)
    marker = output / "owner.marker"
    marker.write_bytes(b"other-invocation")

    with pytest.raises(BenchmarkError, match="partial benchmark directory exists"):
        benchmark(repo, workspace, dataset_root, identity, warmup=30, cycles=4)

    assert _directory_inventory(output) == {"owner.marker": b"other-invocation"}


def _configure_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> tuple[Path, Path, Path, L4RunIdentity]:
    import pcb_defect.benchmark as benchmark_module

    repo, workspace, dataset_root, identity, _ = _fake_l4_benchmark(tmp_path, monkeypatch)
    if failure == "gpu":
        monkeypatch.setattr(
            benchmark_module,
            "_load_gpu_runtime",
            lambda: benchmark_module._GpuRuntime(
                SimpleNamespace(cuda=_FakeCuda([], name="NVIDIA T4")),
                _FakeModel,
                _FakeModel,
                "10.0",
            ),
        )
    elif failure == "cuda":
        monkeypatch.setattr(
            benchmark_module,
            "_load_gpu_runtime",
            lambda: benchmark_module._GpuRuntime(
                SimpleNamespace(cuda=_FakeCuda([], available=False)),
                _FakeModel,
                _FakeModel,
                "10.0",
            ),
        )
    elif failure == "provider":
        monkeypatch.setattr(
            benchmark_module,
            "onnxruntime_state",
            lambda **_: (_ for _ in ()).throw(
                RuntimeContractError("ONNX Runtime is missing CUDAExecutionProvider")
            ),
        )
    elif failure == "tensorrt":
        monkeypatch.setattr(
            benchmark_module,
            "_load_gpu_runtime",
            lambda: benchmark_module._GpuRuntime(
                SimpleNamespace(cuda=_FakeCuda([])), _FakeModel, _FakeModel, ""
            ),
        )
    elif failure == "contract":

        def contract_error(*_: object) -> VerifiedL4Inputs:
            raise L4ContractError("raw deployment-gate SHA-256 mismatch")

        monkeypatch.setattr(
            benchmark_module,
            "verify_l4_inputs",
            contract_error,
        )
    else:
        raise AssertionError(f"unknown failure fixture: {failure}")
    return repo, workspace, dataset_root, identity


def _completed_fake_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, L4RunIdentity, dict[str, Any]]:
    from pcb_defect.benchmark import benchmark

    repo, workspace, dataset_root, identity, _ = _fake_l4_benchmark(tmp_path, monkeypatch)
    report_path = benchmark(repo, workspace, dataset_root, identity, warmup=30, cycles=4)
    return (
        repo,
        workspace,
        dataset_root,
        identity,
        json.loads(report_path.read_text(encoding="utf-8")),
    )


def _mutate_completed_benchmark(
    repo: Path, workspace: Path, dataset_root: Path, report: dict[str, Any], mutation: str
) -> None:
    if mutation == "runner_sha":
        report["runner_git_sha"] = "f" * 40
    elif mutation == "experiment_sha":
        report["experiment_git_sha"] = "f" * 40
    elif mutation == "engine_bytes":
        (workspace / "benchmark_l4" / ("a" * 12) / "best_fp16.engine").write_bytes(b"changed")
    elif mutation == "runtime_state":
        (workspace / "runtime_state_changed").write_text("yes", encoding="utf-8")
    elif mutation == "calibration_bytes":
        (dataset_root / "images" / "calibration.jpg").write_bytes(b"changed")
    elif mutation == "protocol":
        report["protocol"]["warmup"] = 10
    elif mutation == "wrong_raw_count":
        report["timings"]["pytorch_fp32"]["raw_ms"].pop()
    elif mutation == "altered_summary":
        report["timings"]["onnxruntime_cuda_fp32"]["mean_ms"] = 99.0
    elif mutation == "environment":
        report["environment"]["platform"] = "forged"
    elif mutation == "runtime":
        report["runtime"]["onnxruntime_providers"] = ["CPUExecutionProvider"]
    elif mutation == "hardware":
        report["hardware"]["gpu"] = "NVIDIA L40"
    elif mutation == "fidelity":
        report["fidelity"]["absolute_delta_threshold"] = 1.0
    elif mutation == "artifacts":
        report["artifacts"]["engine_committable"] = True
    elif mutation == "command":
        report["command"] = [""]
    elif mutation == "timestamps":
        report["started_at_utc"] = "not-a-timestamp"
    else:
        raise AssertionError(f"unknown benchmark mutation: {mutation}")


from pcb_defect.benchmark import (  # noqa: E402
    BenchmarkError,
    _benchmark_image_paths,
    _time_backend,
    benchmark,
    benchmark_is_complete,
    main,
    summarize_latencies,
)


def test_latency_summary_retains_complete_statistics() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])

    assert summary == {
        "n_runs": 4,
        "mean_ms": 2.5,
        "std_ms": pytest.approx(1.2909944487358056),
        "p50_ms": 2.5,
        "p95_ms": pytest.approx(3.85),
        "min_ms": 1.0,
        "max_ms": 4.0,
        "fps_from_p50": 400.0,
    }


def test_time_backend_warms_then_measures_four_complete_cycles() -> None:
    calls: list[str] = []
    images = [object(), object()]

    result = _time_backend(
        lambda image: calls.append(f"infer:{images.index(image)}"),
        images,
        lambda: calls.append("sync"),
        warmup=30,
        cycles=4,
    )

    assert sum(item.startswith("infer:") for item in calls) == 38
    assert result["n_runs"] == 8
    assert len(result["raw_ms"]) == 8
    assert calls.count("sync") == 17


def test_benchmark_report_records_runner_and_parent_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity, fake_runtime = _fake_l4_benchmark(
        tmp_path, monkeypatch
    )

    report_path = benchmark(repo, workspace, dataset_root, identity, warmup=30, cycles=4)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["runner_git_sha"] == identity.runner_git_sha
    assert report["experiment_git_sha"] == identity.parent.experiment_git_sha
    assert report["deployment_gate_sha256"] == identity.parent.deployment_gate_sha256
    assert report["artifacts"]["source_checkpoint_sha256"] == identity.parent.checkpoint_sha256
    assert report["artifacts"]["onnx_sha256"] == identity.parent.onnx_sha256
    assert report["runtime_contract"]["before"] == fake_runtime
    assert report["runtime_contract"]["after"] == fake_runtime


def test_cli_requires_all_immutable_expectations(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_benchmark(
        repo: Path,
        workspace: Path,
        dataset_root: Path,
        identity: L4RunIdentity,
        *,
        warmup: int,
        cycles: int,
    ) -> Path:
        captured.update(
            repo=repo,
            workspace=workspace,
            dataset_root=dataset_root,
            identity=identity,
            warmup=warmup,
            cycles=cycles,
        )
        return workspace / "benchmark_l4.json"

    monkeypatch.setattr("pcb_defect.benchmark.benchmark", fake_benchmark)
    assert (
        main(
            [
                "--repo",
                "repo",
                "--workspace",
                "workspace",
                "--dataset",
                "dataset",
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
        == 0
    )
    assert captured["identity"] == L4RunIdentity.parse(
        runner_git_sha="a" * 40,
        experiment_git_sha="b" * 40,
        deployment_gate_sha256="c" * 64,
        checkpoint_sha256="d" * 64,
        onnx_sha256="e" * 64,
    )
    assert captured["dataset_root"] == Path("dataset").resolve()
    assert captured["warmup"] == 30
    assert captured["cycles"] == 4


def test_cli_rejects_missing_dataset_argument() -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--repo",
                "repo",
                "--workspace",
                "workspace",
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


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("gpu", "benchmark requires a Colab L4"),
        ("cuda", "CUDA is unavailable"),
        ("provider", "CUDAExecutionProvider"),
        ("tensorrt", "TensorRT runtime is unavailable"),
        ("contract", "raw deployment-gate SHA-256 mismatch"),
    ],
)
def test_benchmark_preflight_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str
) -> None:
    repo, workspace, dataset_root, identity = _configure_preflight_failure(
        tmp_path, monkeypatch, failure
    )

    with pytest.raises((BenchmarkError, L4ContractError), match=message):
        benchmark(repo, workspace, dataset_root, identity, warmup=30, cycles=4)

    assert not (workspace / "benchmark_l4" / identity.runner_git_sha[:12]).exists()


@pytest.mark.parametrize("device_name", ["NVIDIA L40", "NVIDIA L40S", "NVIDIA L4S", "L4"])
def test_benchmark_rejects_similar_but_non_l4_device_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_name: str
) -> None:
    import pcb_defect.benchmark as benchmark_module

    repo, workspace, dataset_root, identity, _ = _fake_l4_benchmark(tmp_path, monkeypatch)
    monkeypatch.setattr(
        benchmark_module,
        "_load_gpu_runtime",
        lambda: benchmark_module._GpuRuntime(
            SimpleNamespace(cuda=_FakeCuda([], name=device_name)),
            _FakeModel,
            _FakeModel,
            "10.0",
        ),
    )

    with pytest.raises(BenchmarkError, match="benchmark requires a Colab L4"):
        benchmark(repo, workspace, dataset_root, identity, warmup=30, cycles=4)

    assert not (workspace / "benchmark_l4" / identity.runner_git_sha[:12]).exists()


def test_benchmark_rejects_noncanonical_protocol_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity, _ = _fake_l4_benchmark(tmp_path, monkeypatch)

    with pytest.raises(BenchmarkError, match="warmup=30 and cycles=4"):
        benchmark(repo, workspace, dataset_root, identity, warmup=10, cycles=2)

    assert not (workspace / "benchmark_l4" / identity.runner_git_sha[:12]).exists()


@pytest.mark.parametrize(
    "mutation",
    ["runner_sha", "experiment_sha", "engine_bytes", "runtime_state", "calibration_bytes"],
)
def test_completed_benchmark_rejects_changed_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    _mutate_completed_benchmark(repo, workspace, dataset_root, report, mutation)

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, report) is False


def test_completed_benchmark_rejects_different_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, _, identity, report = _completed_fake_benchmark(tmp_path, monkeypatch)
    different_dataset_root = tmp_path / "different-dataset" / "pcb"
    different_dataset_root.mkdir(parents=True)

    assert benchmark_is_complete(repo, workspace, different_dataset_root, identity, report) is False


@pytest.mark.parametrize(
    "required_field",
    [
        "schema_version",
        "status",
        "started_at_utc",
        "completed_at_utc",
        "command",
        "runner_git_sha",
        "experiment_git_sha",
        "deployment_gate_sha256",
        "dataset_sha256",
        "manifest_sha256",
        "environment",
        "runtime",
        "runtime_contract",
        "hardware",
        "protocol",
        "artifacts",
        "fidelity",
        "timings",
        "int8",
    ],
)
def test_completed_benchmark_rejects_missing_required_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required_field: str
) -> None:
    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    report.pop(required_field)

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, report) is False


@pytest.mark.parametrize(
    "mutation",
    [
        "protocol",
        "wrong_raw_count",
        "altered_summary",
        "environment",
        "runtime",
        "hardware",
        "fidelity",
        "artifacts",
        "command",
        "timestamps",
    ],
)
def test_completed_benchmark_rejects_forged_required_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    _mutate_completed_benchmark(repo, workspace, dataset_root, report, mutation)

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, report) is False


@pytest.mark.parametrize(
    "summary_field",
    ["n_runs", "mean_ms", "std_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "fps_from_p50"],
)
def test_completed_benchmark_rejects_boolean_timing_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summary_field: str
) -> None:
    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    report["timings"]["pytorch_fp32"][summary_field] = True

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, report) is False


@pytest.mark.parametrize(
    "mutation",
    ["batch_boolean", "engine_zero", "extra_runtime_contract_key", "trailing_provider"],
)
def test_completed_benchmark_rejects_nested_schema_or_type_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    if mutation == "batch_boolean":
        report["protocol"]["batch"] = True
    elif mutation == "engine_zero":
        report["artifacts"]["engine_committable"] = 0
    elif mutation == "extra_runtime_contract_key":
        report["runtime_contract"]["extra"] = "forged"
    elif mutation == "trailing_provider":
        report["runtime"]["onnxruntime_providers"].append("ForgedExecutionProvider")
    else:
        raise AssertionError(f"unknown nested mutation: {mutation}")

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, report) is False


@pytest.mark.parametrize("command", [[], [True], [""], "not-a-list"])
def test_completed_benchmark_rejects_invalid_informational_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: object
) -> None:
    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    report["command"] = command

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, report) is False


def test_completed_benchmark_reuses_valid_evidence_from_different_process_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pcb_defect.benchmark as benchmark_module

    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(benchmark_module.sys, "argv", ["l4-package"])

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, report) is True


def test_completed_benchmark_allows_active_provider_subset_of_available_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity, _ = _fake_l4_benchmark(
        tmp_path,
        monkeypatch,
        available_providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    report_path = benchmark(repo, workspace, dataset_root, identity, warmup=30, cycles=4)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, report) is True


@pytest.mark.parametrize("root", [{}, [], "scalar", None])
def test_benchmark_completion_rejects_noncomplete_json_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: object
) -> None:
    repo, workspace, dataset_root, identity, _ = _fake_l4_benchmark(tmp_path, monkeypatch)

    assert benchmark_is_complete(repo, workspace, dataset_root, identity, root) is False


@pytest.mark.parametrize("serialized", ["[]", '"scalar"', "null"])
def test_existing_nonobject_report_raises_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serialized: str
) -> None:
    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    report_path = workspace / "benchmark_l4" / identity.runner_git_sha[:12] / "benchmark_l4.json"
    report_path.write_text(serialized, encoding="utf-8")

    with pytest.raises(BenchmarkError, match="invalid existing benchmark report"):
        benchmark(repo, workspace, dataset_root, identity, warmup=30, cycles=4)


def test_existing_unreadable_report_raises_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace, dataset_root, identity, report = _completed_fake_benchmark(
        tmp_path, monkeypatch
    )
    report_path = workspace / "benchmark_l4" / identity.runner_git_sha[:12] / "benchmark_l4.json"
    report_path.write_bytes(b"\xff")

    with pytest.raises(BenchmarkError, match="invalid existing benchmark report"):
        benchmark(repo, workspace, dataset_root, identity, warmup=30, cycles=4)


def test_benchmark_inputs_come_from_calibration_val_list(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.jpg"
    forbidden_final = tmp_path / "final.jpg"
    calibration.write_bytes(b"calibration")
    forbidden_final.write_bytes(b"final")
    calibration_list = tmp_path / "calibration.txt"
    final_list = tmp_path / "final.txt"
    calibration_list.write_text(f"{calibration}\n", encoding="utf-8")
    final_list.write_text(f"{forbidden_final}\n", encoding="utf-8")
    data_yaml = tmp_path / "calibration.yaml"
    data_yaml.write_text(
        f"val: {calibration_list.as_posix()}\ntest: {final_list.as_posix()}\n",
        encoding="utf-8",
    )

    assert _benchmark_image_paths(data_yaml) == [calibration]


def test_benchmark_module_forces_ultralytics_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import pcb_defect.benchmark as benchmark_module

    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.delenv("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", raising=False)
    importlib.reload(benchmark_module)

    assert os.environ["YOLO_AUTOINSTALL"] == "false"
    assert os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] == "1"
