from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest


def test_configure_hermetic_ultralytics_forces_both_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pcb_defect.runtime_contract import configure_hermetic_ultralytics

    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.delenv("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", raising=False)

    configure_hermetic_ultralytics()

    assert os.environ["YOLO_AUTOINSTALL"] == "false"
    assert os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] == "1"


def _fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    platform: str,
    distributions: dict[str, str],
    module_version: str = "1.26.0",
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> None:
    import pcb_defect.runtime_contract as contract

    monkeypatch.setattr(contract.sys, "platform", platform)

    def version(name: str) -> str:
        if name not in distributions:
            raise contract.metadata.PackageNotFoundError(name)
        return distributions[name]

    fake_ort = SimpleNamespace(
        __version__=module_version,
        get_available_providers=lambda: list(providers),
    )
    monkeypatch.setattr(contract.metadata, "version", version)
    monkeypatch.setattr(contract.importlib, "import_module", lambda name: fake_ort)


@pytest.mark.parametrize(
    ("platform", "distributions", "expected_distribution"),
    [
        ("linux", {"onnxruntime-gpu": "1.26.0"}, "onnxruntime-gpu"),
        ("win32", {"onnxruntime": "1.26.0"}, "onnxruntime"),
    ],
)
def test_runtime_state_accepts_exact_platform_contract(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    distributions: dict[str, str],
    expected_distribution: str,
) -> None:
    from pcb_defect.runtime_contract import onnxruntime_state

    _fake_runtime(monkeypatch, platform=platform, distributions=distributions)

    state = onnxruntime_state()

    assert state["expected_distribution"] == expected_distribution
    assert state["module_version"] == "1.26.0"
    assert state["available_providers"] == ["CPUExecutionProvider"]


@pytest.mark.parametrize(
    ("distributions", "module_version", "providers", "message"),
    [
        ({"onnxruntime-gpu": "1.25.0"}, "1.26.0", ("CPUExecutionProvider",), "1.26.0"),
        (
            {"onnxruntime-gpu": "1.26.0", "onnxruntime": "1.26.0"},
            "1.26.0",
            ("CPUExecutionProvider",),
            "conflicting",
        ),
        ({"onnxruntime-gpu": "1.26.0"}, "1.28.0", ("CPUExecutionProvider",), "module"),
        (
            {"onnxruntime-gpu": "1.26.0"},
            "1.26.0",
            ("CUDAExecutionProvider",),
            "CPUExecutionProvider",
        ),
    ],
)
def test_runtime_state_rejects_wrong_or_conflicting_linux_state(
    monkeypatch: pytest.MonkeyPatch,
    distributions: dict[str, str],
    module_version: str,
    providers: tuple[str, ...],
    message: str,
) -> None:
    from pcb_defect.runtime_contract import RuntimeContractError, onnxruntime_state

    _fake_runtime(
        monkeypatch,
        platform="linux",
        distributions=distributions,
        module_version=module_version,
        providers=providers,
    )

    with pytest.raises(RuntimeContractError, match=message):
        onnxruntime_state()


def test_runtime_state_requires_cuda_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pcb_defect.runtime_contract import RuntimeContractError, onnxruntime_state

    _fake_runtime(
        monkeypatch,
        platform="linux",
        distributions={"onnxruntime-gpu": "1.26.0"},
    )

    with pytest.raises(RuntimeContractError, match="CUDAExecutionProvider"):
        onnxruntime_state(require_cuda_provider=True)


def test_windows_runtime_rejects_gpu_distribution_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pcb_defect.runtime_contract import RuntimeContractError, onnxruntime_state

    _fake_runtime(
        monkeypatch,
        platform="win32",
        distributions={
            "onnxruntime": "1.26.0",
            "onnxruntime-gpu": "1.26.0",
        },
    )

    with pytest.raises(RuntimeContractError, match="conflicting"):
        onnxruntime_state()


def test_runtime_contract_cli_prints_validated_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import pcb_defect.runtime_contract as contract

    _fake_runtime(
        monkeypatch,
        platform="linux",
        distributions={"onnxruntime-gpu": "1.26.0"},
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )

    assert contract.main(["--require-cuda-provider"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cuda_required"] is True
    assert payload["distribution_versions"] == {
        "onnxruntime": None,
        "onnxruntime-gpu": "1.26.0",
    }
