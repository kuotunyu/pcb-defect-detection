# Ultralytics AutoInstall Hermeticity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every intended ONNX deployment and benchmark stage fail closed on the locked ONNX Runtime environment while preventing Ultralytics 8.4.89 from installing a conflicting CPU runtime.

**Architecture:** Add one dependency-light runtime-contract module that sets the two Ultralytics environment controls and validates the exact platform-specific ONNX Runtime distribution, imported module version, and providers. Deployment, diagnostic-probe, and L4 benchmark entry points record and compare validated state before and after inference; Colab notebooks establish the policy before any subprocess and independently gate the relevant stages. Existing model artifacts, CPU-to-CPU parity semantics, thresholds, and the already verified short probe remain unchanged.

**Tech Stack:** Python 3.11, `importlib.metadata`, Ultralytics 8.4.89, ONNX Runtime 1.26.0, pytest, Ruff, uv 0.11.18, Jupyter notebook JSON, Git bundle, Colab A100/L4.

## Global Constraints

- Force `YOLO_AUTOINSTALL=false` before any intended workflow can import Ultralytics.
- Force `ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1`; repository-owned validation replaces the known false-negative CPU distribution-name check.
- On Linux, require only `onnxruntime-gpu==1.26.0`; `onnxruntime` must be absent.
- On Windows, require only `onnxruntime==1.26.0`; `onnxruntime-gpu` must be absent.
- Require imported `onnxruntime.__version__ == "1.26.0"` and `CPUExecutionProvider` for every ONNX stage.
- Require `CUDAExecutionProvider` only for explicit A100/L4 environment gates; runtime parity remains CPU-to-CPU.
- Keep 60 calibration images, parity confidence 0.25, match IoU 0.5, minimum IoU 0.90, and maximum confidence delta 0.15 unchanged.
- Do not install both ORT distributions, split dependency groups, patch Ultralytics, alter parity backends, or relax any gate.
- Do not rerun the independently verified short parity probe and do not retrain before the regenerated handoff passes all local gates.
- Do not use a local GPU, push, publish, create a release, or call a paid API.
- Preserve release identity `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no co-author trailer.
- Use `apply_patch` for repository edits; keep generated handoff and verification artifacts ignored under `dist/`.

## File Structure

- Create `src/pcb_defect/runtime_contract.py`: the only owner of Ultralytics environment controls and platform-specific ORT validation.
- Create `tests/test_runtime_contract.py`: isolated distribution/module/provider and CLI tests that require neither ORT nor Ultralytics in CPU-safe CI.
- Modify `src/pcb_defect/deployment.py`: guard ONNX validation and same-ONNX parity, persist before/after state, and reject stale completed gates without the contract.
- Modify `src/pcb_defect/deployment_probe.py`: guard future diagnostic parity runs without changing the validity of the already completed real probe.
- Modify `src/pcb_defect/benchmark.py`: guard CUDA ORT benchmarking and persist the exact runtime state.
- Modify `tests/test_deployment_gate.py`, `tests/test_deployment_probe.py`, and `tests/test_benchmark.py`: exercise stable and mutated workflow states without importing a real runtime in CI.
- Modify `notebooks/paired_experiment_a100.ipynb`, `notebooks/deployment_parity_probe_a100.ipynb`, and `notebooks/deployment_benchmark_l4.ipynb`: establish the policy in the first code cell and run repository-owned runtime gates.
- Modify `tests/test_release_contract.py`: enforce notebook ordering, before/after gates, locked reinstall, and rendered-notebook safety.
- Do not modify `pyproject.toml` or `uv.lock`; their existing platform markers are the intended contract.

---

### Task 1: Add the hermetic runtime contract

**Files:**
- Create: `src/pcb_defect/runtime_contract.py`
- Create: `tests/test_runtime_contract.py`

**Interfaces:**
- Produces: `RuntimeContractError(RuntimeError)`
- Produces: `configure_hermetic_ultralytics() -> None`
- Produces: `onnxruntime_state(require_cuda_provider: bool = False) -> dict[str, object]`
- Produces: `main(argv: list[str] | None = None) -> int`, invoked as `python -m pcb_defect.runtime_contract [--require-cuda-provider]`
- Return schema: `platform`, `expected_distribution`, `distribution_versions`, `module_version`, `available_providers`, and `cuda_required`.

- [ ] **Step 1: Write the failing environment-control test**

Create `tests/test_runtime_contract.py` with:

```python
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
```

- [ ] **Step 2: Run the focused test and confirm the red state**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_contract.py::test_configure_hermetic_ultralytics_forces_both_controls -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pcb_defect.runtime_contract'`.

- [ ] **Step 3: Add mocked platform/runtime tests before implementation**

Append this helper and tests to `tests/test_runtime_contract.py`:

```python
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
        ({"onnxruntime-gpu": "1.26.0"}, "1.26.0", ("CUDAExecutionProvider",), "CPUExecutionProvider"),
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
```

- [ ] **Step 4: Implement the minimum contract module**

Create `src/pcb_defect/runtime_contract.py`:

```python
"""Hermetic Ultralytics and ONNX Runtime environment contract."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from importlib import metadata
from typing import Any

EXPECTED_ORT_VERSION = "1.26.0"
ORT_DISTRIBUTIONS = ("onnxruntime", "onnxruntime-gpu")


class RuntimeContractError(RuntimeError):
    """The installed ONNX runtime cannot produce trustworthy evidence."""


def configure_hermetic_ultralytics() -> None:
    """Force Ultralytics to leave the locked environment unchanged."""
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def onnxruntime_state(require_cuda_provider: bool = False) -> dict[str, object]:
    """Validate and return a JSON-serializable snapshot of the exact ORT state."""
    expected_by_platform = {"linux": "onnxruntime-gpu", "win32": "onnxruntime"}
    expected_distribution = expected_by_platform.get(sys.platform)
    if expected_distribution is None:
        raise RuntimeContractError(f"unsupported ONNX Runtime platform: {sys.platform}")

    versions = {name: _distribution_version(name) for name in ORT_DISTRIBUTIONS}
    observed = versions[expected_distribution]
    if observed != EXPECTED_ORT_VERSION:
        raise RuntimeContractError(
            f"{expected_distribution} must equal {EXPECTED_ORT_VERSION}, found {observed!r}"
        )
    conflicting = next(name for name in ORT_DISTRIBUTIONS if name != expected_distribution)
    if versions[conflicting] is not None:
        raise RuntimeContractError(
            f"conflicting ONNX Runtime distribution is installed: {conflicting}=={versions[conflicting]}"
        )

    ort: Any = importlib.import_module("onnxruntime")
    module_version = getattr(ort, "__version__", None)
    if module_version != EXPECTED_ORT_VERSION:
        raise RuntimeContractError(
            f"onnxruntime module must equal {EXPECTED_ORT_VERSION}, found {module_version!r}"
        )
    providers = list(ort.get_available_providers())
    if "CPUExecutionProvider" not in providers:
        raise RuntimeContractError("ONNX Runtime is missing CPUExecutionProvider")
    if require_cuda_provider and "CUDAExecutionProvider" not in providers:
        raise RuntimeContractError("ONNX Runtime is missing CUDAExecutionProvider")

    return {
        "platform": sys.platform,
        "expected_distribution": expected_distribution,
        "distribution_versions": versions,
        "module_version": module_version,
        "available_providers": providers,
        "cuda_required": require_cuda_provider,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda-provider", action="store_true")
    args = parser.parse_args(argv)
    configure_hermetic_ultralytics()
    print(json.dumps(onnxruntime_state(args.require_cuda_provider), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add CLI JSON coverage and run Task 1 tests**

Append:

```python
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
```

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_contract.py -q
.\.venv\Scripts\ruff.exe check src/pcb_defect/runtime_contract.py tests/test_runtime_contract.py
.\.venv\Scripts\ruff.exe format --check src/pcb_defect/runtime_contract.py tests/test_runtime_contract.py
```

Expected: all runtime-contract tests and style checks pass without importing a real ORT installation.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/pcb_defect/runtime_contract.py tests/test_runtime_contract.py
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "Enforce hermetic ONNX runtime contract"
```

---

### Task 2: Guard deployment, probe, and benchmark workflows

**Files:**
- Modify: `src/pcb_defect/deployment.py:17-33,92-204,207-236`
- Modify: `src/pcb_defect/deployment_probe.py:17-20,131-182`
- Modify: `src/pcb_defect/benchmark.py:18-22,52-178,181-205`
- Modify: `tests/test_deployment_gate.py`
- Modify: `tests/test_deployment_probe.py`
- Modify: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `configure_hermetic_ultralytics()` and `onnxruntime_state(require_cuda_provider=False)` from Task 1.
- Deployment/probe requirement: `require_cuda_provider=False` because both same-ONNX parity backends remain CPU-bound.
- Benchmark requirement: `require_cuda_provider=True` because the L4 ONNX backend must activate CUDA.
- Report field: `runtime_contract = {"before": <state>, "after": <state>}`.
- Existing `runtime.onnxruntime_providers` in the benchmark report remains for backward-readable benchmark evidence.

- [ ] **Step 1: Write the failing probe mutation test and CPU-safe fixture**

In `tests/test_deployment_probe.py`, add a stable state fixture so the suite still runs in CI without the eval dependency group:

```python
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
```

Then add:

```python
def test_probe_rejects_runtime_mutation_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
```

- [ ] **Step 2: Run the mutation test and confirm the red state**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_deployment_probe.py::test_probe_rejects_runtime_mutation_before_publication -q
```

Expected: FAIL because `deployment_probe` has no `onnxruntime_state` boundary and writes the report.

- [ ] **Step 3: Configure Ultralytics before dynamic imports in all three modules**

In each workflow module, import the Task 1 interfaces with the existing repository imports, then call the configuration function once after imports and before any workflow function can dynamically import Ultralytics:

```python
from pcb_defect.runtime_contract import configure_hermetic_ultralytics, onnxruntime_state

configure_hermetic_ultralytics()
```

Do not import ORT at module scope. This keeps base/dev CI dependency-free while ensuring any later `from ultralytics import YOLO` observes both environment controls.

- [ ] **Step 4: Add deployment before/after validation and evidence**

In `export_and_gate`, snapshot immediately before the first ONNX-backed Ultralytics validation and again after same-ONNX parity:

```python
runtime_before = onnxruntime_state()
onnx_metrics = _validate_model(onnx_path, calibration_yaml, config["imgsz"])
fidelity = {
    "split": "calibration",
    "threshold": config["fidelity_absolute_delta"],
    "pt": pt_metrics,
    "onnx": onnx_metrics,
    "delta_map50": onnx_metrics["map50"] - pt_metrics["map50"],
    "delta_map50_95": onnx_metrics["map50_95"] - pt_metrics["map50_95"],
}
parity = _standalone_parity(onnx_path, [Path(path) for path in calibration_paths], config)
runtime_after = onnxruntime_state()
if runtime_after != runtime_before:
    raise DeploymentError("ONNX Runtime state changed during deployment inference")
```

Persist:

```python
"runtime_contract": {"before": runtime_before, "after": runtime_after},
```

In `_deployment_gate_is_complete`, compute `current_runtime = onnxruntime_state()` and require:

```python
report["runtime_contract"]["before"]
== report["runtime_contract"]["after"]
== current_runtime
```

This intentionally rejects old completed deployment evidence that predates the hermetic contract; it does not affect the old failed gate or the independently verified external probe.

- [ ] **Step 5: Add probe before/after validation and evidence**

In `run_probe`, verify immutable inputs, config, and frozen thresholds first. Take the first state before reserving the external output or staging the ONNX, so a broken environment leaves no new report path. Take the second state immediately after parity and ONNX hash revalidation:

```python
runtime_before = onnxruntime_state()
parity = _standalone_parity(staged.path, list(inputs.calibration_paths), config)
_verify_onnx_unchanged_after_inference(staged.path, inputs.onnx_path, expected_onnx_sha256)
runtime_after = onnxruntime_state()
if runtime_after != runtime_before:
    raise ProbeError("ONNX Runtime state changed during parity inference")
```

Persist the same `runtime_contract` field before exclusive report publication. Extend the existing successful-report test with:

```python
assert report["runtime_contract"] == {
    "before": _runtime_state(),
    "after": _runtime_state(),
}
```

- [ ] **Step 6: Add L4 benchmark before/after validation and evidence**

In `benchmark`, insert this call after verifying the L4 device and before creating a partial benchmark output directory:

```python
runtime_before = onnxruntime_state(require_cuda_provider=True)
```

Immediately after the existing `timings = {...}` comprehension and before `_hardware(...)`, insert:

```python
runtime_after = onnxruntime_state(require_cuda_provider=True)
if runtime_after != runtime_before:
    raise BenchmarkError("ONNX Runtime state changed during the L4 benchmark")
```

Persist the pair under `runtime_contract`, keep the actual active session providers under the existing `runtime.onnxruntime_providers`, and make `benchmark_is_complete` require both states to equal a freshly validated CUDA-required state.

- [ ] **Step 7: Add focused workflow and import-order assertions**

Append to `tests/test_deployment_gate.py`:

```python
def test_deployment_module_forces_ultralytics_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import os
    import pcb_defect.deployment as deployment

    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.delenv("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", raising=False)
    importlib.reload(deployment)

    assert os.environ["YOLO_AUTOINSTALL"] == "false"
    assert os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] == "1"
```

Append equivalent reload assertions for `pcb_defect.benchmark` in `tests/test_benchmark.py`. The probe import is already covered transitively and by its mutation test.

- [ ] **Step 8: Run all affected tests and style checks**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_contract.py tests/test_deployment_gate.py tests/test_deployment_probe.py tests/test_benchmark.py -q
.\.venv\Scripts\ruff.exe check src/pcb_defect/runtime_contract.py src/pcb_defect/deployment.py src/pcb_defect/deployment_probe.py src/pcb_defect/benchmark.py tests/test_runtime_contract.py tests/test_deployment_gate.py tests/test_deployment_probe.py tests/test_benchmark.py
.\.venv\Scripts\ruff.exe format --check src/pcb_defect/runtime_contract.py src/pcb_defect/deployment.py src/pcb_defect/deployment_probe.py src/pcb_defect/benchmark.py tests/test_runtime_contract.py tests/test_deployment_gate.py tests/test_deployment_probe.py tests/test_benchmark.py
```

Expected: all focused tests pass without training, inference, a real ORT import, or GPU access.

- [ ] **Step 9: Commit Task 2**

```powershell
git add src/pcb_defect/deployment.py src/pcb_defect/deployment_probe.py src/pcb_defect/benchmark.py tests/test_deployment_gate.py tests/test_deployment_probe.py tests/test_benchmark.py
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "Guard deployment runtime state"
```

---

### Task 3: Harden the A100, probe, and L4 notebooks

**Files:**
- Modify: `notebooks/paired_experiment_a100.ipynb`
- Modify: `notebooks/deployment_parity_probe_a100.ipynb`
- Modify: `notebooks/deployment_benchmark_l4.ipynb`
- Modify: `tests/test_release_contract.py`
- Test unchanged renderer behavior: `tests/test_handoff.py`

**Interfaces:**
- Consumes: `python -m pcb_defect.runtime_contract --require-cuda-provider` from Task 1.
- Notebook helper: `runtime_contract_state(label: str) -> dict[str, object]` parses the final nonempty stdout line as JSON and raises on a nonzero command or malformed output.
- A100 persisted names: `DEPLOYMENT_RUNTIME_BEFORE` and `DEPLOYMENT_RUNTIME_AFTER`.
- L4 persisted names: `BENCHMARK_RUNTIME_BEFORE` and `BENCHMARK_RUNTIME_AFTER`.
- Probe notebook performs a locked-runtime preflight only; future probe execution is internally guarded by Task 2.

- [ ] **Step 1: Write failing first-cell and stage-contract tests**

In `tests/test_release_contract.py`, add helpers:

```python
def _notebook(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
```

Add:

```python
def test_gpu_notebooks_disable_ultralytics_install_before_external_actions() -> None:
    for relative in (
        "notebooks/paired_experiment_a100.ipynb",
        "notebooks/deployment_parity_probe_a100.ipynb",
        "notebooks/deployment_benchmark_l4.ipynb",
    ):
        notebook = _notebook(relative)
        first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        source = "".join(first_code["source"])
        auto = source.index("os.environ['YOLO_AUTOINSTALL'] = 'false'")
        checks = source.index("os.environ['ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS'] = '1'")
        mount = source.index("drive.mount(")
        assert auto < mount
        assert checks < mount


def test_full_gpu_notebooks_gate_runtime_before_and_after_onnx_stages() -> None:
    a100 = _code(_notebook("notebooks/paired_experiment_a100.ipynb"))
    l4 = _code(_notebook("notebooks/deployment_benchmark_l4.ipynb"))

    assert "DEPLOYMENT_RUNTIME_BEFORE = runtime_contract_state(" in a100
    assert "DEPLOYMENT_RUNTIME_AFTER = runtime_contract_state(" in a100
    assert "if DEPLOYMENT_RUNTIME_AFTER != DEPLOYMENT_RUNTIME_BEFORE:" in a100
    assert "BENCHMARK_RUNTIME_BEFORE = runtime_contract_state(" in l4
    assert "BENCHMARK_RUNTIME_AFTER = runtime_contract_state(" in l4
    assert "if BENCHMARK_RUNTIME_AFTER != BENCHMARK_RUNTIME_BEFORE:" in l4
```

Replace the old direct `import onnxruntime as ort` notebook assertion with assertions that every GPU notebook invokes `pcb_defect.runtime_contract` and passes `--require-cuda-provider`.

- [ ] **Step 2: Run the new notebook tests and confirm the red state**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_release_contract.py::test_gpu_notebooks_disable_ultralytics_install_before_external_actions tests/test_release_contract.py::test_full_gpu_notebooks_gate_runtime_before_and_after_onnx_stages -q
```

Expected: both tests fail because the environment controls and before/after state variables are absent.

- [ ] **Step 3: Establish the environment policy in every first code cell**

Add `json` to the standard-library imports and place these lines immediately before `MPLBACKEND` and `drive.mount` in all three templates:

```python
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
```

No subprocess, package import, Drive operation, source checkout, or installer command may precede these assignments.

- [ ] **Step 4: Add one strict runtime CLI helper after locked reinstall**

After `VENV_PYTHON` validation in each notebook, define:

```python
def runtime_contract_state(label: str) -> dict[str, object]:
    command = [
        str(VENV_PYTHON),
        '-m',
        'pcb_defect.runtime_contract',
        '--require-cuda-provider',
    ]
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    print(f'[{label}] returncode={result.returncode}')
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f'{label} FAILED')
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f'{label} returned no runtime state')
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'{label} returned invalid runtime JSON') from exc


LOCKED_RUNTIME_STATE = runtime_contract_state('LOCKED RUNTIME CONTRACT')
```

Keep `uv sync --locked --no-editable --reinstall-package pcb-defect --extra train --group eval` exactly intact. Remove the A100 notebook's direct ORT `import_probe`; the repository CLI supersedes it with stricter distribution, module, CPU-provider, and CUDA-provider validation.

- [ ] **Step 5: Wrap full A100 deployment with an external before/after gate**

In the final A100 cell, leave final evaluation unchanged, then wrap the deployment command:

```python
DEPLOYMENT_RUNTIME_BEFORE = runtime_contract_state('DEPLOYMENT RUNTIME BEFORE')
deployment_result = run_logged(
    'deployment',
    [str(VENV_PYTHON), '-m', 'pcb_defect.deployment', *common],
    deployment_log,
)
DEPLOYMENT_RUNTIME_AFTER = runtime_contract_state('DEPLOYMENT RUNTIME AFTER')
if DEPLOYMENT_RUNTIME_AFTER != DEPLOYMENT_RUNTIME_BEFORE:
    raise RuntimeError('ONNX Runtime state changed across the deployment command')
```

Perform the comparison before handling `deployment_result.returncode`, so an installation side effect is reported even if the deployment command also failed. Preserve full deployment evidence logging and package verification.

- [ ] **Step 6: Wrap the L4 benchmark and preflight the probe**

In the L4 command cell:

```python
BENCHMARK_RUNTIME_BEFORE = runtime_contract_state('BENCHMARK RUNTIME BEFORE')
benchmark_result = run_logged(
    'L4 benchmark',
    [
        str(VENV_PYTHON), '-m', 'pcb_defect.benchmark',
        '--workspace', str(WORKSPACE), '--warmup', '30', '--cycles', '4',
    ],
    benchmark_log,
)
BENCHMARK_RUNTIME_AFTER = runtime_contract_state('BENCHMARK RUNTIME AFTER')
if BENCHMARK_RUNTIME_AFTER != BENCHMARK_RUNTIME_BEFORE:
    raise RuntimeError('ONNX Runtime state changed across the L4 benchmark command')
```

The probe notebook runs `LOCKED_RUNTIME_STATE = runtime_contract_state(...)` after reinstall and relies on the probe process's internal before/after gate from Task 2. Do not add training or change any immutable parent values.

- [ ] **Step 7: Verify templates and rendered notebooks**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_release_contract.py tests/test_handoff.py -q
.\.venv\Scripts\ruff.exe check tests/test_release_contract.py
.\.venv\Scripts\ruff.exe format --check tests/test_release_contract.py
```

Expected: all templates remain unexecuted, JSON-valid, placeholder-safe after rendering, and every code cell compiles.

- [ ] **Step 8: Commit Task 3**

```powershell
git add notebooks/paired_experiment_a100.ipynb notebooks/deployment_parity_probe_a100.ipynb notebooks/deployment_benchmark_l4.ipynb tests/test_release_contract.py
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "Harden Colab runtime environment"
```

---

### Task 4: Verify the clean snapshot and regenerate the full handoff

**Files:**
- Generated, ignored: `dist/colab-handoff-<commit-prefix>/`
- Generated, ignored: `dist/verification-<commit-prefix>/`
- No tracked edits unless a verification failure starts a new red-green cycle and receives its own commit.

**Interfaces:**
- Consumes: clean committed Tasks 1-3 and `pcb_defect.handoff`.
- Produces: a one-commit source bundle, rendered full A100 notebook, rendered L4 notebook, and manifest with matching hashes.
- Does not produce a new short-probe result; the verified report SHA-256 remains `1996f7b8f9d11aee24ca02c0ed4cc49974dc6804f3155a63e6c3d432208fa521` and is historical diagnostic evidence for its exact earlier probe snapshot.

- [ ] **Step 1: Run the full CPU-safe repository gates**

```powershell
uv sync --locked --no-editable --reinstall-package pcb-defect --extra train --group eval
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
uv build
git diff --check
git status --short --branch
```

Expected: the complete suite passes, Ruff and build pass, no GPU is accessed, `uv.lock` is unchanged, and the tracked worktree is clean.

- [ ] **Step 2: Generate the immutable full-rerun handoff**

```powershell
$handoffLabel = git rev-parse --short=7 HEAD
$handoffDir = "dist/colab-handoff-$handoffLabel"
.\.venv\Scripts\python.exe -m pcb_defect.handoff --repo . --output $handoffDir
```

Expected: a new, non-overwritten directory containing `pcb-defect-source.bundle`, `handoff_manifest.json`, `paired_experiment_a100.ipynb`, and `deployment_benchmark_l4.ipynb`. Do not pass the old probe-parent arguments because the short probe is already complete.

- [ ] **Step 3: Verify manifest hashes and rendered notebook contracts**

Run this read-only PowerShell verification:

```powershell
$manifest = Get-Content -LiteralPath "$handoffDir\handoff_manifest.json" -Raw | ConvertFrom-Json
$bundleHash = (Get-FileHash -LiteralPath "$handoffDir\pcb-defect-source.bundle" -Algorithm SHA256).Hash.ToLowerInvariant()
$a100Hash = (Get-FileHash -LiteralPath "$handoffDir\paired_experiment_a100.ipynb" -Algorithm SHA256).Hash.ToLowerInvariant()
$l4Hash = (Get-FileHash -LiteralPath "$handoffDir\deployment_benchmark_l4.ipynb" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($bundleHash -ne $manifest.bundle_sha256) { throw 'bundle hash mismatch' }
if ($a100Hash -ne $manifest.a100_notebook_sha256) { throw 'A100 notebook hash mismatch' }
if ($l4Hash -ne $manifest.l4_notebook_sha256) { throw 'L4 notebook hash mismatch' }
if (Select-String -Path "$handoffDir\*.ipynb" -Pattern 'PASTE_' -Quiet) { throw 'rendered placeholder remains' }
```

Then parse and compile every rendered code cell:

```powershell
$env:PCB_HANDOFF_VERIFY_DIR = (Resolve-Path $handoffDir).Path
@'
import json, os, pathlib
root = pathlib.Path(os.environ["PCB_HANDOFF_VERIFY_DIR"])
for path in sorted(root.glob("*.ipynb")):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"{path}:{index}", "exec")
print("RENDERED NOTEBOOK CONTRACT PASS")
'@ | .\.venv\Scripts\python.exe -
```

Expected: every hash matches, no `PASTE_` sentinel remains, and both rendered notebooks compile without stored outputs.

- [ ] **Step 4: Verify the one-commit bundle clone in isolation**

```powershell
$verifyDir = "dist/verification-$handoffLabel"
git bundle verify "$handoffDir\pcb-defect-source.bundle"
git clone "$handoffDir\pcb-defect-source.bundle" $verifyDir
if ((git -C $verifyDir rev-list --count HEAD) -ne '1') { throw 'bundle must contain one commit' }
if (git -C $verifyDir status --porcelain) { throw 'bundle clone is dirty' }
uv sync --locked --no-editable --reinstall-package pcb-defect --extra train --group eval --directory $verifyDir
& "$verifyDir\.venv\Scripts\python.exe" -m pytest -q "$verifyDir\tests"
& "$verifyDir\.venv\Scripts\ruff.exe" check $verifyDir
& "$verifyDir\.venv\Scripts\ruff.exe" format --check $verifyDir
uv build --directory $verifyDir
```

Expected: bundle verification passes, history contains exactly one clean commit, locked reinstall succeeds, the complete test suite and Ruff pass, and the package builds.

- [ ] **Step 5: Audit final provenance and handoff contents**

```powershell
git log -1 --format='%H%n%an <%ae>%n%cn <%ce>%n%B'
git status --short --branch
Get-ChildItem -LiteralPath $handoffDir -File | Select-Object Name,Length
```

Expected: exact release identity, no co-author trailer, clean tracked worktree, and only the expected bundle, manifest, A100 notebook, and L4 notebook in the new handoff.

- [ ] **Step 6: Deliver the new full A100 handoff and stop before cloud execution**

Give the user clickable links and SHA-256 values for the bundle, manifest, and full A100 notebook. State that the old Drive bundle and notebook should be replaced together, that a fresh A100 runtime is required, and that the L4 notebook remains gated on successful full A100 deployment evidence. Do not run Colab, train locally, publish, merge, or push.
