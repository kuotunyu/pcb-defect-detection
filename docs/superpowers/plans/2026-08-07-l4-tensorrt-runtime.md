# L4 TensorRT Runtime Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the private Colab L4 notebook install and verify an exact CUDA 12 TensorRT 10 runtime before starting its three-backend benchmark.

**Architecture:** Add a Linux-only `l4` dependency group without changing normal train/eval installs. Strengthen the notebook contract so its locked sync includes that group and a subprocess probe verifies the TensorRT import, exact version, and Builder construction before the benchmark cell can run.

**Tech Stack:** Python 3.11, uv lockfile, NVIDIA TensorRT CUDA 12 Python wheels, pytest, Jupyter notebook JSON.

## Global Constraints

- Pin `tensorrt-cu12==10.13.3.9` only for Linux in dependency group `l4`.
- Do not add TensorRT to ordinary `train`, `eval`, `dev`, or CPU CI installs.
- Do not introduce a CUDA-only fallback; the L4 evidence must still cover PyTorch FP32, ORT CUDA FP32, and TensorRT FP16.
- Do not run a local GPU benchmark, retrain, publish, or modify public claims.
- All commits must retain `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` as author and committer.

---

### Task 1: Locked L4 Runtime Contract

**Files:**
- Modify: `tests/test_release_contract.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `notebooks/deployment_benchmark_l4.ipynb`

**Interfaces:**
- Consumes: the existing L4 notebook's `VENV_PYTHON`, `run_project_json`, and `runtime_contract_state` helpers.
- Produces: dependency group `l4` and notebook output `LOCKED_TENSORRT_STATE` with keys `version` and `builder_available`.

- [ ] **Step 1: Write the failing contract test**

Add a test which parses `pyproject.toml` and the notebook source:

```python
def test_l4_notebook_installs_and_probes_locked_tensorrt_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["dependency-groups"]["l4"] == [
        "tensorrt-cu12==10.13.3.9; sys_platform == 'linux'"
    ]

    code = _code(_notebook("notebooks/deployment_benchmark_l4.ipynb"))
    sync = "[UV, 'sync', '--locked', '--no-editable', '--extra', 'train', '--group', 'eval', '--group', 'l4'"
    probe = "LOCKED_TENSORRT_STATE = run_project_json('LOCKED TENSORRT CONTRACT'"
    _assert_in_order(code, sync, probe, "LOCKED_RUNTIME_STATE = runtime_contract_state(")
    assert "import tensorrt as trt" in code
    assert "trt.__version__ == '10.13.3.9'" in code
    assert "bool(trt.Builder(trt.Logger()))" in code
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_release_contract.py::test_l4_notebook_installs_and_probes_locked_tensorrt_runtime -q`

Expected: FAIL because dependency group `l4` does not exist.

- [ ] **Step 3: Add the minimal locked dependency**

Add to `pyproject.toml`:

```toml
l4 = [
    "tensorrt-cu12==10.13.3.9; sys_platform == 'linux'",
]
```

Run: `uv lock --check` first to confirm the old lock is stale, then `uv lock` to resolve the exact Linux wheels while preserving existing pins.

- [ ] **Step 4: Add the notebook sync flag and early probe**

Extend the existing `uv sync` argument list with `--group`, `l4`. Immediately after `run_project_json` is defined, invoke it with a self-contained script that imports TensorRT, asserts version `10.13.3.9`, constructs a Builder, and emits JSON:

```python
TENSORRT_PROBE_SCRIPT = r'''
import json
import tensorrt as trt
if trt.__version__ != '10.13.3.9':
    raise RuntimeError(f'unexpected TensorRT version: {trt.__version__}')
builder_available = bool(trt.Builder(trt.Logger()))
if not builder_available:
    raise RuntimeError('TensorRT Builder is unavailable')
print(json.dumps({'version': trt.__version__, 'builder_available': builder_available}, sort_keys=True))
'''
LOCKED_TENSORRT_STATE = run_project_json(
    'LOCKED TENSORRT CONTRACT', TENSORRT_PROBE_SCRIPT
)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `uv run --locked --no-editable --extra train --group eval pytest tests/test_release_contract.py tests/test_l4_handoff.py tests/test_handoff.py -q`

Expected: PASS on CPU without importing or installing TensorRT because group `l4` is not selected by the test command.

- [ ] **Step 6: Run full CPU verification**

Run: `uv run --locked --no-editable --extra train --group eval pytest -q`

Run: `uv run --locked --no-editable --extra train --group eval ruff check .`

Expected: both exit 0.

- [ ] **Step 7: Commit**

```powershell
git add tests/test_release_contract.py pyproject.toml uv.lock notebooks/deployment_benchmark_l4.ipynb
git commit -m "fix: bootstrap locked TensorRT for L4"
```

### Task 2: Immutable Replacement Handoff

**Files:**
- Generate ignored artifacts under: `dist/colab-handoff-l4-<runner-prefix>/`
- Verify only: the generated notebook, manifest, and source bundle.

**Interfaces:**
- Consumes: clean committed Task 1 HEAD and the frozen A100 parent identities.
- Produces: exactly three same-version files ready for the user's Google Drive handoff directory.

- [ ] **Step 1: Generate from clean HEAD**

Run `python -m pcb_defect.l4_handoff` through the locked non-editable train/eval environment using the frozen parent experiment SHA `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`, gate SHA `466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803`, checkpoint SHA `44646b130b8b42282b752f77659cabfc1c484dc3aaa9a2dc8f710da8468f511a`, and ONNX SHA `b62590a14e2e88a414eb06389058d13d69ff1ea3998232996877088951fe3bb8`.

- [ ] **Step 2: Independently verify generated files**

Verify `git bundle verify` succeeds; bundle and notebook SHA-256 values match the manifest; the rendered notebook contains no `PASTE_` sentinel, execution output, or execution count; and its Drive directory embeds the new runner prefix.

- [ ] **Step 3: Verify identity and cleanliness**

Run full author/committer/co-author checks on the L4 branch and confirm every commit identity is `kuotunyu`. Confirm `git status --short` is empty and no generated `dist` artifact is tracked.

- [ ] **Step 4: Hand off one unambiguous action list**

Give the user absolute links to only the new notebook, bundle, and manifest. Instruct them to create the exact new Drive directory, upload those three files together, start a fresh Colab L4 runtime, open the new notebook, and choose **Run all**. They must not delete the dataset or parent workspace and must download the final ZIP plus `.sha256` only after `L4 HANDOFF COMPLETE` appears.
