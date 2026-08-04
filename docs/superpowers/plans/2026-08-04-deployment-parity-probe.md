# Deployment Parity Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct standalone parity to compare two runtimes of the same ONNX artifact, prove the correction against the existing failed deployment without retraining, and generate immutable probe and full-rerun Colab handoffs.

**Architecture:** Keep PyTorch-to-ONNX metric fidelity and same-ONNX runtime parity as separate gates. Add a read-only probe CLI that verifies the existing workspace and writes a new content-addressed report outside it. Extend the handoff generator with an optional, all-or-none parent-probe contract so one source bundle can render a probe notebook and the normal full-rerun notebooks.

**Tech Stack:** Python 3.11, Ultralytics 8.4.89, ONNX Runtime 1.26.0, PyYAML, Pillow, pytest, uv 0.11.18, Git bundle, Colab A100/L4.

## Global Constraints

- Do not change `fidelity_absolute_delta: 0.02`, `parity_min_iou: 0.90`, or `parity_max_confidence_delta: 0.15`.
- Do not modify or delete `/content/drive/MyDrive/pcb-defect-paired/workspaces/378e3925a8af`.
- Do not train during the probe and do not use a local GPU.
- Do not treat the probe as release evidence; only the later full rerun can produce promotable deployment evidence.
- Do not push, publish, create a release, or call paid APIs.
- Preserve exact release identity `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no co-author trailer.
- Use `apply_patch` for repository file edits and keep generated handoff artifacts ignored under `dist/`.

---

### Task 1: Separate export fidelity from same-ONNX runtime parity

**Files:**
- Modify: `src/pcb_defect/deployment.py:39-52`
- Modify: `src/pcb_defect/deployment.py:154-159`
- Modify: `src/pcb_defect/deployment.py:272-327`
- Modify: `tests/test_deployment_gate.py`

**Interfaces:**
- Produces: `parity_passes(parity: dict[str, Any]) -> bool`
- Produces: `_build_runtime_parity_models(onnx_path: Path, *, reference_factory: Callable | None = None, standalone_factory: Callable | None = None) -> tuple[Any, Any]`
- Changes: `_standalone_parity(onnx_path: Path, image_paths: list[Path], config: dict[str, Any]) -> dict[str, Any]`
- Consumes: `OnnxYoloModel`, `boxes_from_ultralytics`, `greedy_match`, and the unchanged deployment thresholds.

- [ ] **Step 1: Write the failing same-artifact model-construction test**

Add to `tests/test_deployment_gate.py`:

```python
def test_runtime_parity_builds_both_backends_from_the_same_onnx(tmp_path: Path) -> None:
    from pcb_defect import deployment

    onnx_path = tmp_path / "best.onnx"
    onnx_path.write_bytes(b"onnx")
    observed: dict[str, Path] = {}

    def reference_factory(path: str) -> object:
        observed["reference"] = Path(path)
        return object()

    def standalone_factory(path: Path) -> object:
        observed["standalone"] = Path(path)
        return object()

    build_models = getattr(deployment, "_build_runtime_parity_models", None)
    assert build_models is not None
    build_models(
        onnx_path,
        reference_factory=reference_factory,
        standalone_factory=standalone_factory,
    )

    assert observed == {"reference": onnx_path, "standalone": onnx_path}
```

- [ ] **Step 2: Run the focused test and verify the red state**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_deployment_gate.py::test_runtime_parity_builds_both_backends_from_the_same_onnx -q
```

Expected: FAIL because `_build_runtime_parity_models` does not exist.

- [ ] **Step 3: Implement the same-ONNX model boundary**

In `src/pcb_defect/deployment.py`, import `Callable` from `collections.abc` and add:

```python
def _build_runtime_parity_models(
    onnx_path: Path,
    *,
    reference_factory: Callable[[str], Any] | None = None,
    standalone_factory: Callable[[Path], Any] | None = None,
) -> tuple[Any, Any]:
    if reference_factory is None:
        from ultralytics import YOLO

        reference_factory = YOLO
    if standalone_factory is None:
        from pcb_defect.e2e_onnx import OnnxYoloModel

        standalone_factory = OnnxYoloModel
    return reference_factory(str(onnx_path)), standalone_factory(onnx_path)
```

Change `_standalone_parity` to receive only `onnx_path`, build both models with this helper, and call
the reference predictor with the same ONNX artifact:

```python
reference_model, standalone_model = _build_runtime_parity_models(onnx_path)
reference_result = reference_model.predict(
    str(image_path), conf=config["parity_confidence"], device="cpu", verbose=False
)[0]
reference_boxes = boxes_from_ultralytics(reference_result)
standalone_boxes = standalone_model.predict(
    image, conf=config["parity_confidence"]
)
match = greedy_match(standalone_boxes, reference_boxes, iou_thr=config["parity_match_iou"])
```

Rename per-image count fields from `n_pt` to `n_reference` while retaining `n_onnx` for backward
readability, and add these top-level fields:

```python
"reference_backend": "ultralytics-onnx",
"candidate_backend": "standalone-onnxruntime",
"onnx_sha256": _sha256_file(onnx_path),
```

- [ ] **Step 4: Extract and test the parity-only gate predicate**

Add this predicate and make `gate_passes` call it after checking fidelity:

```python
def parity_passes(parity: dict[str, Any]) -> bool:
    return (
        int(parity["n_images"]) == int(parity["required_images"])
        and int(parity["n_failed"]) == 0
        and float(parity["min_iou"]) >= float(parity["required_min_iou"])
        and float(parity["max_conf_delta"])
        <= float(parity["allowed_max_conf_delta"])
    )
```

Extend `tests/test_deployment_gate.py`:

```python
def test_parity_predicate_keeps_existing_threshold_contract() -> None:
    from pcb_defect.deployment import parity_passes

    parity = _report()["parity"]
    assert parity_passes(parity)
    parity["min_iou"] = 0.899
    assert not parity_passes(parity)
```

- [ ] **Step 5: Run deployment-gate tests and verify green**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_deployment_gate.py -q
```

Expected: all deployment-gate tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/pcb_defect/deployment.py tests/test_deployment_gate.py
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "Fix same-ONNX runtime parity boundary"
```

---

### Task 2: Add a read-only existing-artifact parity probe

**Files:**
- Create: `src/pcb_defect/deployment_probe.py`
- Create: `tests/test_deployment_probe.py`
- Modify: `pyproject.toml` only if an explicit console entry point is needed; prefer `python -m pcb_defect.deployment_probe` so no entry point is required.

**Interfaces:**
- Produces: `ProbeInputs` frozen dataclass containing verified parent paths and JSON records.
- Produces: `verify_probe_inputs(parent_workspace: Path, *, expected_parent_git_sha: str, expected_gate_sha256: str, expected_onnx_sha256: str) -> ProbeInputs`
- Produces: `run_probe(repo: Path, parent_workspace: Path, output: Path, *, expected_parent_git_sha: str, expected_gate_sha256: str, expected_onnx_sha256: str) -> dict[str, Any]`
- Consumes: `deployment._standalone_parity`, `deployment.parity_passes`, `experiment._git_provenance`, and `experiment._sha256_file`.

- [ ] **Step 1: Write failing parent-verification tests**

Create `tests/test_deployment_probe.py` with a helper that writes an input lock, failed deployment
gate, ONNX bytes, source checkpoint, calibration YAML, and a 60-line calibration list. Add:

```python
def test_probe_rejects_parent_gate_or_onnx_hash_mismatch(tmp_path: Path) -> None:
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
```

Also assert that a valid fixture returns resolved paths and that the failed gate has
`passed is False`.

- [ ] **Step 2: Run the focused test and verify red**

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_deployment_probe.py -q
```

Expected: FAIL at `assert spec is not None` because `pcb_defect.deployment_probe` does not exist.

- [ ] **Step 3: Implement fail-closed input verification**

Create `src/pcb_defect/deployment_probe.py` with:

```python
@dataclass(frozen=True)
class ProbeInputs:
    parent_workspace: Path
    input_lock: InputLock
    failed_gate_path: Path
    failed_gate: dict[str, Any]
    onnx_path: Path
    source_checkpoint: Path
    calibration_paths: tuple[Path, ...]
```

`verify_probe_inputs` must:

1. resolve the parent workspace;
2. read `inputs/input_lock.json` and require its `git_sha` to equal the expected parent SHA;
3. require `deployment/deployment_gate.json` to have the exact expected SHA-256 and
   `passed is False`;
4. resolve `artifacts.onnx` under the deployment directory and require both the report hash and the
   explicit expected ONNX hash to match its bytes;
5. resolve the source checkpoint under the parent workspace and require its report hash to match;
6. read `deployment/calibration.yaml`, require its `val` list to contain exactly 60 existing image
   paths, and reject any path outside the declared list;
7. perform no writes.

- [ ] **Step 4: Write the failing no-overwrite and provenance test**

Add a test that monkeypatches `_standalone_parity` to return `_report()["parity"]`, snapshots every
parent file hash before and after `run_probe`, and asserts:

```python
assert report["schema_version"] == "1.0"
assert report["status"] == "complete"
assert report["passed"] is True
assert report["parent"]["experiment_git_sha"] == expected["git_sha"]
assert report["parent"]["deployment_gate_sha256"] == expected["gate_sha256"]
assert report["parent"]["onnx_sha256"] == expected["onnx_sha256"]
assert parent_hashes_after == parent_hashes_before
assert output.with_suffix(".json.sha256").is_file()
```

Call `run_probe` a second time and require `ProbeError` with `refusing to overwrite`.

- [ ] **Step 5: Implement atomic probe reporting and CLI**

`run_probe` must verify the current probe repository is clean, call `verify_probe_inputs`, load
`configs/deployment_gate.yaml`, call `_standalone_parity` with the verified ONNX and calibration
paths, and write this report:

```python
report = {
    "schema_version": "1.0",
    "status": "complete",
    "passed": parity_passes(parity),
    "probe_git_sha": probe_git_sha,
    "parent": {
        "experiment_git_sha": inputs.input_lock.git_sha,
        "deployment_gate_sha256": expected_gate_sha256,
        "onnx_sha256": expected_onnx_sha256,
        "source_checkpoint_sha256": inputs.failed_gate["artifacts"][
            "source_checkpoint_sha256"
        ],
    },
    "config_sha256": _sha256_file(repo / "configs" / "deployment_gate.yaml"),
    "parity": parity,
}
```

Write JSON to a sibling temporary file, write `<output>.sha256` to a sibling temporary sidecar,
then atomically replace both destinations. The CLI arguments are:

```text
--repo
--parent-workspace
--output
--expected-parent-git-sha
--expected-gate-sha256
--expected-onnx-sha256
```

Exit nonzero after persisting the report when `passed` is false, so the notebook stops before a
full rerun recommendation.

- [ ] **Step 6: Run probe and deployment tests**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_deployment_probe.py tests/test_deployment_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
git add src/pcb_defect/deployment_probe.py tests/test_deployment_probe.py
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "Add read-only deployment parity probe"
```

---

### Task 3: Render probe and diagnostic full-rerun notebooks

**Files:**
- Create: `notebooks/deployment_parity_probe_a100.ipynb`
- Modify: `notebooks/paired_experiment_a100.ipynb`
- Modify: `notebooks/deployment_benchmark_l4.ipynb`
- Modify: `src/pcb_defect/handoff.py:105-184`
- Modify: `tests/test_handoff.py`
- Modify: `tests/test_release_contract.py`

**Interfaces:**
- Changes: `_render_notebook(template: Path, destination: Path, replacements: dict[str, str]) -> str`
- Adds optional all-or-none handoff CLI arguments: `--probe-parent-git-sha`,
  `--probe-parent-deployment-gate-sha256`, and `--probe-parent-onnx-sha256`.
- Produces: rendered `deployment_parity_probe_a100.ipynb` and probe metadata in
  `handoff_manifest.json` when all three probe arguments are present.

- [ ] **Step 1: Write failing generalized-renderer tests**

Update `tests/test_handoff.py` so `_render_notebook` receives:

```python
replacements={
    "PASTE_FINAL_BUNDLE_SHA256": "a" * 64,
    "PASTE_FINAL_GIT_SHA": "b" * 40,
}
```

Add a test with five placeholders and assert each is replaced exactly once. Add a second test that
passes only one of the three probe CLI arguments to `handoff.main` and expects `HandoffError` with
`probe arguments must be supplied together`.

- [ ] **Step 2: Run handoff tests and verify red**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_handoff.py -q
```

Expected: FAIL because `_render_notebook` still has fixed keyword parameters and the probe arguments
do not exist.

- [ ] **Step 3: Generalize notebook rendering and validate immutable values**

Change `_render_notebook` to iterate over `replacements.items()`, require every placeholder count to
equal one, validate rendered JSON, reject persisted outputs, and return the destination SHA-256.
Validate each hash argument as lowercase hexadecimal with the exact expected length before creating
the output directory.

- [ ] **Step 4: Create the thin probe notebook**

Create `notebooks/deployment_parity_probe_a100.ipynb` with unexecuted cells that:

1. mount Drive and set `os.environ["MPLBACKEND"] = "Agg"`;
2. define and verify the new source bundle SHA and snapshot SHA;
3. derive the immutable parent workspace from
   `PARENT_EXPERIMENT_GIT_SHA = "PASTE_PARENT_EXPERIMENT_GIT_SHA"`;
4. install uv 0.11.18 and perform the locked non-editable train/eval sync;
5. invoke `python -m pcb_defect.deployment_probe` with:

```text
PARENT_EXPERIMENT_GIT_SHA = 378e3925a8af9b5b1efba40cdfcf2fae4490f59b
PARENT_DEPLOYMENT_GATE_SHA256 = 017a1d36b214ff75eae8d01dec971a1c71d5273799b1c599ea0614345de518ec
PARENT_ONNX_SHA256 = 2195d0e3c2f0c7cafb7532af7909074781886484c67badeb38ef19eacdaedcb3
```

6. capture stdout/stderr to
   `probes/378e3925a8af-to-<new-snapshot-prefix>/probe_command.log`;
7. verify the report and `.sha256` sidecar before printing `PARITY PROBE PASS`.

The template stores placeholders for these three parent values; the exact values above are supplied
only by the handoff renderer.

- [ ] **Step 5: Make all long notebook stages diagnostic**

In `paired_experiment_a100.ipynb`, define one `run_logged(label, command, log_path)` helper using
`subprocess.run(..., text=True, capture_output=True)`. Use it for final evaluation, deployment, and
result packaging as well as the existing gate stage. On deployment failure, print
`deployment/deployment_gate.json` and `model_contract.candidate.json` when present.

In `deployment_benchmark_l4.ipynb`, import `os`, set `MPLBACKEND=Agg`, and use the same logged-command
pattern for the L4 benchmark and final package.

- [ ] **Step 6: Extend the handoff manifest and rendering loop**

When all probe arguments are present, add these manifest fields:

```python
"probe_notebook": "deployment_parity_probe_a100.ipynb",
"probe_parent_git_sha": args.probe_parent_git_sha,
"probe_parent_deployment_gate_sha256": args.probe_parent_deployment_gate_sha256,
"probe_parent_onnx_sha256": args.probe_parent_onnx_sha256,
```

Render the probe notebook with the source bundle/snapshot replacements and the three parent
replacements. Record `probe_notebook_sha256` in the final manifest.

- [ ] **Step 7: Strengthen notebook contract tests**

Update `tests/test_release_contract.py` to assert all three templates are unexecuted, parse as JSON,
compile every code cell, use the locked non-editable environment, set `MPLBACKEND=Agg`, and include
captured command logs. Assert the full A100 notebook prints deployment evidence on failure and the
probe notebook contains no training command (`train-all` or `experiment train`).

- [ ] **Step 8: Run notebook and handoff tests**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
.\.venv\Scripts\python.exe -m pytest tests/test_handoff.py tests/test_release_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 3**

```powershell
git add notebooks/deployment_parity_probe_a100.ipynb notebooks/paired_experiment_a100.ipynb notebooks/deployment_benchmark_l4.ipynb src/pcb_defect/handoff.py tests/test_handoff.py tests/test_release_contract.py
git -c user.name="kuotunyu" -c user.email="61350295+kuotunyu@users.noreply.github.com" commit -m "Add immutable deployment parity probe handoff"
```

---

### Task 4: Verify, package, and hand off the probe before full retraining

**Files:**
- Generated, ignored: `dist/colab-handoff-<commit-prefix>/`
- Generated, ignored: `dist/verification-<commit-prefix>/`
- No tracked source edits unless verification exposes a defect; any defect follows a new red-green
  cycle and its own commit.

**Interfaces:**
- Consumes: clean committed repository, all three probe parent hashes, `pcb_defect.handoff`.
- Produces: one bundle, manifest, ready probe notebook, ready full A100 notebook, and ready L4
  notebook with mutually consistent hashes.

- [ ] **Step 1: Run the complete local verification suite**

```powershell
uv sync --locked --no-editable --extra train --group eval --reinstall-package pcb-defect
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
git diff --check
git status --short --branch
```

Expected: all tests pass, lint and formatting pass, and the tracked worktree is clean.

- [ ] **Step 2: Generate the immutable handoff**

Set the output label from the committed HEAD prefix and run:

```powershell
$handoffLabel = git rev-parse --short=7 HEAD
$handoffDir = "dist/colab-handoff-$handoffLabel"
.\.venv\Scripts\python.exe -m pcb_defect.handoff `
  --repo . `
  --output $handoffDir `
  --probe-parent-git-sha 378e3925a8af9b5b1efba40cdfcf2fae4490f59b `
  --probe-parent-deployment-gate-sha256 017a1d36b214ff75eae8d01dec971a1c71d5273799b1c599ea0614345de518ec `
  --probe-parent-onnx-sha256 2195d0e3c2f0c7cafb7532af7909074781886484c67badeb38ef19eacdaedcb3
```

Expected: the command prints source SHA, snapshot SHA, bundle SHA, probe notebook SHA, and output
directory without overwriting an existing directory.

- [ ] **Step 3: Verify rendered notebook and manifest integrity**

Use a Python verification script to:

- recompute bundle and all notebook SHA-256 values from the manifest;
- assert no `PASTE_` placeholder remains;
- parse all notebooks as JSON;
- assert every output list is empty;
- compile every code cell;
- assert the probe notebook contains the exact parent hashes and no training command.

Expected: `rendered handoff integrity gates passed`.

- [ ] **Step 4: Verify a one-commit clean bundle clone**

```powershell
$verifyDir = "dist/verification-$handoffLabel"
git bundle verify "$handoffDir\pcb-defect-source.bundle"
git clone "$handoffDir\pcb-defect-source.bundle" $verifyDir
git -C $verifyDir rev-list --count HEAD
git -C $verifyDir status --porcelain
uv sync --locked --no-editable --extra train --group eval --directory $verifyDir
& "$verifyDir\.venv\Scripts\python.exe" -m pytest -q "$verifyDir\tests"
& "$verifyDir\.venv\Scripts\ruff.exe" check $verifyDir
```

Expected: bundle valid, exactly one commit, clean clone, full tests pass, and lint passes.

- [ ] **Step 5: Deliver only the probe as the next action**

Give the user links to the new source bundle, manifest, and
`deployment_parity_probe_a100.ipynb`. Tell them to preserve all existing Drive data, replace the
bundle, open a fresh A100 runtime, and run only the probe notebook. Explicitly tell them not to run
the full A100 notebook yet.

- [ ] **Step 6: Gate the full-rerun recommendation on real probe evidence**

After the user returns `parity_probe.json` and its `.sha256` sidecar, verify:

```text
status == complete
passed == true
parity.reference_backend == ultralytics-onnx
parity.candidate_backend == standalone-onnxruntime
parity.n_images == parity.required_images == 60
parity.n_failed == 0
parent hashes equal the immutable values in this plan
sidecar hash equals the report bytes
```

Only then provide the ready `paired_experiment_a100.ipynb` for tomorrow's full rerun. If the probe
fails, stop and diagnose the custom preprocessing/postprocessing with the persisted per-image data;
do not recommend retraining.
