# L4 Parent Workspace Layout Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the private L4 handoff verify and package the existing production A100 workspace layout without mutating parent evidence or rerunning training.

**Architecture:** Extend the L4 verification boundary with an explicit dataset root, verify all six run-scoped manifest copies byte-for-byte, and constrain external calibration pixels to the declared dataset image root with manifest content hashes. Propagate the dataset root through benchmark, package, and notebook resume paths, then generate a new immutable handoff and fast-forward the public single-contributor repository.

**Tech Stack:** Python 3.11, pathlib, argparse, pytest, Ultralytics integration boundaries, Git bundles, nbformat-compatible JSON notebooks, uv, Ruff, GitHub CLI.

## Global Constraints

- Parent experiment Git SHA remains `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`.
- Parent workspace remains `/content/drive/MyDrive/pcb-defect-paired/workspaces/9e3a1ed5827a` and is read-only.
- Dataset root is explicit: `/content/drive/MyDrive/pcb-defect-paired/dataset/pcb`.
- Both arms (`grouped`, `leaky_control`) and seeds 42, 43, 44 must provide byte-identical manifests.
- Checkpoint and ONNX paths remain contained by the parent workspace.
- Calibration pixels must resolve beneath `<dataset-root>/images` and match frozen manifest hashes.
- No local implementation or verification step may use a GPU.
- No dataset pixel, checkpoint, ONNX model, TensorRT engine, result ZIP, log, secret, or `.env` value may be tracked or published.
- Every published commit must use `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` as author and committer, with no co-author trailer.
- The public GitHub Contributors API must continue to return only `kuotunyu`.

---

### Task 1: Production-shaped parent evidence contract

**Files:**
- Modify: `tests/test_l4_contract.py`
- Modify: `src/pcb_defect/l4_contract.py`

**Interfaces:**
- Produces: `verify_l4_parent_inputs(workspace: Path, dataset_root: Path, parent: L4ParentIdentity) -> VerifiedL4ParentInputs`
- Produces: `verify_l4_inputs(repo: Path, workspace: Path, dataset_root: Path, identity: L4RunIdentity) -> VerifiedL4Inputs`
- Extends: `VerifiedL4ParentInputs.manifest_path: Path`
- Consumes: fixed arms `grouped`, `leaky_control` and seeds `42`, `43`, `44`

- [ ] **Step 1: Rewrite the fixture to match production layout**

Create an external `dataset/pcb/images` tree beside the workspace. Write the same canonical manifest bytes to all six paths:

```python
for arm in ("grouped", "leaky_control"):
    for seed in (42, 43, 44):
        path = workspace / "runs" / arm / f"seed{seed}" / "inputs" / "paired_split_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(manifest_bytes)
```

Render `runtime_data/grouped/calibration.txt` with the absolute external image path and return `dataset_root` from the fixture.

- [ ] **Step 2: Add focused acceptance and rejection tests**

Add tests that call the wished-for four-argument contract and assert:

```python
verified = verify_l4_parent_inputs(workspace, dataset_root, identity.parent)
assert verified.manifest_path == (
    workspace / "runs/grouped/seed42/inputs/paired_split_manifest.json"
)
assert verified.calibration_images == (dataset_root / "images/01_missing_hole_02.jpg",)
```

Add parametrized mutations for one missing manifest, one byte-different manifest, invalid self-hash, calibration path outside `dataset_root/images`, symlink escape when supported, changed image bytes, wrong order, duplicate stem, and final-test overlap.

- [ ] **Step 3: Run the focused tests and prove RED**

Run:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
uv run --locked --extra train --group eval pytest tests/test_l4_contract.py -q
```

Expected: failures because current signatures lack `dataset_root`, current code requires a root manifest, and external calibration images are rejected as workspace escapes.

- [ ] **Step 4: Implement the minimal production-layout verifier**

Add a fixed manifest-path generator, read and compare all six raw byte sequences before parsing, validate the existing self-hash and gate identities, require a canonical existing dataset root distinct from the workspace, and resolve calibration images only beneath `dataset_root/images`. Preserve canonical list-byte, partition-order, final-test-disjointness, and image-content checks.

- [ ] **Step 5: Run focused tests and prove GREEN**

Run the Task 1 command again. Expected: all `test_l4_contract.py` tests pass, with platform-only symlink tests skipped when privilege is unavailable.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- tests/test_l4_contract.py src/pcb_defect/l4_contract.py
git commit -m "fix: verify production A100 parent layout"
```

---

### Task 2: Dataset-aware benchmark and resume contract

**Files:**
- Modify: `tests/test_benchmark.py`
- Modify: `src/pcb_defect/benchmark.py`

**Interfaces:**
- Consumes: Task 1 `verify_l4_inputs(repo, workspace, dataset_root, identity)`
- Produces: `benchmark(repo, workspace, dataset_root, identity, *, warmup, cycles) -> Path`
- Produces: `benchmark_is_complete(repo, workspace, dataset_root, identity, report) -> bool`
- CLI: required `--dataset PATH`

- [ ] **Step 1: Add CLI and resume-path failing tests**

Assert the parser rejects a command without `--dataset`, a benchmark passes the resolved dataset root to `verify_l4_inputs`, and an existing report is not accepted when the current dataset root is wrong or calibration bytes have changed.

Use the real function boundary rather than asserting only on parser internals:

```python
result = benchmark_module.main([
    "--repo", str(repo), "--workspace", str(workspace),
    "--dataset", str(dataset_root), *identity_arguments,
])
assert result == 0
```

- [ ] **Step 2: Run the focused tests and prove RED**

```powershell
$env:CUDA_VISIBLE_DEVICES=''
uv run --locked --extra train --group eval pytest tests/test_benchmark.py -q
```

Expected: signature and missing-argument failures at the dataset propagation boundary.

- [ ] **Step 3: Implement dataset propagation**

Add required `--dataset`, resolve it once in `main`, and pass it through fresh benchmark verification, completed-report reuse, and `benchmark_is_complete`. Do not add a default or infer the path from Drive environment state.

- [ ] **Step 4: Run focused plus contract tests**

```powershell
$env:CUDA_VISIBLE_DEVICES=''
uv run --locked --extra train --group eval pytest tests/test_benchmark.py tests/test_l4_contract.py -q
```

Expected: PASS with no CUDA initialization.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- tests/test_benchmark.py src/pcb_defect/benchmark.py
git commit -m "fix: bind L4 benchmark to dataset root"
```

---

### Task 3: Dataset-aware package and rendered notebook

**Files:**
- Modify: `tests/test_l4_package.py`
- Modify: `tests/test_l4_handoff.py`
- Modify: `src/pcb_defect/l4_package.py`
- Modify: `notebooks/deployment_benchmark_l4.ipynb`

**Interfaces:**
- Consumes: Task 1 verified `manifest_path`
- Consumes: Task 2 `benchmark_is_complete(repo, workspace, dataset_root, identity, report)`
- Produces: `collect_l4_files(repo, workspace, dataset_root, identity) -> list[Path]`
- Produces: `create_or_verify_l4_package(repo, workspace, dataset_root, output_root, identity) -> Path`
- CLI: required `--dataset PATH`
- Notebook constant: `PARENT_DATASET = DRIVE_ROOT / 'dataset' / 'pcb'`

- [ ] **Step 1: Add package inventory failing tests**

Assert the package contains the verified canonical path:

```python
assert Path("runs/grouped/seed42/inputs/paired_split_manifest.json") in files
assert Path("inputs/paired_split_manifest.json") not in files
```

Assert package creation and existing-package reuse both fail when `dataset_root` is missing, wrong, or mutated, and that no pixel suffix enters the inventory.

- [ ] **Step 2: Add notebook AST and command failing tests**

Parse every code cell and require `PARENT_DATASET` to flow to initial parent verification, embedded input verification, benchmark verification, benchmark CLI, package CLI, and package verification. Require the notebook to remain unexecuted and free of `PASTE_` placeholders.

- [ ] **Step 3: Run Task 3 tests and prove RED**

```powershell
$env:CUDA_VISIBLE_DEVICES=''
uv run --locked --extra train --group eval pytest tests/test_l4_package.py tests/test_l4_handoff.py -q
```

Expected: current root-manifest inventory and missing dataset arguments fail.

- [ ] **Step 4: Implement package and notebook propagation**

Use `verified.parent.manifest_path.relative_to(workspace)` in the inventory. Add required `--dataset` and pass it through collection, creation, verification, and resume. Update all notebook embedded scripts and commands to receive the same immutable `PARENT_DATASET`; do not copy or rewrite parent files.

- [ ] **Step 5: Run Tasks 1-3 focused tests**

```powershell
$env:CUDA_VISIBLE_DEVICES=''
uv run --locked --extra train --group eval pytest tests/test_l4_contract.py tests/test_benchmark.py tests/test_l4_package.py tests/test_l4_handoff.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- tests/test_l4_package.py tests/test_l4_handoff.py src/pcb_defect/l4_package.py notebooks/deployment_benchmark_l4.ipynb
git commit -m "fix: package production L4 parent evidence"
```

---

### Task 4: Full verification, new handoff, review, and public fast-forward

**Files:**
- Modify only if verification finds a scoped defect: files already listed in Tasks 1-3
- Generate ignored artifact: `dist/colab-handoff-l4-<new-runner-prefix>/`
- Update ignored receipt: `.superpowers/sdd/2026-08-04-l4-private-benchmark-handoff/progress.md`

**Interfaces:**
- Consumes: corrected clean feature HEAD and the existing immutable parent identities
- Produces: exact three-file L4 handoff and a `kuotunyu`-only public `main`

- [ ] **Step 1: Run full CPU verification**

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:NVIDIA_VISIBLE_DEVICES='none'
$env:YOLO_AUTOINSTALL='false'
$env:ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS='1'
uv sync --locked --no-editable --extra train --group eval --reinstall-package pcb-defect
uv run --locked --extra train --group eval pytest -q
uv run --locked --extra train --group eval ruff check .
uv run --locked --extra train --group eval ruff format --check .
uv build
```

Expected: zero failures, zero lint findings, formatting clean, wheel and sdist created only in ignored build output.

- [ ] **Step 2: Run safety and identity gates**

Verify clean tracked status, no `.env` access, no tracked model/export/result/pixel/log artifacts, no unresolved notebook placeholders or outputs, exact `kuotunyu` author/committer on every new commit, and zero co-author trailers.

- [ ] **Step 3: Generate a new stage-specific handoff**

Use `python -m pcb_defect.l4_handoff` with the reviewed parent experiment, deployment-gate, checkpoint, and ONNX hashes. Require a new runner-derived directory containing exactly:

```text
deployment_benchmark_l4.ipynb
handoff_manifest.json
pcb-defect-source.bundle
```

- [ ] **Step 4: Audit the generated handoff from a clean bundle clone**

Verify bundle SHA-256, `git bundle verify`, detached clean clone, notebook AST, exact rendered identities, dataset-root propagation, manifest inventory, no outputs, and CPU-safe `--help` for the benchmark and package CLIs.

- [ ] **Step 5: Obtain independent code and artifact review**

Review against the approved spec. Critical and Important findings must be fixed with focused tests before continuing; rerun the full CPU verification and artifact audit after any correction.

- [ ] **Step 6: Publish a single-contributor fast-forward commit**

Create a `kuotunyu` author/committer commit whose parent is the current public `main` and whose tree equals the corrected verified feature HEAD. Push only that commit to `refs/heads/main`; do not use `--all`, `--mirror`, tags, or legacy refs.

- [ ] **Step 7: Verify GitHub state**

Use authenticated and anonymous checks to confirm public visibility, default `main`, only one branch, zero tags, corrected tree, all reachable commits authored and committed by `kuotunyu`, no co-author trailers, and Contributors API exactly `[{"login":"kuotunyu"}]`.

- [ ] **Step 8: Hand off the corrected Colab files**

Report the new Drive directory and instruct the user to upload only the new three files, keep the existing A100 workspace and dataset, start a fresh Colab L4 runtime, and download both the final ZIP and `.sha256` after `L4 HANDOFF COMPLETE`.

