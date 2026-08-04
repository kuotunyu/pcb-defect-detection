# Ultralytics AutoInstall Hermeticity Design

## Status

Approved direction: keep the current Linux `onnxruntime-gpu==1.26.0` lock, preserve CPU deployment parity, and prevent Ultralytics from mutating the environment.

## Evidence and root cause

The real A100 parity probe passed independently: 60/60 images, zero failures, minimum IoU 1.0, maximum confidence delta 0.0, and exact parent/checkpoint/ONNX provenance. Its log also showed an unintended package installation attempt:

1. The Linux lock installs the `onnxruntime-gpu==1.26.0` distribution.
2. The standalone runtime constructs first and imports its `onnxruntime` module from that GPU distribution.
3. The Ultralytics ONNX reference is explicitly CPU-bound, so Ultralytics 8.4.89 checks for the distribution named `onnxruntime`, not the interchangeable GPU distribution.
4. Ultralytics therefore attempts to install CPU `onnxruntime==1.28.0` into the locked environment.
5. The running process continues to use the already-imported 1.26.0 module, so the completed probe remains valid, but the environment is mutated for any later process or restart.

Installing both CPU and GPU distributions is not acceptable because they provide the same Python module tree. Running the reference on CUDA is also rejected because runtime parity is intentionally CPU-to-CPU.

## Decision

Use a defense-in-depth hermetic runtime policy:

- set `YOLO_AUTOINSTALL=false` before any Ultralytics import;
- set `ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1` because the repository performs a stricter lock check itself and Ultralytics' CPU-only distribution-name check is a known false negative for this environment;
- verify the exact ORT distribution, imported module version, and providers before an ONNX runtime stage;
- verify the same state after the stage so any environment mutation fails the run;
- retain `onnxruntime-gpu==1.26.0` on Linux and `onnxruntime==1.26.0` on Windows;
- do not relax parity thresholds or change the two CPU backends.

## Components

### Runtime contract module

Add `src/pcb_defect/runtime_contract.py` with three focused interfaces:

- `configure_hermetic_ultralytics() -> None` forces both environment controls.
- `onnxruntime_state(require_cuda_provider: bool = False) -> dict[str, object]` returns and validates distribution names/versions, imported module version, and providers.
- a small CLI prints the validated state as JSON for notebook before/after gates.

On Linux, the CPU distribution must be absent and `onnxruntime-gpu` must equal 1.26.0. On Windows, the inverse applies. The imported `onnxruntime.__version__` must equal 1.26.0 and `CPUExecutionProvider` must exist. `--require-cuda-provider` additionally requires `CUDAExecutionProvider` for A100/L4 environment gates.

### Deployment entry points

Call `configure_hermetic_ultralytics()` before the first possible Ultralytics import in deployment and deployment-probe execution paths. Snapshot the validated ORT state before and after same-ONNX parity and fail if it changes.

The already completed probe will not be rerun. The fix is for the new full A100 handoff and all later deployment/benchmark runs.

### Notebook templates

In the first code cell of the A100, probe, and L4 templates, set both environment controls before invoking repository commands. Continue using locked, non-editable reinstall.

The full A100 and L4 notebooks run the runtime-contract CLI before the relevant ONNX stage and again afterward. The preflight requires CUDA availability on the GPU runtimes, while the actual parity reference remains CPU-bound.

## Failure behavior

- Missing, duplicated, wrong-version, or conflicting ORT distributions fail before inference.
- A missing CPU provider fails every runtime stage.
- A missing CUDA provider fails only an explicitly GPU-required environment gate.
- Any before/after state change fails before a result package is accepted.
- Ultralytics cannot auto-install or run its conflicting distribution-name check.
- Logs retain the exact validated state and failure reason.

## Tests

Use TDD to cover:

- environment controls are forced before importing Ultralytics;
- Linux accepts only GPU 1.26.0 and rejects CPU presence, wrong versions, missing CPU provider, or missing required CUDA provider;
- Windows accepts only CPU 1.26.0 and rejects GPU presence;
- deployment parity rejects a changed before/after state;
- all three notebook templates set both controls before commands and force package reinstall;
- A100/L4 notebook contracts include before/after runtime state gates;
- rendered notebooks remain unexecuted, placeholder-free, valid, and compilable.

Then run the complete CPU-safe suite, Ruff checks, regenerate an immutable handoff, verify every hash, and test a one-commit clean bundle clone.

## Non-goals

- no dependency-group split;
- no simultaneous CPU/GPU ORT installations;
- no Ultralytics source patching;
- no parity threshold or backend change;
- no new training before the regenerated handoff passes local release gates;
- no rerun of the already verified short probe.

## Acceptance criteria

The change is complete when tests prove the policy, source and clean-clone verification pass, the regenerated notebooks contain the gates, and no code path can emit `attempting AutoUpdate` or install a second ORT distribution during the intended workflow. The new complete A100 handoff is then ready for user execution.
