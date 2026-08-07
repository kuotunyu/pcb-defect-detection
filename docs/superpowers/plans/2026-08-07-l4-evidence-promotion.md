# L4 Evidence Promotion Plan

## Task 2: Record the CUDA TensorRT distribution accurately

The L4 runner installs `tensorrt-cu12`, while the generic environment recorder currently
looks up only the distribution name `tensorrt`. Preserve the stable report key `tensorrt`, but
fall back to `tensorrt-cu12` when the generic distribution is absent. Add a unit test that
proves the fallback and preserves the generic distribution when it is present.

Files:

- Modify `src/pcb_defect/experiment.py`.
- Modify `tests/test_experiment.py`.

No GPU, training, benchmark, or external publish is allowed.

## Task 3: Promote metadata-only L4 evidence

Create a metadata-only summary from the locally verified L4 result package. Do not add the
ZIP, checkpoint, ONNX, TensorRT engine, dataset, images, or logs to Git. Update the public
README, claims registry, model card, release checklist, and contract tests so the L4 claim is
explicitly private/unreleased, calibration-only for fidelity and latency, and bounded by the
nonportable-engine and licensing constraints.

The summary must retain exact provenance and metrics from the verified package, including package
SHA-256, runner/experiment SHAs, deployment/checkpoint/ONNX hashes, L4 hardware/runtime, 60-image
calibration protocol, backend p50/p95 latency and FPS, and fidelity deltas. It must state that
the engine is not committed and that no public model or hosted demo is claimed.
