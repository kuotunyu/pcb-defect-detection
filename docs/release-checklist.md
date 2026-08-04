# Release checklist

This checklist separates code readiness from scientific/release evidence. A checked CPU item does
not imply the Colab or account-migration items are complete.

## CPU and clean-clone gates

- [x] Dependency graph is locked in `uv.lock`; release installs are non-editable and pinned to
  Python 3.11 for clean-clone portability.
- [x] CI installs base/dev dependencies only; no dataset, weight, API key, torch, or GPU required.
- [x] CI runs lint, format, tests, and wheel/sdist build.
- [x] Dataset and split manifests are portable and content-addressed.
- [x] Protocol tests cover exact final-test exclusion, sibling exposure, equal train size/class
  counts, deterministic hashes, and 10-board fail-closed behavior.
- [x] Claim registry blocks legacy failed export/parity evidence from deployment claims.
- [x] Base initialization is pinned to an immutable release URL, byte count, and SHA-256; download
  and preflight fail closed on any mismatch.
- [x] Current candidate tree excludes dataset pixels, pixel-derived examples, weights, exports, and
  engines.
- [x] App refuses floating or hash-mismatched model artifacts.

## Colab evidence

- [x] A100 clean-runtime, data/hash, tiny-train, resume, and speed gates pass.
- [x] Six runs complete with matching run records and checkpoint hashes.
- [x] Deployment checkpoint is selected from grouped validation before final-test access.
- [x] One-shot common final evaluation completes and reports three-seed mean/std and paired image
  bootstrap intervals.
- [x] Calibration-only ONNX fidelity and standalone parity gates pass.
- [ ] L4 PyTorch/ORT CUDA/TensorRT FP16 raw-timing benchmark uses calibration images only and
  passes calibration fidelity.
- [x] Final result ZIP and sidecar SHA-256 are returned from Drive.

## License, identity, and official migration

- [ ] Upstream dataset redistribution/training/weight-release rights are confirmed in writing.
- [ ] Fine-tuned weight and export obligations are reviewed against both dataset and Ultralytics
  terms.
- [ ] Official GitHub and Hugging Face namespaces are selected and independently verified.
- [ ] Official publication uses a clean reviewed snapshot, not the legacy Git history.
- [ ] Official model artifact and immutable Hub revision are written into the app contract only
  after the deployment gate passes.

The legacy history is deliberately not rewritten in this branch. It contains prior personal/test
identity, AI co-author trailers, and removed dataset images. The configured remote is a practice
remote and is not an official publication target. These are release blockers, not reasons to alter
or conceal the audit trail.
