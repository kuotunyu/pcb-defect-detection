# Release checklist

This checklist records the completed metadata-only portfolio release. It separates code readiness
from scientific evidence and does not imply that model binaries or a hosted endpoint are public.

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
- [x] L4 PyTorch/ORT CUDA/TensorRT FP16 benchmark uses calibration images only and passes
  aggregate calibration fidelity; verified public metadata and complete raw timings are recorded
  in [`reports/benchmark_l4.json`](../reports/benchmark_l4.json) and
  [`reports/benchmark_l4_raw.json`](../reports/benchmark_l4_raw.json).
- [x] The frozen strict per-box prediction-parity gate failed for both exported backends; the
  failure, thresholds, summaries, and pseudonymized per-image evidence are retained in
  [`reports/backend_parity_l4.json`](../reports/backend_parity_l4.json) without relaxing the gate.
- [x] Final result ZIP and sidecar SHA-256 are returned from Drive.

## Identity and official publication

- [x] Official GitHub namespace is independently verified as `kuotunyu/pcb-defect-detection`.
- [x] This candidate is based on official `main`; the unrelated prototype history is not merged.
- [x] Official push/review is completed.
- [x] The metadata-only portfolio release is published without dataset pixels, weights, exports,
  engines, result packages, or secrets.

## Intentional non-goals

- Dataset, fine-tuned weights, ONNX exports, and TensorRT engines are not redistributed.
- Hugging Face publication and hosted inference are intentional non-goals for this portfolio.
- Factory-line generalization, production throughput, and AOI acceptance/SLA claims are out of
  scope.
- Backend prediction equivalence is not claimed because the strict per-box prediction-parity gate failed.

The candidate's reachable history contains only the `kuotunyu` author/committer identity, has no
co-author trailers, and excludes removed dataset pixels. The unrelated prototype history remains
available for private audit but is not part of this clean promotion. The clean promotion is published on official `main`; the release is intentionally metadata-only.
