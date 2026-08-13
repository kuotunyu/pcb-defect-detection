# Release checklist

This checklist records the completed technical gates for a metadata-only portfolio release candidate.
It separates code readiness from scientific evidence and does not imply that model binaries or a
hosted endpoint have been published. External source-release facts are recorded explicitly below.

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

## Identity and external publication

- [x] Official GitHub namespace is independently verified as `kuotunyu/pcb-defect-detection`.
- [x] This candidate is based on official `main`; the unrelated prototype history is not merged.
- [x] The annotated `v0.1.0` tag resolves to
  `56c086206eab9be1a9c6a4e36410fd13ed42f5ec`; the corresponding
  [GitHub Release](https://github.com/kuotunyu/pcb-defect-detection/releases/tag/v0.1.0)
  publishes the deterministic source-and-metadata research archive (1,505,971 bytes; SHA-256
  `21abbe3c71c5f7b962a8c33a8bc649dbe98757199a6ae17b5a6af0bbe27998e1`).
- [x] Zenodo version DOI [`10.5281/zenodo.21877497`](https://doi.org/10.5281/zenodo.21877497)
  and all-versions DOI [`10.5281/zenodo.21877496`](https://doi.org/10.5281/zenodo.21877496) resolve
  to the same 1,505,971-byte archive recorded above.

## v0.2.0 portfolio release preparation

- [x] Final desktop and mobile workstation screenshots are captured from the running app at
  exactly 1440×900 and 390×844 and are embedded in the README.
- [x] The responsive UI preserves the recorded-evidence mode, readable typography, keyboard
  semantics, and visible `BLOCKED` Promotion Gate.
- [x] Software, citation, and Zenodo ingestion metadata identify version `0.2.0` and preserve the
  immutable `v0.1.0` DOI as historical evidence rather than reusing it for the new version.
- [x] The scientific claims and redistribution boundary are unchanged: no dataset media, weights,
  exports, engines, or hosted inference are added.
- [x] The annotated `v0.2.0` tag resolves to
  `7fc1777d306584fc1f3ffe0c05989296370fe6df`; the corresponding
  [GitHub Release](https://github.com/kuotunyu/pcb-defect-detection/releases/tag/v0.2.0)
  publishes the deterministic research package and checksum sidecar (1,699,878 bytes; SHA-256
  `89d82a6ab8737193f8c59614d2a04c68f07b02fca3bc7d3ee7178c56ff882f29`).
- [x] Zenodo version DOI [`10.5281/zenodo.21912370`](https://doi.org/10.5281/zenodo.21912370)
  preserves concept DOI [`10.5281/zenodo.21877496`](https://doi.org/10.5281/zenodo.21877496)
  and publishes the same 1,699,878-byte archive with the verified SHA-256 above.

## Intentional non-goals

- Dataset, fine-tuned weights, ONNX exports, and TensorRT engines are not redistributed.
- Hugging Face publication and hosted inference are intentional non-goals for this portfolio.
- Factory-line generalization, production throughput, and AOI acceptance/SLA claims are out of
  scope.
- Backend prediction equivalence is not claimed because the strict per-box prediction-parity gate failed.

The candidate's reachable history contains only the `kuotunyu` author/committer identity, has no
co-author trailers, and excludes removed dataset pixels. The unrelated prototype history remains
available for private audit but is not part of this clean promotion. The `v0.1.0` source tag,
GitHub Release, and Zenodo record are intentionally metadata-only and do not publish dataset
pixels, labels, model weights, ONNX exports, TensorRT engines, or hosted inference.
