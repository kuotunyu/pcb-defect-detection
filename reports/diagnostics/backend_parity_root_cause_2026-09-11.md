# Backend prediction parity: root-cause diagnosis (2026-09-11)

Scope: the failed strict per-box PyTorch-reference parity gate recorded in
[`reports/backend_parity_l4.json`](../backend_parity_l4.json) (ONNX Runtime CUDA FP32 and
TensorRT FP16 each failed 40/60 calibration images). This document records what was verified,
what was ruled out, what remains unverified, and where to stop. It does not alter any historical
report, threshold, model, or result. Source tree at diagnosis time: `main` @ `19c973a` plus the
runner fix committed together with this document.

## Conclusion

The failure is dominated by an **input-geometry mismatch inside the parity harness**, not by
ONNX export loss, ONNX Runtime, TensorRT, or FP16 precision.

- The L4 runner (`src/pcb_defect/benchmark.py`) called Ultralytics `predict()` on the PyTorch
  checkpoint with library defaults. In Ultralytics 8.4.89, `Model.predict()` injects `rect=True`
  (`engine/model.py:528`), and `BasePredictor.pre_transform()` enables the minimal-rectangle
  letterbox only for `format == "pt"` (`engine/predictor.py:197-203`, `LetterBox(auto=True)`,
  `data/augment.py:120-121`).
- Every calibration image is 3034x1586. Under `auto=True` the PyTorch reference received a
  **1x3x352x640** tensor (17 padding rows); the ONNX export, the TensorRT engine, and the
  standalone runtime are fixed at **1x3x640x640** (305 padding rows). Resize ratio and resized
  content are identical; only the padding extent, and therefore the feature-map grid and border
  context, differ.
- The two candidates agree with each other far better than either agrees with the reference,
  which is the signature of a reference-side difference rather than a backend-side one.

| Evidence (from the committed L4 JSON) | ORT CUDA FP32 | TensorRT FP16 |
|---|---:|---:|
| Matched IoU vs PyTorch, median / min | 0.9426 / 0.8410 | 0.9383 / 0.8451 |
| Confidence delta vs PyTorch, median / max | 0.0406 / 0.1984 | 0.0408 / 0.1961 |
| Failed images: count mismatch only / IoU < 0.9 / delta > 0.15 | 32 / 7 / 2 | 31 / 8 / 2 |
| ORT-vs-TRT IoU gap on the same PyTorch box, median / max | 0.0068 / 0.0389 | (same pair) |
| ORT-vs-TRT confidence-delta gap, median / max | 0.0016 / 0.0067 | (same pair) |
| Images where ORT and TRT disagree on candidate count / pass flag | 1 / 0 of 60 | (same pair) |

## Verified facts

1. **Artifact identity is consistent.** The parity JSON, `benchmark_l4.json`, and
   `deployment_gate.public.json` bind the same checkpoint (`44646b13...`) and ONNX
   (`b62590a1...`); `verify_l4_inputs` hash-checks both before the run.
   `configs/backend_parity.yaml` and `configs/deployment_gate.yaml` at HEAD hash to the
   `config_sha256` values recorded in the evidence (`ef05243a...`, `1c279957...`).
2. **Calibration inputs are reproducible locally.** All 60 frozen calibration stems exist in the
   local converted dataset with SHA-256 equal to the protocol manifest; all are 3034x1586.
3. **Preprocessing and coordinate inversion of the standalone path match the Ultralytics non-PT
   path exactly** (historical same-ONNX gate: 60/60, IoU 1.0, delta 0.0), and `scale_boxes`
   uses the same `round(pad - 0.1)` convention as `e2e_onnx.postprocess`.
4. **Head semantics are the same on both sides**: the YOLO26 end2end head emits `(1, 300, 6)`
   xyxy rows; Ultralytics applies only a confidence filter for end2end models
   (`utils/nms.py:54-55`), as does `e2e_onnx.postprocess`.
5. **Mechanism check on this machine**
   ([`backend_parity_mechanism_check_2026-09-11.json`](backend_parity_mechanism_check_2026-09-11.json),
   produced by `scripts/diagnostics/backend_parity_mechanism_check.py`). Because the L4
   checkpoint is not present locally, the check uses the local July prototype pair (YOLO26s
   checkpoint `9d2d1b64...`, its ONNX export `057c61db...`) on the same 60 SHA-verified
   calibration images, the locked Ultralytics 8.4.89 / torch 2.12.1+cu126 / ONNX Runtime 1.26.0
   stack, an RTX 4090 for PyTorch and the CPU provider for ORT. It is a mechanism demonstration,
   **not** a reproduction of the L4 numbers.

   | Comparison (frozen evaluator and thresholds) | Ref / cand / matched | Min IoU | Max delta | Failed |
   |---|---:|---:|---:|---:|
   | A: PyTorch default `predict()` (observed 1x3x352x640) vs C: standalone ORT 640x640 | 100 / 99 / 98 | 0.8290 | 0.1237 | 7 / 60 |
   | A vs B: same weights, `predict(rect=False)` (observed 1x3x640x640) | 100 / 99 / 98 | 0.8294 | 0.1237 | 7 / 60 |
   | B vs C: same geometry | 99 / 99 / 99 | 0.9991 | 0.0031 | 0 / 60 |
   | D: patched runner helper vs C | 99 / 99 / 99 | 0.9991 | 0.0031 | 0 / 60 |
   | D vs B | 99 / 99 / 99 | 1.0000 | 0.0000 | 0 / 60 |

   The geometry-only comparison (A vs B) reproduces the reference-vs-export discrepancy almost
   exactly, and with equal geometry the export matches PyTorch to within numerical noise.

## Hypotheses ruled out

- Wrong or mismatched source model / export: excluded by the hash bindings in (1).
- Resize, colour, or normalisation differences: the standalone path was already shown to be
  identical to the Ultralytics ONNX path (3); the resize ratio and resized pixels are identical
  in both geometries.
- Coordinate inversion or confidence-filter bugs in `e2e_onnx.postprocess`: excluded by (3),
  (4), and by B vs C above.
- FP16 or TensorRT-specific error: ORT FP32 fails identically to TRT FP16; their mutual gap is an
  order of magnitude below the thresholds.
- Evaluator (`compare_backend_predictions` / `greedy_match`) defect: the same evaluator returns
  60/60 for B vs C and D vs C.

## Fix applied

`src/pcb_defect/benchmark.py` now pins every Ultralytics-driven inference (PyTorch and TensorRT,
timing and parity) to the export input contract via
`ULTRALYTICS_INPUT_CONTRACT = {"imgsz": 640, "rect": False}`. Thresholds, the evaluator, the
report schema, and the standalone ONNX path are unchanged. Regression tests in
`tests/test_benchmark.py` fail against the previous runner and pass with the fix:

```bash
uv run --locked --extra app pytest tests/test_benchmark.py -k input_contract -q
```

## Not verified

- The exact L4 checkpoint and ONNX are not on this machine; they exist only in the private Drive
  workspace and result packages. The recorded 40/60 failure has therefore **not** been re-run,
  and the patched runner has not been exercised on that model. The historical evidence and the
  "parity failed" statements in the docs remain correct for the run as performed.
- The runner commit `fe9005d7` is not in the public history; behavioural equivalence to HEAD was
  established via config hashes and the recorded protocol scope, not via a source diff.
- Whether the weaker L4 model (calibration mAP50-95 0.12) reaches 60/60 under equal geometry, or
  leaves a small residual, can only be answered by a re-run.

## Related observations (not changed)

- The A100 aggregate fidelity gate (`deployment.py::_validate_model`) has the same asymmetry:
  `Model.val()` defaults to `rect=True` for `.pt` while the validator forces `rect=False` for
  non-PT formats (`engine/validator.py:213-214`). The recorded -0.0128 / -0.0145 mAP50-95 deltas
  therefore include a geometry component and understate export fidelity. The gate passed and is
  left untouched.
- The published PyTorch FP32 L4 latency (p50 60.86 ms) was measured on a 1x3x352x640 input; it
  is not the same workload as the 640x640 ORT/TRT timings. A re-run with the fixed runner will
  change the PyTorch timing.
- `e2e_onnx.postprocess` keeps rows with `score >= conf` while Ultralytics keeps `score > conf`;
  this only matters at exactly 0.25 and is not a contributor here.

## Reproduction

Static geometry (pure Python, mirrors `ultralytics/data/augment.py::LetterBox`):
3034x1586 -> ratio 0.21094 -> resized 640x335 -> `auto=True` pads 8/9 rows (352x640),
`auto=False` pads 152/153 rows (640x640).

Mechanism check (requires the locked train+eval environment, a checkpoint/ONNX pair, and the
converted dataset; run from the repository root):

```bash
uv run --locked --extra train --group eval python scripts/diagnostics/backend_parity_mechanism_check.py --repo . --pt weights/grouped/best.pt --onnx exports/best.onnx --out reports/diagnostics/backend_parity_mechanism_check_<date>.json --device 0
```

## Recommendation and stopping point

- Worth one more bounded step, only if a Colab L4 session is acceptable: re-run the L4 benchmark
  with the fixed runner through the existing handoff flow on the same parent workspace, producing
  a new private package and new public metadata. Expected outcome: parity passes or leaves a
  residual of the ORT-vs-TRT scale. Timing tables would be regenerated, not edited.
- If no further GPU time is to be spent: stop here. The failure has a checkable explanation, the
  runner defect is fixed with regression coverage, and the published claims stay truthful as
  written. Do not attempt to reduce the ORT-vs-TRT residual; it is inherent runtime/precision
  noise well inside the frozen thresholds.
