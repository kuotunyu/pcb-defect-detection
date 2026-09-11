# Corrected NVIDIA L4 re-run (runner `2abe78fe2b54`)

This directory records public, path-free metadata derived from a locally verified private and
unreleased result package produced on 2026-09-11. It re-runs the L4 benchmark and the frozen strict
per-box prediction-parity gate after the runner correction described in
[`backend_parity_root_cause_2026-09-11.md`](../diagnostics/backend_parity_root_cause_2026-09-11.md).
The historical first session in [`benchmark_l4.md`](../benchmark_l4.md) and its JSON files are
retained unchanged.

## What changed and what did not

The only behavioural change from the first session is that every Ultralytics-driven inference
(PyTorch and TensorRT, timing and parity) is pinned to the export input contract `imgsz=640`,
`rect=False`. In the first session the PyTorch reference received a rectangular 1x3x352x640
letterbox while both exported backends ran at 1x3x640x640.

Unchanged: the parent experiment (`9e3a1ed5827a`), deployment gate (`466bf152…`), checkpoint
(`44646b13…`), ONNX export (`b62590a1…`), the 60 calibration images (identical image-list and
image-content hashes), the parity configuration (`ef05243a…`) and thresholds, the evaluator, and
the L4 software stack. The TensorRT FP16 engine was rebuilt (`5c8274e6…`); engine bytes remain
private and non-portable.

## Strict per-box prediction parity

The gate requires every detection to match by class at IoU ≥ `0.5`, minimum IoU ≥ `0.9`, maximum
confidence delta ≤ `0.15`, and zero unmatched detections, for both candidates. Thresholds were
frozen before the first session and were not changed.

| Candidate | Ref / candidate boxes | Matched | Unmatched ref / candidate | Min IoU | Max confidence delta | Failed images | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| ONNX Runtime CUDA FP32 | 62 / 62 | 62 | 0 / 0 | 0.9989099779610657 | 0.0012852251529693604 | 0 / 60 | Passed |
| TensorRT FP16 | 62 / 61 | 61 | 1 / 0 | 0.9585281265862317 | 0.007691502571105957 | 1 / 60 | **Failed** |

The overall gate **failed**, because it requires both candidates to pass. ONNX Runtime CUDA FP32
reproduced every PyTorch detection on all 60 calibration images within the frozen tolerances.
TensorRT FP16 matched 61 of 62 reference detections; the only failure is one reference detection
on `image_028` whose confidence exceeded the `0.25` threshold by less than `0.002`. ONNX Runtime
kept that detection. TensorRT FP16 confidence deviations on the 61 matched detections ranged from
`-0.0042` to `+0.0077`, so the missing detection is consistent with an FP16 score falling just
below the hard threshold. The candidate's sub-threshold score is not recorded, so this last point
is an inference rather than a measurement.

Per-image, pseudonymized evidence is in [`backend_parity_l4.json`](backend_parity_l4.json); it
excludes image paths, coordinates, and raw confidence values.

## Timing result

One NVIDIA L4 session, CUDA 12.6, cuDNN 91002, driver `580.82.07`, TensorRT `10.13.3.9`, ONNX
Runtime GPU `1.26.0`, batch 1, 30 warmup iterations, four cycles, interleaved rotating backend
order, 240 timed observations per backend. All three backends now run on a 640x640 input.

| Backend | Precision | p50 (ms) | p95 (ms) | Mean ± std (ms) | FPS from p50 | Runs |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | FP32 | 61.03977350005607 | 63.331417949893876 | 61.341780795844635 ± 2.017326584263651 | 16.382760660130586 | 240 |
| ONNX Runtime CUDA | FP32 | 20.1839745000143 | 20.822142999963944 | 20.221899662496412 ± 0.35809289344429635 | 49.544256013566184 | 240 |
| TensorRT | FP16 | 50.64236800001254 | 52.65300454998396 | 50.87315350833895 ± 1.147111874142175 | 19.74631202079161 | 240 |

The PyTorch p50 changed by less than 0.2 ms from the first session, so the first session's
rectangular PyTorch input did not materially affect the published timing comparison. Complete raw
observations are in [`benchmark_l4_raw.json`](benchmark_l4_raw.json).

## Aggregate fidelity

TensorRT FP16 calibration mAP50-95 was `0.10711279775211151` against the recorded source value
`0.12125399555038195`, a delta of `-0.014141197798270444`, within the frozen absolute `0.02`
threshold. The source and ONNX values are carried over from the A100 deployment gate.

## Provenance and boundaries

| Item | Value |
|---|---|
| Private package | `paired-results-l4-9e3a1ed5827a-runner-2abe78fe2b54.zip`, 23,921,329 bytes |
| Package SHA-256 | `8e22c52f9a6a184353631509aba011bfd23e319fb67f00d3f17a6a8d6b28ed61` |
| Private raw report SHA-256 | `2b0473ef17daec2dd99a27061c2940a7384b8f8aea07857729649915eddfa167` |
| Runner snapshot Git SHA | `2abe78fe2b54fb8015dbcc97ca37ecf0f7a6a5a5`, built from source commit `c247603e` |
| Parent experiment Git SHA | `9e3a1ed5827ac3759cbb15632f041e3e5c183b51` |

This is one calibration-only session. It is not a production SLA, a final-test benchmark, or an
estimate of between-session, machine, driver, or thermal variance. The runner correction was made
after the first session failed; it changes only the reference input geometry and leaves the
thresholds and evaluator untouched. No public model, checkpoint, ONNX export, TensorRT engine,
hosted demo, or deployment endpoint is claimed.
