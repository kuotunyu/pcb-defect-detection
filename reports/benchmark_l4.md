# Verified NVIDIA L4 deployment evidence

This report records candidate public metadata, path-free and derived from a locally verified
private and unreleased result package. The source package is
`paired-results-l4-9e3a1ed5827a-runner-fe9005d77920.zip`, has 24,150,052 bytes, and has SHA-256
`482c3bc35d8069bc3301a34f483ed599206a488626e48f34de4c8c9b7619572b`. The package itself,
checkpoint, ONNX export, TensorRT engine, calibration images, and command logs are not committed.

## Timing result

The run used one NVIDIA L4 session with CUDA 12.6, cuDNN 91002, driver `580.82.07`, TensorRT
`10.13.3.9`, and ONNX Runtime GPU `1.26.0`. It evaluated the same 60 calibration images at batch
size 1 after 30 warmup iterations. Four cycles yielded 240 timed observations per backend.
Backend order rotated within an interleaved schedule. Each timing covers a predecoded PIL image,
preprocessing, inference, postprocessing, and explicit CUDA synchronization.

| Backend | Precision | p50 (ms) | p95 (ms) | Mean ± std (ms) | FPS from p50 | Runs |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | FP32 | 60.85868150000806 | 62.36269444993923 | 60.99905142917 ± 0.7251713341237158 | 16.431509447010736 | 240 |
| ONNX Runtime CUDA | FP32 | 20.277195000005577 | 20.87069180000185 | 20.307759533336405 ± 0.41710544001963573 | 49.316485835428665 | 240 |
| TensorRT | FP16 | 51.12191199998506 | 52.25180029992771 | 51.2870281708364 ± 1.565330754377219 | 19.561083709081387 | 240 |

ONNX Runtime CUDA FP32 was the fastest backend in this measured configuration. The complete 720
timing observations are committed in [`benchmark_l4_raw.json`](benchmark_l4_raw.json); the compact,
machine-readable summary is [`benchmark_l4.json`](benchmark_l4.json).

## Aggregate fidelity

The calibration mAP50-95 values were `0.12125399555038195` for the PyTorch source,
`0.1084447068676515` for ONNX, and `0.10671685845629254` for TensorRT FP16. TensorRT minus source
was `-0.014537137094089408`, within the frozen absolute `0.02` threshold, so the aggregate
fidelity gate passed.

## Strict per-box prediction parity

The strict per-box prediction-parity gate failed for both candidate backends. The
gate required all detections to match by class at IoU ≥ `0.5`, aggregate minimum IoU ≥ `0.9`,
maximum confidence delta ≤ `0.15`, and zero unmatched detections. Thresholds were frozen before
the run and were not relaxed afterward.

| Candidate | Ref / candidate boxes | Matched | Unmatched ref / candidate | Min IoU | Max confidence delta | Failed images | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| ONNX Runtime CUDA FP32 | 95 / 62 | 57 | 38 / 5 | 0.8410022500497364 | 0.1983642280101776 | 40 / 60 | **Failed** |
| TensorRT FP16 | 95 / 61 | 56 | 39 / 5 | 0.8451429620055757 | 0.1961173713207245 | 40 / 60 | **Failed** |

Aggregate mAP fidelity and per-box prediction equivalence answer different questions. Passing the
former does not override failure of the latter. Therefore this repository does not claim that
PyTorch, ONNX Runtime CUDA, and TensorRT produce equivalent prediction sets. Pseudonymized
per-image and per-match evidence is committed in
[`backend_parity_l4.json`](backend_parity_l4.json); it excludes image paths, coordinates, and raw
confidence values.

## Provenance and boundaries

The runner Git SHA is `fe9005d7792036460029a376bbd9f97d7159ed41`; the parent experiment SHA is
`9e3a1ed5827ac3759cbb15632f041e3e5c183b51`. The private raw report SHA-256 is
`6080de5237755444ed516e46fd903e20016b3fc562bbdf0c33dfcf90f4e718ee`. Dataset, protocol,
deployment-gate, checkpoint, ONNX, and TensorRT engine identities remain in the JSON evidence.

This is one calibration-only session and is not a production SLA, final-test benchmark,
factory-line throughput result, or estimate of machine/driver/thermal variance. Interleaving and
switching backend wrappers may influence cache state. The TensorRT engine is non-portable and
remains private. No public model, checkpoint, ONNX export, hosted demo, or deployment endpoint is
claimed.
