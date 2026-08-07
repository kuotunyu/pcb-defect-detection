# Private L4 deployment benchmark

This is private/unreleased evidence, summarized from a locally verified result package. The
package filename is `paired-results-l4-9e3a1ed5827a-runner-4d533bcdbf31.zip` and its SHA-256 is
`d988fc6dad3f97d29a52a92cf8024f4919377e9d99011be837bd42a297f85d30`. The package itself,
checkpoint, ONNX export, TensorRT engine, calibration images, and command log are not committed
to this repository.

## Result

The run completed on an NVIDIA L4 with CUDA 12.6, cuDNN 91002, driver `580.82.07`, TensorRT
runtime `10.13.3.9`, Python `3.11.15`, and ONNX Runtime GPU `1.26.0`. It used 60 calibration
images, batch size 1, 30 warmup iterations, and four 60-image cycles. Timings include predecoded
PIL image preprocessing, inference, postprocessing, and CUDA synchronization.

| Backend | Precision | p50 (ms) | p95 (ms) | FPS from p50 | Timed runs |
|---|---|---:|---:|---:|---:|
| PyTorch | FP32 | 60.562907999951676 | 62.54312460007441 | 16.51175666797238 | 240 |
| ONNX Runtime CUDA | FP32 | 20.05252949993519 | 20.71104239996657 | 49.869020265160664 | 240 |
| TensorRT | FP16 | 50.88519949993042 | 52.37864604996503 | 19.65207977619047 | 240 |

TensorRT FP16 passed the calibration split only fidelity gate: source mAP50-95 was
`0.12125399555038195`, TensorRT FP16 was `0.10714582797330278`, and the delta was
`-0.014108167577079167`, within the absolute `0.02` threshold. This does not evaluate the
one-shot final test and must not be interpreted as production throughput, factory-line latency,
or generalization evidence.

## Provenance and boundaries

The runner SHA is `4d533bcdbf3152e71ec2e617dd5d2073ad7666e3`; the parent experiment SHA is
`9e3a1ed5827ac3759cbb15632f041e3e5c183b51`. The deployment-gate, selected-checkpoint, and ONNX
SHA-256 values are retained in [`benchmark_l4.json`](benchmark_l4.json). The TensorRT engine is
not portable across arbitrary GPU/software combinations and is intentionally not tracked.

The raw package records its generic `tensorrt` environment field as `not-installed`. That is a
historical recording limitation, not a rewrite of the raw evidence: Task 2 corrected the runner
source to record the installed `tensorrt-cu12` distribution. TensorRT runtime `10.13.3.9` was
present and used by this completed benchmark.

No public model, public checkpoint, public ONNX export, no hosted demo, or deployment endpoint is
claimed. Distribution and licensing approval remain separate release gates.
