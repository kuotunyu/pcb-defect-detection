# Model card — paired grouped candidate

## Status

Unreleased, technically gate-passed candidate; distribution rights and official model publication remain pending.

## Model and provenance

- Architecture: Ultralytics YOLO26n.
- Source snapshot Git SHA: `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`.
- Dataset SHA-256: `8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`.
- Frozen protocol manifest SHA-256:
  `5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5`.
- Paired-training config SHA-256:
  `6ba44a0024884c11de37a29b294543c9736cb30b6e96b4a6d27dcb93ebcf185b`.
- Shared base initialization: YOLO26n `v8.4.0`, SHA-256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

These identities are bound in [`input_lock.json`](../reports/paired_a100/input_lock.json); package
receipt provenance is in
[`result_package_receipt.json`](../reports/paired_a100/result_package_receipt.json).

## Evaluation design

The frozen paired protocol trains grouped and leaky-control arms at seeds 42, 43, and 44. Both
arms use equal 513-image, class-matched training sets, shared validation and calibration splits,
and the same one-shot 30-image final test from board 08. The controlled difference is exposure to
predeclared same-board siblings. Aggregate metrics are three-seed mean and sample standard
deviation; paired intervals resample the 30 final-test images and do not measure between-board
variance. See [`final_metrics.json`](../reports/paired_a100/final_metrics.json) and
[`input_lock.json`](../reports/paired_a100/input_lock.json).

## Paired result

| Arm | mAP50, mean ± std | mAP50-95, mean ± std |
|---|---:|---:|
| Grouped | `0.6330 ± 0.1491` | `0.2882 ± 0.0654` |
| Leaky control | `0.8456 ± 0.0375` | `0.4008 ± 0.0252` |

Leaky control exceeded grouped mAP50 by 21.3 percentage points. The paired final-image bootstrap
F1 delta was `0.2546`, with 95% CI `[0.2102, 0.3005]` from 10,000 resamples. These are controlled
results for the frozen dataset and recipe, not a universal leakage or model-performance estimate.
Source: [`final_metrics.json`](../reports/paired_a100/final_metrics.json).

## Selected deployment candidate

Grouped seed 42 was selected before final-test access by highest grouped validation mAP50-95, with
lower seed as the declared tie-break. Its recorded validation mAP50-95 was `0.12385345383632197`.
The selected checkpoint SHA-256 is
`44646b130b8b42282b752f77659cabfc1c484dc3aaa9a2dc8f710da8468f511a`; the derived ONNX candidate
SHA-256 is `b62590a14e2e88a414eb06389058d13d69ff1ea3998232996877088951fe3bb8`.
See [`deployment_selection.json`](../reports/paired_a100/deployment_selection.json) and
[`deployment_gate.public.json`](../reports/paired_a100/deployment_gate.public.json).

## ONNX deployment gate

On the 60-image calibration split, PyTorch-to-ONNX deltas were `-0.0186` mAP50 and `-0.0128`
mAP50-95, within the absolute `0.02` fidelity threshold. Standalone ONNX Runtime parity passed
60/60 images, with minimum IoU `1.0`, maximum confidence delta `0.0`, and zero failed images.
This verifies a technical candidate only. Source:
[`deployment_gate.public.json`](../reports/paired_a100/deployment_gate.public.json).

## Intended use

Research and portfolio review of board-aware split design, controlled leakage measurement,
content-addressed experiment evidence, and deployment-gate engineering for six synthetic PCB
defect classes.

## Limitations and non-claims

- The final test contains 30 images from a single PCB template board; it does not establish
  between-board, factory-line, or production generalization.
- Image-bootstrap intervals do not estimate board-level uncertainty.
- No L4 latency, ONNX Runtime CUDA, or TensorRT FP16 benchmark is complete.
- No production AOI acceptance, escape-rate SLA, calibration-drift, or safety claim is made.
- No public checkpoint, ONNX export, official Hub revision, hosted demo, or official namespace is
  claimed.

The numerical limitations are recorded with the results in
[`final_metrics.json`](../reports/paired_a100/final_metrics.json).

## Release boundary

The repository distributes neither the selected checkpoint nor its ONNX export. Technical gate
passage does not establish redistribution rights. Release remains blocked on upstream dataset and
model obligations, identity review, and official publication approval. See
[`result_package_receipt.json`](../reports/paired_a100/result_package_receipt.json) and
[`docs/license-boundary.md`](license-boundary.md).
