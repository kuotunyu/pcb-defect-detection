# Paired A100 evidence

This directory is the portable, machine-readable evidence index for the completed paired A100
experiment and its unreleased ONNX deployment candidate. It does not contain dataset assets,
checkpoints, exports, or the returned package itself.

## Receipt and provenance

- Source ZIP: `paired-results-a100-9e3a1ed5827a.zip` (`75767467` bytes)
- Source ZIP SHA-256: `c6158e5017bbec02c089ad42f55f83b20a46fb8ebd3bd1c35115d2940bb74c92`
- Source snapshot Git SHA: `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`
- Dataset SHA-256: `8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`
- Frozen protocol manifest SHA-256:
  `5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5`
- Paired-training config SHA-256:
  `6ba44a0024884c11de37a29b294543c9736cb30b6e96b4a6d27dcb93ebcf185b`
- Embedded package manifest: 52 verified entries, SHA-256
  `2387496eb57a8126515f13cde72a83fab799e4a5b9e7a8d93ee1a680d6a7132d`

The returned ZIP and sidecar matched, and every embedded manifest entry verified. See the
[portable receipt](result_package_receipt.json).

## Evidence map

| Artifact | Role |
|---|---|
| [`result_package_receipt.json`](result_package_receipt.json) | Returned ZIP/sidecar identity and package-manifest verification |
| [`package_manifest.json`](package_manifest.json) | Byte counts and SHA-256 values for all 52 source-package entries |
| [`input_lock.json`](input_lock.json) | Source snapshot, dataset, protocol, and base-model input lock |
| [`gate_report.json`](gate_report.json) | Clean runtime, data/hash, tiny-train, resume, and speed-gate results |
| [`deployment_selection.json`](deployment_selection.json) | Pre-final-test grouped checkpoint selection |
| [`final_metrics.json`](final_metrics.json) | Six-run aggregate metrics and paired image-bootstrap results |
| [`finalization_record.json`](finalization_record.json) | One-shot final-evaluation completion and result-file hash |
| [`deployment_gate.public.json`](deployment_gate.public.json) | Path-free ONNX fidelity/parity evidence bound to the raw package-manifest entry |

Claim states and licensing boundaries are tracked separately in
[`reports/claims.yaml`](../claims.yaml) and
[`docs/license-boundary.md`](../../docs/license-boundary.md).

## Paired result

All six runs completed: grouped and leaky-control arms at seeds 42, 43, and 44. The common
30-image final test contains only board 08.

| Arm | mAP50, mean ± std | mAP50-95, mean ± std |
|---|---:|---:|
| Grouped | `0.6330 ± 0.1491` | `0.2882 ± 0.0654` |
| Leaky control | `0.8456 ± 0.0375` | `0.4008 ± 0.0252` |

The leaky-control minus grouped mAP50 difference is 21.3 percentage points. The paired
final-image bootstrap F1 delta is `0.2546`, with 95% CI `[0.2102, 0.3005]` (10,000 resamples;
image, not board, is the resampling unit). The source is
[`final_metrics.json`](final_metrics.json).

## Selected deployment candidate

Grouped seed 42 was selected before final-test access using the highest grouped validation
mAP50-95, with lower seed as the declared tie-break. Its checkpoint is identified by SHA-256
`44646b130b8b42282b752f77659cabfc1c484dc3aaa9a2dc8f710da8468f511a`; the corresponding ONNX
candidate is identified by SHA-256
`b62590a14e2e88a414eb06389058d13d69ff1ea3998232996877088951fe3bb8`. See
[`deployment_selection.json`](deployment_selection.json) and
[`deployment_gate.public.json`](deployment_gate.public.json).

On the 60-image calibration split, PyTorch-to-ONNX aggregate fidelity deltas were `-0.0186` mAP50
and `-0.0128` mAP50-95, both within the absolute `0.02` gate. Standalone ONNX Runtime parity
passed 60/60 images with minimum IoU `1.0`, maximum confidence delta `0.0`, and no failed image;
that historical check compares two execution paths over the same ONNX artifact and is not a
PyTorch-reference per-box equivalence result.

## Non-claims

- This A100 report itself does not include an L4 benchmark. Verified public metadata derived from
  the private L4 package is in [`benchmark_l4.md`](../benchmark_l4.md),
  [`benchmark_l4.json`](../benchmark_l4.json),
  [`benchmark_l4_raw.json`](../benchmark_l4_raw.json), and
  [`backend_parity_l4.json`](../backend_parity_l4.json). The strict PyTorch-reference per-box gate
  failed and is reported without threshold changes.
- No checkpoint, ONNX file, TensorRT engine, or returned package binary is distributed here.
- Technical gate passage does not establish rights to release the dataset, selected checkpoint,
  or ONNX candidate.
- External publication is not asserted by this candidate tree; no public model revision, Hugging
  Face model publication, or hosted demo is claimed.
- The final test is one board, so these results do not establish factory-line or between-board
  generalization.
