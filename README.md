[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Code license: AGPL-3.0](https://img.shields.io/badge/code-AGPL--3.0-lightgrey)](LICENSE)

# Leakage-aware PCB defect detection

An industrial computer-vision experiment that measures how same-board sibling exposure changes
PCB defect detection under a frozen, paired evaluation protocol.

> **Status: paired A100 and ONNX candidate evidence complete; release hardening continues.** Six
> paired runs, the one-shot final evaluation, and the hash-pinned ONNX deployment gate have passed.
> L4/TensorRT benchmarking, distribution-rights approval, official model publication, and the live
> demo remain pending.

## The 30-second evidence

1. **Leakage-aware experimental design.** Both arms use the same 30 final-test images, the same
   60 validation images, the same 60 calibration images, and equal 513-image training sets with
   exactly matching class histograms. The grouped arm sees no final-board sibling; the control arm
   sees exactly 30 predeclared siblings.
2. **Content-addressed reproducibility.** The 693-image normalized dataset fingerprint is
   `8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`; the frozen protocol
   manifest hash is `5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5`.
   YOLO26n initialization is pinned to the official `v8.4.0` asset, SHA-256
   `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
3. **Current paired A100 result.** Across seeds 42/43/44, grouped achieved
   `0.6330 ± 0.1491` mAP50 versus `0.8456 ± 0.0375` for leaky control: a 21.3
   percentage-point paired difference on the same final images. The paired F1 delta was `0.2546`
   with 95% CI `[0.2102, 0.3005]` under final-image bootstrap resampling.
4. **Unreleased ONNX candidate.** Grouped seed 42 was selected by validation before final-test
   access. Its calibration-only ONNX gate passed 60/60 standalone parity images with minimum IoU
   `1.0`, maximum confidence delta `0.0`, and mAP50/mAP50-95 fidelity deltas of `-0.0186` and
   `-0.0128`, both within the absolute `0.02` gate.

The canonical assignments are in
[`reports/protocol/paired_split_manifest.json`](reports/protocol/paired_split_manifest.json), and
the scientific contract is in [`configs/paired_protocol.yaml`](configs/paired_protocol.yaml).
The promoted machine evidence and its provenance are indexed in
[`reports/paired_a100/README.md`](reports/paired_a100/README.md).

## Frozen experiment

| Partition | Images | Board policy |
|---|---:|---|
| Common final test | 30 (5/class) | Board 08; exact images excluded from both training arms |
| Validation | 60 (10/class) | Board 01; shared by both arms |
| Calibration | 60 (10/class) | Board 01; disjoint from validation and training |
| Grouped train | 513 | Boards 04/05/06/07/09/10/11/12; no board-08 image |
| Leaky-control train | 513 | Class-wise replacement of 30 grouped samples by 30 board-08 siblings |

The control is not a generic random split. It changes one factor only: exposure to other images
from the final-test board. Final images, validation/calibration policy, architecture, initial
weights, training configuration, train size, and image-class counts remain paired.

## Historical context: the earlier 12.1-point observation

The legacy prototype observed mAP50 0.8390 on a board-grouped split and 0.9603 on an image-random
split. Those runs used **different test images and different test sizes**, so the 12.1-point gap is
retained only as **legacy observed split sensitivity**. It is not a causal leakage estimate and is
not a current model claim. The current 21.3-point result above replaces that comparison as the
paired claim.

See [`reports/claims.yaml`](reports/claims.yaml) for claim status and evidence boundaries.

## Reproduce the CPU-side contract

```bash
uv sync --locked --no-editable
uv run python -m pytest -v
uv run ruff check .
uv run ruff format --check .

# Anonymous upstream download, VOC-to-YOLO conversion, and structural tripwires
uv run python -m pcb_defect.data_prep.prepare --out data/pcb --strategy grouped --seed 42

# Recompute content hashes, verify the frozen protocol, and write ignored runtime lists
uv run python -m pcb_defect.data_prep.paired \
  --source data/pcb \
  --config configs/paired_protocol.yaml \
  --artifacts reports/protocol \
  --runtime data/paired
```

The dataset command downloads about 2 GB and does not require an API key. Dataset pixels, trained
weights, ONNX files, TensorRT engines, and run directories are intentionally untracked.

## GPU handoff

Create the private Colab handoff only from a clean committed branch:

```bash
uv run python -m pcb_defect.handoff --output dist/colab-handoff
```

This produces a one-commit Git bundle of the current tree, deliberately excluding legacy history,
removed dataset pixels, ignored data, weights, exports, caches, and `.env`. It also renders
ready-to-run notebook copies with the exact bundle hash and snapshot Git SHA already embedded and
records their hashes in `handoff_manifest.json`. Upload the bundle to the recorded Drive path and
open the rendered notebook from the same handoff directory; no manual code editing is required.

The repository-native runner provides these gates and resumable operations:

```bash
python -m pcb_defect.experiment resolve-base ...
python -m pcb_defect.experiment preflight ...
python -m pcb_defect.experiment gates ...       # tiny train + interruption/resume + speed probe
python -m pcb_defect.experiment train-all ...   # six hash-locked runs
python -m pcb_defect.final_evaluation ...       # one-shot common final test
```

`resolve-base` downloads only the immutable URL in [`configs/base_model.yaml`](configs/base_model.yaml)
and refuses any byte-count or SHA-256 mismatch. It does not ask Ultralytics for a mutable latest
asset.

The [A100 experiment notebook](notebooks/paired_experiment_a100.ipynb) and
[L4 benchmark notebook](notebooks/deployment_benchmark_l4.ipynb) are deliberately thin: they
install from `uv.lock` and call these modules. The
first final-test access creates an irreversible marker. A partial final evaluation fails closed
instead of silently spending the test set again.

## Deployment status

The validation-selected grouped seed-42 candidate passed both technical deployment gates:

- calibration-set PyTorch-to-ONNX metric fidelity; and
- standalone ONNX Runtime preprocessing/postprocessing parity.

This is an unreleased candidate, not a public model. Distribution rights, an official immutable
model revision, and hosted deployment remain unresolved; technical gate passage does not authorize
redistribution of the checkpoint or ONNX binary. The app model contract therefore remains
`blocked`, and no floating `main/best.onnx` or account-specific model URL is used.
The legacy ONNX evidence did not pass its own gates and does not support this candidate claim.
TensorRT engines are device-specific run artifacts and must never be committed.
L4 latency measurements use the predeclared calibration set, never the one-shot final test, and a
completed benchmark is resumable only while its source checkpoint, ONNX, engine, calibration
inputs, deployment gate, CUDA provider, and recorded hashes still match.

## License and release boundary

`LICENSE` covers this repository's original code. It does **not** grant rights to HRIPCB images,
annotations, upstream model weights, newly trained weights, or third-party libraries. The observed
Kaggle distribution has no verified dataset license in this project, so raw and pixel-derived
dataset assets are excluded from the candidate tree. See
[`docs/license-boundary.md`](docs/license-boundary.md) and [`docs/data-card.md`](docs/data-card.md).

The existing Git history still contains legacy dataset images and personal/test-account identity.
Do not migrate that history to an official public account. Official publication requires a clean,
reviewed source snapshot after the license and identity gates are resolved.

## Scope and limitations

- Six synthetic defect classes on ten PCB template boards; no factory-line domain validation.
- The new final test contains one board, so image-bootstrap intervals do not estimate
  between-board variance.
- Not a production AOI acceptance system; no escape-rate SLA, calibration drift study, or process
  capability evidence.
- No segmentation, tracking, dashboard, architecture sweep, or hyperparameter search is planned;
  those features do not strengthen the leakage/deployment thesis.

More detail: [`docs/model-card.md`](docs/model-card.md),
[`docs/limitations.md`](docs/limitations.md), [`docs/release-checklist.md`](docs/release-checklist.md),
and [`schemas/run_record.schema.json`](schemas/run_record.schema.json).
