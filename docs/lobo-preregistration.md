# Pre-registration: leave-one-board-out replication of the paired leakage experiment

Registered on 2026-09-11, before any fold other than Board 08 was trained. This document is
committed first; `reports/lobo/summary.json` records its SHA-256 so the analysis below can be
checked against what was declared. Nothing in this document may change after a fold has been
run; corrections, if ever needed, go into a new dated section that leaves the original text intact.

## Question

The published paired experiment holds out one board (Board 08) and measures how much final-test
mAP50 rises when the training arm is allowed to see 30 sibling images from that board:
grouped `0.6330 ± 0.1491` versus leaky control `0.8456 ± 0.0375`, a `+21.3` percentage-point
difference over three seeds. That result cannot say whether the effect is specific to Board 08.
This replication asks: **does same-board sibling exposure raise final-test mAP50 on every board
that the frozen protocol can hold out, and by how much does the effect vary between boards?**

## Eligibility rule and folds

One rule, applied to every board in the frozen dataset manifest: a board is eligible when it has
exactly `final_per_class + exposure_per_class = 10` images in every one of the six classes, and it
is neither the validation/calibration board (01) nor the legacy-excluded board (04). The rule is
implemented in `pcb_defect.lobo.eligible_boards`; `configs/lobo/folds.yaml` records its output and
`tests/test_lobo.py` re-derives it from the committed manifest on every CI run.

<!-- FOLD_TABLE_START -->
| Held-out board | Protocol config | Frozen manifest SHA-256 | Evidence directory | Status |
|---|---|---|---|---|
| 05 | `configs/lobo/board05.yaml` | `d0ccb65f61ad9173126594d00617a7993f5d6f25fe50c6689ad75a38040e5e08` | `reports/lobo/board05/` | pending Colab run |
| 07 | `configs/lobo/board07.yaml` | `89ff93656fb7b9231872caecca8a3454602982be77c329954e53f79cd189d41d` | `reports/lobo/board07/` | pending Colab run |
| 08 | `configs/paired_protocol.yaml` | `5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5` | `reports/paired_a100/` | published (reused) |
| 09 | `configs/lobo/board09.yaml` | `a498503bf75d80a678373e5c4809318a8ce214773c51ea3afc0ce9ad1c7c94ad` | `reports/lobo/board09/` | pending Colab run |
| 11 | `configs/lobo/board11.yaml` | `3e983c3ed85f5956c97185aefc33603eb7b362f152685211dac7fd26f42ce3b6` | `reports/lobo/board11/` | pending Colab run |
| 12 | `configs/lobo/board12.yaml` | `64b112051f60cf96227de28e7628be9e95e40431303714646f5ddeac03dd99e2` | `reports/lobo/board12/` | pending Colab run |
<!-- FOLD_TABLE_END -->

Boards outside the rule and why: 01 is the validation and calibration board; 04 was excluded
from final-test candidacy by the original protocol; 06 has 11 `short` images where the protocol
requires exactly 10; 10 has only 5 to 6 images per class. Board 06 is excluded rather than
relaxing the exact-count rule, because relaxing it would change the protocol for every fold.

Board 08 is not re-run. Its fold is the published evidence in `reports/paired_a100/`, whose
manifest hash equals the regenerated fold hash byte for byte.

## Fixed factors

Everything below is identical to the Board 08 run and is bound by hashes in each fold's input lock:

- Training recipe `configs/train_paired.yaml` (YOLO26n, 100 epochs, imgsz 640, batch 32,
  deterministic, seeds 42, 43, 44), SHA-256
  `6ba44a0024884c11de37a29b294543c9736cb30b6e96b4a6d27dcb93ebcf185b`.
- Base checkpoint YOLO26n `v8.4.0`, SHA-256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- Dataset identity SHA-256 `8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`,
  validation board 01 (60 validation and 60 calibration images), 513-image training arms with
  identical class histograms, 30-image final test and 30-image sibling pool per held-out board.
- Evaluation `configs/final_evaluation.yaml`: AP confidence 0.001, validation IoU 0.7, operating
  confidence 0.25, matching IoU 0.5, 10,000 image-bootstrap resamples with seed 20260803.
- Pipeline per fold, unchanged: data gate, GPU gates, six hash-locked runs, pre-final grouped
  seed selection, one-shot final evaluation, ONNX deployment gate, verifiable result package.
  The deployment gate is kept only so the package contract stays identical; fold deployment
  candidates are not deployment claims.

## Endpoints

Unit of analysis: held-out board, n = 6.

- **Primary.** For board b and arm a, m(b, a) is the mean over seeds 42, 43, 44 of final-test
  mAP50. Δ_b = m(b, leaky_control) − m(b, grouped). Report every Δ_b, the mean and sample
  standard deviation across the six boards, and a percentile bootstrap 95% interval obtained by
  resampling the six boards with replacement (10,000 draws, seed 20260803). With n = 6 the
  interval is labelled approximate.
- **Secondary.** The same statistics for mAP50-95; per-board grouped mAP50 as a measure of board
  difficulty spread; the number of boards with Δ_b > 0; the per-board paired image-bootstrap F1
  deltas that `final_evaluation` already produces.

The computation is `pcb_defect.lobo.aggregate_folds`, committed with this document, and its
output is `reports/lobo/summary.json` plus `reports/lobo/README.md`.

## Reporting rules

- All six boards are reported whatever the sign or size of Δ_b.
- No seed is added, no run is restarted from scratch except through the existing resume path, and
  no threshold, configuration, or evaluator change is made after any fold result is seen.
- A fold that cannot complete is reported as incomplete with the reason recorded in the summary;
  it is never dropped silently, and the cross-board statistics are then computed over the
  completed boards with the summary status set to `incomplete`.
- Wording fixed in advance: if every Δ_b > 0, the claim reads "same-board sibling exposure
  increased final-test mAP50 on every one of six held-out boards, mean +X pp (range a to b)";
  otherwise it reads "the direction of the exposure effect varies across held-out boards", with
  the per-board table.

## Limitations declared in advance

- Boards 01, 04, 06, and 10 are never held out; the estimate covers the 60-image boards only.
- Each fold has one 30-image final test; seeds capture training noise, not between-image
  structure beyond what the image bootstrap already reports.
- Six boards give a wide, approximate board-level interval; the result is a replication across
  the boards this dataset allows, not a population estimate.
- No dataset pixels, weights, ONNX exports, engines, or result packages are published; the
  selected deployment candidate and the L4 parity evidence are untouched.

## Reproduction

```bash
# regenerate and verify the fold configs, manifests, and registry (requires the converted dataset)
python -m pcb_defect.lobo folds --repo . --dataset data/pcb
# create the Colab handoff for every pending fold
python -m pcb_defect.lobo handoff --repo . --output-root ../handoff-lobo
# after Colab: promote each returned package, then aggregate
python -m pcb_defect.lobo promote --repo . --package <paired-results-a100-<sha12>-board05.zip> --board 05
python -m pcb_defect.lobo aggregate --repo .
```
