# Evaluation Claim Hardening Design

## Purpose

Make the public portfolio narrative match the strongest inference supported by the committed
artifacts. The paired experiment is a controlled measurement of same-board sibling exposure on a
single held-out board. It is not evidence of population-level between-board or factory-line
generalization.

This phase is documentation and release-contract hardening only. It does not change metrics,
splits, artifacts, model weights, or Git history, and it requires no GPU or model training.

## Problem statement

The machine-readable registry and model card already state the material limitations:

- the common final test has 30 images from board 08;
- the paired bootstrap resamples images, not boards;
- the interval therefore does not estimate between-board variance; and
- the result is specific to the frozen dataset and training recipe.

The top-level README currently overstates those artifacts in three places by describing the
grouped result as real production-relevant cross-board generalization and the image-level interval
as strict proof of board-level statistical significance. A reviewer can reasonably interpret that
language as treating correlated images as independent board-level evidence.

## Considered approaches

### 1. Wording-only correction

Edit the README and stop. This is the smallest patch, but a later visual rewrite could reintroduce
the same claims because CI currently checks metric values, not inferential scope.

### 2. Claim-boundary correction plus executable guardrails (selected)

Correct the README, preserve the precise claims registry and model-card language, and extend the
release contract so future edits must retain the sampling-unit and generalization limitations.
This is still a small CPU-only change while making the scientific boundary durable.

### 3. Immediate multi-board retraining

Start board-level cross-validation before correcting the public narrative. This would eventually
produce stronger evidence, but it leaves the current overclaim public during a much larger GPU
project and mixes two independently reviewable concerns.

## Scope

### In scope

- Replace absolute language such as `true cross-board generalization`, `production deployment
  performance`, and `proof of board-level statistical significance` with artifact-supported
  wording.
- State next to the headline metric that the final-test sampling scope is 30 images from one board.
- Describe the interval precisely as a paired image-bootstrap interval whose resampling unit is an
  image and which does not estimate between-board uncertainty.
- Preserve the distinction between the legacy 12.1-point split sensitivity and the controlled
  21.3-point paired result.
- Add release-contract tests that fail if the README drops these boundaries or reintroduces the
  known overclaims.
- Keep the existing visual hierarchy and evidence links intact.

### Out of scope

- Recomputing metrics or confidence intervals.
- Changing any JSON evidence, frozen hash, split manifest, checkpoint, export, or benchmark.
- Training, evaluation, GPU use, or Colab work.
- Claiming that the current result estimates new-product, factory-line, or population-level
  generalization.
- Redesigning the rest of the README or adding new portfolio features.
- Publishing weights, datasets, packages, releases, or hosted demos.

## Claim language contract

The public narrative will use the following hierarchy:

1. **Verified observation:** under the frozen protocol, the three-seed mean final-test mAP50 is
   `0.6330` for the grouped arm and `0.8456` for the leaky-control arm, a `21.3` percentage-point
   difference.
2. **Controlled interpretation:** the arms have equal size and class counts and differ by
   predeclared same-board sibling exposure, so the result measures leakage sensitivity for this
   frozen dataset and recipe.
3. **Inference boundary:** the final test is 30 images from board 08. The image-level bootstrap
   interval measures within-board final-image variation and does not estimate between-board
   variance.
4. **Explicit non-claim:** the result does not establish population-level cross-board,
   new-product, factory-line, or production generalization.

The phrase `statistically significant` will not be used as an unqualified headline. Where the
interval is described, the sampling unit and the single-board boundary must appear in the same
result section.

## Files and responsibilities

- `README.md`
  - Correct the executive summary, workflow node, result table, and paired-result explanation.
  - Add a compact `What this proves / What this does not prove` boundary adjacent to the metrics.
  - Preserve current evidence anchors, hashes, deployment caveats, and layout.
- `reports/claims.yaml`
  - Remain the machine-readable source of truth.
  - Change only if needed to make the existing inference boundary directly testable; no status or
    metric promotion is permitted.
- `docs/model-card.md`
  - Retain its existing single-board and image-bootstrap limitations.
  - Receive only terminology alignment if the README uses a new canonical phrase.
- `docs/limitations.md`
  - Retain the one-board and unresolved board-level uncertainty statements.
- `reports/paired_a100/README.md`
  - Remain the artifact-local explanation and retain `image, not board, is the resampling unit`.
- `tests/test_release_contract.py`
  - Assert that the README contains the single-board, sampling-unit, and non-production boundary.
  - Reject the exact known overclaim phrases.
  - Continue deriving all displayed metric values from committed machine artifacts.

## Data flow

`reports/paired_a100/final_metrics.json` remains the numeric source. `reports/claims.yaml` records
the supported statement and limitations. The README and model card communicate those claims to
humans. `tests/test_release_contract.py` binds the human-facing documents back to both the numeric
artifact and its inference boundary.

No document becomes an independent source of metrics, and no metric is copied from a historical
README or cache as fresh evidence.

## Failure handling

- If a displayed number differs from the machine artifact, the existing release contract fails.
- If the README removes the one-board or image-bootstrap limitation, the new contract fails.
- If an absolute production/cross-board claim returns, the new contract fails.
- If satisfying the wording requires changing a metric or artifact, implementation stops; that is
  outside this phase and requires a separate evidence-producing protocol.

## Test strategy

Implementation will follow test-driven development:

1. Add focused contract assertions and observe them fail against the current README.
2. Make the minimum documentation edits needed for the tests to pass.
3. Run the focused release-contract suite.
4. Run the complete CPU-safe test suite in the locked environment.
5. Run Ruff check and format check, lock validation, and a final diff/identity audit.

No test may require a dataset, weights, API key, network access, GPU, or `.env` contents.

## Acceptance criteria

- A reader cannot mistake the 21.3-point result for a ten-board or production estimate.
- The 30-image board-08 scope and image-level resampling unit are visible beside the result.
- The legacy 12.1-point comparison remains explicitly non-paired and non-causal.
- `reports/claims.yaml`, README, model card, limitations, and artifact-local report do not
  contradict one another.
- Release-contract tests prevent recurrence of the known overclaims.
- Locked CPU tests and Ruff checks pass.
- The worktree contains no new dataset, image, model, export, result package, log, secret, local
  path, or non-`kuotunyu` commit identity.

## Follow-on project: board-level generalization evidence

After this hardening is merged, a separate design will predeclare a board-rotating evaluation. Its
primary unit will be the board, not the image. It will compare a grouped arm with a size- and
class-matched same-board-exposure control across multiple held-out boards, report per-board effects,
and separate board variation from training-seed variation. That design must address historical
analyst exposure to the ten existing boards, validation-board rotation, license boundaries, compute
budget, and whether an untouched external board set is available before any GPU run begins.

The follow-on experiment is the evidence required before upgrading claims to cross-board
generalization. More model variants, UI work, hosted demos, and extra hardware benchmarks remain
lower priority.
