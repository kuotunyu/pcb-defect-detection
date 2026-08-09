# Portfolio Final Polish Design

## Context

The official GitHub `main` is clean, single-author, CPU-CI green, and contains verified paired A100
and private L4 metadata. The remaining risk is presentation drift rather than missing ML work:
README GPU commands omit required arguments, the README highlights TensorRT without showing the
faster recorded ONNX Runtime CUDA result, several public documents still describe optional model
publication as pending work, and GitHub has no repository topics while an empty Wiki remains enabled.

## Approaches considered

1. Stop with the current tree. This avoids churn but leaves copy-paste failures and an unfinished
   release impression.
2. Apply a targeted portfolio closure. Correct the runnable path, expose all recorded backends,
   declare model hosting intentionally out of scope, and improve GitHub discoverability. This is
   the selected approach because it materially improves recruiter trust without new ML work.
3. Publish a model/demo or run more experiments. This adds licensing, hosting, and GPU work with
   little incremental value for the leakage-aware evaluation story, so it is rejected.

## Design

### Recruiter-facing evidence

Add a compact evidence index near the top of `README.md`. Replace the TensorRT-only L4 presentation
with the complete PyTorch FP32, ONNX Runtime CUDA FP32, and TensorRT FP16 timing comparison derived
from `reports/benchmark_l4.json`. State explicitly that ONNX Runtime CUDA FP32 was the fastest
observed backend and that TensorRT validates an additional deployment path rather than winning this
measurement. Remove the unrecorded `Opset 17` label and consistently call the 60 images the
calibration split.

### Runnable onboarding

Keep CPU clean-clone verification as the primary Quick Start. Retain the valid data-preparation
commands. Remove bare GPU commands that exit with missing required arguments and replace them with a
safe `--help` discovery command plus links to the immutable A100 and L4 notebooks. No README command
may start training merely to prove onboarding works.

### Intentional release boundary

The public deliverable is a metadata-only portfolio release. Keep the app contract fail-closed and
all artifact/Hub fields null, but describe this as an intentional release decision instead of an
unfinished future launch. Convert optional rights review, Hugging Face publication, and hosted demo
items from unchecked release tasks into explicit non-goals. Synchronize `docs/model-card.md`,
`app/README.md`, `app/model_contract.json`, `reports/claims.yaml`,
`reports/paired_a100/README.md`, and legacy-report wording without changing any evidence JSON,
metric, frozen hash, model artifact, or dataset artifact.

### Repository surface

Add concise GitHub topics for Computer Vision, Industrial AI, evaluation, and deployment. Disable
the unused Wiki. Remove the local `practice` remote after confirming `origin` is the official
`kuotunyu/pcb-defect-detection` repository. Preserve old local branches/worktrees because deleting
private audit history is not required for the public portfolio and would be unnecessarily
destructive.

## Verification

Extend the release contract before changing production documents. The test must fail on the current
TensorRT-only README, unchecked release items, invalid bare GPU commands, and stale publication
wording. It must bind all displayed backend metrics to `reports/benchmark_l4.json`, keep the app
contract blocked with null artifact identities, and ensure the metadata-only decision is consistent
across public documents. Run the focused RED/GREEN cycle, the full CPU suite, Ruff, lock/diff checks,
Markdown local-link validation, package build, identity scan, independent review, merged-main tests,
GitHub Actions, remote-head, topics, Wiki, and Contributors verification.

## Non-goals

- No training, evaluation, export, benchmark, GPU, Colab, dataset download, or paid API.
- No weights, ONNX, TensorRT engine, dataset pixels, result package, or secret publication.
- No model hosting, Hugging Face Space, release, tag, history rewrite, or contributor change.
- No modification of evidence JSON, metrics, manifests, or frozen hashes.
