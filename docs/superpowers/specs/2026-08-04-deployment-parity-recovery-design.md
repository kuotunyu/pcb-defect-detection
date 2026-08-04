# Deployment Parity Probe and Full-Rerun Design

## Context

The immutable A100 experiment at Git snapshot
`378e3925a8af9b5b1efba40cdfcf2fae4490f59b` completed all six training runs and the
final evaluation. Its ONNX export also completed. The committed deployment report shows that the
aggregate PyTorch-to-ONNX fidelity gate passed:

- `delta_map50 = -0.018590364845378238`
- `delta_map50_95 = -0.012809288682730458`
- allowed absolute delta: `0.02`

The deployment was blocked because the standalone parity gate compared the standalone ONNX
runtime against the PyTorch checkpoint. That comparison duplicates export fidelity and mixes it
with runtime parity. It reported 40 failing images out of 60. The failed report, candidate contract,
ONNX model, six runs, and final-evaluation artifacts must remain unchanged as evidence.

## Goal

Correct the parity responsibility boundary, verify the correction against the existing ONNX
artifact before spending more A100 time, and then produce an immutable handoff for one clean full
rerun.

## Non-goals

- Do not relax the `0.02` fidelity threshold, `0.90` IoU threshold, or `0.15` confidence-delta
  threshold.
- Do not mark the existing blocked deployment report as passed.
- Do not overwrite or delete anything under the existing `378e3925a8af` workspace.
- Do not retrain during the probe.
- Do not run local GPU work, create a release, push, or publish artifacts.

## Approaches Considered

### Selected: read-only probe, then full rerun

First run a corrected same-artifact parity probe against the existing ONNX file. The probe writes a
new, content-addressed diagnostic report outside the existing workspace. Only after that report
passes should the user run the new full A100 notebook.

This approach proves the proposed gate semantics on real artifacts before another multi-hour run,
while the eventual full rerun keeps the release evidence on one immutable source snapshot.

### Rejected: deployment-only promotion of the old workspace

A downstream-only recovery could preserve compute, but it would require cross-snapshot lineage and
changes to the result-package and benchmark contracts. The user prefers a complete rerun, so this
additional release architecture is unnecessary.

### Rejected: threshold relaxation

The observed failure is caused by comparing different model backends at the wrong gate boundary.
Relaxing thresholds would conceal that design error and weaken the portfolio evidence.

## Architecture

### 1. Isolated fidelity and runtime-parity gates

The deployment stage keeps its existing aggregate fidelity comparison:

```text
selected PyTorch checkpoint -> Ultralytics validation metrics
same exported ONNX artifact -> Ultralytics validation metrics
absolute metric deltas       -> fidelity gate
```

Standalone parity becomes a same-artifact comparison:

```text
same best.onnx -> Ultralytics ONNX predictor  --+
same best.onnx -> standalone OnnxYoloModel    --+--> per-image box parity
```

Both paths use the same calibration images and confidence threshold. The existing IoU,
confidence-delta, unmatched-box, and 60-image requirements remain unchanged. The deployment report
identifies the parity reference as `ultralytics-onnx` so reviewers can distinguish runtime parity
from export fidelity.

### 2. Read-only probe

A repository CLI accepts:

- the existing parent workspace;
- the exact parent experiment Git SHA;
- the exact failed deployment-gate SHA-256;
- the exact ONNX SHA-256;
- a new output report path.

Before inference, it verifies the parent input lock, failed deployment report, ONNX bytes,
calibration list, and source checkpoint reference. It then runs only the corrected same-ONNX parity
comparison. It refuses to overwrite an existing output and records the new probe-code Git SHA plus
all parent hashes.

The probe output is stored under:

```text
/content/drive/MyDrive/pcb-defect-paired/probes/
  378e3925a8af-to-<new-snapshot-prefix>/parity_probe.json
```

It never writes inside `workspaces/378e3925a8af`.

### 3. Three immutable notebooks from one source bundle

The handoff generator produces:

- `deployment_parity_probe_a100.ipynb`: clones the new source bundle, verifies all immutable parent
  hashes, and runs only the probe;
- `paired_experiment_a100.ipynb`: performs the complete clean rerun in the new source snapshot's
  workspace after the probe passes;
- `deployment_benchmark_l4.ipynb`: continues to consume only a passed deployment gate from the new
  full-rerun workspace.

The probe notebook and full-rerun notebook use the same locked environment and force
`MPLBACKEND=Agg`. Every long-running subprocess captures stdout/stderr to a Drive log and prints the
underlying failure instead of exposing only `CalledProcessError`.

## Data and Provenance Flow

1. The probe notebook verifies the new source bundle and its history-free snapshot.
2. The probe CLI verifies the old experiment snapshot and failed deployment bytes without changing
   them.
3. The corrected parity calculation reads the old `best.onnx` and calibration images.
4. A new probe report records both old artifact hashes and the new probe-code Git SHA.
5. If and only if the probe passes, the user runs the full-rerun notebook.
6. The full rerun creates a new workspace keyed by the new snapshot SHA and regenerates all six
   runs, final metrics, deployment evidence, and result package without referencing the probe as
   release evidence.

## Failure Handling

- Parent hash mismatch: stop before loading a model.
- Existing probe output: refuse to overwrite it.
- Same-ONNX parity failure: preserve the report and do not advise full retraining.
- Full-rerun deployment failure: preserve the deployment report and print the exact report/log path.
- Any incomplete package pair: retain the existing fail-closed behavior.

## Testing

- Unit-test that runtime parity uses one ONNX path for both reference and standalone predictors.
- Unit-test unchanged thresholds and unmatched-box behavior.
- Unit-test probe rejection for parent gate, ONNX, and input-lock hash mismatches.
- Unit-test probe output provenance and no-overwrite behavior.
- Contract-test all notebook placeholders, headless backend, captured logs, empty outputs, and valid
  Python syntax.
- Verify locked CPU-safe installation, complete test suite, lint, formatting, bundle integrity,
  one-commit clean clone, and rendered notebook hashes locally.
- Treat the A100 probe report as the required real-artifact confirmation before starting the full
  rerun.

## Acceptance Criteria

- The original blocked deployment directory remains byte-for-byte untouched.
- The probe completes without training and writes one content-addressed report.
- The probe report compares Ultralytics ONNX with standalone ONNX for exactly 60 calibration images.
- A failed probe prevents the full-rerun recommendation.
- A passed probe enables a new ready-to-run A100 handoff whose source, bundle, notebook, and parent
  probe inputs all have recorded SHA-256 values.
- No release claim is promoted until the new full rerun and subsequent L4 benchmark pass their own
  gates.
