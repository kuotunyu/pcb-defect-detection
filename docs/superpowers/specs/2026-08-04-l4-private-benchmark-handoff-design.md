# Private L4 Benchmark Handoff Design

## Purpose

Create a fail-closed, immutable handoff for one private Colab L4 benchmark that reuses the
completed paired A100 experiment without retraining. The benchmark compares PyTorch FP32, ONNX
Runtime CUDA FP32, and TensorRT FP16 on the frozen calibration split. It produces private runtime
artifacts and a metadata report suitable for later, separate evidence review.

This phase does not publish a model, dataset, ONNX file, TensorRT engine, result ZIP, or hosted
demo. It does not resolve dataset redistribution rights or authorize commercial use.

## Decision

Use the current reviewed source as the benchmark runner while preserving the completed A100
experiment as an independent immutable parent.

Two rejected alternatives are:

1. Run the old L4 notebook from the A100 source snapshot. It predates the current runtime,
   failure-diagnostic, and handoff hardening.
2. Rerun the complete A100 experiment at the current source SHA. The A100 artifacts are already
   complete and content-addressed; retraining would add cost without improving the L4 comparison.

## Release and license boundary

The repository code remains `AGPL-3.0-or-later`. Ultralytics' current official guidance treats its
code and fine-tuned models as AGPL-3.0 by default when the entire derivative project is openly
released. That does not establish rights to redistribute the HRIPCB/Kaggle dataset or artifacts
derived from it. The HRIPCB paper describes the dataset as public for research and evaluation, but
the project has not found an explicit upstream redistribution license.

Therefore:

- the L4 run is private technical evaluation;
- the source bundle, notebook, result ZIP, checkpoint, ONNX export, TensorRT engine, dataset
  pixels, calibration images, and logs remain untracked and private;
- a later evidence-promotion phase may consider only reviewed, path-free metadata;
- model publication, hosted inference, and official-account migration remain blocked;
- this design is an engineering boundary, not legal advice.

Primary references:

- <https://www.ultralytics.com/license>
- <https://docs.ultralytics.com/help/contributing/>
- <https://robotics.pkusz.edu.cn/static/papers/ACAIT2019-huangweibo1.pdf>

## Immutable identities

The handoff must distinguish the code that runs the benchmark from the experiment that produced
the artifacts.

### Benchmark runner

- The runner Git SHA is the history-free snapshot SHA produced by the new clean source bundle.
- The bundle SHA-256 and runner SHA are rendered into the L4 notebook.
- Colab clones the bundle and checks out the exact runner SHA in detached mode.
- The benchmark records the observed runner SHA from the checked-out repository.

### Parent A100 experiment

- Experiment Git SHA:
  `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`
- Raw deployment-gate SHA-256:
  `466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803`
- Selected checkpoint SHA-256:
  `44646b130b8b42282b752f77659cabfc1c484dc3aaa9a2dc8f710da8468f511a`
- ONNX SHA-256:
  `b62590a14e2e88a414eb06389058d13d69ff1ea3998232996877088951fe3bb8`
- Dataset SHA-256:
  `8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`
- Frozen protocol-manifest SHA-256:
  `5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5`

The L4 workspace is derived only from the parent experiment SHA:

```text
/content/drive/MyDrive/pcb-defect-paired/workspaces/9e3a1ed5827a
```

It must never be derived from the benchmark runner SHA.

## Handoff architecture

Extend the repository handoff boundary with an explicit L4 stage. The stage accepts the parent
experiment SHA plus the raw deployment-gate, checkpoint, and ONNX SHA-256 values. These four
parent values are an atomic group: missing, extra, malformed, or mixed-stage arguments fail before
bundle creation.

The L4 handoff contains only:

- a history-free source bundle for the reviewed benchmark runner;
- one rendered `deployment_benchmark_l4.ipynb`;
- one manifest containing the runner bundle identity, notebook template/rendered hashes, parent
  identities, protocol identity, and base-model contract identity.

It does not render or include an A100 training notebook or parity-probe notebook. A stage-specific
handoff prevents the user from accidentally rerunning training.

Use a distinct local output directory and Drive directory named with the runner SHA prefix. Do not
overwrite the earlier A100 handoff. The rendered notebook must have no unresolved placeholders,
outputs, execution counts, or hidden account-specific values.

## Notebook data flow

The rendered notebook performs these steps in order:

1. Set the hermetic Ultralytics controls and non-interactive Matplotlib backend.
2. Mount Drive and locate the stage-specific source bundle.
3. Verify the bundle SHA-256 before cloning.
4. Clone and check out the exact runner SHA; reject a dirty or different checkout.
5. Derive `PARENT_WORKSPACE` from the parent experiment SHA, never from the runner SHA.
6. Verify the parent workspace and immutable parent inputs before installing or exporting:
   - raw deployment-gate bytes and SHA-256;
   - gate `passed=true` state;
   - gate experiment, dataset, and protocol-manifest identities;
   - selected checkpoint path containment and SHA-256;
   - ONNX path containment and SHA-256;
   - calibration YAML and image-list identity.
7. Install the locked, non-editable project environment with the project wheel forcibly refreshed.
8. Verify an L4 GPU, CUDA availability, ONNX Runtime CUDA provider, Ultralytics version, and
   detected TensorRT runtime before creating the benchmark output directory.
9. Run the benchmark with the immutable values repeated as command-line expectations.
10. Verify the completed benchmark report and its source/engine/runtime bindings.
11. Create an atomic private ZIP and SHA-256 sidecar with a name containing both parent and runner
    SHA prefixes.

The notebook prints diagnostics and exact log paths on failure. It must not delete, rewrite, or
repair the parent A100 evidence.

## Benchmark contract

The benchmark CLI receives exact expected values for:

- runner Git SHA;
- parent experiment Git SHA;
- raw deployment-gate SHA-256;
- selected checkpoint SHA-256;
- ONNX SHA-256.

It independently observes the checked-out runner SHA and all parent bytes. Passing values from the
notebook is not sufficient by itself.

The report distinguishes:

- `runner_git_sha`: code used to execute and validate the benchmark;
- `experiment_git_sha`: source snapshot that produced the checkpoint and ONNX;
- raw deployment-gate SHA-256;
- dataset and protocol-manifest SHA-256;
- checkpoint, ONNX, and TensorRT engine SHA-256;
- exact hardware, CUDA, ONNX Runtime, provider, Ultralytics, and TensorRT versions;
- calibration image-list and image-content SHA-256;
- warmup count, cycles, raw per-image timings, and summary statistics;
- TensorRT FP16 calibration fidelity and the predeclared absolute threshold.

The measured system TensorRT runtime is not treated as a portable locked wheel. The notebook
rejects its absence, records the exact detected version, and performs no floating auto-install.
The resulting claim is tied to that recorded Colab L4 environment.

## Benchmark semantics

- Backends: PyTorch FP32, ONNX Runtime CUDA FP32, TensorRT FP16.
- GPU: NVIDIA L4 only.
- Inputs: the frozen 60-image calibration split only.
- Warmup: 30 inferences per backend.
- Measured cycles: four complete passes over the calibration list per backend.
- Synchronization: synchronize CUDA before and after each timed inference.
- Statistics: preserve raw timings and report count, mean, standard deviation, median, p95, min,
  and max.
- Fidelity: compare TensorRT FP16 calibration mAP50-95 with the source PyTorch value using the
  threshold already recorded by the passed deployment gate.
- INT8, throughput batching, final-test inference, power measurements, and production SLA claims
  are out of scope.

## Resume and failure behavior

- Any immutable identity mismatch stops before TensorRT export.
- A non-L4 GPU, missing CUDA provider, missing TensorRT runtime, or runtime-state change stops the
  run.
- The benchmark output directory is created exclusively. An incomplete directory is never
  overwritten or accepted.
- A completed report is reusable only if every parent, engine, calibration, environment, and
  runner binding still verifies.
- Logs are stored in Drive and displayed in the notebook on command failure.
- A partial ZIP/sidecar pair is rejected. A complete pair is accepted only when the sidecar name
  and SHA-256 match exactly.
- Colab disconnection is not success. Only the final verified completion message authorizes
  download of the result package and sidecar.

## Private result package

The package name includes both identities:

```text
paired-results-l4-9e3a1ed5827a-runner-<runner-prefix>.zip
```

The package remains private and may contain the TensorRT engine, parent model artifacts, logs, and
calibration references needed for independent local verification. It is not a release artifact.

After download, a separate read-only audit verifies the sidecar, package manifest, benchmark
report, and all internal hashes. Promotion of a public metadata projection is a separate design
and implementation cycle; it is not automatic after an L4 pass.

## Testing strategy

Implementation follows strict red-green TDD.

CPU-safe tests must cover:

- runner SHA and parent experiment SHA remain different and are used for their declared roles;
- L4 parent arguments are required together and validated as lowercase fixed-length hex;
- the L4 stage renders only the L4 notebook and stage-specific manifest;
- every placeholder is replaced exactly once and rendered notebook JSON is unexecuted and
  compilable;
- missing or mismatched gate, checkpoint, ONNX, runner, dataset, manifest, and calibration
  identities fail closed;
- parent paths cannot escape the parent workspace;
- benchmark reports record both runner and experiment provenance;
- only calibration images are accepted and final-test inputs are rejected;
- incomplete output, mutated engine, changed runtime state, and partial package pairs are rejected;
- success and failure diagnostics preserve durable logs;
- tracked-file scans reject `.pt`, `.onnx`, `.engine`, `.plan`, `.trt`, result ZIPs, sidecars,
  dataset pixels, notebook outputs, and generated logs.

All GPU libraries and inference backends are replaced at the narrow external boundary in unit
tests. CI and local verification do not require a GPU, dataset, checkpoint, ONNX file, API key, or
network access.

Before handoff, run:

- locked non-editable synchronization;
- the complete CPU pytest suite;
- Ruff lint and format checks;
- wheel and source-distribution build to a unique temporary directory;
- clean-tree, tracked-binary, placeholder, local-path, secret, commit-identity, and manifest audit;
- a clean-bundle clone verification of the rendered L4 handoff.

## Definition of done

The local phase is complete only when:

1. all tests, style checks, and source builds pass in a CPU-only environment;
2. the rendered L4 notebook binds a reviewed runner to the immutable `9e3a1ed...` parent;
3. the bundle, notebook, and manifest hashes verify from a clean clone;
4. no generated binary, model, export, engine, dataset pixel, package, log, or secret is tracked;
5. the handoff directory is ready for the user to upload without manual placeholder editing;
6. no Git remote, official account, release, or hosted service has been changed.

The private GPU phase is complete only when the L4 notebook prints its verified completion message
and the user downloads both the named ZIP and matching `.sha256` sidecar.

## Estimated effort

- Local CPU-only hardening, tests, review, and handoff generation: approximately 45–90 minutes.
- User-operated Colab L4 run: approximately 15–40 minutes, dominated by TensorRT export and
  measured inference.
- Returned-package audit and any public metadata promotion are separate follow-up work.
