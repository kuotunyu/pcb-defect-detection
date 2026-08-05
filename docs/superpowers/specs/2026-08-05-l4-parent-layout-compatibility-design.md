# L4 Parent Workspace Layout Compatibility Design

## Purpose

Repair the private L4 benchmark handoff so it verifies the real, already completed A100
workspace without mutating that parent evidence or rerunning training. The current L4 verifier
assumes a synthetic layout used by its tests: a root-level protocol manifest and calibration
pixels contained inside the workspace. The production A100 workflow instead stores one frozen
manifest copy under every completed run and keeps dataset pixels in the separate Drive dataset
root. The mismatch makes the rendered L4 notebook fail before benchmarking.

This change preserves the original fail-closed identity contract. It changes only how canonical
parent evidence is located and bounded; it does not relax any Git, gate, dataset, manifest,
checkpoint, ONNX, calibration-partition, or content-hash check.

## Verified production layout

The immutable parent experiment remains:

- experiment Git SHA: `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`;
- workspace: `/content/drive/MyDrive/pcb-defect-paired/workspaces/9e3a1ed5827a`;
- dataset root: `/content/drive/MyDrive/pcb-defect-paired/dataset/pcb`.

The existing A100 workspace contains the protocol manifest at run-scoped paths such as:

```text
runs/grouped/seed42/inputs/paired_split_manifest.json
```

The A100 experiment creates equivalent copies for both arms and seeds 42, 43, and 44. Runtime
partition lists are stored below `runtime_data`, while their absolute image entries point into
the external dataset root. The parent workspace must remain read-only during the L4 stage.

## Decision

Adopt an explicit production-layout contract.

1. The dataset root becomes a required L4 input alongside the repository and parent workspace.
2. The verifier reads all six run-scoped manifest copies:
   - `runs/grouped/seed{42,43,44}/inputs/paired_split_manifest.json`;
   - `runs/leaky_control/seed{42,43,44}/inputs/paired_split_manifest.json`.
3. Every manifest copy must exist, have identical bytes, have a valid self-hash, and match the
   deployment gate's dataset and manifest identities.
4. Calibration list entries may be outside the workspace only when their canonical resolved paths
   are contained by `<dataset-root>/images`.
5. Calibration stems, order, disjointness from final test, and image SHA-256 values must continue
   to match the frozen manifest exactly.
6. Checkpoint and ONNX paths remain strictly contained by the parent workspace.

Two alternatives are rejected:

- Copying or synthesizing a root-level manifest and dataset tree in Drive would mutate parent
  evidence and could hide further layout mismatches.
- Rerunning the A100 experiment would spend GPU time without changing the already verified model
  artifacts and would not fix the verifier's false assumption.

## Interface changes

`verify_l4_parent_inputs` and `verify_l4_inputs` receive an explicit `dataset_root: Path`. The
benchmark and L4-package CLIs add a required `--dataset` argument. Every internal verification or
resume path propagates the same resolved dataset root; no component infers it from the current
working directory or from mutable environment variables.

The rendered notebook binds:

```text
PARENT_DATASET = /content/drive/MyDrive/pcb-defect-paired/dataset/pcb
```

and passes it to the initial parent check, benchmark command, benchmark completion verifier,
package command, and package verifier. The notebook continues to derive the parent workspace only
from the immutable parent experiment SHA.

## Manifest resolution

The verifier uses a fixed ordered list of the six production run paths. It reads raw bytes first
and rejects a missing file or any byte difference before parsing the canonical first copy. The
parsed payload must retain the existing self-hash, dataset SHA-256, partition, and sample-content
checks. A convenient root-level copy, if present, is neither required nor authoritative.

The verified parent result exposes the canonical manifest path. L4 packaging includes that actual
run-scoped path instead of the nonexistent `inputs/paired_split_manifest.json`. This preserves the
source location in the private audit package and avoids any staging mutation.

## Dataset and calibration boundary

The dataset root must exist, resolve canonically, and be distinct from the parent workspace. Each
calibration entry must:

- be an absolute canonical path;
- resolve beneath `<dataset-root>/images`;
- name the exact ordered calibration stem declared by the manifest;
- be a regular image file whose SHA-256 matches the manifest sample record.

Paths outside the dataset image root, symlink escapes, blank or noncanonical list bytes, duplicate
stems, final-test overlap, reordered entries, or changed image bytes fail closed. Dataset pixels
remain outside the private result ZIP.

## Packaging

The L4 package inventory uses the verified run-scoped manifest path. It continues to include the
input lock, deployment contract, checkpoint, ONNX model, benchmark report, TensorRT engine, and
durable command log required by the existing private evidence design. No dataset pixels are added.
Existing package reuse remains valid only when the complete inventory and all content hashes match
the current verified workspace.

## Notebook and handoff migration

A corrected handoff is generated into a new runner-derived local and Drive directory. The failed
`2c87d70f3c18` handoff is not overwritten or accepted. The new rendered notebook contains no
manual placeholders and is run in a fresh Colab L4 runtime. The user uploads exactly the new
notebook, manifest, and source bundle, while retaining the existing A100 workspace and dataset.

The corrected source is published to the existing public GitHub repository through a new
`kuotunyu`-authored fast-forward commit. No legacy history, tags, additional branches, co-author
trailers, or non-`kuotunyu` identities are published. Post-push verification must confirm that the
GitHub Contributors API still returns only `kuotunyu`.

## Testing strategy

Implementation follows red-green TDD. Production-shaped CPU fixtures place manifests under all
six run directories and pixels under an external dataset root. Tests cover:

- current code fails against the real A100 layout before the fix;
- six identical run manifests are accepted;
- one missing, byte-different, self-hash-invalid, gate-mismatched, or reordered manifest fails;
- an external calibration image inside the explicit dataset root is accepted;
- paths outside the dataset root, symlink escapes, changed bytes, wrong order, duplicates, or
  final-test images fail;
- benchmark and package CLIs require and propagate `--dataset` on fresh and resume paths;
- the package inventory contains the verified run-scoped manifest and no pixels;
- the rendered notebook passes the immutable dataset root to every verification boundary;
- old synthetic fixtures remain supported only after they adopt the explicit production layout.

After focused red-green cycles, run the complete CPU pytest suite, Ruff check and format check,
wheel and sdist build, notebook AST and placeholder checks, clean-bundle clone verification,
tracked-artifact and secret scans, and Git identity audit. GPU execution is not part of local
verification.

## Failure and recovery behavior

Errors identify the missing or mismatched production path rather than collapsing every layout
problem into a generic parent-evidence error. No verifier creates, copies, rewrites, or deletes
parent files. A failed Colab run can be resumed only when the completed benchmark and package pass
the same dataset-aware identity checks; otherwise it fails without overwriting prior evidence.

## Definition of done

The local correction is complete only when:

1. production-shaped tests reproduce the current failure and pass after the fix;
2. all six parent manifests and the external dataset boundary are independently verified;
3. all CPU tests, Ruff checks, and builds pass;
4. a new exact-three-file handoff passes bundle, manifest, notebook, and clean-clone audits;
5. both local worktrees are clean and no private model, export, engine, result, pixel, or secret is
   tracked;
6. the public repository is fast-forwarded using only the approved `kuotunyu` identity and its
   Contributors API still contains only `kuotunyu`.

The GPU phase is complete only after the corrected notebook prints `L4 HANDOFF COMPLETE` and the
returned ZIP plus `.sha256` sidecar pass a separate read-only audit.

