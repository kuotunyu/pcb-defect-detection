# A100 Evidence Promotion Design

## Objective

Promote the completed paired A100 experiment from a private result package into small,
content-addressed, reviewable repository evidence. Update the portfolio-facing documents so every
current numerical or deployment claim points to committed machine-readable artifacts, while
keeping model binaries, dataset assets, and release claims outside the candidate tree until their
license and account gates are resolved.

## Verified source package

The promotion source is `paired-results-a100-9e3a1ed5827a.zip` with SHA-256
`c6158e5017bbec02c089ad42f55f83b20a46fb8ebd3bd1c35115d2940bb74c92`. Its downloaded sidecar
matches that digest. The embedded `package_manifest.json` contains 52 files; an independent
read-only verification found zero missing files, size mismatches, or SHA-256 mismatches.

The package is bound to:

- source snapshot Git SHA `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`;
- dataset SHA-256 `8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`;
- paired protocol manifest SHA-256
  `5996d595f5ce17fabd24e631ce580bbf9932a845f9898078267df8c2522892e5`;
- pinned YOLO26n base checkpoint SHA-256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

## Evidence boundary

Create `reports/paired_a100/` and promote only machine-readable metadata:

- `input_lock.json` — immutable dataset, protocol, base model, and source snapshot identities;
- `gate_report.json` — clean runtime, data, GPU, tiny-train, resume, and speed gates;
- `deployment_selection.json` — validation-only grouped checkpoint selection before final-test use;
- `final_metrics.json` — six-run results, final-test aggregates, per-class data, and paired bootstrap;
- `finalization_record.json` — completed final evaluation and its content digest;
- `deployment_gate.public.json` — a path-free public projection of calibration fidelity,
  standalone ONNX parity, and runtime contract;
- `package_manifest.json` — the original 52-file package inventory and hashes;
- `result_package_receipt.json` — a repository-owned receipt for the external ZIP and verification;
- `README.md` — a concise evidence index, limitations, and non-release boundary.

The input lock, gate report, deployment selection, final metrics, finalization record, and package
manifest are copied byte-for-byte from the verified ZIP. The raw deployment gate is not committed
because its `command` field contains Colab and Google Drive absolute paths. Instead,
`deployment_gate.public.json` removes only `command`, preserves every other gate field, and adds:

```json
"_provenance": {
  "source_entry": "deployment/deployment_gate.json",
  "source_bytes": 18615,
  "source_sha256": "466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803",
  "removed_fields": ["command"]
}
```

The package manifest independently records the same raw gate path, byte count, and digest. Tests
bind the public projection's provenance to that manifest entry and reject `command`, `/content/`,
`/root/`, local drive paths, and `MyDrive` in the public projection.

The receipt records only portable facts: package basename, byte count, package SHA-256, source Git SHA, embedded manifest
SHA-256, number of verified entries, verification status, and the corresponding sidecar basename.
It must not contain a local download path, Drive path, user identity, timestamp that cannot be
reproduced, or a claim that binaries are publicly released.

Do not commit:

- the 75 MB result ZIP or its downloaded sidecar;
- any `.pt`, `.onnx`, TensorRT engine, dataset image, annotation, cache, or run directory;
- pixel-derived screenshots or qualitative examples;
- Colab command logs or absolute filesystem paths.

## Claim promotion

Update `reports/claims.yaml` as follows:

1. Keep `paired_protocol` and `base_initialization` verified.
2. Keep `legacy_split_sensitivity` as `legacy_observed`; its 12.1-point observation remains clearly
   separate from the new paired result.
3. Promote `paired_leakage_effect` to `verified`, backed by the input lock, deployment selection,
   final metrics, finalization record, package receipt, and frozen protocol artifacts.
4. Promote `onnx_deployment` to `verified_candidate`, not `released`. Its statement is limited to
   the hash-pinned selected grouped checkpoint passing calibration fidelity and standalone ONNX
   parity. Evidence is the path-free public deployment gate plus the input lock and package receipt. Legacy
   failed export reports are not evidence for this claim.
5. Keep `tensorrt_performance` as `pending_l4`; legacy T4 reports remain historical only.
6. Keep `hosted_demo` blocked.

The current paired result values are:

- grouped mAP50 `0.6330074066602414 ± 0.14914951898858905` across three seeds;
- leaky-control mAP50 `0.8455772487997025 ± 0.037528675571419035`;
- leaky-minus-grouped mAP50 difference `0.2125698421394611`, or 21.3 percentage points;
- grouped mAP50-95 `0.2881752374835738`;
- leaky-control mAP50-95 `0.40084514509687946`;
- paired image-bootstrap F1 delta `0.2545904095904096`, 95% CI
  `[0.2101614203697537, 0.3005499623832958]`, 10,000 resamples over 30 final-test images.

The deployment gate records:

- PyTorch-to-ONNX calibration mAP50 delta `-0.018590364845378238`;
- PyTorch-to-ONNX calibration mAP50-95 delta `-0.012809288682730458`;
- threshold `0.02` for each fidelity delta;
- 60 parity images, zero failures, minimum IoU `1.0`, and maximum confidence delta `0.0`;
- unchanged Linux `onnxruntime-gpu==1.26.0` state before and after deployment;
- ONNX SHA-256 `b62590a14e2e88a414eb06389058d13d69ff1ea3998232996877088951fe3bb8`.

## Portfolio document changes

### README

Change the status to state that paired A100 and ONNX candidate evidence are complete, while L4,
license approval, model release, and hosted demo remain pending. Replace the evidence-first third
item with the measured paired leakage result and add the ONNX gate as a compact deployment result.
Retain the legacy 12.1-point section as historical context, but explicitly identify the new
21.3-point paired result as the current claim.

The deployment section must distinguish technical deployability from authorization to distribute
the selected checkpoint or ONNX file. It must not imply a public model or live endpoint.

### Model card

Replace the pending-model wording with a candidate-model card. Record the six-run design, selected
grouped seed 42, aggregate metrics, paired bootstrap, ONNX gate, source identities, intended use,
limitations, and release boundary. The status remains unreleased.

### Release checklist

Mark the following Colab items complete because their committed artifacts now support them:

- A100 clean-runtime and experiment gates;
- all six runs and hash-bound records;
- pre-final-test grouped checkpoint selection;
- one-shot final evaluation with three-seed and bootstrap output;
- calibration fidelity and standalone parity;
- returned final ZIP and matching sidecar.

Leave the L4 benchmark item unchecked. Leave every license, identity, official-account, model Hub,
and public-release item unchecked.

## Data flow and failure handling

The local result package is the only extraction source. Before promotion, the process must:

1. recompute the ZIP SHA-256 and compare it with the sidecar;
2. open the archive without extracting binaries;
3. validate all 52 manifest entries by path, byte count, and SHA-256;
4. require every selected JSON entry;
5. require `status=complete` in final metrics and finalization record;
6. require all A100 gates and the raw deployment gate to pass;
7. require the package input lock to match the committed protocol and base-model identities;
8. copy the six approved source JSON files byte-for-byte, generate the path-free deployment gate
   projection, and generate the portable receipt;
9. bind the public gate provenance to the raw gate entry in the package manifest;
10. reject any promoted JSON containing the local download root, `/content/`, `/root/`, `MyDrive`,
    a local drive path, or an unapproved binary payload.

Any mismatch stops promotion. Existing repository claims remain unchanged until all evidence and
contract tests pass together.

## Test strategy

Extend `tests/test_release_contract.py` to verify:

- every promoted claim evidence path exists;
- verified status is limited to the protocol, base initialization, paired leakage, and ONNX
  candidate claims;
- the receipt matches the known result package name, bytes, SHA-256, source snapshot, and 52-entry
  verification count;
- the copied input lock matches the frozen protocol configuration;
- finalization hashes the committed `final_metrics.json` bytes;
- the public deployment gate contains no removed command or absolute path, is bound to the raw gate
  entry in the package manifest, is passed, points to the selected grouped seed 42 checkpoint,
  stays within fidelity thresholds, passes 60/60 parity, and records an unchanged runtime contract;
- README values are derived from committed final metrics rather than hard-coded independently;
- the A100 checklist items are checked while L4 and release-rights items remain unchecked;
- no tracked ZIP, model weight, ONNX file, engine, dataset pixel, or forbidden local path is added.

Run the full 132-test suite, Ruff check, Ruff format check, the claims-to-artifacts tests, and Git
status verification in the locked non-editable CPU environment. No GPU work, training, benchmark,
network publication, remote change, release, or paid API is part of this promotion.

## Acceptance criteria

- The repository contains the complete metadata chain needed to audit the A100 leakage and ONNX
  candidate claims without containing model or dataset binaries.
- Every current README number is machine-backed by committed artifacts.
- Legacy evidence remains visibly legacy and cannot satisfy new claims.
- L4, TensorRT performance, license approval, model release, and hosted demo remain explicitly
  pending or blocked.
- The tracked worktree is clean after focused commits, and all CPU-safe checks pass.
