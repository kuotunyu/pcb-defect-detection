# Research package

## Publication status

The [v0.2.0 GitHub Release](https://github.com/kuotunyu/pcb-defect-detection/releases/tag/v0.2.0)
and Zenodo version record
[`10.5281/zenodo.21912370`](https://doi.org/10.5281/zenodo.21912370) publish the same deterministic
archive built from source commit
`7fc1777d306584fc1f3ffe0c05989296370fe6df`: 1,699,878 bytes with SHA-256
`89d82a6ab8737193f8c59614d2a04c68f07b02fca3bc7d3ee7178c56ff882f29`. The all-versions DOI
remains [`10.5281/zenodo.21877496`](https://doi.org/10.5281/zenodo.21877496). This version adds the
final responsive PCB review-workstation presentation without changing the scientific claims,
failed strict parity gate, or redistribution boundary.

### Immutable v0.1.0 record

The [v0.1.0 GitHub Release](https://github.com/kuotunyu/pcb-defect-detection/releases/tag/v0.1.0)
and Zenodo version record
[`10.5281/zenodo.21877497`](https://doi.org/10.5281/zenodo.21877497) publish the same deterministic
archive built from source commit
`56c086206eab9be1a9c6a4e36410fd13ed42f5ec`: 1,505,971 bytes with SHA-256
`21abbe3c71c5f7b962a8c33a8bc649dbe98757199a6ae17b5a6af0bbe27998e1`. The all-versions DOI is
[`10.5281/zenodo.21877496`](https://doi.org/10.5281/zenodo.21877496). The final L4 evidence records
both aggregate fidelity passage and strict prediction-parity failure. Publication does not expand
the package's redistribution or performance claims.

## Scope

The package contains allowlisted UTF-8 source and metadata evidence from the clean, committed tree needed to audit the
paired split, training recipe, statistical results, deployment contracts, and public claim
boundaries. Each included file is recorded by relative path, byte count, and SHA-256 in the
embedded `RESEARCH_PACKAGE_MANIFEST.json`.

Unknown or binary file types fail closed at the redistribution boundary; even an allowlisted text
suffix must decode as UTF-8 and contain no NUL bytes. The package deliberately excludes dataset pixels, pixel-derived previews, checkpoints, ONNX
exports, TensorRT engines, result ZIPs, logs, and caches. It includes the path-free L4 summary,
complete raw timings, and pseudonymized per-box parity evidence. Those exclusions preserve the
unresolved dataset/model redistribution boundary and prevent a hardware-specific engine from
being mistaken for a portable artifact.

## Build after the final commit

From a clean clone at the exact release-candidate commit:

```bash
uv sync --locked --no-editable
uv run python -m pcb_defect.research_package \
  --repo . \
  --output dist/pcb-defect-detection-v0.2.0-research-package.zip
```

The command refuses a dirty tracked worktree, refuses secret-shaped tracked paths, writes a
deterministic ZIP plus `.sha256` sidecar, and verifies every member against the embedded manifest
before returning success. The published archive was generated from the exact `v0.2.0` release commit.
The historical archive generated from tagged commit
`56c086206eab9be1a9c6a4e36410fd13ed42f5ec` remains attached to the `v0.1.0` GitHub Release and
deposited on Zenodo. Running this command from any other commit produces a different archive; do
not present that result as the `v0.1.0` asset, the `v0.2.0` asset, or commit it back into this
repository.

## Citation and deposit metadata

- `CITATION.cff` defines the single software author, current software version, release date,
  version DOI, and GitHub source URL.
- `.zenodo.json` defines the title, keywords, access mode, license, and redistribution note.
- `docs/license-boundary.md` remains authoritative for third-party data and model artifacts.

Use `10.5281/zenodo.21912370` when citing v0.2.0, `10.5281/zenodo.21877497` when citing v0.1.0,
and the all-versions DOI when referring to the evolving software record. Every new version must
receive its own Zenodo version DOI; do not reuse an earlier version DOI or add a public-model URL
unless the corresponding artifact has actually been published.
