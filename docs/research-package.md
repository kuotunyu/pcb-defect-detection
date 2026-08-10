# Research package

## Publication status

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
  --output dist/pcb-defect-detection-research.zip
```

The command refuses a dirty tracked worktree, refuses secret-shaped tracked paths, writes a
deterministic ZIP plus `.sha256` sidecar, and verifies every member against the embedded manifest
before returning success. The verified archive generated from tagged commit
`56c086206eab9be1a9c6a4e36410fd13ed42f5ec` is attached to the `v0.1.0` GitHub Release and
deposited on Zenodo. Running this command from any other commit produces a different archive; do
not present that result as the `v0.1.0` asset or commit it back into this repository.

## Citation and deposit metadata

- `CITATION.cff` defines the single software author, v0.1.0 DOI, release date, and GitHub source URL.
- `.zenodo.json` defines the title, keywords, access mode, license, and redistribution note.
- `docs/license-boundary.md` remains authoritative for third-party data and model artifacts.

Use the version DOI when citing v0.1.0 and the all-versions DOI when referring to the evolving
software record. Future versions must receive their own Zenodo version DOI; do not reuse the
v0.1.0 DOI or add a public-model URL unless the corresponding artifact has actually been published.
