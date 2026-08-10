# Research package

## Publication status

This archive design is **not yet published** and has no DOI. The final L4 evidence has been
reviewed and records both aggregate fidelity passage and strict prediction-parity failure. After
the corresponding Git commit is frozen and the deterministic archive is verified, it is ready for
a Zenodo draft deposit; external publication remains a separate deliberate action.

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
before returning success. The generated archive belongs in a Zenodo draft or GitHub Release
attachment; it must not be committed back into this repository.

## Citation and deposit metadata

- `CITATION.cff` defines the single software author and GitHub source URL.
- `.zenodo.json` defines the title, keywords, access mode, license, and redistribution note.
- `docs/license-boundary.md` remains authoritative for third-party data and model artifacts.

Create a DOI only after the final evidence commit is immutable. Do not enter a fabricated DOI,
release date, or public-model URL before the corresponding external publication exists.
