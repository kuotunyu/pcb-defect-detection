# A100 Evidence Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the verified paired A100 experiment into small committed metadata artifacts and update portfolio claims and documents without committing model or dataset binaries.

**Architecture:** Copy six authoritative JSON entries byte-for-byte from the independently verified result ZIP into `reports/paired_a100/`, derive one path-free public deployment gate that is content-bound to the raw gate, then add one portable receipt and one evidence index. Extend release-contract tests so claims, README values, finalization hashes, ONNX gates, checklist state, and the binary exclusion boundary are all derived from those committed artifacts.

**Tech Stack:** Python 3.11, pytest, PyYAML, JSON, Markdown, PowerShell/.NET ZIP APIs, Git, uv 0.11.18.

## Global Constraints

- Source ZIP basename: `paired-results-a100-9e3a1ed5827a.zip`.
- Source ZIP byte count: `75767467`.
- Source ZIP SHA-256: `c6158e5017bbec02c089ad42f55f83b20a46fb8ebd3bd1c35115d2940bb74c92`.
- Source snapshot Git SHA: `9e3a1ed5827ac3759cbb15632f041e3e5c183b51`.
- Embedded package manifest: 52 entries, SHA-256 `2387496eb57a8126515f13cde72a83fab799e4a5b9e7a8d93ee1a680d6a7132d`.
- Preserve exact release identity `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` and add no co-author trailer.
- Do not commit the ZIP, sidecar, `.pt`, `.onnx`, TensorRT engine, dataset pixels, annotations, caches, logs, absolute local paths, or Drive paths.
- Do not commit the raw `deployment/deployment_gate.json`; publish `deployment_gate.public.json`
  with only `command` removed and exact raw-entry provenance added.
- Do not use a GPU, train, benchmark, push, publish, create a release, change remotes, call paid APIs, or edit Git history.
- `paired_leakage_effect` may become `verified`; `onnx_deployment` may become `verified_candidate`; L4/TensorRT remains `pending_l4`; hosted demo remains `blocked`.
- Technical ONNX candidate verification does not authorize redistribution of the selected checkpoint or ONNX binary.
- Run Python commands through `uv run --locked --no-editable` because editable installs under the repository's non-ASCII Windows path are not portable.

---

### Task 1: Promote the byte-exact A100 metadata chain

**Files:**
- Create: `reports/paired_a100/input_lock.json`
- Create: `reports/paired_a100/gate_report.json`
- Create: `reports/paired_a100/deployment_selection.json`
- Create: `reports/paired_a100/final_metrics.json`
- Create: `reports/paired_a100/finalization_record.json`
- Create: `reports/paired_a100/deployment_gate.public.json`
- Create: `reports/paired_a100/package_manifest.json`
- Create: `reports/paired_a100/result_package_receipt.json`
- Modify: `tests/test_release_contract.py`

**Interfaces:**
- Consumes: the verified external ZIP and sidecar, addressed through task-specific shell variables rather than a committed local path.
- Produces: `PAIRED_A100 = ROOT / "reports" / "paired_a100"`, six byte-exact source JSON files, one path-free public gate, and a portable receipt used by Tasks 2 and 3.

- [ ] **Step 1: Add JSON and hashing helpers plus failing evidence-chain tests**

Add near the constants in `tests/test_release_contract.py`:

```python
PAIRED_A100 = ROOT / "reports" / "paired_a100"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

Add these tests:

```python
def test_promoted_a100_package_receipt_is_portable_and_complete() -> None:
    receipt = _read_json(PAIRED_A100 / "result_package_receipt.json")

    assert receipt == {
        "schema_version": "1.0",
        "package": {
            "name": "paired-results-a100-9e3a1ed5827a.zip",
            "bytes": 75767467,
            "sha256": "c6158e5017bbec02c089ad42f55f83b20a46fb8ebd3bd1c35115d2940bb74c92",
            "sidecar": "paired-results-a100-9e3a1ed5827a.zip.sha256",
        },
        "source_git_sha": "9e3a1ed5827ac3759cbb15632f041e3e5c183b51",
        "package_manifest": {
            "path": "package_manifest.json",
            "sha256": "2387496eb57a8126515f13cde72a83fab799e4a5b9e7a8d93ee1a680d6a7132d",
            "verified_entries": 52,
            "failed_entries": 0,
        },
        "verification": {
            "sidecar_match": True,
            "internal_manifest_passed": True,
        },
    }


def test_promoted_a100_input_and_finalization_chain_is_hash_bound() -> None:
    input_lock = _read_json(PAIRED_A100 / "input_lock.json")
    protocol = yaml.safe_load(
        (ROOT / "configs" / "paired_protocol.yaml").read_text(encoding="utf-8")
    )
    gate = _read_json(PAIRED_A100 / "gate_report.json")
    selection = _read_json(PAIRED_A100 / "deployment_selection.json")
    metrics = _read_json(PAIRED_A100 / "final_metrics.json")
    finalization = _read_json(PAIRED_A100 / "finalization_record.json")

    assert input_lock["git_sha"] == "9e3a1ed5827ac3759cbb15632f041e3e5c183b51"
    assert input_lock["dataset_sha256"] == protocol["frozen_hashes"]["dataset_sha256"]
    assert input_lock["manifest_sha256"] == protocol["frozen_hashes"]["manifest_sha256"]
    assert gate["passed"] is True
    assert all(gate["checks"].values())
    assert gate["input_lock"] == input_lock
    assert selection["arm"] == "grouped"
    assert selection["seed"] == 42
    assert selection["selected_before_final_test"] is True
    assert metrics["status"] == "complete"
    assert metrics["deployment_selection"] == selection
    assert metrics["git_sha"] == input_lock["git_sha"]
    assert finalization["status"] == "complete"
    assert finalization["results"] == "final_metrics.json"
    assert finalization["results_bytes"] == (PAIRED_A100 / "final_metrics.json").stat().st_size
    assert finalization["results_sha256"] == _sha256(PAIRED_A100 / "final_metrics.json")


def test_public_deployment_gate_is_path_free_and_bound_to_raw_manifest_entry() -> None:
    public_path = PAIRED_A100 / "deployment_gate.public.json"
    public_text = public_path.read_text(encoding="utf-8")
    gate = json.loads(public_text)
    manifest = _read_json(PAIRED_A100 / "package_manifest.json")
    raw = next(
        item for item in manifest["files"] if item["path"] == "deployment/deployment_gate.json"
    )

    assert gate["_provenance"] == {
        "source_entry": "deployment/deployment_gate.json",
        "source_bytes": 18615,
        "source_sha256": "466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803",
        "removed_fields": ["command"],
    }
    assert gate["_provenance"]["source_bytes"] == raw["bytes"]
    assert gate["_provenance"]["source_sha256"] == raw["sha256"]
    assert "command" not in gate
    assert not any(token in public_text for token in ("/content/", "/root/", "MyDrive", "C:\\"))
```

- [ ] **Step 2: Run the focused tests and confirm the missing-evidence failure**

Run:

```powershell
uv run --locked --no-editable pytest tests/test_release_contract.py::test_promoted_a100_package_receipt_is_portable_and_complete tests/test_release_contract.py::test_promoted_a100_input_and_finalization_chain_is_hash_bound tests/test_release_contract.py::test_public_deployment_gate_is_path_free_and_bound_to_raw_manifest_entry -q
```

Expected: all three tests fail because `reports/paired_a100/` does not exist.

- [ ] **Step 3: Reverify the external ZIP and sidecar before extraction**

Set `$a100Zip` and `$a100Sidecar` only in the current PowerShell session. Do not put their absolute values in tracked files. Run:

```powershell
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $a100Zip).Hash.ToLowerInvariant()
$sidecarFields = (Get-Content -LiteralPath $a100Sidecar -Raw).Split(
  [char[]]" `t`r`n", [System.StringSplitOptions]::RemoveEmptyEntries
)
if ($actual -ne 'c6158e5017bbec02c089ad42f55f83b20a46fb8ebd3bd1c35115d2940bb74c92') {
  throw 'Unexpected A100 package SHA-256'
}
if ($actual -ne $sidecarFields[0].ToLowerInvariant()) {
  throw 'A100 package sidecar mismatch'
}
if ((Get-Item -LiteralPath $a100Zip).Length -ne 75767467) {
  throw 'Unexpected A100 package byte count'
}
```

Expected: no output and exit code 0.

- [ ] **Step 4: Reverify all embedded manifest entries in memory**

Run this read-only `.NET` verification before any repository extraction:

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($a100Zip)
try {
  $manifestEntry = $archive.GetEntry('package_manifest.json')
  if ($null -eq $manifestEntry) { throw 'Missing package_manifest.json' }
  $reader = [System.IO.StreamReader]::new($manifestEntry.Open())
  try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
  $failures = @()
  foreach ($file in $manifest.files) {
    $entry = $archive.GetEntry([string]$file.path)
    if ($null -eq $entry) { $failures += ('missing:' + $file.path); continue }
    if ($entry.Length -ne [int64]$file.bytes) { $failures += ('size:' + $file.path); continue }
    $stream = $entry.Open()
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
      $hash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    } finally {
      $sha.Dispose()
      $stream.Dispose()
    }
    if ($hash -ne [string]$file.sha256) { $failures += ('sha:' + $file.path) }
  }
  Write-Output ('MANIFEST_FILES=' + $manifest.files.Count)
  Write-Output ('MANIFEST_FAILURES=' + $failures.Count)
  if ($manifest.files.Count -ne 52 -or $failures.Count -ne 0) {
    $failures
    throw 'A100 package manifest verification failed'
  }
  Write-Output 'PACKAGE_INTERNAL_INTEGRITY=PASS'
} finally {
  $archive.Dispose()
}
```

The final command output must be exactly equivalent to:

```text
MANIFEST_FILES=52
MANIFEST_FAILURES=0
PACKAGE_INTERNAL_INTEGRITY=PASS
```

- [ ] **Step 5: Extract only the six approved source JSON entries byte-for-byte**

Create `reports/paired_a100/` and use this exact mapping with ZIP entry streams:

```powershell
$mapping = [ordered]@{
  'inputs/input_lock.json' = 'input_lock.json'
  'gates/gate_report.json' = 'gate_report.json'
  'final/deployment_selection.json' = 'deployment_selection.json'
  'final/final_metrics.json' = 'final_metrics.json'
  'final/finalization_record.json' = 'finalization_record.json'
  'package_manifest.json' = 'package_manifest.json'
}
```

Extract without JSON reserialization:

```powershell
$destinationRoot = Join-Path (Resolve-Path '.') 'reports\paired_a100'
New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($a100Zip)
try {
  foreach ($item in $mapping.GetEnumerator()) {
    $entry = $archive.GetEntry([string]$item.Key)
    if ($null -eq $entry) { throw ('Missing approved entry: ' + $item.Key) }
    $destination = Join-Path $destinationRoot ([string]$item.Value)
    $sourceStream = $entry.Open()
    $destinationStream = [System.IO.File]::Create($destination)
    try { $sourceStream.CopyTo($destinationStream) } finally {
      $destinationStream.Dispose()
      $sourceStream.Dispose()
    }
  }
} finally {
  $archive.Dispose()
}
```

After extraction, require these exact digests:

```text
input_lock.json            51e5d26e93633041f46283b3bcfaa1bc06ee982d831235cdd4e1b0260da2e080
gate_report.json           eb423f34c6935a21022c7e895105548e4ac1d98cca1d0b53c227e7d95b5bdd11
deployment_selection.json  088af5b72ab326d248209e214c6315d12766bc454534a8507383602e42c5d81d
final_metrics.json         2090d3e7069f2ecffaf2e820216c054075327a6440251878e96b0258b617b022
finalization_record.json   25b7d4e114cbe127fe827c22fa5015c1504cff92299ec502c0dbc2ae48ec2132
package_manifest.json      2387496eb57a8126515f13cde72a83fab799e4a5b9e7a8d93ee1a680d6a7132d
```

- [ ] **Step 6: Create the path-free public deployment gate**

Read `deployment/deployment_gate.json` directly from the ZIP. Require its byte count and SHA-256 to
equal `18615` and `466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803`.
Parse the JSON, remove only `command`, add the exact `_provenance` dictionary asserted in Step 1,
serialize with two-space indentation and a final newline, and write UTF-8 without a BOM to
`reports/paired_a100/deployment_gate.public.json`. Require that none of `/content/`, `/root/`,
`MyDrive`, or `C:\` occurs in the serialized bytes.

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($a100Zip)
try {
  $entry = $archive.GetEntry('deployment/deployment_gate.json')
  if ($null -eq $entry -or $entry.Length -ne 18615) { throw 'Unexpected raw deployment gate' }
  $source = $entry.Open()
  $sha = [System.Security.Cryptography.SHA256]::Create()
  try { $rawHash = [BitConverter]::ToString($sha.ComputeHash($source)).Replace('-', '').ToLowerInvariant() }
  finally { $sha.Dispose(); $source.Dispose() }
  if ($rawHash -ne '466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803') {
    throw 'Unexpected raw deployment gate SHA-256'
  }
  $reader = [System.IO.StreamReader]::new($entry.Open())
  try { $gate = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
} finally {
  $archive.Dispose()
}
$gate.PSObject.Properties.Remove('command')
$gate | Add-Member -NotePropertyName '_provenance' -NotePropertyValue ([ordered]@{
  source_entry = 'deployment/deployment_gate.json'
  source_bytes = 18615
  source_sha256 = '466bf152a30e7efe1768542a71647e8982d18df253b2b170aaa2a13d087c1803'
  removed_fields = @('command')
})
$publicJson = ($gate | ConvertTo-Json -Depth 100) + "`n"
if ($publicJson -match '/content/|/root/|MyDrive|[A-Za-z]:\\') {
  throw 'Public deployment gate still contains an absolute path'
}
$publicPath = Join-Path $destinationRoot 'deployment_gate.public.json'
[System.IO.File]::WriteAllText($publicPath, $publicJson, [System.Text.UTF8Encoding]::new($false))
```

- [ ] **Step 7: Add the portable result receipt**

Create `reports/paired_a100/result_package_receipt.json` with exactly the dictionary asserted in
Step 1, serialized as indented UTF-8 JSON with a final newline. Do not include a timestamp or path.

- [ ] **Step 8: Run the focused evidence tests**

Run:

```powershell
uv run --locked --no-editable pytest tests/test_release_contract.py::test_promoted_a100_package_receipt_is_portable_and_complete tests/test_release_contract.py::test_promoted_a100_input_and_finalization_chain_is_hash_bound tests/test_release_contract.py::test_public_deployment_gate_is_path_free_and_bound_to_raw_manifest_entry -q
```

Expected: `3 passed`.

- [ ] **Step 9: Commit the metadata chain and contract tests**

```powershell
git add -- reports/paired_a100 tests/test_release_contract.py
git -c user.name='kuotunyu' -c user.email='61350295+kuotunyu@users.noreply.github.com' commit -m 'Add verified A100 metadata evidence'
```

---

### Task 2: Promote the paired leakage and ONNX candidate claims

**Files:**
- Modify: `reports/claims.yaml`
- Modify: `tests/test_release_contract.py`

**Interfaces:**
- Consumes: the Task 1 A100 evidence chain.
- Produces: machine-readable claim statuses and evidence paths used by README and model-card review.

- [ ] **Step 1: Change the claim registry test to the approved status contract**

Replace the exact verified-name assertion in
`test_claim_evidence_paths_exist_and_only_supported_claims_are_verified` with:

```python
    assert {name for name, claim in claims.items() if claim["status"] == "verified"} == {
        "base_initialization",
        "paired_leakage_effect",
        "paired_protocol",
    }
    assert {name for name, claim in claims.items() if claim["status"] == "verified_candidate"} == {
        "onnx_deployment"
    }
    assert claims["tensorrt_performance"]["status"] == "pending_l4"
    assert claims["hosted_demo"]["status"] == "blocked"
```

Keep the existence check for every evidence path. Require non-empty limitations for
`pending_colab`, `pending_l4`, `blocked`, and `verified_candidate` statuses.

- [ ] **Step 2: Add a failing test for claim-to-artifact semantics**

```python
def test_promoted_claims_point_only_to_current_a100_evidence() -> None:
    claims = yaml.safe_load((ROOT / "reports" / "claims.yaml").read_text(encoding="utf-8"))[
        "claims"
    ]
    paired = claims["paired_leakage_effect"]
    onnx = claims["onnx_deployment"]

    assert paired["status"] == "verified"
    assert set(paired["evidence"]) >= {
        "reports/paired_a100/input_lock.json",
        "reports/paired_a100/deployment_selection.json",
        "reports/paired_a100/final_metrics.json",
        "reports/paired_a100/finalization_record.json",
        "reports/paired_a100/result_package_receipt.json",
    }
    assert onnx["status"] == "verified_candidate"
    assert set(onnx["evidence"]) == {
        "reports/paired_a100/input_lock.json",
        "reports/paired_a100/deployment_gate.public.json",
        "reports/paired_a100/result_package_receipt.json",
    }
    assert "reports/export_fidelity.json" not in onnx["evidence"]
    assert "reports/onnx_parity.json" not in onnx["evidence"]
```

- [ ] **Step 3: Run the claim tests and verify they fail on the pending/blocked registry**

Run:

```powershell
uv run --locked --no-editable pytest tests/test_release_contract.py::test_claim_evidence_paths_exist_and_only_supported_claims_are_verified tests/test_release_contract.py::test_promoted_claims_point_only_to_current_a100_evidence -q
```

Expected: failure because `paired_leakage_effect` is `pending_colab`, `onnx_deployment` is
`blocked`, and the current evidence paths are absent.

- [ ] **Step 4: Update `reports/claims.yaml`**

Use these exact status and statement boundaries:

```yaml
  paired_leakage_effect:
    status: verified
    statement: >-
      Under the frozen paired protocol, same-board sibling exposure increased three-seed
      final-test mAP50 from 0.6330 to 0.8456, a 21.3 percentage-point difference.
    evidence:
      - reports/protocol/paired_split_manifest.json
      - reports/paired_a100/input_lock.json
      - reports/paired_a100/deployment_selection.json
      - reports/paired_a100/final_metrics.json
      - reports/paired_a100/finalization_record.json
      - reports/paired_a100/result_package_receipt.json
    limitations:
      - The common final test has 30 images from one board; image-bootstrap intervals do not estimate between-board variance.
      - This is a controlled same-board exposure result for this frozen dataset and training recipe, not a universal YOLO performance claim.

  onnx_deployment:
    status: verified_candidate
    statement: >-
      The validation-selected grouped seed-42 checkpoint passed the hash-pinned calibration
      fidelity gate and standalone ONNX Runtime parity gate as an unreleased deployment candidate.
    evidence:
      - reports/paired_a100/input_lock.json
      - reports/paired_a100/deployment_gate.public.json
      - reports/paired_a100/result_package_receipt.json
    limitations:
      - Technical gate passage does not establish redistribution rights for the checkpoint or ONNX file.
      - No public model revision or hosted endpoint is claimed.
```

Change `tensorrt_performance.status` to `pending_l4`, retain its legacy evidence links, and state
that a new L4 run is required. Leave `hosted_demo` blocked.

- [ ] **Step 5: Run the claim tests**

Run:

```powershell
uv run --locked --no-editable pytest tests/test_release_contract.py::test_claim_evidence_paths_exist_and_only_supported_claims_are_verified tests/test_release_contract.py::test_promoted_claims_point_only_to_current_a100_evidence -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit the claim promotion**

```powershell
git add -- reports/claims.yaml tests/test_release_contract.py
git -c user.name='kuotunyu' -c user.email='61350295+kuotunyu@users.noreply.github.com' commit -m 'Promote paired A100 evidence claims'
```

---

### Task 3: Update the portfolio narrative and release state

**Files:**
- Create: `reports/paired_a100/README.md`
- Modify: `README.md`
- Modify: `docs/model-card.md`
- Modify: `docs/release-checklist.md`
- Modify: `tests/test_release_contract.py`

**Interfaces:**
- Consumes: Task 1 artifacts and Task 2 claim registry.
- Produces: recruiter-facing evidence, candidate model documentation, and a truthful release checklist.

- [ ] **Step 1: Add failing README-number and checklist-state tests**

Add:

```python
def test_portfolio_documents_match_promoted_a100_metrics() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    metrics = _read_json(PAIRED_A100 / "final_metrics.json")
    grouped = metrics["aggregate"]["by_arm"]["grouped"]["map50"]
    leaky = metrics["aggregate"]["by_arm"]["leaky_control"]["map50"]
    difference_pp = (leaky["mean"] - grouped["mean"]) * 100

    for document in (readme, model_card):
        assert f"{grouped['mean']:.4f}" in document
        assert f"{grouped['std']:.4f}" in document
        assert f"{leaky['mean']:.4f}" in document
        assert f"{leaky['std']:.4f}" in document
        assert f"{difference_pp:.1f}" in document
    assert "60/60" in readme
    assert "0.0" in readme
    assert "1.0" in readme


def test_release_checklist_marks_only_returned_a100_evidence_complete() -> None:
    checklist = (ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")

    for text in (
        "A100 clean-runtime, data/hash, tiny-train, resume, and speed gates pass.",
        "Six runs complete with matching run records and checkpoint hashes.",
        "Deployment checkpoint is selected from grouped validation before final-test access.",
        "One-shot common final evaluation completes and reports three-seed mean/std and paired image",
        "Calibration-only ONNX fidelity and standalone parity gates pass.",
        "Final result ZIP and sidecar SHA-256 are returned from Drive.",
    ):
        assert f"- [x] {text}" in checklist
    assert "- [ ] L4 PyTorch/ORT CUDA/TensorRT FP16" in checklist
    assert "- [ ] Upstream dataset redistribution/training/weight-release rights" in checklist
    assert "- [ ] Official GitHub and Hugging Face namespaces" in checklist
```

- [ ] **Step 2: Extend the tracked-binary exclusion test**

Rename `test_candidate_tree_contains_no_dataset_pixel_assets` to
`test_candidate_tree_contains_no_dataset_or_model_binaries` and add:

```python
    model_or_package_files = {
        path
        for path in tracked
        if Path(path).suffix.lower() in {".pt", ".onnx", ".engine", ".plan", ".trt", ".zip"}
    }
    assert model_or_package_files == set()
```

Keep the existing pixel-file assertion and fixture exemption.

- [ ] **Step 3: Run the new document tests and verify failure**

Run:

```powershell
uv run --locked --no-editable pytest tests/test_release_contract.py::test_portfolio_documents_match_promoted_a100_metrics tests/test_release_contract.py::test_release_checklist_marks_only_returned_a100_evidence_complete tests/test_release_contract.py::test_candidate_tree_contains_no_dataset_or_model_binaries -q
```

Expected: the portfolio and checklist tests fail; the binary exclusion test passes.

- [ ] **Step 4: Create `reports/paired_a100/README.md`**

The evidence index must include:

- source ZIP basename, bytes, SHA-256, source snapshot, dataset hash, and protocol hash;
- a table mapping each promoted JSON file to its role;
- grouped and leaky three-seed mAP50/mAP50-95 results;
- 21.3-point paired difference and F1 bootstrap interval;
- selected grouped seed 42 and pre-final-test selection criterion;
- ONNX fidelity deltas and 60/60 standalone parity;
- explicit non-claims: no L4/TensorRT benchmark, public model, release rights, hosted demo, or
  distribution of package/model binaries.

Link the receipt, raw JSON artifacts, `reports/claims.yaml`, and `docs/license-boundary.md` with
relative links.

- [ ] **Step 5: Update the top-level README**

Replace the status block with:

```markdown
> **Status: paired A100 and ONNX candidate evidence complete; release hardening continues.** Six
> paired runs, the one-shot final evaluation, and the hash-pinned ONNX deployment gate have passed.
> L4/TensorRT benchmarking, distribution-rights approval, official model publication, and the live
> demo remain pending.
```

Make the 30-second evidence include:

- grouped `0.6330 ± 0.1491` versus leaky-control `0.8456 ± 0.0375` mAP50;
- the 21.3 percentage-point paired difference;
- paired F1 delta `0.2546`, 95% CI `[0.2102, 0.3005]`;
- the selected grouped seed-42 candidate's 60/60 parity, minimum IoU `1.0`, maximum confidence
  delta `0.0`, and fidelity deltas within the `0.02` gate.

Retitle the legacy section to make the 12.1-point result historical. Update deployment status to
say the candidate passed technical gates but is not released. Link `reports/paired_a100/README.md`.

- [ ] **Step 6: Replace the pending model card with the candidate model card**

Use these sections:

```markdown
# Model card — paired grouped candidate

## Status
Unreleased, technically gate-passed candidate; distribution rights and official model publication remain pending.

## Model and provenance
## Evaluation design
## Paired result
## Selected deployment candidate
## ONNX deployment gate
## Intended use
## Limitations and non-claims
## Release boundary
```

Record the exact source snapshot, dataset/protocol/base-model identities, grouped seed 42 selection,
three-seed metrics, paired bootstrap, fidelity deltas, 60-image parity, and the single-board final
test limitation. Link every numerical section to `reports/paired_a100/` artifacts.

- [ ] **Step 7: Update the release checklist**

Mark the six returned A100 evidence items listed in the test as `[x]`. Keep the L4 benchmark and all
five license/identity/official-migration items `[ ]`. Change the section heading from `Pending Colab
evidence` to `Colab evidence` without implying L4 completion.

- [ ] **Step 8: Run the focused document and release tests**

Run:

```powershell
uv run --locked --no-editable pytest tests/test_release_contract.py -q
```

Expected: all release-contract tests pass.

- [ ] **Step 9: Commit the portfolio narrative**

```powershell
git add -- README.md reports/paired_a100/README.md docs/model-card.md docs/release-checklist.md tests/test_release_contract.py
git -c user.name='kuotunyu' -c user.email='61350295+kuotunyu@users.noreply.github.com' commit -m 'Document paired A100 release evidence'
```

---

### Task 4: Verify the complete promotion and audit the diff

**Files:**
- Verify only; modify only if a verification failure identifies a scoped defect in Tasks 1–3.

**Interfaces:**
- Consumes: all committed artifacts, claims, documents, and tests from Tasks 1–3.
- Produces: a clean, reviewable feature branch ready for integration review and the later L4 task.

- [ ] **Step 1: Verify exact promoted file hashes**

Run `Get-FileHash -Algorithm SHA256` for the six byte-exact source JSON files and compare with the
Task 1 digest table. Re-run the public gate provenance test. Require no mismatches.

- [ ] **Step 2: Run the complete CPU-safe test and style suite**

```powershell
uv sync --locked --no-editable
uv run --locked --no-editable pytest -q
uv run --locked --no-editable ruff check .
uv run --locked --no-editable ruff format --check .
```

Expected: all tests pass, Ruff reports no violations, and all files are formatted.

- [ ] **Step 3: Run the source build gate**

Use a unique temporary output directory outside the candidate tree:

```powershell
$buildOut = Join-Path ([System.IO.Path]::GetTempPath()) (
  'pcb-a100-evidence-build-' + [guid]::NewGuid().ToString('N')
)
uv build --out-dir $buildOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Get-ChildItem -LiteralPath $buildOut | Select-Object Name, Length
```

Expected: wheel and sdist build successfully with exit code 0.

- [ ] **Step 4: Audit tracked content and claim boundaries**

Run:

```powershell
git ls-files
git diff portfolio/pcb-leakage-hardening...HEAD --stat
git diff portfolio/pcb-leakage-hardening...HEAD -- README.md reports/claims.yaml reports/paired_a100 docs/model-card.md docs/release-checklist.md tests/test_release_contract.py
git status --porcelain=v2 --branch
```

Verify:

- no tracked ZIP, sidecar, model/export/engine binary, dataset pixel, log, cache, absolute local path,
  Drive path, test-account name, or secret was added;
- all promoted numerical claims resolve to committed JSON;
- legacy evidence remains labeled legacy;
- L4, TensorRT performance, release rights, official publication, and hosted demo remain pending;
- the worktree is clean.

- [ ] **Step 5: Verify commit identity and history boundary**

```powershell
git log --format='%H%n%an <%ae>%n%cn <%ce>%n%B%n---' 2937186..HEAD
```

Expected: every new commit uses `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` for author
and committer, with no `Co-authored-by` trailer.

- [ ] **Step 6: Request final code review**

Review the complete diff against
`docs/superpowers/specs/2026-08-04-a100-evidence-promotion-design.md`. Any finding must identify an
exact file and evidence mismatch. Address Critical or Important findings in focused commits, then
rerun Steps 1–5. Do not merge, push, release, or start the L4 benchmark as part of this plan.

## Plan self-review result

- Spec coverage: every evidence artifact, claim status, document, checklist item, exclusion boundary,
  and verification gate in the approved design is assigned to Tasks 1–4.
- Placeholder scan: no unresolved implementation placeholders are present.
- Interface consistency: all tasks use `reports/paired_a100/` and the same receipt schema, source SHA,
  package digest, current metrics, and claim statuses.
- Scope: L4 execution, license research, official-account migration, model publication, and hosted
  deployment remain separate later tasks.
