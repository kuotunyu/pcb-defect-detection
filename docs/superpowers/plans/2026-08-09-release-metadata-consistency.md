# Release Metadata Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the public README, app contract, limitations, license boundary, and release checklist to the authoritative YOLO26n, AGPL-3.0-or-later, gate-passed-but-unreleased, clean-official-history state.

**Architecture:** Treat `pyproject.toml`, `LICENSE`, `configs/base_model.yaml`, `reports/claims.yaml`, and committed evidence as authoritative. Add one release-contract test that rejects the stale README-redesign and pre-promotion states, then make the minimum documentation and JSON corrections needed to satisfy it.

**Tech Stack:** Markdown, JSON, TOML, Python 3.11, pytest, PyYAML, Ruff, uv

## Global Constraints

- Do not change model metrics, evidence JSON under `reports/`, frozen hashes, split manifests, weights, exports, or benchmark records.
- Do not train, evaluate a model, use a GPU, access `.env`, call a paid API, publish, release, or push.
- Keep `app/model_contract.json` blocked with all artifact hashes and Hugging Face fields null.
- Code license is `AGPL-3.0-or-later`; this does not grant dataset, base-weight, derived-weight, image, or export redistribution rights.
- The paired model family is Ultralytics YOLO26n `v8.4.0`.
- Runtime floors are Python `>=3.11` and PyTorch `>=2.4` for the training extra.
- Official GitHub push/review is complete; Hugging Face namespace and immutable model revision remain pending.
- Current official reachable history is the clean single-author `kuotunyu` history; unrelated prototype history remains outside it.
- Every commit author and committer must be `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no co-author trailer.

---

### Task 1: Bind public metadata to authoritative repository state

**Files:**
- Modify: `tests/test_release_contract.py`
- Modify: `README.md`
- Modify: `app/model_contract.json`
- Modify: `docs/license-boundary.md`
- Modify: `docs/limitations.md`
- Modify: `docs/release-checklist.md`

**Interfaces:**
- Consumes: project/license/dependency metadata from `pyproject.toml` and `LICENSE`; model identity from `configs/base_model.yaml` and `reports/claims.yaml`; release state from the official `main` promotion record.
- Produces: `test_public_metadata_matches_authoritative_release_state()` as the executable public-metadata contract.

- [ ] **Step 1: Add the failing release-contract test**

Add this test beside the existing portfolio/release checklist contract tests:

```python
def test_public_metadata_matches_authoritative_release_state() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    base_model = yaml.safe_load((ROOT / "configs" / "base_model.yaml").read_text(encoding="utf-8"))
    claims = yaml.safe_load((ROOT / "reports" / "claims.yaml").read_text(encoding="utf-8"))["claims"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")
    limitations = (ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    license_boundary = (ROOT / "docs" / "license-boundary.md").read_text(encoding="utf-8")
    contract = _read_json(ROOT / "app" / "model_contract.json")

    python_floor = project["requires-python"].removeprefix(">=")
    torch_requirement = next(
        requirement
        for requirement in project["optional-dependencies"]["train"]
        if requirement.startswith("torch>=")
    )
    torch_floor = torch_requirement.removeprefix("torch>=")
    license_expression = project["license"]
    license_badge = license_expression.replace("-", "--")
    model_filename = base_model["filename"]
    model_name = Path(model_filename).stem
    model_label = f"YOLO{model_name.removeprefix('yolo')}"

    assert project["requires-python"].startswith(">=")
    assert base_model["source"].endswith(f"/{model_filename}")
    assert base_model["revision"] in claims["base_initialization"]["statement"]
    assert model_label in claims["base_initialization"]["statement"]
    assert claims["onnx_deployment"]["status"] == "verified_candidate"
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (ROOT / "LICENSE").read_text(encoding="utf-8")

    for required in (
        f"Python-{python_floor}%2B",
        f"PyTorch-{torch_floor}%2B",
        model_label,
        f"License-{license_badge}",
        license_expression,
    ):
        assert required in readme
    public_source_paths = (
        "src/pcb_defect/data_prep/paired.py",
        "src/pcb_defect/experiment.py",
        "src/pcb_defect/final_evaluation.py",
    )
    for relative in public_source_paths:
        assert (ROOT / relative).is_file()
        assert relative in readme
    for stale in ("Python-3.10%2B", "PyTorch-2.0%2B", "YOLOv8", "License-MIT"):
        assert stale not in readme

    assert contract["status"] == "blocked"
    assert "Deployment gate passed" in contract["reason"]
    assert "redistribution rights remain unresolved" in contract["reason"]
    for field in (
        "onnx_sha256",
        "source_checkpoint_sha256",
        "deployment_gate_sha256",
        "hf_repo_id",
        "hf_revision",
    ):
        assert contract[field] is None

    assert "fidelity/parity gates are resolved" in license_boundary
    assert "- [x] Official push/review is completed." in checklist
    assert "- [ ] Hugging Face namespace is selected." in checklist
    assert "promotion is published on official `main`" in checklist
    assert "Current official `main` has clean single-author reachable history" in limitations
    assert "future L4 benchmark" not in limitations
    assert "Git history contains legacy identity" not in limitations
```

- [ ] **Step 2: Run the new test and confirm RED**

Run:

```powershell
uv run --locked --no-editable --group dev pytest tests/test_release_contract.py::test_public_metadata_matches_authoritative_release_state -q
```

Expected: FAIL because the current README contains the stale Python, PyTorch, YOLOv8, and MIT metadata before any production file is changed.

- [ ] **Step 3: Correct the public metadata with no evidence promotion**

Apply only these changes:

- `README.md`
  - Change badges to Python 3.11+, PyTorch 2.4+, Ultralytics YOLO26n, and AGPL-3.0-or-later.
  - Replace every YOLOv8 reference with YOLO26n where it names this experiment/base model.
  - Replace MIT code-license text with `AGPL-3.0-or-later`, while retaining the separate unresolved upstream dataset/weight/export boundary.
  - Replace incomplete `pcb_defect/data_prep/` and `pcb_defect/experiment/` entries with exact existing `src/pcb_defect/data_prep/paired.py` and `src/pcb_defect/experiment.py` paths.
  - Describe `reports/benchmark_l4.md` as a private metadata-only summary, not a complete public raw report.
- `app/model_contract.json`
  - Keep status `blocked` and every hash/Hub field null.
  - Replace the stale pending-gate reason with: `Deployment gate passed, but no release-approved public ONNX artifact or immutable Hugging Face revision is configured; redistribution rights remain unresolved.`
- `docs/license-boundary.md`
  - State that fidelity/parity gates are resolved for the unreleased candidate and that distribution remains blocked by rights and missing official immutable publication.
- `docs/limitations.md`
  - Replace the obsolete future-L4 sentence with the current calibration-only/private/non-SLA boundary.
  - Replace the old dirty-history warning with the current official clean-history boundary and keep unrelated prototype history explicitly outside public `main`.
- `docs/release-checklist.md`
  - Split the combined official-push/Hugging-Face item into checked official push/review and unchecked namespace selection items.
  - Replace the stale local/unreleased promotion sentence with the current official-main publication state while keeping model artifacts unreleased.

- [ ] **Step 4: Run focused GREEN verification**

Run:

```powershell
uv run --locked --no-editable --group dev pytest tests/test_release_contract.py -q
```

Expected: all release-contract tests PASS.

- [ ] **Step 5: Run complete CPU and static verification**

Run:

```powershell
uv run --locked --no-editable --group dev pytest -q
uv run --locked --no-editable --group dev ruff check .
uv run --locked --no-editable --group dev ruff format --check .
uv lock --check
git diff --check
```

Expected: complete CPU-safe pytest PASS with a final summary and zero failures; Ruff check and format PASS; lock check PASS; no whitespace errors.

- [ ] **Step 6: Audit scope, identity, and sensitive content**

Run:

```powershell
git status --short
git diff --name-only
git diff -- README.md app/model_contract.json docs/license-boundary.md docs/limitations.md docs/release-checklist.md tests/test_release_contract.py
git config user.name
git config user.email
```

Expected: only the six planned files plus this committed plan are present in the branch history. Added content contains no dataset, weight, export, binary, secret value, local absolute path, test account, or non-`kuotunyu` identity.

- [ ] **Step 7: Commit the verified repair**

Run:

```powershell
git add -- README.md app/model_contract.json docs/license-boundary.md docs/limitations.md docs/release-checklist.md tests/test_release_contract.py
git diff --cached --check
git commit -m "docs: align public release metadata"
git show -1 --pretty=fuller --no-patch
```

Expected: one implementation commit with author and committer both `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`, no co-author trailer, and no push.

---

## Completion boundary

This plan corrects repository-owned metadata only. It does not declare upstream dataset or derived-artifact redistribution rights, choose a Hugging Face namespace, publish a model, or alter scientific results. Those remain separate license-research and release tasks.
