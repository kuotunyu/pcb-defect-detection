# Portfolio Final Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the public repository as an honest, runnable, metadata-only Computer Vision portfolio release.

**Architecture:** Treat `reports/benchmark_l4.json`, `app/model_contract.json`, and the current CLI parsers as authoritative. Enforce their recruiter-facing representation in the release contract, then make the minimum documentation and repository-metadata corrections needed to satisfy it.

**Tech Stack:** Markdown, JSON, YAML, Python 3.11, pytest, PyYAML, Ruff, uv, GitHub Actions

## Global Constraints

- Do not modify evidence JSON, metrics, frozen hashes, manifests, model artifacts, or dataset artifacts.
- Do not train, evaluate, export, benchmark, use a GPU, run Colab, download a dataset, call a paid API, create a release, or deploy a model.
- Keep `app/model_contract.json` blocked and keep every artifact hash and Hub field null.
- The portfolio release is metadata-only; public model artifacts and hosted inference are intentional non-goals.
- Every commit author and committer must be `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no co-author trailer.

---

### Task 1: Bind the public portfolio surface to current evidence and release scope

**Files:**
- Modify: `tests/test_release_contract.py`
- Modify: `README.md`
- Modify: `docs/release-checklist.md`
- Modify: `docs/model-card.md`
- Modify: `app/README.md`
- Modify: `app/model_contract.json`
- Modify: `reports/claims.yaml`
- Modify: `reports/paired_a100/README.md`
- Modify: `reports/benchmark.md`
- Modify: `reports/legacy-evidence.md`

**Interfaces:**
- Consumes: L4 timing fields from `reports/benchmark_l4.json`, app release state from `app/model_contract.json`, and actual CLI signatures from `pcb_defect.experiment` and `pcb_defect.final_evaluation`.
- Produces: a recruiter-facing README and release contract that expose complete evidence, runnable onboarding, and the intentional metadata-only boundary.

- [ ] **Step 1: Write the failing release-contract test**

Extend `test_public_metadata_matches_authoritative_release_state()` so it:

```python
l4 = _read_json(ROOT / "reports" / "benchmark_l4.json")
for backend in l4["timings"].values():
    for key in ("p50_ms", "p95_ms", "fps_from_p50"):
        assert f"{backend[key]:.2f}" in readme
assert "ONNX Runtime CUDA FP32" in readme
assert "Opset 17" not in readme
assert "- [ ]" not in checklist
assert "metadata-only portfolio release" in checklist
assert "python -m pcb_defect.experiment preflight" not in readme
assert "python -m pcb_defect.experiment train-all" not in readme
assert "python -m pcb_defect.final_evaluation" not in readme
assert "--extra train --group eval" in readme
assert "pcb_defect.experiment --help" in readme
```

Also require the metadata-only release decision in the model card and app contract, require the
hosted-demo limitation to say it is out of scope, and reject the stale official-publication and
legacy-rerun wording.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```powershell
uv run --locked --no-editable --group dev pytest tests/test_release_contract.py::test_public_metadata_matches_authoritative_release_state -q
```

Expected: FAIL because the current README omits PyTorch/ORT CUDA values, contains `Opset 17` and
bare GPU commands, and the release checklist still has unchecked publication work.

- [ ] **Step 3: Apply the minimum public-document corrections**

Update the ten production documents listed above. Preserve all numeric values and derive the
three-backend table from the committed L4 JSON. Replace incomplete release wording with an
intentional metadata-only boundary. Replace unsafe bare training commands with:

```bash
uv run --locked --no-editable --extra train --group eval \
  python -m pcb_defect.experiment --help
```

Link the A100 and L4 notebooks instead of presenting argument-free training/evaluation commands.

- [ ] **Step 4: Run focused GREEN and release-contract verification**

Run:

```powershell
uv run --locked --no-editable --group dev pytest tests/test_release_contract.py::test_public_metadata_matches_authoritative_release_state -q
uv run --locked --no-editable --group dev pytest tests/test_release_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Run complete CPU and static verification**

Run:

```powershell
uv run --locked --no-editable --group dev pytest -q
uv run --locked --no-editable --group dev ruff check .
uv run --locked --no-editable --group dev ruff format --check .
uv lock --check
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit the reviewed tracked change**

```powershell
git add README.md app/README.md app/model_contract.json docs/model-card.md docs/release-checklist.md reports/benchmark.md reports/claims.yaml reports/legacy-evidence.md reports/paired_a100/README.md tests/test_release_contract.py
git commit -m "docs: close metadata-only portfolio release"
```

Expected: author and committer are only `kuotunyu`, with no co-author trailer.

---

### Task 2: Close the official GitHub repository surface

**Files:**
- No tracked file changes.
- Modify repository settings: topics and Wiki.
- Modify local Git config: remove `practice` remote.

**Interfaces:**
- Consumes: official repository identity `kuotunyu/pcb-defect-detection` and the reviewed Task 1 commit.
- Produces: discoverable GitHub metadata and one unambiguous local push target.

- [ ] **Step 1: Merge and verify the tracked change on local `main`**

Fast-forward `codex/portfolio-final-polish` into `main`, then rerun the full CPU suite and static
checks before cleanup or push.

- [ ] **Step 2: Clean only the owned feature worktree and branch**

Resolve and verify that `.worktrees/portfolio-final-polish` is under this repository's owned
`.worktrees` directory, remove it, prune registrations, and delete only
`codex/portfolio-final-polish`. Preserve every other branch and worktree.

- [ ] **Step 3: Remove the obsolete local test-account remote**

Verify `origin` equals `https://github.com/kuotunyu/pcb-defect-detection.git`, then run:

```powershell
git remote remove practice
```

Expected: `git remote -v` lists only `origin`.

- [ ] **Step 4: Push official `main` and update repository metadata**

Push `main`, add these GitHub topics, and disable the empty Wiki:

```text
computer-vision, object-detection, industrial-ai, machine-learning,
model-evaluation, mlops, yolo, onnx, tensorrt, pcb-defect-detection
```

Do not set a homepage because no public demo exists.

- [ ] **Step 5: Verify public completion**

Verify the remote `main` SHA matches local HEAD, GitHub Actions succeeds for that SHA, topics match
the expected set, Wiki is disabled, the repository remains public, Contributors contains only
`kuotunyu`, and no remote branch other than `main` exists.
