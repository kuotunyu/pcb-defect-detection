# Evaluation Claim Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the README's leakage result impossible to mistake for population-level cross-board or production evidence, and enforce that boundary in the release contract.

**Architecture:** Keep committed JSON metrics as the numeric source and `reports/claims.yaml` as the claim source of truth. Correct only the human-facing README language that exceeds those artifacts, then add a focused contract test that binds the README, model card, limitations document, artifact-local report, and claims registry to the single-board/image-bootstrap inference boundary.

**Tech Stack:** Markdown, YAML, Python 3.11, pytest, PyYAML, Ruff, uv

## Global Constraints

- Do not modify metrics, JSON evidence, frozen hashes, split manifests, weights, exports, or benchmark records.
- Do not train, evaluate a model, use a GPU, access `.env`, call a paid API, or require network access.
- Preserve the current README visual hierarchy, evidence anchors, hashes, and deployment caveats.
- The legacy 12.1-point result remains an observed, non-paired split sensitivity result.
- The controlled 21.3-point result remains specific to the frozen dataset and recipe.
- The final-test scope is 30 images from board 08; image bootstrap does not estimate between-board variance.
- Do not claim population-level cross-board, new-product, factory-line, or production generalization.
- Every commit author and committer must be `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no co-author trailer.
- Do not push, publish, release, rewrite Git history, or change remotes.

---

### Task 1: Bind public leakage language to its sampling scope

**Files:**
- Modify: `tests/test_release_contract.py`
- Modify: `README.md`
- Verify unchanged semantics: `reports/claims.yaml`
- Verify unchanged semantics: `docs/model-card.md`
- Verify unchanged semantics: `docs/limitations.md`
- Verify unchanged semantics: `reports/paired_a100/README.md`

**Interfaces:**
- Consumes: `reports/paired_a100/final_metrics.json` numeric values and the `paired_leakage_effect` statement/limitations in `reports/claims.yaml`.
- Produces: a README result section whose scope is contract-tested by `test_portfolio_documents_bound_paired_inference_to_single_board()`.

- [ ] **Step 1: Add the failing release-contract test**

Add this test beside the existing portfolio-document contract tests:

```python
def test_portfolio_documents_bound_paired_inference_to_single_board() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    limitations = (ROOT / "docs" / "limitations.md").read_text(encoding="utf-8")
    paired_report = (ROOT / "reports" / "paired_a100" / "README.md").read_text(
        encoding="utf-8"
    )
    paired_claim = yaml.safe_load(
        (ROOT / "reports" / "claims.yaml").read_text(encoding="utf-8")
    )["claims"]["paired_leakage_effect"]

    for forbidden in (
        "真實跨板泛化表現",
        "反映實際產線部署效能",
        "證明板級洩漏顯著性",
        "確認洩漏效應具備嚴格統計顯著性",
    ):
        assert forbidden not in readme

    for required in (
        "30 張",
        "Board 08",
        "image",
        "board",
        "between-board",
        "production",
    ):
        assert required in readme

    assert "one board" in paired_claim["limitations"][0]
    assert "between-board variance" in paired_claim["limitations"][0]
    assert "not a universal" in paired_claim["limitations"][1]
    assert "single PCB template board" in model_card
    assert "Image-bootstrap intervals do not estimate board-level uncertainty" in model_card
    assert "one board and 30 images" in limitations
    assert "image, not board, is the resampling unit" in paired_report
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```powershell
uv run --locked --no-editable --group dev pytest tests/test_release_contract.py::test_portfolio_documents_bound_paired_inference_to_single_board -q
```

Expected: FAIL because the current README still contains at least one forbidden phrase. If the
Windows non-ASCII workspace prevents Python startup, create a disposable locked non-editable venv
under an ASCII-only temporary directory and run the same pytest node against this worktree; do not
modify or delete the repository `.venv`.

- [ ] **Step 3: Apply the minimum README wording correction**

Make these semantic replacements without changing numeric values or evidence links:

- Executive summary: describe `21.3` points as the observed controlled difference under the frozen
  paired protocol, not a population-wide bias estimate.
- Paired-protocol highlight: replace `證明...統計顯著落差` with wording that names the
  one-board final-test scope.
- Mermaid gate: replace proof language with `paired image-bootstrap 95% CI` and `does not estimate
  between-board uncertainty`.
- Result introduction: replace unqualified significance language with `observed difference`.
- Grouped table cell: replace production/cross-board language with `single held-out Board 08;
  does not represent population-level cross-board or production generalization`.
- Leaky-control table cell: describe the arm intervention and the observed `+21.3 pp` difference,
  not an unrestricted causal claim.
- F1 bullet: state that the resampling unit is the image, not the board, and that the interval does
  not estimate between-board uncertainty.
- Add an adjacent compact boundary block with exactly two labels: `What this proves` and
  `What this does not prove`. The first is limited to controlled same-board sibling exposure under
  the frozen dataset/recipe; the second excludes population-level cross-board, new-product,
  factory-line, and production generalization.

- [ ] **Step 4: Run focused GREEN verification**

Run:

```powershell
uv run --locked --no-editable --group dev pytest tests/test_release_contract.py -q
```

Expected: all release-contract tests PASS.

- [ ] **Step 5: Verify claim-to-artifact consistency and formatting**

Run:

```powershell
uv run --locked --no-editable --group dev pytest -q
uv run --locked --no-editable --group dev ruff check .
uv run --locked --no-editable --group dev ruff format --check .
uv lock --check
git diff --check
```

Expected: complete CPU-safe pytest PASS, Ruff check PASS, Ruff format check PASS, lock check PASS,
and no whitespace errors. Do not report the suite as passing without the final pytest summary and
exit code.

- [ ] **Step 6: Audit scope, identity, and tracked content**

Run:

```powershell
git status --short
git diff -- README.md tests/test_release_contract.py
git diff --name-only
git config user.name
git config user.email
```

Expected: only `README.md` and `tests/test_release_contract.py` are modified; identity is the
required `kuotunyu` noreply identity. Inspect added lines to confirm there is no secret, local
absolute path, account name other than the required project identity, dataset, weight, export,
image, ZIP, log, or binary.

- [ ] **Step 7: Commit the independently verified change**

Run:

```powershell
git add -- README.md tests/test_release_contract.py
git diff --cached --check
git commit -m "docs: bound leakage claims to single-board evidence"
git show -1 --pretty=fuller --no-patch
```

Expected: one commit with author and committer both
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`, no co-author trailer, and no push.

---

## Completion boundary

This plan ends when the current public narrative and its automated release contract agree on the
single-board inference boundary. A separate approved spec and plan are required before generating
a board-rotating protocol or starting any GPU training.
