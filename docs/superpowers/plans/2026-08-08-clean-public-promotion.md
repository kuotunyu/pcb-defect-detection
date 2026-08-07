# Clean Public Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean local public-release branch from official `kuotunyu/main` using only the reviewed private tree delta.

**Architecture:** Fetch the official main commit into a local read-only tracking ref, create an isolated worktree and named branch from that commit, then apply a text-only tree patch from the reviewed source. Prove initial tree equality, permit only documented review-driven public-context cleanup, and verify safety boundaries, tests, and identity before stopping prior to push.

**Tech Stack:** Git worktrees, PowerShell, Python 3.11, uv, pytest, Ruff.

## Global Constraints

- Official base URL: `https://github.com/kuotunyu/pcb-defect-detection.git`.
- Expected official base SHA before execution: `eaf0f77be7edca1909bca78221f999892794ec5a`.
- Reviewed source branch: `codex/l4-private-benchmark-handoff`.
- Promotion branch: `codex/public-l4-promotion`.
- Worktree: `.worktrees/public-l4-promotion`.
- Never merge unrelated histories, rebase, force-push, rewrite history, or change remote configuration.
- Do not push, open a pull request, create a release, or update GitHub/Hugging Face in this phase.
- Do not track dataset pixels, images, weights, ONNX, TensorRT engines, ZIPs, logs, caches, or `.env`.
- New commit author and committer must be `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no co-author trailers.
- Do not run GPU, training, benchmark, deployment, or paid APIs.

---

### Task 1: Create and verify the clean public promotion branch

**Files:**

- Create worktree: `.worktrees/public-l4-promotion`
- Modify tracked tree only through the reviewed source-versus-public text patch.

**Interfaces:**

- Consumes: official main SHA and reviewed private source tree.
- Produces: local branch `codex/public-l4-promotion` with a clean public commit.

- [ ] **Step 1: Fetch and pin official main**

Run `git fetch https://github.com/kuotunyu/pcb-defect-detection.git main:refs/remotes/official/main`.
Require `git rev-parse refs/remotes/official/main` to equal
`eaf0f77be7edca1909bca78221f999892794ec5a`; stop if it moved.

- [ ] **Step 2: Verify worktree target safety**

Require `.worktrees` to be ignored, require `.worktrees/public-l4-promotion` not to exist, and
require branch `codex/public-l4-promotion` not to exist.

- [ ] **Step 3: Create isolated worktree**

Run `git worktree add .worktrees/public-l4-promotion -b codex/public-l4-promotion refs/remotes/official/main`.
Verify its HEAD equals the pinned official SHA and its worktree is clean.

- [ ] **Step 4: Apply only the reviewed tree delta**

Generate a binary-capable Git patch from
`eaf0f77be7edca1909bca78221f999892794ec5a` to
`codex/l4-private-benchmark-handoff`. Require the changed-path list to contain no prohibited
artifact extension or `.env`, then apply the patch in the promotion worktree with `git apply --index`.

- [ ] **Step 5: Verify promoted tree equality before commit**

Compare `git write-tree` in the staged promotion worktree with
`git rev-parse codex/l4-private-benchmark-handoff^{tree}`. They must be identical.

- [ ] **Step 6: Run safety and CPU verification**

Use an ASCII temporary `UV_PROJECT_ENVIRONMENT`, force reinstall the local package, and run:

```powershell
uv sync --locked --no-editable --extra train --group eval --reinstall-package pcb-defect
uv run --locked --no-sync --extra train --group eval pytest -q
uv run --locked --no-sync --extra train --group eval ruff check .
uv run --locked --no-sync --extra train --group eval ruff format --check .
uv lock --check
```

Parse `reports/benchmark_l4.json` and `reports/claims.yaml`. Scan tracked paths and staged bytes for
prohibited artifacts, `.env`, absolute local paths, practice-account URLs, and common secret
patterns. Verify no tracked file exceeds 10 MiB.

- [ ] **Step 7: Commit with public identity**

Commit the staged tree as `release: promote verified L4 evidence` using the required author and
committer identity and no trailers.

- [ ] **Step 8: Apply and audit review-driven public-context cleanup**

After independent review, correct stale private-history wording and add actionable recovery guidance
for intentionally retained partial package outputs. Limit divergence from the reviewed source to
`README.md`, `docs/release-checklist.md`, this design/plan pair, `src/pcb_defect/result_package.py`,
`tests/test_result_package.py`, and `tests/test_release_contract.py`. Re-run the full CPU suite and
all static checks, then amend the single promotion commit.

- [ ] **Step 9: Final audit**

Require the diff from the reviewed private source to contain only the seven allowlisted cleanup
paths above. Require all commits from the official base through HEAD to use only `kuotunyu`, require
no co-author trailers, require a clean worktree, and verify no remote branch contains the new HEAD.
Stop before push.
