# Clean Public Promotion Design

## Purpose

Promote the reviewed private L4 hardening tree into the official public repository without
joining or exposing the private development history.

## Repository boundary

- Official public base: `https://github.com/kuotunyu/pcb-defect-detection.git`, branch `main`,
  currently pinned to `eaf0f77be7edca1909bca78221f999892794ec5a` before execution.
- Reviewed private source: branch `codex/l4-private-benchmark-handoff`.
- Public promotion branch: `codex/public-l4-promotion`, created directly from the fetched official
  `main` commit in a separate `.worktrees/public-l4-promotion` worktree.
- The two repositories have unrelated Git histories. Promotion is a tree delta, never a merge,
  rebase, force-push, or `--allow-unrelated-histories` operation.

## Content flow

Compute the text-only tree difference between the pinned official base and the reviewed private
source. Apply that delta in the public promotion worktree and first prove exact tracked-tree
equality. After independent review, allow only narrowly documented public-context cleanup: replace
private-history wording, add actionable partial-package recovery guidance, and update their tests
and this promotion record. Retain only the official three-commit history plus one new `kuotunyu`
promotion commit.

No dataset pixels, images, weights, checkpoints, ONNX exports, TensorRT engines, result ZIPs,
logs, caches, `.env`, or ignored runtime artifacts may enter the public branch.

## Identity and publication safety

Every new commit must use author and committer
`kuotunyu <61350295+kuotunyu@users.noreply.github.com>` and contain no co-author trailers. The
existing public commits must remain unchanged. This phase is local-only: it does not change remote
configuration, push a branch, open a pull request, or modify GitHub/Hugging Face.

## Verification

The promoted worktree must pass the locked full CPU test suite, Ruff check/format, lock check,
claim and metadata parsing, initial tree equality plus an allowlisted post-review diff against the
reviewed source, tracked binary/secret/path scans, commit identity audit, and clean-worktree check.
Any unexplained mismatch stops promotion before push.

## Result

A reviewed local `codex/public-l4-promotion` branch ready for a separate explicit push decision.
