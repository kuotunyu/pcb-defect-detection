# Result Package Portability Plan

**Goal:** Make result-package publication work on Google Drive FUSE while preserving no-overwrite and cleanup guarantees.

## Global constraints

- Change only the result-package publication path and its regression tests.
- Preserve refusal to overwrite an existing package or sidecar.
- Use exclusive destination creation and bounded streaming copy; never use hard links.
- Do not run GPU, training, benchmark, publish, or modify public claims.
- Keep author and committer identity as `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`.

## Task 1: Portable exclusive publication

Files:

- Modify `src/pcb_defect/result_package.py`.
- Modify `tests/test_result_package.py`.

Required behavior:

1. Add a failing regression test showing a package can be published when `os.link` is unavailable or raises `PermissionError`.
2. The publication helper must create the destination with exclusive semantics (`xb`/`O_EXCL`) and stream bytes from the staging file.
3. If the destination already exists, raise `PackageError` and preserve the pre-existing bytes.
4. If publication fails partway, remove only this invocation's staging files. Once a
   public destination exists, leave it in place: portable pathname-based cleanup
   cannot safely distinguish it from a concurrent replacement between identity
   check and unlink.
5. Existing package and sidecar verification tests remain green.

Verification:

- Focused `tests/test_result_package.py`.
- Full CPU pytest with locked train/eval dependencies.
- Ruff clean.
- Worktree clean after commit except ignored generated artifacts.
