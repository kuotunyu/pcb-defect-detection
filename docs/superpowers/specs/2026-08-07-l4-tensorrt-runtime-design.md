# L4 TensorRT Runtime Bootstrap Design

## Problem

The rendered L4 notebook creates a locked environment with the training and evaluation dependency
sets, but those sets do not contain the TensorRT Python runtime. The benchmark imports `tensorrt`
before creating any output, so a fresh Colab L4 runtime fails deterministically with
`ModuleNotFoundError` even though ONNX Runtime advertises `TensorrtExecutionProvider`.

## Decision

Add a Linux-only `l4` dependency group containing an exact CUDA 12 TensorRT 10 package. The L4
notebook will include that group in its locked sync and run a narrow TensorRT probe immediately
after environment creation. TensorRT 10 is intentional: the benchmark requires FP16 engine export,
while TensorRT 11 introduces a separate ModelOpt dependency path that is outside this repair.

The existing `train` and `eval` sets remain unchanged, so CPU CI and non-L4 workflows do not install
TensorRT. There is no CUDA fallback because that would silently remove one of the three benchmarked
backends and invalidate the L4 experiment.

## Runtime contract

On Linux, `uv sync --locked --extra train --group eval --group l4` must install the pinned
`tensorrt-cu12` distribution and its bindings into the project virtual environment. Before running
the benchmark, the notebook must verify from that exact interpreter that:

- `tensorrt` imports successfully;
- the reported version is the pinned TensorRT 10 release;
- `tensorrt.Builder(tensorrt.Logger())` can be constructed.

Any failure stops before benchmark output creation and prints a focused TensorRT bootstrap error.

## Testing and acceptance

- A contract test must fail until `pyproject.toml`, `uv.lock`, and the notebook all include the L4
  runtime dependency and probe.
- The targeted handoff and release-contract tests must pass without a GPU.
- The full test suite and Ruff must pass on CPU.
- A newly generated L4 handoff must have matching bundle/notebook hashes, no template sentinels,
  and a verifiable history-free bundle.
- The user will upload only the three files from the new content-addressed handoff directory and
  rerun in a fresh Colab L4 session. The prior A100 workspace and dataset remain reusable.

## Non-goals

This repair does not retrain models, benchmark locally, publish artifacts, change public claims, or
add TensorRT to ordinary developer/CI environments.
