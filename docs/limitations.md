# Limitations and non-claims

- The new final test has one board and 30 images; board-level uncertainty remains unresolved.
- No final-test threshold tuning, export tuning, seed selection, or architecture selection is allowed.
- No current ONNX, TensorRT, Hugging Face model, or Space is release-approved.
- The private L4 timings cover the 60-image calibration split on one recorded software/hardware
  stack; they are not final-test timings, a portable-engine claim, or a production SLA.
- Aggregate calibration fidelity passed, but strict PyTorch-reference per-box prediction parity
  failed for both ONNX Runtime CUDA FP32 and TensorRT FP16. Backend equivalence is not claimed.
- L4 timings come from one session. The interleaved wrapper schedule reduces fixed-order bias but
  does not estimate between-session, machine, driver, cache, or thermal variance.
- A code license does not establish rights to the dataset, base weights, derived weights, or images.
- Current release candidate has clean single-author reachable history and excludes dataset pixels.
  The unrelated private prototype history remains outside the candidate history.
