# Limitations and non-claims

- The legacy 12.1-point split gap is observed sensitivity, not a pure leakage effect.
- The new final test has one board and 30 images; board-level uncertainty remains unresolved.
- No final-test threshold tuning, export tuning, seed selection, or architecture selection is allowed.
- No current ONNX, TensorRT, Hugging Face model, or Space is release-approved.
- The private L4 timings cover the 60-image calibration split on one recorded software/hardware
  stack; they are not final-test timings, a portable-engine claim, or a production SLA.
- A code license does not establish rights to the dataset, base weights, derived weights, or images.
- Current official `main` has clean single-author reachable history and excludes dataset pixels.
  The unrelated private prototype history remains outside the public repository.
