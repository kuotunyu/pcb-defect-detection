# Limitations and non-claims

- The legacy 12.1-point split gap is observed sensitivity, not a pure leakage effect.
- The new final test has one board and 30 images; board-level uncertainty remains unresolved.
- No final-test threshold tuning, export tuning, seed selection, or architecture selection is allowed.
- No current ONNX, TensorRT, Hugging Face model, or Space is release-approved.
- Legacy GPU timings are not comparable to the future L4 benchmark until rerun with raw timings,
  warmup, environment, checkpoint hash, export hash, and fidelity evidence.
- A code license does not establish rights to the dataset, base weights, derived weights, or images.
- Git history contains legacy identity and dataset pixels. Official migration must use a reviewed
  clean snapshot rather than pushing this history.
