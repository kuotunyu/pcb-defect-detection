---
title: Leakage-aware PCB defect detection
emoji: 🔬
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
python_version: "3.11"
license: agpl-3.0
tags:
  - object-detection
  - pcb
  - model-evaluation
short_description: Metadata-only portfolio; public model artifact intentionally omitted
pinned: false
---

# Deployment status

Aggregate fidelity gate passed, but the strict L4 backend prediction-parity gate failed; this metadata-only portfolio release intentionally provides no public ONNX artifact, hosted model revision, or inference endpoint.

This optional Space scaffold remains fail-closed and does not use a floating model revision. It can
serve a model only if a future, separately reviewed release supplies the exact ONNX SHA-256, source
checkpoint SHA-256, deployment-gate SHA-256, official model repository, and immutable revision in
`model_contract.json`.

Dataset images and examples are not bundled because the upstream dataset license has not been
verified by this project.
