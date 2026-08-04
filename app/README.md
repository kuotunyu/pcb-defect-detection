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
short_description: Hash-pinned ONNX demo; blocked until the paired deployment gate passes
pinned: false
---

# Deployment status

This Space source is intentionally blocked until the newly trained grouped checkpoint passes the
calibration-set PyTorch/ONNX fidelity gate and the standalone ONNX Runtime parity gate.

The app does not use a floating model revision. A release candidate must update
`model_contract.json` with the exact ONNX SHA-256, source checkpoint SHA-256, deployment-gate
SHA-256, official model repository, and immutable repository revision.

Dataset images and examples are not bundled because the upstream dataset license has not been
verified by this project.
