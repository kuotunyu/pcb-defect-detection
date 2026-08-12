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

# PCB review workstation

The Gradio app now presents the project as a Traditional Chinese PCB review workstation instead of replacing the entire page with a deployment warning. The complete portfolio remains visible in every mode:

- `Recorded evidence`: the current public release; committed metrics, review-workstation structure, and Promotion Gate are visible without claiming live inference.
- `Live inference`: appears only after the model contract passes, its ONNX SHA-256 matches, and the runtime session initializes.
- `Degraded`: preserves the page and shows a scoped inline error when evidence or a release-approved model cannot be loaded.

Run from the repository root:

```powershell
uv run --locked --no-editable --extra app python -m app.app
```

The PCB preview is original synthetic artwork labeled `介面示意 · 非模型輸出`. Dataset images and examples are not bundled because the upstream dataset license has not been verified by this project.

## Deployment status

Aggregate fidelity gate passed, but the strict L4 backend prediction-parity gate failed; this metadata-only portfolio release candidate intentionally provides no public ONNX artifact, hosted model revision, or inference endpoint.

This optional Space scaffold remains fail-closed and does not use a floating model revision. It can
serve a model only if a future, separately reviewed release supplies the exact ONNX SHA-256, source
checkpoint SHA-256, deployment-gate SHA-256, official model repository, and immutable revision in
`model_contract.json`.

The app does not use placeholder metrics or a silent fallback model. All public KPI values are read from committed reports at startup.
