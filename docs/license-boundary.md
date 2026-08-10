# License boundary

| Component | Current treatment |
|---|---|
| Original repository code | `AGPL-3.0-or-later` via `LICENSE` and `pyproject.toml` |
| Ultralytics package/model family | Third-party terms; AGPL or a separate enterprise license may apply |
| HRIPCB images and annotations | Upstream license unverified; not distributed in the candidate tree |
| Base checkpoint | Third-party `v8.4.0` artifact; immutable URL/size/hash recorded, bytes not committed |
| Fine-tuned checkpoints | Distribution blocked until dataset and upstream model obligations are resolved |
| ONNX export | Aggregate fidelity passed, but strict backend prediction parity failed; distribution remains blocked until redistribution rights and an official immutable publication are resolved |
| TensorRT engine | Ephemeral, hardware-specific artifact; never commit or publish as portable evidence |

The six raw demo images, GIF, prediction grid, and SAHI pixel comparison are excluded from official
`main` because the dataset license is unresolved. They remain only in the unrelated private
prototype history, which is outside the public repository and must not be merged into it.
