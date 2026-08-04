# License boundary

| Component | Current treatment |
|---|---|
| Original repository code | `AGPL-3.0-or-later` via `LICENSE` and `pyproject.toml` |
| Ultralytics package/model family | Third-party terms; AGPL or a separate enterprise license may apply |
| HRIPCB images and annotations | Upstream license unverified; not distributed in the candidate tree |
| Base checkpoint | Third-party `v8.4.0` artifact; immutable URL/size/hash recorded, bytes not committed |
| Fine-tuned checkpoints | Distribution blocked until dataset and upstream model obligations are resolved |
| ONNX export | Distribution blocked until source license and fidelity/parity gates are resolved |
| TensorRT engine | Ephemeral, hardware-specific artifact; never commit or publish as portable evidence |
| Aggregate plots | Retained only when they do not reproduce source pixels; still labelled legacy when applicable |

The six raw demo images, GIF, prediction grid, and SAHI pixel comparison were removed from the
current candidate tree because the dataset license is unresolved. They remain in earlier Git
history because history rewriting is explicitly out of scope. That history must not be migrated to
an official public account; use a clean reviewed source snapshot after release approval.
