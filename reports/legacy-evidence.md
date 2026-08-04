# Legacy evidence index

These files are preserved to show the prototype's investigation trail. They are not produced by the
new paired experiment and cannot satisfy current release claims.

| Evidence | Status | Reason it is not current evidence |
|---|---|---|
| `test_metrics.json`, `leakage_comparison.md` | Legacy observed | Different test images/sizes across arms |
| `export_fidelity.json` | Failed gate | `fidelity_ok` is false |
| `onnx_parity.json` | Failed gate | `all_passed` is false (9/10) |
| `benchmark_cpu.json`, `benchmark_gpu.json`, `benchmark.md` | Legacy unverified | Missing new run/checkpoint/export/manifest chain |
| `sahi_ablation.json`, `sahi_ablation.md` | Exploratory only | Used the legacy test board; no portfolio value to extend |

The original scripts and notebooks that could respent the legacy test set, deploy to a personal
account, or overwrite these reports were removed from the candidate tree. They remain recoverable
from Git history, which itself is not approved for official migration.
