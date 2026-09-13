# Multi-board replication of the paired leakage experiment

Status: **complete** · unit: held-out board · 6/6 boards complete.

Pre-registered analysis: `docs/lobo-preregistration.md` (SHA-256 `86f026d5fdec94dce26d3808b39444b552cc06314d3216b33cce9f1ae523f73e`).

| Board | Grouped mAP50 | Leaky control mAP50 | Δ mAP50 (pp) | Δ mAP50-95 (pp) | Evidence |
|---|---:|---:|---:|---:|---|
| 05 | 0.6552 ± 0.1000 | 0.8495 ± 0.0450 | +19.4 | +9.9 | `reports/lobo/board05/` |
| 07 | 0.8509 ± 0.0608 | 0.9186 ± 0.0190 | +6.8 | +3.6 | `reports/lobo/board07/` |
| 08 | 0.6330 ± 0.1491 | 0.8456 ± 0.0375 | +21.3 | +11.3 | `reports/paired_a100/` |
| 09 | 0.7679 ± 0.0524 | 0.8411 ± 0.0413 | +7.3 | +5.8 | `reports/lobo/board09/` |
| 11 | 0.6813 ± 0.0618 | 0.8809 ± 0.0260 | +20.0 | +15.5 | `reports/lobo/board11/` |
| 12 | 0.8892 ± 0.0799 | 0.9077 ± 0.0297 | +1.8 | +3.2 | `reports/lobo/board12/` |

Δ mAP50 across boards: mean +12.8 pp, sample SD 8.4 pp, range +1.8 to +21.3 pp, 6/6 boards positive; board bootstrap 95% interval +6.5 to +18.3 pp (10000 resamples, approximate for n = 6).

Pre-registered wording outcome: **all_positive** — same-board sibling exposure increased final-test mAP50 on every held-out board.

Limitations:

- Boards 01, 04, 06, and 10 are never held out; the estimate covers the 60-image boards only.
- With six boards the board-level bootstrap interval is approximate.
- Each fold is one frozen 30-image final test on one board; seeds capture training noise only.
