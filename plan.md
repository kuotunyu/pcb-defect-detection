# Experiment plan status

The original prototype plan mixed planned work, exploratory observations, and deployment claims.
It has been superseded by the following reviewable contracts:

- `configs/paired_protocol.yaml` — frozen sample assignments and seeds
- `configs/train_paired.yaml` — common training configuration
- `configs/final_evaluation.yaml` — predeclared operating point and bootstrap settings
- `reports/protocol/paired_split_manifest.json` — content-addressed dataset and split manifest
- `reports/claims.yaml` — claim status and evidence links
- `schemas/run_record.schema.json` — per-run evidence contract

Historical metrics and benchmark files remain available as explicitly labelled legacy evidence.
They must not be promoted to current results. The next irreversible operation is the Colab A100
training handoff; no final-test, deployment, release, or account migration action is authorized by
this plan.
