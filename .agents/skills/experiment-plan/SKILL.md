---
name: experiment-plan
description: Write artifacts/experiment_plan.md or an explicit no-model plan. Use before any training or eval harness work.
---

# /experiment-plan

Required sections:

- Prediction target (or "no model")
- Data contract and split policy
- Leakage risks
- Baseline
- Metrics that match `artifacts/opportunity.md`
- Kill-criteria

If opportunity.md is missing, stop and run intake first.
