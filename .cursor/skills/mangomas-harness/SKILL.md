---
name: mangomas-harness
description: Run the local MangoMAS agent harness (tool loop). Use when asked to run an agent with tools, collect trajectories, or exercise CodeAct/ReAct locally. Not SageMaker predict_fn.
cli: scripts/harness/run_agent.py
inputs:
  task: User task (CLI --task)
  harness_id: Harness YAML id (CLI --harness-id, default MANGOMAS_HARNESS_ID)
  scripted: Optional JSON list of EchoBackend outputs for tests
---

# Run the MangoMAS harness

Local execution only. Distilled SageMaker endpoints remain single-shot `predict_fn`.

```bash
python scripts/harness/run_agent.py \
  --task "$TASK" \
  --harness-id "${MANGOMAS_HARNESS_ID}"
```

Collect traces with the `mangomas-collect` skill (`scripts/harness/collect_trajectories.py`). Trajectory SFT uses `train_distilled_adapter.py --trajectory_mode True` locally; do not add launcher hyperparameters.
