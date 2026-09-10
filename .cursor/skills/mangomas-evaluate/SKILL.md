---
name: mangomas-evaluate
description: Evaluate a trained MangoMAS agent skill against a test suite. Use when asked to score, gate, or report pass rate for a role.
cli: scripts/evaluation/evaluate_agent_skill.py
inputs:
  role: Agent role name (CLI --role)
  test_suite: Path to evaluation suite (CLI --test_suite)
  threshold: Pass-rate gate from MANGOMAS_EVALUATION_THRESHOLD
  region: AWS region from MANGOMAS_AWS_REGION
  bucket: Training bucket from MANGOMAS_TRAINING_DATA_BUCKET
  table: Registry table from MANGOMAS_DYNAMODB_TABLE
---

# Evaluate a MangoMAS agent skill

```bash
python scripts/evaluation/evaluate_agent_skill.py \
  --role "$ROLE" \
  --test_suite "$TEST_SUITE" \
  --threshold "${MANGOMAS_EVALUATION_THRESHOLD}" \
  --region "${MANGOMAS_AWS_REGION}"
```

Exit status is non-zero when `pass_rate` is below the threshold. Do not invent evaluation APIs; this CLI wraps `AutomatedTrainingSystem.evaluate_agent_skill`.
