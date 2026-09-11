---
name: mangomas-evaluate
description: Evaluate a MangoMAS agent skill pass rate with the existing CLI. Use when asked to score or evaluate a role.
cli: scripts/evaluation/evaluate_agent_skill.py
inputs:
  role: Agent role name (CLI --role)
  test_suite: Path to the evaluation suite (CLI --test_suite)
  threshold: Pass-rate threshold from MANGOMAS_EVALUATION_THRESHOLD
  region: AWS region from MANGOMAS_AWS_REGION
---

# Evaluate a MangoMAS agent skill

```bash
python scripts/evaluation/evaluate_agent_skill.py \
  --role "$ROLE" \
  --test_suite "$SUITE" \
  --threshold "${MANGOMAS_EVALUATION_THRESHOLD}" \
  --region "${MANGOMAS_AWS_REGION}"
```

Do not add live AWS e2e tests. Thresholds come from settings, not hardcoded literals in skills.
