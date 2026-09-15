---
name: mangomas-evaluate
description: Count pre-filled pass/fail rows in a skill JSON suite. Use when asked to score a role fixture file. Does not run AgentRuntime or a model.
cli: scripts/evaluation/evaluate_agent_skill.py
inputs:
  role: Agent role name (CLI --role)
  test_suite: Path to the evaluation suite (CLI --test_suite)
  threshold: Pass-rate threshold from MANGOMAS_EVALUATION_THRESHOLD
  region: AWS region from MANGOMAS_AWS_REGION
---

# Evaluate a MangoMAS agent skill fixture

This CLI counts `passed` / `actual==expected` already stored in a JSON suite. It does **not** load a model or call `AgentRuntime`.

Harness-loop quality (final_answer, tool ids, truncation, faults) is `scripts/harness/eval_harness.py` (`mangomas-eval-harness`).

```bash
python scripts/evaluation/evaluate_agent_skill.py \
  --role "$ROLE" \
  --test_suite "$SUITE" \
  --threshold "${MANGOMAS_EVALUATION_THRESHOLD}" \
  --region "${MANGOMAS_AWS_REGION}"
```

Do not add live AWS e2e tests. Thresholds come from settings, not hardcoded literals in skills.
