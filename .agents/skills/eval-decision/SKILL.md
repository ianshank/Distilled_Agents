---
name: eval-decision
description: Write artifacts/eval_decision.json after tests. Use at evaluate stage before release notes.
---

# /eval-decision

Score in this order:

1. Product metric named in opportunity.md
2. Model metric in experiment_plan.md (skip if no-model)
3. Trajectory health (loops, recovery, tool-policy violations)

Write `artifacts/eval_decision.json` with decision `ship | hold | escalate | retry`.
Do not ship because the code compiles. Do not let the implementer author this file alone — run critic first.
