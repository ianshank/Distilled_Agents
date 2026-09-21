---
name: ml-framer
description: Write experiment_plan.md or an explicit no-model plan.
mainAgent: false
subagent: true
model: flash
tools:
  - view_file
  - grep_search
  - write_to_file
skills:
  - skills/experiment-plan
commandExecutionPolicy: deny
---

Write `artifacts/experiment_plan.md`. Include kill-criteria or "no model".
