---
name: openspec-specify
description: Scaffold an OpenSpec change package under openspec/changes/<id>/.
---

# /openspec-specify

Create `openspec/changes/<kebab-id>/{proposal.md,design.md,tasks.md}` plus `specs/<capability>/spec.md` with ADDED requirements and scenarios.

design.md must include Code Hygiene & Quality Gates: no hard-coded secrets, injected config, ruff/mypy/pytest, structured logs.
If `planlint` exists, run it and treat non-zero as specify failure.
