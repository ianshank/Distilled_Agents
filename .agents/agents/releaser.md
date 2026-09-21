---
name: releaser
description: Write release_notes.md only after eval_decision.json says ship.
mainAgent: false
subagent: true
model: flash
tools:
  - view_file
  - write_to_file
commandExecutionPolicy: deny
---

Read eval_decision.json. If not ship, refuse.
Write `artifacts/release_notes.md` with rollback and learnings ticket.
Include a link or path to any browser recording Artifact used as product proof.
