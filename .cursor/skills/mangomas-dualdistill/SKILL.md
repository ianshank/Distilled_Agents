---
name: mangomas-dualdistill
description: Compose two same-task teacher JSONL files with an expected grader (DualDistill y1+transition+y2). Use when a labeled dual-teacher suite exists. Mixing role harness JSONL is multi-task SFT.
cli: scripts/harness/compose_dualdistill.py
inputs:
  first: Teacher A JSONL (CLI --first)
  second: Teacher B JSONL (CLI --second)
  output: Composed JSONL (CLI --output)
---

# Compose DualDistill trajectories

Requires the same prompt, `expected`, two teachers, and grader `G(y,a)`. Drops `(0,0)`. `planning.style` is unused; do not collect “codeact vs plan_and_solve” as DualDistill.

```bash
python scripts/harness/compose_dualdistill.py \
  --first "$TEACHER_A_JSONL" \
  --second "$TEACHER_B_JSONL" \
  --output "$COMPOSED_JSONL"
```

Rows without `expected` are skipped. Checklist tools cannot stand in for an interpreter teacher.
