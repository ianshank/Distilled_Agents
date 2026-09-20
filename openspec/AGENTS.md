# OpenSpec — coordination contract (Distilled_Agents)

## Lifecycle → compile-down

| OpenSpec phase | Compile-down target | Owner |
|---|---|---|
| `propose` | `docs/plans/<topic>/PLAN.md` | SDLC / tech lead |
| `design` | ADR under `docs/adr/` (at land) | Architect |
| `specs/<cap>/spec.md` | Tests + Makefile/CI job | SQE + harness eng |
| `tasks.md` | PR checklist | Implementing eng |
| `review` | `docs/plans/<topic>/REVIEW.md` | Peer review (advisory) |
| `archive` | Move change under `openspec/changes/archive/` + NEXT_STEPS tick | Tech lead |

## Always-on constraints

- Format triangle: harness generate string == trajectory collator string;
  SageMaker `predict_fn` stays single-shot until a dedicated ADR.
- DualDistill = same-task + `expected` + two teachers (never role JSONL concat).
- No live AWS e2e; contract tests monkeypatch boto3.
- Trajectory `distillation_alpha` stays `0.0` until KL is label-masked **and**
  vocab-aligned with tests green.
- Local SDPO/GKD only after an explicit usage ADR (pin alone is not enough).
