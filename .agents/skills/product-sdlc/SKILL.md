---
name: product-sdlc
description: Run the gated product + SDLC + ML workflow. Use when starting a change, shipping a feature, or the user says /product-sdlc.
---

# /product-sdlc

Run stages in order. Do not skip. Do not implement during discover/specify.

`discover → specify → design → build → evaluate → ship → learn`

## Rules

1. Named files under `artifacts/` (and OpenSpec under `openspec/changes/`) are the only promotion API.
2. Delegate researcher and critic as subagents. Researcher must not get Bash or Write to source.
3. You are not your own critic.
4. Keep `artifacts/stage.json` current: `{"stage":"<id>"}`.
5. After specify, run `planlint` if it is on PATH.
6. After evaluate, only releaser may write `release_notes.md`, and only if `eval_decision.json` decision is `ship`.

## Stage outputs

| Stage | Delegate | Must exist before leaving |
|---|---|---|
| discover | intake, then researcher, then critic | artifacts/opportunity.md |
| specify | spec-writer, critic | proposal.md, design.md, tasks.md |
| design | architect, ml-framer, critic | architecture.md, experiment_plan.md |
| build | main session (thin). Proof command must exit 0 | test_report.json |
| evaluate | critic + this skill's eval section | eval_decision.json |
| ship | releaser | release_notes.md |
| learn | intake | learnings.md |

## Eval JSON shape

```json
{
  "stage": "evaluate",
  "decision": "ship",
  "missing": [],
  "notes": "product metric ...; model metric or no-model; trajectory notes"
}
```

Decision vocabulary is only `ship | hold | escalate | retry`.
