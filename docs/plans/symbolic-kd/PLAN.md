# Implementation Plan — Symbolic KD (disposition-first)

**ID:** PLAN-2026-09-20-symbolic-kd  
**Date:** 2026-09-20 · **Base tip:** `94e6c6a` (`main`, post-#16)  
**Motivated by:** `./REVIEW.md` — multi-model consensus reframed against live
Distilled_Agents evidence.  
**OpenSpec changes:**  
`openspec/changes/{green-trunk-ci,golden-passk-aqa,critic-cascade,symbolic-disposition,trl-gkd-usage}/`  
**Scope:** green trunk; honest golden + pass@k AQA in CI; critic cascade on
collect/compose; frozen solver dispose tools; then optional on-policy KD usage.  
**Non-goals:** neurosymbolic student *network*; unbounded MCP autotools;
Ctrl-G/HMM unless P5 latency falsifies; GRPO/SCoRe-RL/SDAR/AgentArk PAD;
SageMaker trajectory on 4.26/1.13 image; live AWS e2e; Serve-side tool loop
without a new ADR.

---

## Cross-cutting standards

| Standard | Rule |
|---|---|
| Organising principle | Eval meaning before scale; disposition before weights |
| Format triangle | Train string == harness generate; Serve stays single-shot |
| Knobs | `MANGOMAS_*` / `MangoMASSettings` only — no call-site literals |
| Tools | Frozen `TOOL_REGISTRY` only; primary dispose tool id `sqe_constraint_solver` |
| Pass@K gate | Chen unbiased estimator defaults $n=5$, gate $k=3$, exact match only on hard/OOD |
| DualDistill | Same task + `expected` + two teachers |
| Vacuity refusal | A gate that cannot fail must not be called pass@k |
| Teachers | Apache-2.0 / MIT open weights (Qwen3, R1-distills); document license |
| ADR numbers | Claimed at land |

---

## Phase map

| Phase | OpenSpec change | Unlock condition |
|---|---|---|
| P0 | `green-trunk-ci` | none |
| P1 | `golden-passk-aqa` | P0 CI green on main |
| P2 | `critic-cascade` | P1 hard-slice pass@k job exists (may be initially red) |
| P3 | `symbolic-disposition` | P1b green on scripted CI + P2 metrics merged |
| P4 | `trl-gkd-usage` | P1–P2 green; explicit usage ADR |
| P5 | (ops runbook later) | P3+P4 meet edge budgets — out of this PLAN’s code scope |

---

## Phase 0 — Green trunk

See `openspec/changes/green-trunk-ci/`. Hotfix only: ruff format, mypy cast,
Bandit revision pins / justified skips. No feature work on a red trunk.

## Phase 1 — Honest eval

See `openspec/changes/golden-passk-aqa/`. Expand golden schema; implement real
pass@1/pass@k; rename AQA docs; wire scripted gate into CI; add rule
traceability stub + prompt_render hash check.

## Phase 2 — Critic cascade

See `openspec/changes/critic-cascade/`. Config-gated filters on collect/compose;
keep recovery traces when outcome matches; emit reject/keep metrics.

## Phase 3 — Symbolic disposition

See `openspec/changes/symbolic-disposition/`. New frozen tool(s); narrow harness
YAML; ADR 0006 addendum; OOD → tool_error / refuse, never confabulate into
control path.

## Phase 4 — On-policy KD usage

See `openspec/changes/trl-gkd-usage/`. Exact TRL minor pin + GKD/on-policy path
with label-mask/vocab tests; trajectory alpha remains 0 until tests say otherwise.

---

## Team swimlanes

| Role | Owns |
|---|---|
| SDLC | P0, CI wiring, OpenSpec archive hygiene |
| SQE | Golden corpus, pass@k, falsifiers, AQA |
| Harness eng | Tools registry, dispatch, collect/compose hooks |
| ML distill | Critic prompts, teacher extract, P4 trainers |
| ML symbolic | Rule extract/mend, solver budgets (P3) |
| Product | Narrow task id, teacher license OK |

---

## Success snapshot (programme done)

- Main CI green including scripted AQA with **named** pass@1 and pass@k.
- Hard-slice gate uses exact match (semantic optional, not success for hard).
- At least one solver dispose tool behind allowlist; OOD refuses.
- Rule traceability matrix non-empty for every hard golden row.
- No GKD merge without usage ADR and harness gates green.
