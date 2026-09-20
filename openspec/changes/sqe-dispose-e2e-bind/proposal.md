# Proposal: sqe-dispose-e2e-bind

**Status:** proposed  
**Date:** 2026-09-20  
**Depends on:** Phase 0 kill CLEARED on main (PR #29, run 35518614849)  
**Plan:** `docs/plans/symbolic-kd/architecture.md` Section 11 and Section 11.7 (Architect PR https://github.com/ianshank/Distilled_Agents/pull/30)  
**Why:** Phase 0 kill criteria are CLEARED on main, but Pass@K evaluation still does not prove the dispose path. The initial scripted fixtures in `tests/fixtures/mock_responses_sqe_passk.json` emitted `final_answer` directly (Echo shortcut), bypassing the `sqe_dispose` harness and `sqe_constraint_solver` tool execution. To achieve end-to-end functional proof, evaluation on hard and OOD slices must strictly enforce tool-call compliance through `sqe_constraint_solver` before `final_answer`.

## Problem

1. While `configs/golden_sets/sqe_hard_ood.jsonl` contains hard and OOD tasks, tests and mock fixtures permitted evaluating `final_answer` without invoking `sqe_constraint_solver`.
2. A gate that accepts `final_answer` without prior disposition tool execution suffers from vacuity: it tests string matching rather than the cognitive-proposes / harness-disposes contract $G = (H, R_\delta)$.
3. Rule traceability in `configs/rule_traceability/matrix.yaml` must bind all hard/OOD golden IDs to `harness_id: sqe_dispose` with `solver_status` in {`fixture`, `active`}.

## Scope

- Every row in `configs/golden_sets/sqe_hard_ood.jsonl` MUST set `harness_id: sqe_dispose` and `expected_tools` including `sqe_constraint_solver` then `final_answer`.
- Scripted fixtures MUST invoke `sqe_constraint_solver` before `final_answer`.
- OOD rows MUST produce canonical `BLOCKED:<CODE>` refusal tokens per `openspec/changes/_shared/blocked-reject-codes.md` (path kept byte-stable).
- `configs/rule_traceability/matrix.yaml` and `make aqa-gate-passk` MUST bind `sqe_hard_ood.jsonl`; `scripts/harness/run_pass_at_k.py` remains the sole Chen CLI; matrix `solver_status` must be in {`fixture`, `active`}.
- Falsifier: any fixture or trace reaching `final_answer` without prior `sqe_constraint_solver` invocation MUST fail `aqa-gate-passk` (vacuity failure) even if `answers_match` would pass.
- E1 acceptance asserts recorded tool-call / `expected_tools` compliance, not exact-match scores alone.
- Allowlist: `configs/harnesses/sqe_dispose.yaml` (and packaged copy) `tool_ids` = `[sqe_constraint_solver, final_answer]` only.
- Close `symbolic-disposition` tasks item 8 ONLY when E1 is green on dispose-bound fixtures (not Echo/`final_answer`-only).
- Byte-stable IDs: `sqe_constraint_solver`, `sqe_dispose`, `aqa-passk-summary.json`.

## Non-goals and Locks

- GKD / on-policy distillation before E3 BC smoke is out of scope.
- Edge-AI and INV-16 language remain strictly locked out.
- Pass@K via `scripts/harness/run_aqa_gate.py` is forbidden (`run_aqa_gate.py` stays single-pass regression only).
- Defer E3 BC smoke corpus and job naming (do not invent job names).
- No external native solver dependencies (Z3, clingo) in default CI; no `exec`/`eval`/subprocess.
