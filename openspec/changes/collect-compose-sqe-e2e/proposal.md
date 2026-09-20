# Proposal: collect-compose-sqe-e2e

**Status:** proposed  
**Date:** 2026-09-20  
**Depends on:** `sqe-dispose-e2e-bind` (E1 on main, PR #34)  
**Plan:** `docs/plans/symbolic-kd/architecture.md` Section 11.4 (E2 order) and Section 12 (E2 note, PR https://github.com/ianshank/Distilled_Agents/pull/35)  
**Why:** E1 bound `sqe_hard_ood.jsonl` to `harness_id: sqe_dispose`, requiring `sqe_constraint_solver` before `final_answer` on `make aqa-gate-passk`. E2 operationalizes the trajectory collection and DualDistill composition pipelines under `sqe_dispose` using the critic cascade (`MANGOMAS_CRITIC_ENABLED`). This ensures student distillation data is sourced exclusively from dispose-compliant teacher rollouts while capturing reject telemetry in `artifacts/critic_rejects.jsonl`.

## Scope

1. **Makefile Targets:** Add targets `collect-sqe-dispose` and `compose-dualdistill-sqe` (exact target names; not `collect-sqe` or `compose-dual`) invoking `scripts/harness/collect_trajectories.py` and `scripts/harness/compose_dualdistill.py`.
2. **Scripted Teacher Fixtures:** Run collect against `configs/golden_sets/sqe_hard_ood.jsonl` prompts using `tests/fixtures/mock_responses_sqe_passk.json` under `sqe_dispose` harness. Every teacher trace MUST execute `sqe_constraint_solver` before `final_answer`. Out-of-distribution (OOD) tasks produce canonical `BLOCKED:<CODE>` tokens per `openspec/changes/_shared/blocked-reject-codes.md`.
3. **Pipeline Flow:** `scripts/harness/collect_trajectories.py` -> critic cascade (`MANGOMAS_CRITIC_ENABLED`) -> `scripts/harness/compose_dualdistill.py` / `compose_pair`.
4. **Reject Sink Telemetry:** Telemetry sink remains `artifacts/critic_rejects.jsonl` (or `--reject-log` override).
   - Collect outcome filter drops log `critic_reject_code: OUTCOME_MISMATCH` when `final_answer` fails `answers_match`.
   - DualDistill compose drops log `critic_reject_code: DUALDISTILL_DROP_0_0` when both teachers score 0 on the graded outcome (existing `(0,0)` rule only; strictly no second drop rule).
5. **Recovery Trace Retention:** Trajectories with intermediate `parse_error` or `tool_error` are preserved if the final outcome matches `expected`.
6. **Acceptance Proof:** Unit and integration tests prove `OUTCOME_MISMATCH` and `DUALDISTILL_DROP_0_0` fire on forced mismatch and forced (0,0) against `sqe_dispose` traces.

## Non-goals and Locks

- No GKD / `trl-gkd-usage` enable before E3 BC smoke is green.
- No Neuroharness.
- No Edge-AI / INV-16 / Serve tool-loop.
- No new Pass@K CLI; `make aqa-gate-passk` remains the sole Chen Pass@K gate and is not redefined by E2.
- Do not route E2 collect through `run_aqa_gate.py`.
- Defer E3 BC smoke corpus and job naming (do not invent job names).
- No second DualDistill drop rule (no drops on partial credit, schema-only faults, or single-teacher failure).

## Citations and Precedents

- `docs/plans/symbolic-kd/architecture.md` Section 11.4 E2 and Section 12 (E2 note, PR #35).
- `docs/adr/0007-symbolic-dispose-tools.md` (canonical dispose tools).
- `openspec/changes/archive/critic-cascade/` (archived critic and reject telemetry behavior).
- `openspec/changes/sqe-dispose-e2e-bind/` (harness_id and expected_tools continuity).
- `openspec/changes/_shared/blocked-reject-codes.md` (byte-stable refusal codes).
