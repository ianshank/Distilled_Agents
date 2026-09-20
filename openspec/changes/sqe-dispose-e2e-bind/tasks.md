# Tasks: sqe-dispose-e2e-bind

Cites: `docs/plans/symbolic-kd/architecture.md` Section 11 and Section 11.7 (Architect PR https://github.com/ianshank/Distilled_Agents/pull/30).

## E0: Spec and Contract Pinning (This Change)

- [x] 1. Specify normative golden contract: every row in `configs/golden_sets/sqe_hard_ood.jsonl` MUST set `harness_id: "sqe_dispose"` and `expected_tools: ["sqe_constraint_solver", "final_answer"]`.
- [x] 2. Specify scripted fixtures contract: `tests/fixtures/mock_responses_sqe_passk.json` MUST execute `sqe_constraint_solver` before `final_answer`. OOD rows MUST yield canonical `BLOCKED:<CODE>` per `openspec/changes/_shared/blocked-reject-codes.md`.
- [x] 3. Specify rule traceability matrix binding: `configs/rule_traceability/matrix.yaml` MUST map all hard/OOD IDs with `harness_id: "sqe_dispose"` and `solver_status` in {`fixture`, `active`}.
- [x] 4. Specify vacuity falsifier: any fixture or trace reaching `final_answer` without prior `sqe_constraint_solver` call MUST fail `make aqa-gate-passk`.
- [x] 5. Pin byte-stable identifiers: `sqe_constraint_solver`, `sqe_dispose`, `aqa-passk-summary.json`, and shared path `openspec/changes/_shared/blocked-reject-codes.md`.
- [x] 6. Pin allowlist invariant: `configs/harnesses/sqe_dispose.yaml` `action.tool_ids` = `[sqe_constraint_solver, final_answer]` only.
- [x] 7. Document out-of-scope boundaries: GKD enable locked until E3 BC smoke; Edge-AI / INV-16 locked; Pass@K via `run_aqa_gate.py` forbidden.
- [x] 8. Defer E3 BC smoke corpus and job naming (do not invent job names).

## E1: Implementer Milestone (Pass@K x sqe_dispose Binding)

- [ ] 9. Verify every row in `configs/golden_sets/sqe_hard_ood.jsonl` contains `harness_id: "sqe_dispose"` and `expected_tools: ["sqe_constraint_solver", "final_answer"]`.
- [ ] 10. Update scripted fixtures (`tests/fixtures/mock_responses_sqe_passk.json`) so every hard/OOD row executes `sqe_constraint_solver` before `final_answer`, and OOD rows return canonical `BLOCKED:<CODE>`.
- [ ] 11. Implement tool-call compliance assertion in `scripts/harness/run_pass_at_k.py` so trials reaching `final_answer` without prior `sqe_constraint_solver` fail the gate (vacuity falsifier).
- [ ] 12. Update `configs/rule_traceability/matrix.yaml` so every hard/OOD golden ID has `harness_id: "sqe_dispose"` and `solver_status` updated from `pending` to `fixture` or `active`.
- [ ] 13. Ensure `make aqa-gate-passk` is green and outputs `aqa-passk-summary.json` asserting recorded tool-call compliance and zero synthetic-success violations (`ood_synthetic_success_violations == 0`).
- [x] 14. Close symbolic-disposition tasks.md item 8 ONLY when E1 is verified green on dispose-bound fixtures (not Echo/final_answer-only). COMPLETE: E1 green on dispose-bound fixtures (PR #34); not Echo/final_answer-only.

## E2-E5: Follow-On Lifecycle (Deferred)

- [ ] 15. E2: Run collect and compose with critic enabled against `sqe_dispose` traces, sinking `OUTCOME_MISMATCH` and `DUALDISTILL_DROP_0_0` to `artifacts/critic_rejects.jsonl`.
- [ ] 16. E3: Execute behavior cloning (BC) smoke run on `sqe_dispose` harness (corpus and job naming deferred; GKD remains disabled).
- [ ] 17. E4: Harden CI checks to fail if golden rows or fixtures bypass `sqe_constraint_solver`.
- [ ] 18. E5: Archive OpenSpec change packages after E1-E4 completion.
