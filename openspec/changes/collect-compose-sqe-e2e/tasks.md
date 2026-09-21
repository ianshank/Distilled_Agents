# Tasks: collect-compose-sqe-e2e

Cites: `docs/plans/symbolic-kd/architecture.md` Section 11.4 E2 and Section 12 (E2 note, PR https://github.com/ianshank/Distilled_Agents/pull/35); ADR 0007; archived `critic-cascade`; `sqe-dispose-e2e-bind`.

## E2 Specification and Contract Pinning (This Change)

- [x] 1. Pin exact Makefile target names: `collect-sqe-dispose` and `compose-dualdistill-sqe` (strictly rejecting `collect-sqe` and `compose-dual`).
- [x] 2. Pin scripted teacher fixture path: `tests/fixtures/mock_responses_sqe_passk.json` with dispose-bound `sqe_constraint_solver` before `final_answer`.
- [x] 3. Pin reject sink path: `artifacts/critic_rejects.jsonl` (sole sink; override via `--reject-log` or `MANGOMAS_CRITIC_REJECT_LOG` only).
- [x] 4. Pin single DualDistill drop rule: drop pairs iff both teachers score 0 on the graded task, logging `critic_reject_code: DUALDISTILL_DROP_0_0` (strictly no second drop rule).
- [x] 5. Pin collect outcome filter drop: log `critic_reject_code: OUTCOME_MISMATCH` when `final_answer` fails `answers_match`.
- [x] 6. Pin recovery trace retention: intermediate faults with matching final answer are kept (`critic_kept_recovery`).
- [x] 7. Document scope boundaries: lock GKD / `trl-gkd-usage` until E3 BC smoke; forbid Neuroharness, Edge-AI, INV-16, and Serve tool-loop; forbid new Pass@K CLIs; forbid routing collect through `run_aqa_gate.py`; defer E3 BC smoke corpus and job naming.
- [x] 8. Cite architecture.md Section 11.4 E2 and Section 12; ADR 0007; `openspec/changes/archive/critic-cascade/`; `openspec/changes/sqe-dispose-e2e-bind/`.

## E2 Implementer Milestone (Collect -> Critic -> Compose under sqe_dispose)

- [x] 9. Add Makefile target `collect-sqe-dispose` invoking `scripts/harness/collect_trajectories.py` with `--harness-id sqe_dispose`, input prompts from `configs/golden_sets/sqe_hard_ood.jsonl`, scripted fixtures `tests/fixtures/mock_responses_sqe_passk.json`, critic enabled, and `--reject-log artifacts/critic_rejects.jsonl`.
- [x] 10. Add Makefile target `compose-dualdistill-sqe` invoking `scripts/harness/compose_dualdistill.py` with dual dispose teacher JSONLs and `--reject-log artifacts/critic_rejects.jsonl`.
- [x] 11. Verify collected trajectories under `sqe_dispose` record `sqe_constraint_solver` prior to `final_answer`.
- [x] 12. Add unit/integration test verifying `OUTCOME_MISMATCH` is emitted to `artifacts/critic_rejects.jsonl` when collect encounters an outcome failure.
- [x] 13. Add unit/integration test verifying `DUALDISTILL_DROP_0_0` is emitted to `artifacts/critic_rejects.jsonl` when compose encounters a (0,0) teacher pair, and no other drop rule fires.
- [x] 14. Verify recovery traces with intermediate parse/tool errors are retained when the outcome matches.
- [x] 15. Verify `make aqa-gate-passk` remains green and unmodified.

## Follow-On Milestones (Deferred to E3-E5)

- [ ] 16. E3: Execute behavior cloning (BC) smoke run on `sqe_dispose` harness (corpus and job naming deferred; GKD remains locked).
- [ ] 17. E4: Harden CI checks to fail if golden rows or fixtures bypass `sqe_constraint_solver`.
- [ ] 18. E5: Archive OpenSpec changes upon full E2E verification.
