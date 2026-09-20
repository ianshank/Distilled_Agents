# Tasks: golden-passk-aqa

- [x] 1. Document golden JSONL schema (`id`, `slice`, `prompt`, `expected`,
  optional `expected_tools`, `grader`, `tags`, `solver_fixture`, `ood`, `notes`).
  Standardize refusal tokens per `openspec/changes/_shared/blocked-reject-codes.md`.
- [x] 2. Expand `configs/golden_sets/`: create `configs/golden_sets/sqe_hard_ood.jsonl`
  with >=24 rows meeting per-bucket minima (DAG >= 6, tree >= 6, mixed >= 4,
  cycle >= 2, unsat >= 2, schema >= 2, theory >= 2) with `grader: "exact"`;
  keep scripted fixtures in `tests/fixtures/mock_responses_sqe_passk.json`
  deterministic for CI.
- [x] 3. Implement Chen unbiased pass@k estimator (arXiv:2107.03374) with n samples
  (default n=5, gate on k=3; also report k=1) in dedicated CLI `scripts/harness/run_pass_at_k.py`.
  Require exact `answers_match` only for hard/ood; stop counting semantic_match
  or unlabeled non-truncated rows as success on hard/ood. Enforce OOD rule: row is
  OOD iff `slice` equals "ood" OR `ood` is true.
- [x] 4. Update `run_aqa_gate.py`: correct docstring to state single-pass pass_rate;
  point to `run_pass_at_k.py` for Pass@K. No dual-CLI extension of run_aqa_gate.py.
- [x] 5. Wire scripted AQA gate into `.github/workflows/ci.yml` and `Makefile`
  (`aqa-gate-passk` calling `run_pass_at_k.py`). Fail on threshold breach or any
  OOD synthetic-success violation.
- [x] 6. Add `configs/rule_traceability/matrix.yaml` + check script or test.
- [x] 7. Add prompt_render sync test/guard.
- [x] 8. Unit tests for Chen pass@k aggregation, disjunctive OOD detection,
  and OOD synthetic-success refusal check.
- [x] 9. Update `docs/README_AGENT_DISTILLATION.md` + `docs/NEXT_STEPS.md`
  + CHANGELOG; link this change.
- [ ] 10. Archive note in REVIEW when CI job is green on main.
