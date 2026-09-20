# Tasks — golden-passk-aqa

- [ ] 1. Document golden JSONL schema (`id`, `slice`, `prompt`, `expected`,
  optional `expected_tools`, `grader`, `tags`, `solver_fixture`, `ood`, `notes`).
- [ ] 2. Expand `configs/golden_sets/`: create `configs/golden_sets/sqe_hard_ood.jsonl`
  with >=24 rows across hard DAG, hard condition tree, hard mixed, and OOD
  cycle/unsat/schema/theory buckets (`grader: "exact"`); keep scripted fixtures
  in `tests/fixtures/mock_responses_sqe_passk.json` deterministic for CI.
- [ ] 3. Implement Chen unbiased pass@k estimator (arXiv:2107.03374) with n samples
  (default n=5, gate on k=3; also report k=1) in `scripts/harness/run_pass_at_k.py`
  or `eval_harness.py`. Require exact `answers_match` only for hard/ood; stop
  counting semantic_match or unlabeled non-truncated rows as success on hard/ood.
- [ ] 4. Update `run_aqa_gate.py`: correct docstring to state single-pass pass_rate;
  point to `run_pass_at_k.py` for Pass@K; accept `--pass-at-k`, `--samples`, `--slice`.
- [ ] 5. Wire scripted AQA gate into `.github/workflows/ci.yml` and `Makefile`
  (`aqa-gate-passk`). Fail on threshold breach or any OOD synthetic-success violation.
- [ ] 6. Add `configs/rule_traceability/matrix.yaml` + check script or test.
- [ ] 7. Add prompt_render sync test/guard.
- [ ] 8. Unit tests for Chen pass@k aggregation and OOD synthetic-success refusal check.
- [ ] 9. Update `docs/README_AGENT_DISTILLATION.md` + `docs/NEXT_STEPS.md`
  + CHANGELOG; link this change.
- [ ] 10. Archive note in REVIEW when CI job is green on main.
