# Tasks — golden-passk-aqa

- [x] 1. Document golden JSONL schema (`prompt`, `expected`, optional
  `harness_id`, `expected_tools`, `slice`, `allow_semantic`, `id`).
- [x] 2. Expand `configs/golden_sets/` (core + hard); keep scripted fixtures
  in `tests/fixtures/` deterministic for CI.
- [x] 3. Implement pass@1 / pass@k in `scripts/harness/eval_harness.py`;
  emit both in summary JSON; stop counting semantic as success unless
  `allow_semantic`.
- [x] 4. Update `run_aqa_gate.py`: rename docstring/help to honest metrics;
  accept `--pass-k`; fail if hard-slice `pass_at_k` below threshold.
- [x] 5. Wire scripted AQA into `.github/workflows/ci.yml`.
- [x] 6. Add `configs/rule_traceability/matrix.yaml` + check script or test.
- [x] 7. Add prompt_render sync test/guard.
- [x] 8. Unit tests for pass@k aggregation (scripted backend).
- [x] 9. Update `docs/README_AGENT_DISTILLATION.md` + `docs/NEXT_STEPS.md`
  + CHANGELOG; link this change.
- [ ] 10. Archive note in REVIEW when CI job is green on main.
