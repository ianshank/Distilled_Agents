# Tasks — symbolic-disposition

- [ ] 1. Product confirms harness id + first constraint domain.
- [ ] 2. Implement `ConstraintCheckTool` (pure Python) in
  `enhanced_system/harness/tools/registry.py`; register id.
- [ ] 3. Add `configs/harnesses/<id>.yaml` + packaged copy under
  `enhanced_system/config/harnesses/` (schema `additionalProperties: false`
  — update both trees).
- [ ] 4. Unit tests: valid dispose, invalid args → ValueError, unknown tool
  still KeyError.
- [ ] 5. Golden hard rows + matrix rows for the new harness.
- [ ] 6. Optional: `scripts/harness/extract_rules.py` stub writing matrix
  candidates (no auto-promote).
- [ ] 7. ADR at land; update distill README false-friends; CHANGELOG;
  NEXT_STEPS.
- [ ] 8. Eval: scripted pass@k on new harness fixtures green in CI.
