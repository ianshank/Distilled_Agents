# Tasks: symbolic-disposition

- [ ] 1. Product confirms harness id (`sqe_dispose`) and domain fixtures.
- [ ] 2. Implement `SqeConstraintSolverTool` (primary tool id `sqe_constraint_solver`,
  pure Python DAG topo + boolean condition tree evaluation) in
  `enhanced_system/harness/tools/registry.py`; register id in `TOOL_REGISTRY`.
- [ ] 3. Add `configs/harnesses/sqe_dispose.yaml` + packaged copy under
  `enhanced_system/config/harnesses/` (schema `additionalProperties: false`
  - update both trees) allowlisting `sqe_constraint_solver` and `final_answer`;
  keep `sqe_validate.yaml` checklist-only.
- [ ] 4. Unit tests: valid DAG solve (lexicographically least topo order),
  cyclic graph -> `ok: false` with `CYCLE_DETECTED`, unsatisfiable constraints ->
  `ok: false` with `UNSAT`, bad schema/limits/out-of-fragment syntax -> raise
  with reject codes conforming to `openspec/changes/_shared/blocked-reject-codes.md`
  (`CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`,
  `RESOURCE_LIMIT`), unknown tool still KeyError. No exec/eval/subprocess.
- [ ] 5. Golden hard rows + matrix rows for the new harness referencing
  `sqe_constraint_solver` and fail-closed `BLOCKED:<CODE>` refusal tokens.
  OOD detection must treat `slice: "ood"` OR `ood: true` equivalently.
- [ ] 6. Optional: `scripts/harness/extract_rules.py` stub writing matrix
  candidates (no auto-promote).
- [ ] 7. ADR 0007 at land; update distill README false-friends; CHANGELOG;
  NEXT_STEPS.
- [ ] 8. Eval: verify solver is gated against `golden-passk-aqa` hard-slice Pass@K
  gate; solver MUST NOT be marked done without hard-slice gate. Scripted pass@k
  green in CI.
