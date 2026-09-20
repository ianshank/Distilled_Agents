# Tasks: symbolic-disposition

- [x] 1. Product confirms harness id (`sqe_dispose` / `qc_constraints`) and domain fixtures.
- [x] 2. Implement `SqeConstraintSolverTool` (primary tool id `sqe_constraint_solver`,
  pure Python DAG topo + boolean condition tree evaluation) and `ConstraintCheckTool` (`constraint_check`) in
  `enhanced_system/harness/tools/registry.py`; register ids in `TOOL_REGISTRY`.
- [x] 3. Add `configs/harnesses/qc_constraints.yaml` and `sqe_dispose.yaml` + packaged copies under
  `enhanced_system/config/harnesses/` (schema `additionalProperties: false`
  - update both trees) allowlisting dispose tools and `final_answer`;
  keep `sqe_validate.yaml` checklist-only.
- [x] 4. Unit tests: valid DAG solve (lexicographically least topo order),
  cyclic graph -> `ok: false` with `CYCLE_DETECTED`, unsatisfiable constraints ->
  `ok: false` with `UNSAT`, bad schema/limits/out-of-fragment syntax -> raise
  with reject codes conforming to `openspec/changes/_shared/blocked-reject-codes.md`
  (`CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`,
  `RESOURCE_LIMIT`), unknown tool still KeyError. No exec/eval/subprocess.
- [x] 5. Golden hard rows + matrix rows for the new harness referencing
  dispose tools and fail-closed `BLOCKED:<CODE>` refusal tokens.
  OOD detection must treat `slice: "ood"` OR `ood: true` equivalently.
- [x] 6. Optional: `scripts/harness/extract_rules.py` stub writing matrix
  candidates (no auto-promote).
- [x] 7. ADR 0007 at land; update distill README false-friends; CHANGELOG;
  NEXT_STEPS.
- [x] 8. Eval: verify solver is gated against `golden-passk-aqa` hard-slice Pass@K
  gate; solver MUST NOT be marked done without hard-slice gate. Scripted pass@k
  green in CI. *(Note: hard-slice Pass@K gate dependency satisfied by aqa-gate-passk existence on PR #23 / landing on main; solver unit proofs and fail-closed OOD rejection are verified against the gate contract).*
