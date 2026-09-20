# ADR 0007: Symbolic dispose tools and fail-closed OOD refusal

## Status

Accepted

## Context

Following ADR 0006, agent execution is structured as \( G = (H, R_\delta) \), where \( H \) is the deterministic runtime harness and \( R_\delta \) is the distilled policy. Consensus review (REVIEW-2026-09-20-symbolic-kd) rejected training an end-to-end "neurosymbolic student network" in favor of symbolic knowledge distillation:
1. The student policy proposes structured action payloads.
2. A deterministic solver/disposition tool validates constraints, checks invariants, and resolves problem state.
3. On invalid, cyclic, unsatisfiable, or out-of-distribution (OOD) inputs, the agent must fail closed by emitting a canonical refusal token (`BLOCKED:<CODE>`), rather than confabulating a false SAT result.

Prior to this ADR, tools in `TOOL_REGISTRY` were static checklists (e.g. `pytest_runner` checked non-empty strings without subprocess; `sqe_checklist` checked for substring `"assert"`).

## Decision

1. **Tool Registry & Pure-Python Solvers:**
   - Implement `ConstraintCheckTool` (`tool_id = "constraint_check"`) in `enhanced_system/harness/tools/constraint.py` for structured QC schema validation, required-key presence, and predicate evaluations (`==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not_in`, `contains`, `type`, `regex`). Violations raise `ValueError` mapping to runtime `tool_error`.
   - Implement `SqeConstraintSolverTool` (`tool_id = "sqe_constraint_solver"`) implementing pure-Python DAG topological sorting (lexicographically least order via Kahn's algorithm with a min-heap) and boolean condition tree evaluation.
   - CI remains dependency-light: no external Z3 or clingo binary dependencies; zero `exec`, `eval`, shell, or subprocess execution.
2. **Standard Reject Codes:**
   - Conforms strictly to `openspec/changes/_shared/blocked-reject-codes.md`: `CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.
   - In `sqe_constraint_solver`, cyclic and unsatisfiable inputs return `ok: false`, `status: "UNSAT"` with `reject_code` populated; schema, syntax, limits, or unsupported theories raise `ValueError(f"{CODE}: ...")`.
3. **Dedicated Harness Specifications:**
   - Add `configs/harnesses/qc_constraints.yaml` (and packaged copy `enhanced_system/config/harnesses/qc_constraints.yaml`) allowlisting `constraint_check` and `final_answer`.
   - Add `configs/harnesses/sqe_dispose.yaml` (and packaged copy `enhanced_system/config/harnesses/sqe_dispose.yaml`) allowlisting `sqe_constraint_solver` and `final_answer`.
   - Keep `sqe_validate.yaml` checklist-only (`sqe_checklist`, `final_answer`).
4. **Fail-Closed OOD Refusal Policy:**
   - OOD rows (`slice: "ood"` or `ood: true`) requiring refusal map to canonical `BLOCKED:<CODE>` tokens (e.g., `BLOCKED:UNSAT`). Confabulating SAT answers on unsatisfiable or cyclic inputs is strictly rejected.
5. **Rule Traceability & Extraction:**
   - Every hard golden row must be mapped in `configs/rule_traceability/matrix.yaml` with `solver_status` in {`fixture`, `active`, `pending`} (never `none`).
   - Add `scripts/harness/extract_rules.py` CLI to extract candidate rules from teacher traces into candidate manifests without auto-promoting into production matrix.

## Consequences

- Operators can deploy compact student models (0.5B–3B) that reliably propose actions to sound, deterministic disposition tools without security risks or native solver overhead.
- Multi-trial pass@k evaluation in `run_aqa_gate.py` and `eval_harness.py` gates regressions on both valid SAT paths and fail-closed OOD refusal tokens.
- Unlocks Phase P4 (`trl-gkd-usage`).
