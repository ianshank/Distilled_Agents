# ADR 0007: Symbolic dispose tools and fail-closed OOD refusal

## Status

Accepted (consolidated canonical record)

## Context

ADR 0006 established the runtime harness-policy contract $G = (H, R_\delta)$, where policy $R_\delta$ proposes candidate actions and harness $H$ disposes them deterministically via a frozen tool registry without in-process `exec` or arbitrary code execution. Consensus review (REVIEW-2026-09-20-symbolic-kd) rejected training an end-to-end "neurosymbolic student network" in favor of symbolic knowledge distillation:
1. The student policy proposes structured action payloads.
2. A deterministic solver/disposition tool validates constraints, checks invariants, and resolves problem state.
3. On invalid, cyclic, unsatisfiable, or out-of-distribution (OOD) inputs, the agent must fail closed by emitting a canonical refusal token (`BLOCKED:<CODE>`), rather than confabulating a false SAT result.

Prior to Phase 0 I3, tools registered in `TOOL_REGISTRY` were shallow string checklists (e.g. `pytest_runner` checked non-empty strings without subprocess; `sqe_checklist` checked for substring `"assert"`), providing AST and schema validation but no semantic or symbolic soundness.

When agents encounter complex Software Quality Engineering (SQE) tasks involving dependency DAGs or boolean condition trees, or out-of-distribution (OOD) tasks with cycles or unsatisfiable constraints, models risk confabulating solutions or fabricating topological orders. The repository requires a deterministic, sound symbolic disposition layer that enforces fail-closed behavior without introducing external heavy solvers (e.g. Z3, clingo), subprocess sandboxes, or arbitrary code execution.

## Decision

1. **Tool Identity & Pure-Python Solvers:**
   - Register frozen tool ID `sqe_constraint_solver` in `TOOL_REGISTRY` (`enhanced_system/harness/tools/registry.py`), distinct from `sqe_checklist`.
   - Implement `SqeConstraintSolverTool` (`tool_id = "sqe_constraint_solver"`) in `enhanced_system/harness/tools/solver.py` implementing pure-Python DAG topological sorting (Kahn's algorithm with min-heap for deterministic lexicographically least topological order) and propositional boolean condition tree evaluation (`and`, `or`, `not`, `atom`).
   - Implement `ConstraintCheckTool` (`tool_id = "constraint_check"`) in `enhanced_system/harness/tools/constraint.py` for structured QC schema validation, required-key presence, and predicate evaluations (`==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not_in`, `contains`, `type`, `regex`). Violations raise `ValueError` mapping to runtime `tool_error`.

2. **Security Boundary:**
   - CI remains dependency-light: zero external Z3 or clingo binary dependencies; forbid `exec`, `eval`, `subprocess`, `os.system`, network, shell, or mandatory native solver libraries in the solver execution path.
   - Out-of-fragment operations raise `UNSUPPORTED_THEORY`.

3. **Standard Reject Codes & Fail-Closed Behavior:**
   - Conform strictly to `openspec/changes/_shared/blocked-reject-codes.md`: `CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.
   - `CYCLE_DETECTED` and `UNSAT`: Return normal JSON observation with `ok: false`, `status: "UNSAT"`, `reject_code: <CODE>`, `order: null`, `assignment: null`. Never raise; never fabricate SAT or topological order.
   - `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, and `RESOURCE_LIMIT`: Raise `ValueError("<CODE>: ...")` which the harness catches as fault `tool_error`.
   - The runtime records fault token `blocked` on trajectories terminating with canonical refusal `BLOCKED:<CODE>`.

4. **Dedicated Harness Specifications:**
   - Add `configs/harnesses/qc_constraints.yaml` (and packaged copy `enhanced_system/config/harnesses/qc_constraints.yaml`) allowlisting `constraint_check` and `final_answer`.
   - Add `configs/harnesses/sqe_dispose.yaml` (and packaged copy `enhanced_system/config/harnesses/sqe_dispose.yaml`) allowlisting `sqe_constraint_solver` and `final_answer`.
   - Keep `sqe_validate.yaml` checklist-only (`sqe_checklist`, `final_answer`).

5. **Fail-Closed OOD Refusal Policy:**
   - In-distribution tasks are solved to sound SAT or proved UNSAT. OOD tasks (disjunctively identified by `slice == "ood"` or `ood == true`) degrade deterministically to canonical refusal tokens `BLOCKED:<CODE>` with zero synthetic success. Confabulating SAT answers on unsatisfiable or cyclic inputs is strictly rejected.

6. **Rule Traceability & Extraction:**
   - Every hard golden row must be mapped in `configs/rule_traceability/matrix.yaml` with `solver_status` in {`fixture`, `active`, `pending`} (never `none`).
   - Add `scripts/harness/extract_rules.py` CLI to extract candidate rules from teacher traces into candidate manifests without auto-promoting into production matrix.

## Consequences

- Policy $R_\delta$ proposes candidate constraints, graphs, and assignments; harness $H$ disposes them through deterministic AST dispatch and sound solver execution without security risks or native solver overhead.
- Default CI remains lightweight, reproducible, and secure with no C-extension or SMT solver dependencies.
- Multi-trial Pass@K evaluation in `scripts/harness/run_pass_at_k.py` on hard and OOD slices verifies zero synthetic-success violations against `configs/golden_sets/sqe_hard_ood.jsonl`.
- Future SMT / ASP solver integrations remain optional behind explicit `UNSUPPORTED_THEORY` boundaries rather than breaking the pure-Python default.
- Unlocks Phase P4 (`trl-gkd-usage`).
