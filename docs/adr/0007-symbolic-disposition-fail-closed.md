# ADR 0007: Symbolic disposition and fail-closed OOD rejection

## Status

Accepted

## Context

ADR 0006 established the runtime harness–policy contract $G = (H, R_\delta)$, where policy $R_\delta$ proposes candidate actions and harness $H$ disposes them deterministically via a frozen tool registry without in-process `exec` or arbitrary code execution. Prior to Phase 0 I3, tools registered in `TOOL_REGISTRY` were shallow string checklists (`sqe_checklist`, `pytest_runner`) providing AST and schema validation but no semantic or symbolic soundness.

When agents encounter complex Software Quality Engineering (SQE) tasks involving dependency DAGs or boolean condition trees, or out-of-distribution (OOD) tasks with cycles or unsatisfiable constraints, models risk confabulating solutions or fabricating topological orders. The repository requires a deterministic, sound symbolic disposition layer that enforces fail-closed behavior without introducing external heavy solvers (e.g. Z3, clingo), subprocess sandboxes, or arbitrary code execution.

## Decision

1. **Primary Tool Identity:** Register frozen tool ID `sqe_constraint_solver` in `TOOL_REGISTRY` (`enhanced_system/harness/tools/registry.py`), distinct from `sqe_checklist`.
2. **Pure-Python Engine:** Implement DAG topological sorting (Kahn's algorithm with min-heap for deterministic lexicographically least topological order) and propositional boolean condition tree evaluation (`and`, `or`, `not`, `atom`) in pure Python (`enhanced_system/harness/tools/solver.py`).
3. **Security Boundary:** Forbid `exec`, `eval`, `subprocess`, `os.system`, network, shell, or mandatory native solver libraries (Z3, clingo) in the solver execution path.
4. **Dedicated Harness Configuration:** Introduce `configs/harnesses/sqe_dispose.yaml` and packaged duplicate `enhanced_system/config/harnesses/sqe_dispose.yaml` allowlisting ONLY `sqe_constraint_solver` and `final_answer`. Maintain `sqe_validate.yaml` as checklist-only (`sqe_checklist`, `final_answer`).
5. **Standard Reject Codes & Fail-Closed Behavior:** Conform strictly to `openspec/changes/_shared/blocked-reject-codes.md`:
   - `CYCLE_DETECTED` and `UNSAT`: Return normal JSON observation with `ok: false`, `status: "UNSAT"`, `reject_code: <CODE>`, `order: null`, `assignment: null`. Never raise; never fabricate SAT or topological order.
   - `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, and `RESOURCE_LIMIT`: Raise `ValueError("<CODE>: ...")` which the harness catches as fault `tool_error`.
   - The runtime records fault token `blocked` on trajectories terminating with canonical refusal `BLOCKED:<CODE>`.
6. **Fail-Closed OOD Refusal:** In-distribution tasks are solved to sound SAT or proved UNSAT. OOD tasks (disjunctively identified by `slice == "ood"` or `ood == true`) degrade deterministically to canonical refusal tokens `BLOCKED:<CODE>` with zero synthetic success.

## Consequences

- Policy $R_\delta$ proposes candidate constraints, graphs, and assignments; harness $H$ disposes them through deterministic AST dispatch and sound solver execution.
- Default CI remains lightweight, reproducible, and secure with no C-extension or SMT solver dependencies.
- Multi-trial Pass@K evaluation on hard and OOD slices can verify zero synthetic-success violations against `configs/golden_sets/sqe_hard_ood.jsonl`.
- Future SMT / ASP solver integrations remain optional behind explicit `UNSUPPORTED_THEORY` boundaries rather than breaking the pure-Python default.
