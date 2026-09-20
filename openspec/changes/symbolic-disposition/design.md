# Design: symbolic-disposition

## Architecture

```
User task
  -> AgentRuntime (H)
      -> ModelBackend proposes thought + JSON/Call action (R_delta)
      -> dispatch.parse_action (allowlist)
      -> TOOL_REGISTRY[sqe_constraint_solver].run(args) ## deterministic dispose
      -> observation back to loop
  -> final_answer only when solver OK (canonical SAT) or explicit refuse (BLOCKED:<CODE>)
```

## Tool contract (`sqe_constraint_solver`)

- **Primary tool id:** `sqe_constraint_solver` (distinct from `sqe_checklist`).
- **Engine:** Pure-Python DAG topological ordering + boolean condition tree evaluation.
  Raise `UNSUPPORTED_THEORY` for operations outside the supported fragment. No Z3 or
  clingo required for CI; no network, shell, `exec`, `eval`, or subprocess. Optional
  Z3/clingo integrations are deferred or kept behind optional extras emitting
  `UNSUPPORTED_THEORY` when uninstalled; default CI stays lightweight pure Python.
- **Input schema:**
  - `mode`: `"check"` | `"solve"`
  - `graph`: `{"nodes": ["string"], "edges": [["from", "to"]]}` where edge `from -> to` indicates `from` precedes `to`
  - `constraints`: list of constraint nodes (`id`, `op` in `{"and", "or", "not", "atom"}`, `args`, `atom`)
  - `assignment`: optional dictionary of predicate truth values
  - `limits`: `{"max_nodes": 64, "max_edges": 256, "max_constraints": 128}`
- **Output schema (JSON string):**
  - Fields: `ok` (bool), `status` (`"SAT"` | `"UNSAT"`), `order` (list of strings or null), `assignment` (dict or null), `reject_code` (string or null), `details` (dict).
  - SAT path: `ok: true`, `status: "SAT"`, `order` set to lexicographically least valid topological order, `reject_code: null`.
  - UNSAT path: `ok: false`, `status: "UNSAT"`, `order: null`, `assignment: null`, `reject_code` set to code (e.g. `CYCLE_DETECTED` or `UNSAT`), `details` explaining reason.
- **Reject codes and fail-closed mapping:**
  - Standardized reject codes conform to `openspec/changes/_shared/blocked-reject-codes.md`:
    `CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.
  - `SCHEMA_VIOLATION`: Malformed input or limit breach -> raise `ValueError` -> runtime `tool_error` -> expected `BLOCKED:SCHEMA_VIOLATION`.
  - `SYNTAX_INVALID`: Unparseable tree/atom -> raise `ValueError` -> runtime `tool_error` -> expected `BLOCKED:SYNTAX_INVALID`.
  - `CYCLE_DETECTED`: Cyclic graph -> return `ok: false`, `status: "UNSAT"`, `reject_code: "CYCLE_DETECTED"` -> expected `BLOCKED:CYCLE_DETECTED`.
  - `UNSAT`: Contradictory predicates -> return `ok: false`, `status: "UNSAT"`, `reject_code: "UNSAT"` -> expected `BLOCKED:UNSAT`.
  - `UNSUPPORTED_THEORY`: Out-of-fragment operation -> raise `ValueError` -> runtime `tool_error` -> expected `BLOCKED:UNSUPPORTED_THEORY`.
  - `RESOURCE_LIMIT`: Resource cap exceeded -> raise `ValueError` -> runtime `tool_error` -> expected `BLOCKED:RESOURCE_LIMIT`.
- **Fail-closed locks:**
  - Never return `ok: true` or `status: "SAT"` for cyclic or unsatisfiable inputs.
  - Never return a fabricated `order` unless status is SAT.
  - Policy may only emit `final_answer` after solver observation; OOD gold target is `BLOCKED:<CODE>`.
  - Introducing fault token `blocked` on runtime trajectories is recommended.

## Harness configuration

- Dedicated config `configs/harnesses/sqe_dispose.yaml` (and packaged copy under
  `enhanced_system/config/harnesses/sqe_dispose.yaml`) allowlisting `sqe_constraint_solver`
  and `final_answer`.
- `configs/harnesses/sqe_validate.yaml` remains checklist-only (`sqe_checklist`, `final_answer`).

## Dependencies and gate

- Depends on: `golden-passk-aqa` hard-slice gate exists in CI (scripted backend green initially).
- The solver MUST NOT be marked done without the hard-slice gate in place.

## ADR

Land ADR 0007 (or amend ADR 0006) at merge time describing dispose tools, fail-closed
execution, and OOD refusal (`BLOCKED:<CODE>`). Number claimed at land.
