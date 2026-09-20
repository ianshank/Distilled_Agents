# Spec delta — solver-dispose

## ADDED Requirements

### Requirement: Primary tool identity and pure-Python registry

The harness SHALL register primary tool id `sqe_constraint_solver` in `TOOL_REGISTRY`,
distinct from `sqe_checklist`. Any reference to generic `ConstraintCheckTool` is deprecated
or aliased to `sqe_constraint_solver`. The tool default engine SHALL be pure Python
implementing DAG topological ordering and boolean condition trees with no Z3 or clingo
dependency, and MUST NOT use `exec`, `eval`, shell, or subprocess execution.

#### Scenario: Registry lookup

- **WHEN** `get_tool("sqe_constraint_solver")` is called after this change
- **THEN** it returns an instance whose `run` executes pure-Python constraint solving without subprocess or external native solver

### Requirement: Input and output contract

The tool SHALL accept JSON payload fields `mode` (`"check"` | `"solve"`), `graph` (`nodes`, `edges`),
`constraints` (tree with `op`, `args`, `atom`), optional `assignment`, and `limits`.
The tool SHALL return a JSON string with `ok` (bool), `status` (`"SAT"` | `"UNSAT"`),
`order` (list or null), `assignment` (dict or null), `reject_code` (string or null),
and `details` (dict).

#### Scenario: SAT solve returns topological order

- **GIVEN** a valid DAG with mode `"solve"`
- **WHEN** the tool runs successfully
- **THEN** `ok` is true, `status` is `"SAT"`, `order` contains the lexicographically least valid topological order, and `reject_code` is null

### Requirement: Reject codes and fail-closed behavior

The tool SHALL employ reject codes: `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `CYCLE_DETECTED`,
`UNSAT`, `UNSUPPORTED_THEORY`, and `RESOURCE_LIMIT`.
It SHALL never return `ok: true` or fabricate an `order` for cyclic or unsatisfiable inputs.

#### Scenario: Cycle detection is not SAT

- **GIVEN** a graph containing a cycle
- **WHEN** the tool runs
- **THEN** `ok` is false, `status` is `"UNSAT"`, `reject_code` is `"CYCLE_DETECTED"`, and `order` is null

#### Scenario: Unsupported theory raises

- **GIVEN** a constraint operation outside the DAG+boolean-tree fragment (e.g. numeric ILP)
- **WHEN** the tool runs
- **THEN** it raises `ValueError("UNSUPPORTED_THEORY: ...")` and the runtime records fault `tool_error`

### Requirement: Dedicated dispose harness YAML

The repository SHALL provide `configs/harnesses/sqe_dispose.yaml` (and packaged copy)
whose `action.tool_ids` allowlist contains `sqe_constraint_solver` and `final_answer`.
`configs/harnesses/sqe_validate.yaml` SHALL remain checklist-only (`sqe_checklist`, `final_answer`).

#### Scenario: Dispose harness allows solver

- **WHEN** `configs/harnesses/sqe_dispose.yaml` is loaded
- **THEN** allowlisted tool ids include `sqe_constraint_solver` and `final_answer`, and do not conflate with `sqe_validate.yaml`

### Requirement: OOD degrades to canonical refusal, not confabulation

Harness instructions and policy MUST map OOD conditions to explicit canonical refusal
tokens `BLOCKED:<CODE>` (e.g. `BLOCKED:CYCLE_DETECTED`, `BLOCKED:UNSAT`). The policy
MUST NOT fabricate a SAT solution after receiving an error or UNSAT result.

#### Scenario: Refusal token emitted on OOD

- **WHEN** an OOD problem with a cycle or unsatisfiable constraint is processed
- **THEN** the model observation leads to `final_answer` with `BLOCKED:<CODE>` matching gold refusal expectation

### Requirement: Gate dependency on hard-slice Pass@K

The solver implementation and disposition change MUST NOT be marked complete without
the `golden-passk-aqa` hard-slice Pass@K gate in place and verified green on scripted fixtures.

#### Scenario: Incomplete gate blocks solver sign-off

- **WHEN** evaluating completion criteria for `symbolic-disposition`
- **THEN** completion is rejected if the hard-slice Pass@K gate is absent

### Requirement: Traceability

Every hard golden for the dispose harness MUST reference a matrix row with
`solver_status` in {`fixture`,`active`,`pending`} and MUST NOT use `none`.

## Falsifier

- Returning `ok: true` or any non-null `order` on cyclic or unsatisfiable graph inputs MUST fail unit tests.
- Claiming `symbolic-disposition` is done without a working hard-slice Pass@K gate MUST fail review.
