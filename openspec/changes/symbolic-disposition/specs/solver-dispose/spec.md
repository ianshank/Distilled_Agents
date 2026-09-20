# Spec delta — solver-dispose

## ADDED Requirements

### Requirement: Dispose tools are frozen and deterministic

New constraint/solver tools MUST live in `TOOL_REGISTRY` with stable string
ids and MUST NOT use `exec`, `eval`, or shell.

#### Scenario: Registration

- **WHEN** `get_tool("constraint_check")` is called after this change
- **THEN** it returns an instance whose `run` validates args without subprocess

### Requirement: Fail-closed on constraint violation

When constraints are not satisfied, `run` MUST raise `ValueError` so the
runtime records `tool_error` rather than returning a success observation.

#### Scenario: Violation

- **WHEN** payload fails the tool’s predicates
- **THEN** `run` raises `ValueError` and no ok observation is returned

### Requirement: Harness allowlist is explicit

A dedicated harness YAML MUST list the dispose tool ids under
`action.tool_ids` and MUST NOT rely on `planning.style` to enable them.

#### Scenario: Style ignored

- **WHEN** `planning.style` is set to any schema-allowed value
- **THEN** available tools remain exactly `action.tool_ids`

### Requirement: OOD degrades to refusal, not confabulation

Harness instruction MUST tell the policy to call `final_answer` with an
explicit refusal when the dispose tool repeatedly errors or constraints cannot
be met.

#### Scenario: Instruction present

- **WHEN** the new harness YAML is loaded
- **THEN** `planning.instruction` mentions refusal on constraint failure

### Requirement: Traceability

Every hard golden for the dispose harness MUST reference a matrix row with
`solver_status` in {`fixture`,`active`,`pending`} and MUST NOT use `none`.

## Falsifier

A test that expects `run` to return ok on violated constraints MUST fail.
