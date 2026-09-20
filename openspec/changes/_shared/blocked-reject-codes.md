# Shared Contract: BLOCKED and Reject Codes

Shared normative reference for rejection codes, failure tokens, and telemetry across:
- `openspec/changes/archive/golden-passk-aqa/`
- `openspec/changes/archive/critic-cascade/`
- `openspec/changes/archive/symbolic-disposition/`

## Standard Reject Codes

The following six canonical reject codes represent deterministic rejection reasons:

| Reject Code | Trigger Condition | Tool Run Behavior | Trajectory Fault | Canonical Refusal Token |
| --- | --- | --- | --- | --- |
| `CYCLE_DETECTED` | Dependency graph contains directed cycle | Return `ok: false`, `status: "UNSAT"`, `reject_code: "CYCLE_DETECTED"` | None (normal return) | `BLOCKED:CYCLE_DETECTED` |
| `UNSAT` | Constraints unsatisfiable or contradictory | Return `ok: false`, `status: "UNSAT"`, `reject_code: "UNSAT"` | None (normal return) | `BLOCKED:UNSAT` |
| `SCHEMA_VIOLATION` | Malformed input payload or limits exceeded | Raise `ValueError("SCHEMA_VIOLATION: ...")` | `tool_error` | `BLOCKED:SCHEMA_VIOLATION` |
| `SYNTAX_INVALID` | Unparseable constraint tree or atom syntax | Raise `ValueError("SYNTAX_INVALID: ...")` | `tool_error` | `BLOCKED:SYNTAX_INVALID` |
| `UNSUPPORTED_THEORY` | Operation outside supported fragment (e.g. non-DAG/boolean) | Raise `ValueError("UNSUPPORTED_THEORY: ...")` | `tool_error` | `BLOCKED:UNSUPPORTED_THEORY` |
| `RESOURCE_LIMIT` | Resource caps exceeded during execution | Raise `ValueError("RESOURCE_LIMIT: ...")` | `tool_error` | `BLOCKED:RESOURCE_LIMIT` |

## Upstream Pipeline Codes

In addition to the six deterministic solver reject codes, the trace critic and data pipelines emit explicit codes for upstream drop events:
- `OUTCOME_MISMATCH`: Emitted by `collect_trajectories.py` outcome filter when `final_answer` fails `answers_match`.
- `DUALDISTILL_DROP_0_0`: Emitted by `compose_dualdistill.py` / `compose_pair` when both teachers score 0 on the task.

## Rules Across Changes

1. **Golden Expected Tokens (`golden-passk-aqa`):** OOD rows expecting solver refusal must set `expected` to canonical `BLOCKED:<CODE>` using one of the six standard codes above. Any sample on an OOD row producing a non-empty `final_answer` that is not the canonical refusal token is a synthetic-success violation.
2. **Critic Telemetry (`critic-cascade`):** When dropping a trace or pair, append JSONL records to `artifacts/critic_rejects.jsonl` with field `critic_reject_code` populated from this standard set (`OUTCOME_MISMATCH`, `DUALDISTILL_DROP_0_0`, or teacher-rule codes `CYCLE_DETECTED`, `UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`).
3. **Solver Rejection (`symbolic-disposition`):** `sqe_constraint_solver` must only return or raise reject codes from this table. Cyclic and unsat inputs must return `ok: false` with `reject_code` set; malformed schema, syntax, unsupported theories, or resource limits must raise `ValueError` with the code prefix.
