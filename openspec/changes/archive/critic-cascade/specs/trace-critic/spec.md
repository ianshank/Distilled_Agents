# Spec delta: trace-critic

## ADDED Requirements

### Requirement: Critic is mandatory when enabled

When `MANGOMAS_CRITIC_ENABLED` is true, collect MUST NOT write a training row
that fails allowlist validity or outcome grade (when `expected` present).

#### Scenario: Bad tool id

- **WHEN** a trajectory step references a tool not in the harness allowlist
- **THEN** the row is rejected with a logged reason code

#### Scenario: Outcome mismatch

- **WHEN** `expected` is present and `final_answer` fails `answers_match`
- **THEN** the row is skipped and logged with `critic_reject_code: "OUTCOME_MISMATCH"`

### Requirement: Reject telemetry and sink logging

The critic and filtering stages SHALL record an explicit `critic_reject_code` field
(not only aggregated counters) and append JSONL records to `artifacts/critic_rejects.jsonl`
(or configured `--reject-log` sink).

#### Scenario: Reject log sink append

- **WHEN** collect or compose drops a row or pair
- **THEN** an entry is appended to the reject log containing `critic_reject_code`, prompt, and metadata

### Requirement: DualDistill drops (0,0) with explicit telemetry

Compose MUST continue to drop pairs where both teachers fail the grade by returning
None on (0,0) without inventing a second drop rule, and MUST log `critic_reject_code: "DUALDISTILL_DROP_0_0"`.

#### Scenario: DualDistill (0,0) logged

- **WHEN** `compose_pair` drops a pair because both teachers scored 0
- **THEN** compose returns None and logs `critic_reject_code: "DUALDISTILL_DROP_0_0"`

### Requirement: Recovery traces retained on success

Intermediate `parse_error` / `tool_error` MUST NOT alone cause rejection if
final outcome matches `expected`.

#### Scenario: Recovered parse

- **WHEN** a trajectory has an early parse_error and a matching final_answer
- **THEN** the row is kept and `critic_kept_recovery` increments

### Requirement: Teacher-rule critic codes defined

Later teacher-rule critic stages SHALL draw reject codes from the shared standard
specification `openspec/changes/_shared/blocked-reject-codes.md`: `CYCLE_DETECTED`,
`UNSAT`, `SCHEMA_VIOLATION`, `SYNTAX_INVALID`, `UNSUPPORTED_THEORY`, `RESOURCE_LIMIT`.

#### Scenario: Standardized reason code emitted

- **WHEN** a teacher-rule check fails
- **THEN** the logged `critic_reject_code` is one of the standardized enum tokens from `openspec/changes/_shared/blocked-reject-codes.md`

## Sequencing note

This change is landable after `golden-passk-aqa` metrics exist and does not wait
on `symbolic-disposition`.

## Falsifier

A unit test that injects an unknown tool_id MUST fail if collect writes the row
with critic enabled.
A unit test dropping a (0,0) pair without emitting `DUALDISTILL_DROP_0_0` MUST fail.
