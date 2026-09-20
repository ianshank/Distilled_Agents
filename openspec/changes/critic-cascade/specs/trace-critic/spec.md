# Spec delta — trace-critic

## ADDED Requirements

### Requirement: Critic is mandatory when enabled

When `MANGOMAS_CRITIC_ENABLED` is true, collect MUST NOT write a training row
that fails allowlist validity or outcome grade (when `expected` present).

#### Scenario: Bad tool id

- **WHEN** a trajectory step references a tool not in the harness allowlist
- **THEN** the row is rejected with a logged reason code

#### Scenario: Outcome mismatch

- **WHEN** `expected` is present and `final_answer` fails `answers_match`
- **THEN** the row is skipped (existing behavior retained as a named critic stage)

### Requirement: Recovery traces retained on success

Intermediate `parse_error` / `tool_error` MUST NOT alone cause rejection if
final outcome matches `expected`.

#### Scenario: Recovered parse

- **WHEN** a trajectory has an early parse_error and a matching final_answer
- **THEN** the row is kept and `critic_kept_recovery` increments

### Requirement: DualDistill still drops (0,0)

Compose MUST continue to drop pairs where both teachers fail the grade.

## Falsifier

A unit test that injects an unknown tool_id MUST fail if collect writes the row
with critic enabled.
