# Spec delta: collect-compose-sqe-e2e

Cites: `docs/plans/symbolic-kd/architecture.md` Section 11.4 E2 and Section 12 (E2 note, PR https://github.com/ianshank/Distilled_Agents/pull/35); ADR 0007; `openspec/changes/archive/critic-cascade/`; `openspec/changes/sqe-dispose-e2e-bind/`.

## ADDED Requirements

### Requirement: Makefile targets for dispose-bound collect and compose

The repository Makefile MUST provide target `collect-sqe-dispose` for trajectory collection under the `sqe_dispose` harness and target `compose-dualdistill-sqe` for DualDistill trajectory composition. Target names `collect-sqe` and `compose-dual` SHALL NOT be used.

#### Scenario: Running collect-sqe-dispose target

- **WHEN** an operator runs `make collect-sqe-dispose`
- **THEN** `scripts/harness/collect_trajectories.py` executes with `--harness-id sqe_dispose`
- **AND** input prompts are derived from `configs/golden_sets/sqe_hard_ood.jsonl`
- **AND** deterministic execution uses `--scripted tests/fixtures/mock_responses_sqe_passk.json`
- **AND** critic filtering is active with reject logging directed to `artifacts/critic_rejects.jsonl`
- **AND** trajectories are output under `artifacts/trajectories/`

#### Scenario: Running compose-dualdistill-sqe target

- **WHEN** an operator runs `make compose-dualdistill-sqe`
- **THEN** `scripts/harness/compose_dualdistill.py` executes with two teacher JSONLs from dispose collect
- **AND** reject logging is directed to `artifacts/critic_rejects.jsonl`
- **AND** composed DualDistill pairs are output under `artifacts/trajectories/`

### Requirement: Scripted teacher traces enforce solver execution before final answer

Trajectories collected under `sqe_dispose` MUST record a tool call to `sqe_constraint_solver` before `final_answer`. Any teacher rollout that emits `final_answer` directly without prior disposition tool execution is non-compliant.

#### Scenario: Scripted teacher trajectory compliance

- **GIVEN** prompts from `configs/golden_sets/sqe_hard_ood.jsonl`
- **WHEN** `collect_trajectories.py` runs with `--scripted tests/fixtures/mock_responses_sqe_passk.json`
- **THEN** each generated trajectory step sequence includes `sqe_constraint_solver` before `final_answer`
- **AND** for OOD prompts, `final_answer` matches canonical refusal token `BLOCKED:<CODE>` per `openspec/changes/_shared/blocked-reject-codes.md`

### Requirement: Reject telemetry sink with exact reject codes

Collect and compose pipelines MUST record dropped trajectories and pairs to `artifacts/critic_rejects.jsonl` using standardized `critic_reject_code` identifiers.

#### Scenario: Collect drops on outcome mismatch

- **GIVEN** a prompt with an expected answer
- **WHEN** the trajectory's `final_answer` fails `answers_match` against `expected`
- **THEN** collect filters out the trajectory
- **AND** a record is appended to `artifacts/critic_rejects.jsonl` with `critic_reject_code: "OUTCOME_MISMATCH"`

#### Scenario: DualDistill drops exclusively on dual failure (0,0)

- **GIVEN** a task evaluated against two teacher trajectories
- **WHEN** both teachers score 0 on the graded task
- **THEN** `compose_pair` returns None
- **AND** a record is appended to `artifacts/critic_rejects.jsonl` with `critic_reject_code: "DUALDISTILL_DROP_0_0"`

#### Scenario: Prohibition of second DualDistill drop rule

- **GIVEN** a task evaluated against two teacher trajectories
- **WHEN** at least one teacher scores non-zero on the graded task
- **THEN** compose SHALL NOT drop the pair on grounds of partial credit, schema fault, or asymmetric teacher failure
- **AND** the pair is composed into a valid DualDistill transition sequence

### Requirement: Recovery traces retained on outcome match

Intermediate faults (`parse_error` or `tool_error`) during rollout MUST NOT cause trajectory rejection if `final_answer` matches `expected`.

#### Scenario: Recovery trace retention

- **WHEN** a rollout experiences a retryable parse or tool error but produces a matching `final_answer`
- **THEN** the trajectory is retained in the output JSONL
- **AND** `critic_kept_recovery` is incremented

## Falsifiers

1. Any trajectory written by `collect-sqe-dispose` where `final_answer` was emitted without a preceding `sqe_constraint_solver` tool call MUST fail validation.
2. An intentional outcome mismatch that is dropped without recording `critic_reject_code: "OUTCOME_MISMATCH"` in `artifacts/critic_rejects.jsonl` MUST fail test acceptance.
3. A synthetic (0,0) teacher pair that is dropped without recording `critic_reject_code: "DUALDISTILL_DROP_0_0"` in `artifacts/critic_rejects.jsonl` MUST fail test acceptance.
4. Any pair dropped by DualDistill when at least one teacher achieved success MUST fail test acceptance.
5. Any introduction of GKD dependencies, new Pass@K CLIs, or routing through `run_aqa_gate.py` MUST fail architectural gate review.
