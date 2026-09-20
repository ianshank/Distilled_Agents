# Spec delta: sqe-dispose-e2e-bind

Cites: `docs/plans/symbolic-kd/architecture.md` Section 11 and Section 11.7 (Architect PR https://github.com/ianshank/Distilled_Agents/pull/30).

## ADDED Requirements

### Requirement: Golden row binding to sqe_dispose harness with ordered tool sequence

Every problem entry in `configs/golden_sets/sqe_hard_ood.jsonl` MUST explicitly declare `harness_id: "sqe_dispose"` and `expected_tools` containing `sqe_constraint_solver` followed by `final_answer`. The identifier spelling `sqe_constraint_solver` and `sqe_dispose` SHALL remain strictly byte-stable.

#### Scenario: Golden row schema compliance

- **GIVEN** the golden dataset file `configs/golden_sets/sqe_hard_ood.jsonl`
- **WHEN** any row is parsed and validated for evaluation
- **THEN** `row.harness_id` equals `"sqe_dispose"`
- **AND** `row.expected_tools` is a list where `sqe_constraint_solver` precedes `final_answer`
- **AND** `row.grader` equals `"exact"` with `row.allow_semantic` equal to `false`

### Requirement: Scripted fixtures execute solver before terminal answer

Scripted fixtures in `tests/fixtures/mock_responses_sqe_passk.json` (or successor fixtures keyed by golden row ID) MUST simulate multi-turn interaction by issuing a tool call to `sqe_constraint_solver` prior to returning `final_answer`. For OOD tasks, `final_answer` MUST equal canonical refusal token `BLOCKED:<CODE>` defined in byte-stable contract `openspec/changes/_shared/blocked-reject-codes.md`.

#### Scenario: In-distribution fixture execution

- **GIVEN** a hard in-distribution row in `sqe_hard_ood.jsonl`
- **WHEN** the runtime processes the prompt against scripted fixtures
- **THEN** step 1 emits a tool call to `sqe_constraint_solver` with valid graph or constraint arguments
- **AND** step 2 receives solver observation and emits `final_answer` matching expected output exactly

#### Scenario: Out-of-distribution refusal fixture execution

- **GIVEN** an OOD row (where `slice == "ood"` OR `ood == true`) in `sqe_hard_ood.jsonl`
- **WHEN** the runtime processes the prompt against scripted fixtures
- **THEN** step 1 calls `sqe_constraint_solver` triggering UNSAT or error reject code
- **AND** step 2 emits `final_answer` equal to `BLOCKED:<CODE>` from `openspec/changes/_shared/blocked-reject-codes.md`
- **AND** `ood_synthetic_success_violations` remains 0

### Requirement: Vacuity rejection on solver bypass

The evaluation gate `make aqa-gate-passk` MUST reject any fixture or trajectory that reaches `final_answer` without prior recorded invocation of `sqe_constraint_solver`. A bypass of the solver constitutes a gate failure regardless of whether `answers_match` passes.

#### Scenario: Solver bypass falsification

- **GIVEN** a scripted fixture or student rollout that immediately returns `final_answer` matching `expected` without calling `sqe_constraint_solver`
- **WHEN** `evaluate_pass_at_k` or `make aqa-gate-passk` executes
- **THEN** the gate marks the problem as failed due to solver bypass (vacuity failure)
- **AND** the gate fails with nonzero exit code

### Requirement: E1 acceptance asserts recorded tool-call compliance

Acceptance for milestone E1 SHALL assert recorded tool-call and `expected_tools` compliance across all evaluated trials, in addition to exact-match score thresholds. Exact-match final answers alone without verified solver steps SHALL NOT satisfy E1 acceptance.

#### Scenario: Gate verifies tool sequence

- **WHEN** `scripts/harness/run_pass_at_k.py` evaluates $n$ samples per row
- **THEN** each sample trajectory is inspected for tool execution order
- **AND** only trials containing both `sqe_constraint_solver` and `final_answer` in proper sequence count toward pass rate $c$

### Requirement: Rule matrix binding and sole Chen CLI

Every hard and OOD row in `configs/golden_sets/sqe_hard_ood.jsonl` MUST be tracked in `configs/rule_traceability/matrix.yaml` with `harness_id: "sqe_dispose"` and `solver_status` in {`fixture`, `active`}. `scripts/harness/run_pass_at_k.py` remains the sole Chen Pass@K CLI and outputs summary artifact `aqa-passk-summary.json`.

#### Scenario: Matrix coverage verification

- **WHEN** `scripts/harness/check_rule_matrix.py` validates `configs/rule_traceability/matrix.yaml` against `sqe_hard_ood.jsonl`
- **THEN** every golden ID exists in the matrix
- **AND** each entry sets `harness_id: "sqe_dispose"` and `solver_status` in {`fixture`, `active`}

### Requirement: Harness configuration allowlist restriction

`configs/harnesses/sqe_dispose.yaml` and its packaged copy `enhanced_system/config/harnesses/sqe_dispose.yaml` MUST restrict `action.tool_ids` strictly to `sqe_constraint_solver` and `final_answer`.

#### Scenario: Harness allowlist enforcement

- **WHEN** `sqe_dispose.yaml` is loaded by the harness runtime
- **THEN** `spec.action.tool_ids` contains exactly `["sqe_constraint_solver", "final_answer"]`

## Falsifiers

1. Any test or gate run where `final_answer` is reached without prior `sqe_constraint_solver` call on a `sqe_hard_ood` task MUST fail the gate.
2. Any OOD row producing a non-empty answer different from canonical `BLOCKED:<CODE>` MUST fail with `ood_synthetic_success_violations > 0`.
3. Attempting to run Pass@K via `scripts/harness/run_aqa_gate.py` MUST be rejected.
4. Marking `symbolic-disposition` task 8 complete before E1 is green on dispose-bound fixtures MUST fail review.
