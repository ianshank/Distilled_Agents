# Spec delta: harness-eval

## ADDED Requirements

### Requirement: Golden rows carry eval metadata and hard/OOD golden set exists

The system SHALL provide `configs/golden_sets/sqe_hard_ood.jsonl` containing at
least 24 rows across hard and OOD buckets. Each golden JSONL object MUST include
`id`, `slice` (`easy` | `hard` | `ood`), `prompt`, `expected`, and `grader`.
Rows used for hard-slice and OOD gating MUST set `slice` to `hard` or `ood`,
and `grader` to `exact` (`allow_semantic: false`).

The golden set MUST meet or exceed the following normative per-bucket minima:
- Hard DAG satisfy: >= 6 rows (lexicographically least topological order, space-separated node ids)
- Hard condition tree satisfy: >= 6 rows (assignment JSON with sorted keys)
- Hard mixed graph + predicates: >= 4 rows
- OOD cycle: >= 2 rows (expected refusal `BLOCKED:CYCLE_DETECTED`)
- OOD unsat predicates: >= 2 rows (expected refusal `BLOCKED:UNSAT`)
- OOD schema/syntax: >= 2 rows (expected refusal `BLOCKED:SCHEMA_VIOLATION` or `BLOCKED:SYNTAX_INVALID`)
- OOD unknown theory: >= 2 rows (expected refusal `BLOCKED:UNSUPPORTED_THEORY`)
Total rows in `configs/golden_sets/sqe_hard_ood.jsonl` MUST be >= 24. Standard
refusal tokens and reject codes MUST conform to `openspec/changes/_shared/blocked-reject-codes.md`.

#### Scenario: Hard row schema and minimum composition

- **WHEN** `configs/golden_sets/sqe_hard_ood.jsonl` is loaded for AQA
- **THEN** every row has stable `id`, `prompt`, `expected`, `slice` in {`hard`, `ood`}, and `grader` equal to `exact`
- **AND** row counts across every individual bucket meet or exceed the specified minima (DAG >= 6, tree >= 6, mixed >= 4, cycle >= 2, unsat >= 2, schema >= 2, theory >= 2; total >= 24)

### Requirement: Deterministic OOD row detection

A golden row is defined as OOD if and only if `slice` equals `"ood"` OR `ood` is `true`.
The evaluation and gate logic SHALL treat both fields as equivalent detectors for OOD classification.
When either field is set to indicate OOD, datasets and fixtures SHALL maintain consistency across both fields.
Implementer tests MUST treat both fields equivalently for OOD synthetic-success checks.

#### Scenario: Consistent OOD identification

- **GIVEN** a row with `slice: "ood"` and `ood: false`, or `slice: "hard"` and `ood: true`
- **WHEN** the OOD evaluation filter or synthetic-success check executes
- **THEN** the row is classified as OOD under the disjunctive rule
- **AND** test suites verify that either field triggers OOD handling

### Requirement: Chen unbiased pass@k gate with n samples

The system SHALL expose Pass@K evaluation via dedicated CLI `scripts/harness/run_pass_at_k.py`
calculating the Chen et al. unbiased estimator using $n$ independent samples per problem ($n \ge k$):
$$\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$
Default CI configuration MUST use $n=5$ and gate on $k=3$. The evaluation MUST
report both `pass_at_1` and `pass_at_k` (for $k=3$).

#### Scenario: Gate reports Chen pass@k and exact-only metrics

- **GIVEN** hard and OOD rows with $n=5$ samples per row
- **WHEN** `scripts/harness/run_pass_at_k.py` runs with exact-only grading
- **THEN** output JSON includes `pass_at_k` for $k \in \{1, 3\}$, `exact_only: true`, and `semantic_counted: false`
- **AND** the process exits nonzero if `pass_at_k["3"]` is below threshold

#### Scenario: Vacuity refusal

- **WHEN** docs or CLI help text mention Pass@K
- **THEN** they describe multi-trial Chen unbiased semantics and reject single-pass or biased $1 - (1 - \hat{p})^k$ formulas

### Requirement: Hard-slice success is exact match only

For rows with `slice` in {`hard`, `ood`} or `grader: exact`, a trial success MUST
require exact `answers_match` and `security_failed: false`. Eval paths SHALL NOT
treat `semantic_match` as success on hard/OOD rows. Unlabeled non-truncated rows
SHALL NOT increment success on hard/OOD rows.

#### Scenario: Semantic-only hard row fails

- **WHEN** final_answer is a semantic near-miss that passes `semantic_match` but fails `answers_match` on a hard row
- **THEN** the trial counts as failure for pass@k

#### Scenario: Unlabeled non-truncated row does not count as success

- **WHEN** a hard/OOD row is evaluated and completes without truncation but lacks matching answer
- **THEN** it is counted as failure

### Requirement: OOD synthetic-success prohibition

For every row identified as OOD (`slice` equals `"ood"` OR `ood` is `true`), a sample SHALL
pass only if `answers_match(final_answer, expected)` where `expected` is a canonical refusal
token `BLOCKED:<CODE>` defined in `openspec/changes/_shared/blocked-reject-codes.md`.
Non-empty answers that do not match the expected refusal token SHALL be recorded as
synthetic-success violations and fail the gate.

#### Scenario: Invented SAT on OOD fails CI gate

- **GIVEN** an OOD row with expected `BLOCKED:CYCLE_DETECTED`
- **WHEN** a sample emits a non-empty `final_answer` that is not that refusal string
- **THEN** the gate increments `ood_synthetic_success_violations` and exits nonzero

### Requirement: Single-pass AQA gate docs are accurate

`scripts/harness/run_aqa_gate.py` documentation and help text SHALL NOT claim
Pass@K while the implementation remains single-pass pass_rate. The module docstring
SHALL be updated for accuracy only; multi-trial Pass@K logic SHALL reside exclusively
in `scripts/harness/run_pass_at_k.py`.

#### Scenario: Docstring corrected

- **WHEN** a reader inspects `scripts/harness/run_aqa_gate.py` docstring
- **THEN** it describes single-pass pass_rate and references `scripts/harness/run_pass_at_k.py` for multi-sample Pass@K

### Requirement: Scripted AQA runs in CI

CI MUST execute the scripted AQA gate (`aqa-gate-passk`) on PRs/pushes using
mock responses under `tests/fixtures/mock_responses_sqe_passk.json` via
`scripts/harness/run_pass_at_k.py`.

#### Scenario: Mock backend

- **WHEN** CI runs AQA via `make aqa-gate-passk`
- **THEN** no network model calls occur and the job is deterministic

### Requirement: Rule traceability covers hard goldens

Every hard golden `id` MUST appear in `configs/rule_traceability/matrix.yaml`.

#### Scenario: Missing matrix row

- **WHEN** the matrix check runs against a hard golden missing from the matrix
- **THEN** the check exits non-zero

### Requirement: prompt_render copies stay synchronized

The harness and distill `prompt_render` modules MUST stay behaviorally identical.

#### Scenario: Drift

- **WHEN** a test compares the two modules after docstring normalization
- **THEN** a unilateral body edit fails CI

## Falsifiers

- Relabeling pass_rate as pass@k without multi-trial Chen estimator code MUST fail review.
- Any golden set with <24 total rows or failing any per-bucket minimum MUST fail review.
- Allowing `semantic_match` or unlabeled non-truncated rows to count toward hard/OOD pass@k MUST fail tests.
- Treating a row as non-OOD when either `slice: "ood"` or `ood: true` is set MUST fail tests.
- Emitting non-empty non-refusal `final_answer` on an OOD row without tripping `ood_synthetic_success_violations` MUST fail CI.
- Removing the CI AQA step MUST be caught by PLAN checklist before archive.
