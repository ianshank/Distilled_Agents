# Spec delta — harness-eval

## ADDED Requirements

### Requirement: Golden rows carry eval metadata

Each golden JSONL object MUST include `prompt` and `expected`. Rows used for
hard-slice gating MUST include `slice: hard` and a stable `id`.

#### Scenario: Hard row schema

- **WHEN** a hard-slice golden file is loaded for AQA
- **THEN** every row has `id`, `expected`, and `slice` equal to `hard`

### Requirement: pass@k is multi-trial and named honestly

`eval_harness` MUST report `pass_at_1` and `pass_at_k` for configured k≥1.
A single-trial pass_rate MUST NOT be labeled pass@k in docs or CLI.

#### Scenario: k=1 equivalence

- **WHEN** `MANGOMAS_EVAL_PASS_K=1`
- **THEN** `pass_at_1` equals `pass_at_k` and both use the hard/core success rule

#### Scenario: Vacuity refusal

- **WHEN** docs or CLI help text mention Pass@K
- **THEN** they describe multi-trial semantics matching this spec

### Requirement: Hard-slice success is exact

For `slice: hard`, a trial success MUST require exact `answers_match` and MUST
NOT treat `semantic_match` alone as success.

#### Scenario: Semantic-only hard row

- **WHEN** final_answer is a semantic near-miss on a hard row
- **THEN** the trial counts as failure for pass@k

### Requirement: Scripted AQA runs in CI

CI MUST execute the scripted AQA gate on every PR/push to default branch
paths that affect harness or golden sets (or unconditionally).

#### Scenario: Mock backend

- **WHEN** CI runs AQA with `--scripted tests/fixtures/mock_responses.json`
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

- Relabeling pass_rate as pass@k without multi-trial code MUST fail review.
- Removing the CI AQA step MUST be caught by PLAN checklist before archive.
