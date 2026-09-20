# Spec delta - ci-hygiene

## ADDED Requirements

### Requirement: Trunk hygiene gates are green

The default branch MUST pass the existing CI jobs `lint`, `types`, `security`,
and `test` without weakening their commands.

#### Scenario: Format gate

- **WHEN** `ruff format --check enhanced_system scripts tests` runs
- **THEN** it exits 0

#### Scenario: Types gate

- **WHEN** `mypy enhanced_system/ops enhanced_system/core/cache enhanced_system/harness` runs
- **THEN** it reports 0 errors

#### Scenario: Bandit gate

- **WHEN** `bandit -r enhanced_system scripts ... -c pyproject.toml` runs
- **THEN** it exits 0 (no unresolved Medium/High findings under current policy)

## Falsifier

Reverting the format commit alone MUST turn `lint` red.
