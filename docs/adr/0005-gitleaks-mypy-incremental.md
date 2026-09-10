# ADR 0005: Gitleaks and incremental mypy

## Status

Accepted

## Context

The nested `enhanced_system/.pre-commit-config.yaml` still pointed at Black/flake8 after the root switched to ruff. Types across the legacy tree are uneven. Training JSONL contains fictional passwords that trip generic secret rules.

## Decision

- Root `.pre-commit-config.yaml` keeps ruff and adds the official Gitleaks hook (`gitleaks/gitleaks` v8) plus mypy limited to `enhanced_system/ops` and `enhanced_system/core/cache`.
- Nested enhanced_system pre-commit file is a stub pointing at the root.
- `.gitleaks.toml` extends the default rules and allowlists `.env.example` plus `data/training/*.jsonl`.
- CI `make validate` runs Gitleaks detect and mypy on the same incremental scope. `ignore_missing_imports` stays on. Strict mypy for all of `enhanced_system` is later.
- pip-audit remains non-blocking.

## Consequences

Secret scanning and a typed ops/cache surface land without blocking the rest of the legacy tree. Training datasets must stay fictional; real credentials still fail the scan.
