# ADR 0005: Gitleaks and incremental mypy

## Status

Accepted

## Context

The nested `enhanced_system/.pre-commit-config.yaml` still pointed at Black/flake8 after the root switched to ruff. Types across the legacy tree are uneven. Training JSONL contains fictional passwords that trip generic secret rules. The runtime harness package is new and should be typed with ops/cache rather than waiting for a whole-tree mypy ratchet.

## Decision

- Root `.pre-commit-config.yaml` keeps ruff and adds the official Gitleaks hook (`gitleaks/gitleaks` v8) plus mypy on `enhanced_system/ops`, `enhanced_system/core/cache`, and `enhanced_system/harness`.
- Nested enhanced_system pre-commit file is a stub pointing at the root.
- `.gitleaks.toml` extends the default rules and allowlists `.env.example` plus the **checked-in** `data/training/*.jsonl` fixtures by filename. New training JSONL is scanned.
- Tool mypy `python_version` is **3.11** (CI). Package `requires-python` stays `>=3.9`. Incremental mypy uses `follow_imports = silent` (not `skip`) so harness/ops/cache errors still fail the gate.
- `make validate` runs gitleaks and scoped mypy. GitHub Actions **keeps** the dedicated `pytest -m harness --cov-config=.coveragerc.harness` job (fail_under 90). The composite `mangomas-validate` action is for operators/skills and must not replace that job.
- pip-audit remains non-blocking. Global coverage `fail_under` stays **60**.

## Consequences

Secret scanning and a typed ops/cache/harness surface land without blocking the rest of the legacy tree. Training datasets must stay fictional; real credentials still fail the scan.
