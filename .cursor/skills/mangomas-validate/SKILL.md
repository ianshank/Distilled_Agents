---
name: mangomas-validate
description: Run the full pre-PR suite (ruff, mypy, pytest AQA, bandit, gitleaks). Use when asked to validate, make validate, or run operator AQA. Does not replace CI jobs.
cli: Makefile
inputs:
  target: Make target (default validate)
---

# Validate MangoMAS

From the repository root:

```bash
make validate
```

That is lint + scoped mypy (`ops`, `core/cache`, `harness`) + pytest AQA (global `fail_under=60` then harness `.coveragerc.harness` 90) + bandit + gitleaks.

The composite `.github/actions/mangomas-validate` wraps the same target for skills. GitHub Actions still runs lint, types, unit+integration@60, harness@90, and security as first-class jobs.

Bandit-only scans stay on `mangomas-scan` (`scripts/infrastructure/security_scan.py`).
