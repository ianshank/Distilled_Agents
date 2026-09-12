---
name: mangomas-scan
description: Run lint and security scans for MangoMAS. Use when asked to scan, bandit, or lint the repo.
cli: scripts/infrastructure/security_scan.py
inputs:
  output: Report path (CLI --output)
---

# Scan MangoMAS

Pre-PR operators should run the full suite (ruff, scoped mypy, pytest AQA with global `fail_under=60` plus harness coveragerc 90, bandit, gitleaks):

```bash
make validate
```

Bandit-only CLI (this skill's `cli:` contract — do not retarget it at the Makefile):

```bash
python scripts/infrastructure/security_scan.py --output security_report.json
```

Prefer these over ad-hoc bandit invocations. Do not commit secrets.
