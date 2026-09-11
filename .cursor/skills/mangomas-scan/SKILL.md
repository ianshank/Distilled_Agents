---
name: mangomas-scan
description: Run lint and security scans for MangoMAS. Use when asked to scan, bandit, or lint the repo.
cli: scripts/infrastructure/security_scan.py
inputs:
  output: Report path (CLI --output)
---

# Scan MangoMAS

```bash
python scripts/infrastructure/security_scan.py --output security_report.json
```

Prefer this CLI over ad-hoc bandit invocations. Do not commit secrets.
