---
name: mangomas-scan
description: Run local security and secret scans for MangoMAS. Use before opening a PR or when asked to audit dependencies and leaked secrets.
cli: scripts/infrastructure/security_scan.py
inputs:
  output: Report path (CLI --output, default security_report.json)
---

# Scan the repository

Run the full pre-PR suite when possible:

```bash
make validate
```

Dependency / Bandit report only:

```bash
python scripts/infrastructure/security_scan.py --output security_report.json
```

Secret scan (same engine as CI):

```bash
gitleaks detect --source . --config .gitleaks.toml --redact
```

Do not treat a green Bandit run as a security certification. pip-audit is non-blocking in CI. Never commit `.env` or live credentials; use `.env.example` as the template.
