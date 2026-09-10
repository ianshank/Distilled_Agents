# C4 — Container (L2)

Installable library `mangomas` (`enhanced_system`) plus thin `scripts/` CLIs, GitHub Actions CI, and optional Compose (Redis/Prometheus).

```mermaid
C4Container
    title MangoMAS — containers
    Person(operator, "Operator")
    Container(lib, "mangomas library", "Python", "enhanced_system: routing, cache, errors, ops")
    Container(cli, "Thin CLIs", "Python scripts/", "train, evaluate, launch, scan")
    Container(ci, "CI", "GitHub Actions", "make validate composite action")
    Container(compose, "Compose sidecar", "Docker", "Redis + Prometheus for local ops")
    System_Ext(aws, "AWS", "SageMaker, S3, IAM roles")
    Rel(operator, cli, "python scripts/... / make validate")
    Rel(cli, lib, "imports enhanced_system.ops")
    Rel(ci, lib, "pytest, ruff, mypy, bandit, gitleaks")
    Rel(lib, aws, "boto3 / SageMaker SDK")
    Rel(compose, lib, "L2 cache and metrics")
```

SageMaker training images copy `scripts/training` (entrypoint + `distill/` package). They may not have the full library on `PYTHONPATH`; distill/inference keep a try/except around `get_settings()`.
