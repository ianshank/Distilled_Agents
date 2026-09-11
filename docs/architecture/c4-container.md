# C4 — Container (L2)

Installable library `mangomas` (`enhanced_system`) plus thin `scripts/` CLIs. Distill, Serve, and Harness H are separate containers of behavior.

```mermaid
C4Container
    title MangoMAS — Distill / Serve / Harness H
    Person(operator, "Operator")
    Container(distill, "Distill", "scripts/training + distill/", "LoRA / trajectory SFT; SageMaker source_dir")
    Container(serve, "Serve", "scripts/inference.py", "SageMaker predict_fn single-shot; no tools")
    Container(harnessH, "Harness H", "enhanced_system.harness", "YAML specs, TOOL_REGISTRY, AgentRuntime")
    Container(cli, "Thin CLIs", "scripts/harness + ops", "run_agent, collect, tailor, train, launch, scan")
    Container(ci, "CI", "GitHub Actions", "lint / unit+integration@60 / harness@90 / types / security")
    System_Ext(aws, "AWS", "SageMaker, S3, IAM roles")
    Rel(operator, cli, "python scripts/... / make validate")
    Rel(cli, harnessH, "run / collect / tailor")
    Rel(cli, distill, "train_distilled_adapter --trajectory_mode")
    Rel(distill, aws, "optional training jobs")
    Rel(serve, aws, "endpoint invoke")
    Rel(harnessH, distill, "JSONL trajectories")
    Rel(ci, harnessH, "pytest -m harness --cov-config=.coveragerc.harness")
    Rel(serve, harnessH, "not wired in v1")
```

SageMaker training images copy `scripts/training` (entrypoint + `distill/` package). They may not have the full library on `PYTHONPATH`; `train_distilled_adapter.py` and `scripts/inference.py` read `MANGOMAS_*` from the environment directly (no `get_settings()` import). Composite `mangomas-validate` is for operators/skills and does not replace the CI jobs.
