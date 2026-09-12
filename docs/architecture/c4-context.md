# C4 — Context (L1)

Operators distill role LoRAs, serve them as single-shot SageMaker `predict_fn`, and run tool loops only on the local harness H. Distill, Serve, and Harness are three surfaces of one system.

```mermaid
C4Context
    title MangoMAS — Distill vs Serve vs Harness H
    Person(operator, "Operator", "Trains adapters, collects traces, runs local agents")
    Person(developer, "Developer", "Changes library, CLIs, skills, and CI")
    System(mangomas, "MangoMAS", "Distill + Serve predict_fn + local harness H")
    System_Ext(sagemaker, "Amazon SageMaker", "Training jobs and optional endpoints")
    System_Ext(s3, "Amazon S3", "Datasets, adapters, L3 cache")
    Rel(operator, mangomas, "CLIs / Cursor skills / make validate")
    Rel(developer, mangomas, "PRs; CI keeps harness 90% job")
    Rel(mangomas, sagemaker, "Create training jobs; optional single-shot invoke")
    Rel(mangomas, s3, "Upload data, store adapters")
```

Serve does **not** call tools. The local harness (`enhanced_system.harness` + `scripts/harness/run_agent.py`) is the only v1 tool loop.
