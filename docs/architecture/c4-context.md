# C4 — Context (L1)

Operators interact with MangoMAS to train, evaluate, and serve distilled agents. The system talks to AWS SageMaker for training, S3 for artifacts and L3 cache, Redis for L2 cache, and Prometheus for metrics.

```mermaid
C4Context
    title MangoMAS distilled agents — system context
    Person(operator, "Operator", "Trains, evaluates, and launches agents")
    Person(developer, "Developer", "Changes library, CLIs, and CI")
    System(mangomas, "MangoMAS", "Distilled LoRA agents + enhanced inference")
    System_Ext(sagemaker, "Amazon SageMaker", "Training jobs and optional endpoints")
    System_Ext(s3, "Amazon S3", "Datasets, model artifacts, L3 cache")
    System_Ext(redis, "Redis", "L2 inference cache")
    System_Ext(prom, "Prometheus", "Metrics scrape")
    Rel(operator, mangomas, "CLI / skills / SageMaker jobs")
    Rel(developer, mangomas, "make validate, pull requests")
    Rel(mangomas, sagemaker, "Create and monitor training jobs")
    Rel(mangomas, s3, "Upload data, store adapters")
    Rel(mangomas, redis, "Get/set JSON cache entries")
    Rel(prom, mangomas, "Scrape metrics")
```

If the C4 renderer is unavailable, the same relationships are: Operator → MangoMAS → SageMaker/S3/Redis; Prometheus scrapes MangoMAS.
