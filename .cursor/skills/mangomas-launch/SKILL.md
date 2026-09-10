---
name: mangomas-launch
description: Launch SageMaker training jobs for MangoMAS agents using the shared ops launcher. Use when asked to start, dry-run, or monitor SageMaker jobs.
cli: scripts/deployment/simple_launch_sagemaker.py
inputs:
  region: AWS region from MANGOMAS_AWS_REGION
  instance_type: MANGOMAS_GPU_INSTANCE_TYPE or MANGOMAS_CPU_INSTANCE_TYPE with --cpu
  role_arn: SAGEMAKER_ROLE_ARN (never commit the ARN with a real account id)
  max_concurrent: MANGOMAS_MAX_CONCURRENT_JOBS
---

# Launch SageMaker training

Prefer the thin CLIs over duplicating launcher code.

GPU (default instance from settings):

```bash
python scripts/deployment/simple_launch_sagemaker.py
```

CPU / quota-friendly:

```bash
python scripts/deployment/simple_launch_sagemaker.py --cpu
```

Dry-run job specs without calling SageMaker:

```bash
python scripts/deployment/run_sagemaker_training.py --dry-run --region "${MANGOMAS_AWS_REGION}"
```

The implementation lives in `enhanced_system.ops.MangoMASSageMakerLauncher`. Do not import `MangoMASSageMakerLauncher` from `launch_all_agents_sagemaker` (that module is a CLI wrapper only).
