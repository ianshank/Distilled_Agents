---
name: mangomas-launch
description: Launch SageMaker training jobs with the thin CLI wrapper. Use when asked to launch, deploy, or start training jobs.
cli: scripts/deployment/simple_launch_sagemaker.py
inputs:
  cpu: Use CPU instance from MANGOMAS_CPU_INSTANCE_TYPE (CLI --cpu)
  region: AWS region from settings (`MANGOMAS_AWS_REGION`); not a CLI flag
---

# Launch SageMaker jobs

```bash
python scripts/deployment/simple_launch_sagemaker.py --cpu
```

Import `MangoMASSageMakerLauncher` from `enhanced_system.ops`. Instance types and region come from `get_settings()`.
