#!/usr/bin/env python3
"""CPU SageMaker launcher (compatibility wrapper)."""

from __future__ import annotations

import sys

try:
    from scripts.deployment.simple_launch_sagemaker import launch_sagemaker_training_jobs
except ImportError:
    from simple_launch_sagemaker import launch_sagemaker_training_jobs

if __name__ == "__main__":
    jobs = launch_sagemaker_training_jobs(use_cpu=True)
    print(f"Launched {len(jobs)} CPU jobs")
    sys.exit(0 if jobs else 1)
