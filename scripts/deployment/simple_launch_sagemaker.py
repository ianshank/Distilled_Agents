#!/usr/bin/env python3
"""Simple SageMaker launcher (GPU by default, --cpu for quota-friendly instances)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from enhanced_system.ops.sagemaker_launcher import MangoMASSageMakerLauncher
from enhanced_system.ops.settings import get_settings


def launch_sagemaker_training_jobs(use_cpu: bool = False):
    settings = get_settings()
    instance_type = settings.cpu_instance_type if use_cpu else settings.gpu_instance_type
    launcher = MangoMASSageMakerLauncher(
        region=os.getenv("AWS_REGION", settings.aws_region),
        instance_type=instance_type,
    )
    if use_cpu:
        launcher.agent_configs = launcher.agent_configs[:3]
        for config in launcher.agent_configs:
            config.instance_type = settings.cpu_instance_type
            config.model_name = settings.cpu_model
            config.student_model = settings.cpu_model
            config.epochs = 1
            config.batch_size = 1
            config.use_spot_instances = False
    return asyncio.run(launcher.launch_all_jobs())


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple MangoMAS SageMaker launcher")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instance type")
    args = parser.parse_args()
    jobs = launch_sagemaker_training_jobs(use_cpu=args.cpu)
    print(f"Launched {len(jobs)} jobs")
    return 0 if jobs else 1


if __name__ == "__main__":
    sys.exit(main())
