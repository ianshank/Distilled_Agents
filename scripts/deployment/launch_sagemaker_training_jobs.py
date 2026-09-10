#!/usr/bin/env python3
"""Launch SageMaker training jobs (thin CLI over shared launcher)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from enhanced_system.ops.sagemaker_launcher import MangoMASSageMakerLauncher
from enhanced_system.ops.settings import get_settings


async def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Launch MangoMAS SageMaker training jobs")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", settings.aws_region))
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--terraform-config", default=None)
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()

    launcher = MangoMASSageMakerLauncher(
        region=args.region,
        terraform_config_path=args.terraform_config,
    )
    if args.monitor:
        launcher.monitor_training_jobs()
        return 0
    results = await launcher.launch_all_jobs(
        parallel=args.parallel, max_concurrent=args.max_concurrent
    )
    print(launcher.generate_summary_report())
    launcher.save_results()
    return 0 if results else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
