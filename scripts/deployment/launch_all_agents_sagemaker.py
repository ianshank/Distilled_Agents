#!/usr/bin/env python3
"""Launch all agent SageMaker jobs (thin CLI over shared launcher)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from enhanced_system.ops.sagemaker_launcher import MangoMASSageMakerLauncher
from enhanced_system.ops.settings import get_settings


async def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Launch all MangoMAS agent training jobs")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", settings.aws_region))
    parser.add_argument("--role-arn", default=os.getenv("SAGEMAKER_ROLE_ARN"))
    args = parser.parse_args()
    launcher = MangoMASSageMakerLauncher(region=args.region, role_arn=args.role_arn)
    results = await launcher.launch_all_jobs()
    print(launcher.generate_summary_report())
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
