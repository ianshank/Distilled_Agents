#!/usr/bin/env python3
"""Set up AWS infrastructure for agent training."""

from __future__ import annotations

import asyncio
import os
import sys

import click

from enhanced_system.ops.settings import get_settings
from enhanced_system.ops.training_system import AutomatedTrainingSystem, InfrastructureConfig


@click.command()
@click.option("--region", default=lambda: os.getenv("AWS_REGION", get_settings().aws_region))
@click.option("--bucket", default=lambda: get_settings().training_data_bucket)
@click.option("--table", default=lambda: get_settings().dynamodb_table)
def setup_infrastructure(region, bucket, table):
    config = InfrastructureConfig(aws_region=region, s3_bucket=bucket, dynamodb_table=table)
    result = asyncio.run(AutomatedTrainingSystem(config).setup_infrastructure())
    if result.get("status") == "completed":
        print("Infrastructure setup completed")
        sys.exit(0)
    print(f"Infrastructure setup failed: {result.get('error', 'Unknown')}")
    sys.exit(1)


if __name__ == "__main__":
    setup_infrastructure()
