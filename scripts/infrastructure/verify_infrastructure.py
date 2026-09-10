#!/usr/bin/env python3
"""Verify AWS infrastructure is reachable."""

from __future__ import annotations

import os
import sys

import click
from enhanced_system.ops.settings import get_settings
from enhanced_system.ops.training_system import AutomatedTrainingSystem, InfrastructureConfig


@click.command()
@click.option("--region", default=lambda: os.getenv("AWS_REGION", get_settings().aws_region))
@click.option("--bucket", default=lambda: get_settings().training_data_bucket)
@click.option("--table", default=lambda: get_settings().dynamodb_table)
def verify_infrastructure(region, bucket, table):
    config = InfrastructureConfig(aws_region=region, s3_bucket=bucket, dynamodb_table=table)
    result = AutomatedTrainingSystem(config).verify_resources()
    if result.get("status") == "verified":
        print("AWS infrastructure verified")
        sys.exit(0)
    print(f"AWS infrastructure not available: {result.get('error', result)}")
    sys.exit(1)


if __name__ == "__main__":
    verify_infrastructure()
