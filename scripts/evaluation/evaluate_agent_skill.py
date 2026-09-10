#!/usr/bin/env python3
"""Evaluate a trained agent skill."""

from __future__ import annotations

import asyncio
import os
import sys

import click
from enhanced_system.ops.settings import get_settings
from enhanced_system.ops.training_system import AutomatedTrainingSystem, InfrastructureConfig


@click.command()
@click.option("--role", required=True)
@click.option("--test_suite", required=True)
@click.option("--threshold", default=85, type=float)
@click.option("--region", default=lambda: os.getenv("AWS_REGION", get_settings().aws_region))
@click.option("--bucket", default=lambda: get_settings().training_data_bucket)
@click.option("--table", default=lambda: get_settings().dynamodb_table)
def evaluate_agent(role, test_suite, threshold, region, bucket, table):
    config = InfrastructureConfig(aws_region=region, s3_bucket=bucket, dynamodb_table=table)
    result = asyncio.run(
        AutomatedTrainingSystem(config).evaluate_agent_skill(role, test_suite, threshold)
    )
    print(f"pass_rate={result.pass_rate} passed={result.passed_tests}")
    sys.exit(0 if result.pass_rate >= threshold else 1)


if __name__ == "__main__":
    evaluate_agent()
