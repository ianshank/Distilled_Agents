#!/usr/bin/env python3
"""Register a trained agent skill."""

from __future__ import annotations

import asyncio
import os
import sys

import click

from enhanced_system.ops.settings import get_settings
from enhanced_system.ops.training_system import (
    AutomatedTrainingSystem,
    InfrastructureConfig,
    RegisteredSkill,
)


@click.command()
@click.option("--role", required=True)
@click.option("--adapter_uri", required=True)
@click.option("--pass_rate", required=True, type=float)
@click.option("--model", required=True)
@click.option("--region", default=lambda: os.getenv("AWS_REGION", get_settings().aws_region))
@click.option("--bucket", default=lambda: get_settings().training_data_bucket)
@click.option("--table", default=lambda: get_settings().dynamodb_table)
def register_agent(role, adapter_uri, pass_rate, model, region, bucket, table):
    config = InfrastructureConfig(aws_region=region, s3_bucket=bucket, dynamodb_table=table)
    skill = RegisteredSkill(
        role=role, adapter_uri=adapter_uri, pass_rate=pass_rate, model=model
    )
    result = asyncio.run(AutomatedTrainingSystem(config).register_skill(skill))
    print(result)
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    register_agent()
