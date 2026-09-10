#!/usr/bin/env python3
"""Package a trained adapter to ONNX."""

from __future__ import annotations

import asyncio
import os
import sys

import click

from enhanced_system.ops.settings import get_settings
from enhanced_system.ops.training_system import AutomatedTrainingSystem, InfrastructureConfig


@click.command()
@click.option("--role", required=True)
@click.option("--adapter_path", required=True)
@click.option("--output_path", required=True)
@click.option("--region", default=lambda: os.getenv("AWS_REGION", get_settings().aws_region))
@click.option("--bucket", default=lambda: get_settings().training_data_bucket)
@click.option("--table", default=lambda: get_settings().dynamodb_table)
def package_to_onnx(role, adapter_path, output_path, region, bucket, table):
    config = InfrastructureConfig(aws_region=region, s3_bucket=bucket, dynamodb_table=table)
    result = asyncio.run(
        AutomatedTrainingSystem(config).package_onnx(role, adapter_path, output_path)
    )
    if result.get("status") == "success":
        print(f"ONNX packaging completed: {result['onnx_model_path']}")
        sys.exit(0)
    print(f"ONNX packaging failed: {result.get('error', 'Unknown')}")
    sys.exit(1)


if __name__ == "__main__":
    package_to_onnx()
