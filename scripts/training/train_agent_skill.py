#!/usr/bin/env python3
"""Train an agent skill (thin CLI)."""

from __future__ import annotations

import asyncio
import os
import sys

import click
from enhanced_system.ops.settings import get_settings
from enhanced_system.ops.training_system import (
    AutomatedTrainingSystem,
    InfrastructureConfig,
    SkillTrainingConfig,
)


@click.command()
@click.option("--role", required=True)
@click.option("--dataset", required=True)
@click.option("--model", required=True)
@click.option("--adapter_type", default="ALoRA")
@click.option("--instance_type", default=lambda: get_settings().gpu_instance_type)
@click.option("--epochs", default=lambda: get_settings().skill_epochs, type=int)
@click.option("--batch_size", default=lambda: get_settings().skill_batch_size, type=int)
@click.option("--learning_rate", default=lambda: get_settings().skill_learning_rate, type=float)
@click.option("--output_dir", required=True)
@click.option(
    "--mock",
    is_flag=True,
    default=False,
    help="Return synthetic artifacts without launching SageMaker.",
)
@click.option("--region", default=lambda: os.getenv("AWS_REGION", get_settings().aws_region))
@click.option("--bucket", default=lambda: get_settings().training_data_bucket)
@click.option("--table", default=lambda: get_settings().dynamodb_table)
def train_agent(
    role,
    dataset,
    model,
    adapter_type,
    instance_type,
    epochs,
    batch_size,
    learning_rate,
    output_dir,
    mock,
    region,
    bucket,
    table,
):
    config = InfrastructureConfig(aws_region=region, s3_bucket=bucket, dynamodb_table=table)
    system = AutomatedTrainingSystem(config)
    training = SkillTrainingConfig(
        role=role,
        dataset_path=dataset,
        base_model=model,
        adapter_type=adapter_type,
        output_path=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        instance_type=instance_type,
        use_mock_training=mock,
    )
    result = asyncio.run(system.train_agent_skill(training))
    if result.get("status") == "success":
        print(result["model_artifacts"])
        sys.exit(0)
    print(result.get("error", "Training failed"))
    sys.exit(1)


if __name__ == "__main__":
    train_agent()
