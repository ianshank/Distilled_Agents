"""Environment-driven MangoMAS operational settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MangoMASSettings(BaseSettings):
    """AWS, model, and runtime defaults loaded from env."""

    model_config = SettingsConfigDict(
        env_prefix="MANGOMAS_",
        env_file=".env",
        extra="ignore",
    )

    aws_region: str = Field(default="us-east-1")
    execution_role_name: str = Field(default="mangomas-sagemaker-sagemaker-execution-role")
    s3_bucket_prefix: str = Field(default="mangomas-sagemaker-production")
    training_data_bucket: str = Field(default="mangomas-training-data")
    dynamodb_table: str = Field(default="agent-skill-registry")
    gpu_instance_type: str = Field(default="ml.g4dn.xlarge")
    cpu_instance_type: str = Field(default="ml.m5.large")
    teacher_model: str = Field(default="mistralai/Mistral-7B-v0.1")
    student_model: str = Field(default="microsoft/DialoGPT-medium")
    cpu_model: str = Field(default="distilgpt2")
    max_run_seconds: int = Field(default=7200)
    trust_remote_code: bool = Field(default=False)
    bind_host: str = Field(default="127.0.0.1")
    grafana_admin_password: str = Field(default="")
    model_revision: str = Field(default="")
    skill_epochs: int = Field(default=5)
    skill_batch_size: int = Field(default=16)
    skill_learning_rate: float = Field(default=1e-4)
    evaluation_threshold: float = Field(default=85.0)
    max_concurrent_jobs: int = Field(default=3)
    bind_port: int = Field(default=8080)

    def role_arn(self, account_id: str) -> str:
        return f"arn:aws:iam::{account_id}:role/{self.execution_role_name}"

    def production_bucket(self, account_id: str) -> str:
        return f"{self.s3_bucket_prefix}-{account_id}"


@lru_cache(maxsize=1)
def get_settings() -> MangoMASSettings:
    return MangoMASSettings()
