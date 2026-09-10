"""Lightweight training-system API used by ops CLIs."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from enhanced_system.ops.settings import MangoMASSettings, get_settings

logger = logging.getLogger(__name__)


def _case_passed(case: Any) -> bool:
    """Score one recorded suite case (pre-labeled or expected/actual)."""
    if not isinstance(case, dict):
        return False
    if "passed" in case:
        return case.get("passed") is True
    if "expected" in case:
        observed = case.get("actual", case.get("output"))
        return observed == case.get("expected")
    return False


@dataclass
class InfrastructureConfig:
    aws_region: str
    s3_bucket: str
    dynamodb_table: str


@dataclass
class SkillTrainingConfig:
    role: str
    dataset_path: str
    base_model: str
    adapter_type: str = "ALoRA"
    output_path: str = ""
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 5
    evaluation_threshold: float = 0.85
    use_mock_training: bool = False
    instance_type: str = "ml.g4dn.xlarge"


def AgentTrainingConfig(*args: Any, **kwargs: Any) -> SkillTrainingConfig:
    """Deprecated alias for SkillTrainingConfig (one-release compatibility)."""
    warnings.warn(
        "training_system.AgentTrainingConfig is deprecated; use SkillTrainingConfig",
        DeprecationWarning,
        stacklevel=2,
    )
    return SkillTrainingConfig(*args, **kwargs)


@dataclass
class RegisteredSkill:
    role: str
    adapter_uri: str
    pass_rate: float
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    pass_rate: float
    passed_tests: int
    total_tests: int
    role: str
    error: str = ""


class AutomatedTrainingSystem:
    """Boto3-backed ops facade with an explicit mock path when AWS is unavailable."""

    def __init__(self, config: InfrastructureConfig, settings: Optional[MangoMASSettings] = None):
        self.config = config
        self.settings = settings or get_settings()
        self.aws_available = self._probe_aws()
        logger.info(
            "AutomatedTrainingSystem ready region=%s aws_available=%s",
            self.config.aws_region,
            self.aws_available,
        )

    def _probe_aws(self) -> bool:
        try:
            import boto3

            boto3.client("sts", region_name=self.config.aws_region).get_caller_identity()
            return True
        except Exception as exc:
            logger.debug("AWS probe failed: %s", exc)
            return False

    def verify_resources(self) -> dict[str, Any]:
        """Confirm the configured S3 bucket and DynamoDB table exist."""
        if not self.aws_available:
            return {"status": "unavailable", "error": "STS identity probe failed"}
        try:
            import boto3

            boto3.client("s3", region_name=self.config.aws_region).head_bucket(
                Bucket=self.config.s3_bucket
            )
            boto3.client("dynamodb", region_name=self.config.aws_region).describe_table(
                TableName=self.config.dynamodb_table
            )
            return {
                "status": "verified",
                "bucket": self.config.s3_bucket,
                "table": self.config.dynamodb_table,
            }
        except Exception as exc:
            logger.warning("Resource verification failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

    def _ensure_dynamodb_table(self, dynamodb: Any) -> None:
        try:
            dynamodb.describe_table(TableName=self.config.dynamodb_table)
            return
        except Exception:
            logger.info("Creating DynamoDB table %s", self.config.dynamodb_table)
        dynamodb.create_table(
            TableName=self.config.dynamodb_table,
            KeySchema=[{"AttributeName": "role", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "role", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        try:
            dynamodb.get_waiter("table_exists").wait(
                TableName=self.config.dynamodb_table,
                WaiterConfig={"Delay": 2, "MaxAttempts": 30},
            )
        except Exception as exc:
            logger.warning(
                "Timed out waiting for DynamoDB table %s: %s", self.config.dynamodb_table, exc
            )

    async def setup_infrastructure(self) -> dict[str, Any]:
        if not self.aws_available:
            logger.info("Infrastructure setup using mock mode")
            return {
                "status": "mock",
                "mode": "mock",
                "error": "AWS unavailable; refusing to report setup as completed",
                "region": self.config.aws_region,
                "bucket": self.config.s3_bucket,
                "table": self.config.dynamodb_table,
            }
        try:
            import boto3

            s3 = boto3.client("s3", region_name=self.config.aws_region)
            try:
                s3.head_bucket(Bucket=self.config.s3_bucket)
            except Exception:
                kwargs: dict[str, Any] = {"Bucket": self.config.s3_bucket}
                if self.config.aws_region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": self.config.aws_region
                    }
                s3.create_bucket(**kwargs)
            dynamodb = boto3.client("dynamodb", region_name=self.config.aws_region)
            self._ensure_dynamodb_table(dynamodb)
            return {
                "status": "completed",
                "mode": "aws",
                "bucket": self.config.s3_bucket,
                "table": self.config.dynamodb_table,
            }
        except Exception as exc:
            logger.warning("Infrastructure setup failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

    async def train_agent_skill(self, training_config: SkillTrainingConfig) -> dict[str, Any]:
        logger.info(
            "Train skill role=%s mock=%s", training_config.role, training_config.use_mock_training
        )
        if not training_config.role or not training_config.dataset_path:
            return {"status": "failed", "error": "role and dataset_path are required"}
        dataset = Path(training_config.dataset_path)
        if training_config.use_mock_training:
            return {
                "status": "success",
                "model_artifacts": training_config.output_path or "s3://mock/models",
                "training_time": 0,
                "instance_type": training_config.instance_type,
                "mock": True,
            }
        if not dataset.exists():
            return {
                "status": "failed",
                "error": f"dataset not found: {training_config.dataset_path}",
            }
        if not self.aws_available:
            return {"status": "failed", "error": "AWS unavailable"}
        from enhanced_system.ops.sagemaker_launcher import (
            AgentTrainingConfig as LauncherConfig,
        )
        from enhanced_system.ops.sagemaker_launcher import MangoMASSageMakerLauncher

        launcher = MangoMASSageMakerLauncher(
            region=self.config.aws_region,
            instance_type=training_config.instance_type,
            settings=self.settings,
        )
        launched = await launcher.launch_training_job(
            LauncherConfig(
                agent_name=training_config.role.replace(" ", "_").lower(),
                training_file=str(dataset),
                model_name=training_config.base_model,
                instance_type=training_config.instance_type,
                epochs=training_config.epochs,
                batch_size=training_config.batch_size,
                learning_rate=training_config.learning_rate,
            )
        )
        if launched.get("status") == "failed":
            return {"status": "failed", "error": launched.get("error", "launch failed")}
        return {
            "status": "success",
            "model_artifacts": launched.get("model_output") or training_config.output_path,
            "training_time": 0,
            "instance_type": training_config.instance_type,
            "mock": False,
            "job": launched,
        }

    async def evaluate_agent_skill(
        self, role: str, test_suite: str, threshold: float
    ) -> EvalResult:
        logger.info("Evaluate skill role=%s suite=%s threshold=%s", role, test_suite, threshold)
        path = Path(test_suite)
        if not path.exists():
            return EvalResult(
                pass_rate=0.0,
                passed_tests=0,
                total_tests=0,
                role=role,
                error=f"test suite not found: {test_suite}",
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return EvalResult(
                pass_rate=0.0,
                passed_tests=0,
                total_tests=0,
                role=role,
                error=str(exc),
            )
        cases = payload if isinstance(payload, list) else payload.get("tests", [])
        total = len(cases)
        passed = sum(1 for case in cases if _case_passed(case))
        rate = (passed / total) * 100 if total else 0.0
        error = ""
        if total and rate < threshold:
            error = f"pass_rate {rate:.1f} below threshold {threshold}"
        return EvalResult(
            pass_rate=rate,
            passed_tests=passed,
            total_tests=total,
            role=role,
            error=error,
        )

    async def register_skill(self, skill: RegisteredSkill) -> dict[str, Any]:
        logger.info("Register skill role=%s", skill.role)
        if not self.aws_available:
            return {"status": "failed", "error": "AWS unavailable", "mode": "mock"}
        try:
            import boto3

            client = boto3.client("dynamodb", region_name=self.config.aws_region)
            client.put_item(
                TableName=self.config.dynamodb_table,
                Item={
                    "role": {"S": skill.role},
                    "adapter_uri": {"S": skill.adapter_uri},
                    "model": {"S": skill.model},
                    "pass_rate": {"N": str(skill.pass_rate)},
                },
            )
            return {"status": "success", "role": skill.role, "adapter_uri": skill.adapter_uri}
        except Exception as exc:
            logger.warning("Skill registration failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

    async def package_onnx(self, role: str, adapter_path: str, output_path: str) -> dict[str, Any]:
        logger.info("Package ONNX role=%s", role)
        source = Path(adapter_path)
        if not source.exists():
            return {"status": "failed", "error": f"adapter not found: {adapter_path}"}
        return {
            "status": "failed",
            "error": "ONNX conversion is not implemented in this package",
            "role": role,
            "source": adapter_path,
            "onnx_model_path": output_path,
        }
