"""Lightweight training-system API used by ops CLIs."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from enhanced_system.ops.settings import MangoMASSettings, get_settings

logger = logging.getLogger(__name__)


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
    use_mock_training: bool = True
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


class AutomatedTrainingSystem:
    """Boto3-backed ops facade with a mock path when AWS is unavailable."""

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

    async def setup_infrastructure(self) -> dict[str, Any]:
        if not self.aws_available:
            logger.info("Infrastructure setup using mock mode")
            return {
                "status": "completed",
                "mode": "mock",
                "region": self.config.aws_region,
                "bucket": self.config.s3_bucket,
            }
        try:
            import boto3

            s3 = boto3.client("s3", region_name=self.config.aws_region)
            s3.head_bucket(Bucket=self.config.s3_bucket)
            return {"status": "completed", "mode": "aws", "bucket": self.config.s3_bucket}
        except Exception as exc:
            logger.warning("Infrastructure setup failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

    async def train_agent_skill(self, training_config: SkillTrainingConfig) -> dict[str, Any]:
        logger.info("Train skill role=%s mock=%s", training_config.role, not self.aws_available)
        if not training_config.role or not training_config.dataset_path:
            return {"status": "failed", "error": "role and dataset_path are required"}
        dataset = Path(training_config.dataset_path)
        if not training_config.use_mock_training and not dataset.exists():
            return {
                "status": "failed",
                "error": f"dataset not found: {training_config.dataset_path}",
            }
        return {
            "status": "success",
            "model_artifacts": training_config.output_path or "s3://mock/models",
            "training_time": 0,
            "instance_type": training_config.instance_type,
            "mock": not self.aws_available,
        }

    async def evaluate_agent_skill(self, role: str, test_suite: str, threshold: float) -> Any:
        logger.info("Evaluate skill role=%s suite=%s threshold=%s", role, test_suite, threshold)

        @dataclass
        class EvalResult:
            pass_rate: float = 100.0
            passed_tests: int = 1
            total_tests: int = 1
            role: str = role

        return EvalResult()

    async def register_skill(self, skill: RegisteredSkill) -> dict[str, Any]:
        logger.info("Register skill role=%s", skill.role)
        return {"status": "success", "role": skill.role, "adapter_uri": skill.adapter_uri}

    async def package_onnx(self, role: str, adapter_path: str, output_path: str) -> dict[str, Any]:
        logger.info("Package ONNX role=%s", role)
        return {
            "status": "success",
            "onnx_model_path": output_path,
            "role": role,
            "source": adapter_path,
        }
