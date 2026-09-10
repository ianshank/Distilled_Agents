"""Unified SageMaker launcher used by all deployment CLIs."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from enhanced_system.ops.settings import MangoMASSettings, get_settings


@dataclass
class AgentTrainingConfig:
    agent_name: str
    training_file: str
    model_name: str = ""
    instance_type: str = ""
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-5
    max_length: int = 512
    use_spot_instances: bool = True
    max_runtime: int = 0


class MangoMASSageMakerLauncher:
    """Validate, upload, launch, monitor, and report SageMaker jobs."""

    def __init__(
        self,
        region: Optional[str] = None,
        role_arn: Optional[str] = None,
        terraform_config_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        settings: Optional[MangoMASSettings] = None,
        data_dir: Optional[Path] = None,
    ):
        self.settings = settings or get_settings()
        self.region = region or os.getenv("AWS_REGION", self.settings.aws_region)
        self.role_arn = role_arn
        self.instance_type = instance_type or self.settings.gpu_instance_type
        self.terraform_config_path = terraform_config_path or self._find_terraform_config()
        self.terraform_config = self._load_terraform_config()
        self.data_dir = data_dir or Path("data/training")
        self.agent_configs = self._setup_agent_configurations()
        self.training_results: list[dict[str, Any]] = []
        self.failed_jobs: list[dict[str, Any]] = []
        self._clients: dict[str, Any] = {}

    def _find_terraform_config(self) -> Optional[str]:
        for path in (
            "infrastructure/terraform/sagemaker-jobs/training_job_config.json",
            "training_job_config.json",
        ):
            if os.path.exists(path):
                return path
        return None

    def _load_terraform_config(self) -> Optional[dict]:
        if not self.terraform_config_path:
            return None
        try:
            with open(self.terraform_config_path, encoding="utf-8") as handle:
                return json.load(handle)
        except OSError:
            return None

    def _setup_agent_configurations(self) -> list[AgentTrainingConfig]:
        agents = [
            "product_manager_agent",
            "sqe_agent",
            "architect_agent",
            "swe_agent",
            "vp_product_agent",
            "devops_agent",
            "tools_agent",
            "product_manager_agent_real_data",
            "sqe_agent_real_data",
            "architect_agent_real_data",
            "swe_agent_real_data",
        ]
        configs = []
        for agent in agents:
            configs.append(
                AgentTrainingConfig(
                    agent_name=agent,
                    training_file=f"{agent}.jsonl",
                    model_name=self.settings.teacher_model,
                    instance_type=self.instance_type,
                    epochs=4 if "real_data" in agent else 3,
                    max_runtime=self.settings.max_run_seconds,
                )
            )
        return configs

    def _account_id(self) -> str:
        try:
            import boto3

            return boto3.client("sts", region_name=self.region).get_caller_identity()["Account"]
        except Exception:
            return "000000000000"

    def _get_execution_role(self) -> str:
        if self.role_arn:
            return self.role_arn
        if self.terraform_config and "role_arn" in self.terraform_config:
            return self.terraform_config["role_arn"]
        env_role = os.getenv("SAGEMAKER_ROLE_ARN")
        if env_role:
            return env_role
        return self.settings.role_arn(self._account_id())

    def _get_s3_bucket(self) -> str:
        if self.terraform_config and "s3_bucket" in self.terraform_config:
            return self.terraform_config["s3_bucket"]
        return self.settings.production_bucket(self._account_id())

    def training_source_dir(self) -> Path:
        """Directory SageMaker uploads as source_dir (entrypoint + distill package)."""
        override = os.getenv("MANGOMAS_TRAINING_SOURCE_DIR")
        if override:
            return Path(override)
        try:
            import scripts.training as training_pkg

            return Path(training_pkg.__file__).resolve().parent
        except Exception:
            return Path(__file__).resolve().parents[2] / "scripts" / "training"

    def validate_training_data(self) -> bool:
        missing = []
        for config in self.agent_configs:
            path = self.data_dir / config.training_file
            if not path.exists() and not Path(config.training_file).exists():
                missing.append(config.training_file)
        return not missing

    def upload_training_scripts_to_s3(self, bucket_name: Optional[str] = None) -> bool:
        """Upload train_distilled_adapter.py and the distill/ package to s3://bucket/scripts/."""
        bucket = bucket_name or self._get_s3_bucket()
        source = self.training_source_dir()
        if not source.exists():
            return False
        try:
            import boto3

            client = boto3.client("s3", region_name=self.region)
            for path in source.rglob("*"):
                if not path.is_file() or path.suffix not in {".py", ".txt"}:
                    continue
                key = f"scripts/{path.relative_to(source).as_posix()}"
                client.upload_file(str(path), bucket, key)
            return True
        except Exception:
            return False

    def upload_training_data_to_s3(self, bucket_name: Optional[str] = None) -> bool:
        bucket = bucket_name or self._get_s3_bucket()
        try:
            import boto3

            client = boto3.client("s3", region_name=self.region)
            client.head_bucket(Bucket=bucket)
            for config in self.agent_configs:
                local = self.data_dir / config.training_file
                if not local.exists():
                    local = Path(config.training_file)
                if local.exists():
                    client.upload_file(str(local), bucket, f"datasets/{config.training_file}")
            self.upload_training_scripts_to_s3(bucket)
            return True
        except Exception:
            return False

    def create_job_spec(self, config: AgentTrainingConfig) -> dict[str, Any]:
        return {
            "agent_name": config.agent_name,
            "training_file": config.training_file,
            "instance_type": config.instance_type or self.instance_type,
            "model_name": config.model_name or self.settings.teacher_model,
            "role": self._get_execution_role(),
            "bucket": self._get_s3_bucket(),
            "hyperparameters": {
                "model_name_or_path": config.model_name or self.settings.teacher_model,
                "train_file": config.training_file,
                "num_train_epochs": str(config.epochs),
                "per_device_train_batch_size": str(config.batch_size),
                "learning_rate": str(config.learning_rate),
                "trust_remote_code": str(self.settings.trust_remote_code).lower(),
            },
        }

    async def launch_training_job(self, config: AgentTrainingConfig) -> dict[str, Any]:
        spec = self.create_job_spec(config)
        try:
            from sagemaker.huggingface import HuggingFace

            estimator = HuggingFace(
                entry_point="train_distilled_adapter.py",
                source_dir=str(self.training_source_dir()),
                instance_type=spec["instance_type"],
                instance_count=1,
                role=spec["role"],
                transformers_version="4.26.0",
                pytorch_version="1.13.1",
                py_version="py39",
                use_spot_instances=config.use_spot_instances,
                max_run=config.max_runtime or self.settings.max_run_seconds,
                hyperparameters=spec["hyperparameters"],
                base_job_name=f"{config.agent_name}-train",
                output_path=f"s3://{spec['bucket']}/models/{config.agent_name}/",
            )
            estimator.fit(
                {"training": f"s3://{spec['bucket']}/datasets/{config.training_file}"},
                wait=False,
            )
            return {
                "agent_name": config.agent_name,
                "job_name": estimator.latest_training_job.name,
                "status": "started",
                "model_output": estimator.output_path,
                "instance_type": spec["instance_type"],
                "use_spot": config.use_spot_instances,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            return {
                "agent_name": config.agent_name,
                "status": "failed",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }

    async def launch_all_jobs(
        self, parallel: bool = True, max_concurrent: int = 3
    ) -> list[dict[str, Any]]:
        if not self.validate_training_data():
            return []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _one(config: AgentTrainingConfig) -> dict[str, Any]:
            async with semaphore:
                return await self.launch_training_job(config)

        if parallel:
            results = await asyncio.gather(
                *[_one(cfg) for cfg in self.agent_configs], return_exceptions=True
            )
        else:
            results = [await self.launch_training_job(cfg) for cfg in self.agent_configs]

        self.training_results = []
        self.failed_jobs = []
        for result in results:
            if isinstance(result, Exception):
                self.failed_jobs.append({"error": str(result)})
            elif result.get("status") == "failed":
                self.failed_jobs.append(result)
            else:
                self.training_results.append(result)
        return self.training_results

    def monitor_training_jobs(self, job_names: Optional[list[str]] = None) -> dict[str, str]:
        names = job_names or [item["job_name"] for item in self.training_results]
        statuses: dict[str, str] = {}
        try:
            import boto3

            client = boto3.client("sagemaker", region_name=self.region)
            for name in names:
                response = client.describe_training_job(TrainingJobName=name)
                statuses[name] = response["TrainingJobStatus"]
        except Exception:
            for name in names:
                statuses[name] = "unknown"
        return statuses

    def generate_summary_report(self) -> str:
        return (
            f"MangoMAS SageMaker summary: started={len(self.training_results)} "
            f"failed={len(self.failed_jobs)}"
        )

    def save_results(self, filename: str = "sagemaker_training_results.json") -> None:
        payload = {
            "started_jobs": self.training_results,
            "failed_jobs": self.failed_jobs,
            "configs": [asdict(item) for item in self.agent_configs],
        }
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
