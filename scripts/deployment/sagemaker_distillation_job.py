#!/usr/bin/env python3
"""Configure and launch SageMaker training jobs for agent distillation."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import sagemaker
from enhanced_system.ops.archive import safe_extract_tar
from enhanced_system.ops.settings import get_settings
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

logger = logging.getLogger(__name__)


class AgentDistillationJob:
    """SageMaker training job manager for agent distillation."""

    def __init__(self, region: Optional[str] = None):
        settings = get_settings()
        self.region = region or settings.aws_region
        self.settings = settings
        self.sagemaker_session = sagemaker.Session()
        self.role = get_execution_role()
        self.s3_bucket = self.sagemaker_session.default_bucket()

    def create_training_job(
        self,
        job_name: str,
        teacher_model: str,
        student_model: str,
        train_data_s3: str,
        eval_data_s3: Optional[str] = None,
        instance_type: str = "ml.g5.2xlarge",
        instance_count: int = 1,
        hyperparameters: Optional[Dict[str, Any]] = None,
        use_spot_instances: bool = True,
        max_wait_time: int = 3600,
        **kwargs,
    ) -> str:
        """Create and launch a SageMaker training job for agent distillation."""
        default_hyperparameters = {
            "teacher_model_name": teacher_model,
            "student_model_name": student_model,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "distillation_alpha": 0.5,
            "temperature": 2.0,
            "max_length": 512,
            "logging_steps": 100,
            "save_steps": 500,
            "save_total_limit": 2,
            "eval_steps": 500,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
            "use_fp16": True,
            "use_device_map": True,
            "use_wandb": False,
            "trust_remote_code": str(self.settings.trust_remote_code).lower(),
        }
        if hyperparameters:
            default_hyperparameters.update(hyperparameters)
        if eval_data_s3:
            default_hyperparameters["eval_file"] = "/opt/ml/input/data/eval/train.jsonl"

        estimator = PyTorch(
            entry_point="train_distilled_adapter.py",
            source_dir="training",
            role=self.role,
            instance_count=instance_count,
            instance_type=instance_type,
            framework_version="2.0.1",
            py_version="py310",
            hyperparameters=default_hyperparameters,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait_time,
            max_run=max_wait_time,
            output_path=f"s3://{self.s3_bucket}/mangomas-distillation-output",
            code_location=f"s3://{self.s3_bucket}/mangomas-distillation-code",
            base_job_name=job_name,
            **kwargs,
        )
        inputs = {
            "train": sagemaker.inputs.TrainingInput(
                s3_data=train_data_s3, content_type="application/json"
            )
        }
        if eval_data_s3:
            inputs["eval"] = sagemaker.inputs.TrainingInput(
                s3_data=eval_data_s3, content_type="application/json"
            )
        logger.info("Launching distillation job %s in %s", job_name, self.region)
        estimator.fit(inputs, job_name=job_name)
        return job_name

    def create_distillation_pipeline(
        self,
        pipeline_name: str,
        teacher_model: str,
        student_model: str,
        train_data_s3: str,
        eval_data_s3: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Create a two-stage distillation pipeline."""
        stage1_job = f"{pipeline_name}-stage1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.create_training_job(
            job_name=stage1_job,
            teacher_model=teacher_model,
            student_model=student_model,
            train_data_s3=train_data_s3,
            eval_data_s3=eval_data_s3,
            hyperparameters={
                "distillation_alpha": 0.7,
                "temperature": 3.0,
                "num_train_epochs": 2,
                "learning_rate": 1e-4,
            },
            **kwargs,
        )
        stage2_job = f"{pipeline_name}-stage2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        stage1_output = (
            f"s3://{self.s3_bucket}/mangomas-distillation-output/{stage1_job}/output/model.tar.gz"
        )
        self.create_training_job(
            job_name=stage2_job,
            teacher_model=teacher_model,
            student_model=stage1_output,
            train_data_s3=train_data_s3,
            eval_data_s3=eval_data_s3,
            hyperparameters={
                "distillation_alpha": 0.3,
                "temperature": 1.5,
                "num_train_epochs": 1,
                "learning_rate": 5e-5,
            },
            **kwargs,
        )
        return pipeline_name

    def monitor_training_job(self, job_name: str) -> dict[str, Any]:
        """Return SageMaker training job status and metrics."""
        client = boto3.client("sagemaker", region_name=self.region)
        try:
            response = client.describe_training_job(TrainingJobName=job_name)
            status = response["TrainingJobStatus"]
            logger.info("Training job %s status=%s", job_name, status)
            return {
                "job_name": job_name,
                "status": status,
                "metrics": response.get("FinalMetricDataList", []),
                "output": response.get("OutputDataConfig", {}),
            }
        except Exception:
            logger.exception("Error monitoring job %s", job_name)
            return {"job_name": job_name, "status": "error"}

    def list_training_jobs(self, name_contains: str = "mangomas") -> list[dict[str, Any]]:
        """List training jobs with optional name filter."""
        client = boto3.client("sagemaker", region_name=self.region)
        try:
            response = client.list_training_jobs(NameContains=name_contains, MaxResults=20)
            summaries = response.get("TrainingJobSummaries", [])
            logger.info("Found %s jobs matching %s", len(summaries), name_contains)
            return summaries
        except Exception:
            logger.exception("Error listing jobs")
            return []

    def download_model(self, job_name: str, local_path: str) -> str:
        """Download a trained model archive and extract it safely."""
        client = boto3.client("sagemaker", region_name=self.region)
        response = client.describe_training_job(TrainingJobName=job_name)
        model_data = response["ModelArtifacts"]["S3ModelArtifacts"]
        s3_client = boto3.client("s3", region_name=self.region)
        bucket = model_data.split("/")[2]
        key = "/".join(model_data.split("/")[3:])
        dest = Path(local_path)
        dest.mkdir(parents=True, exist_ok=True)
        local_tar = dest / "model.tar.gz"
        s3_client.download_file(bucket, key, str(local_tar))
        safe_extract_tar(local_tar, dest)
        logger.info("Model downloaded to %s", dest)
        return str(dest)

    def create_inference_endpoint(
        self,
        model_name: str,
        endpoint_name: str,
        instance_type: str = "ml.m5.large",
        instance_count: int = 1,
    ):
        """Create a SageMaker inference endpoint for a distilled model."""
        model = sagemaker.pytorch.PyTorchModel(
            model_data=(
                f"s3://{self.s3_bucket}/mangomas-distillation-output/"
                f"{model_name}/output/model.tar.gz"
            ),
            role=self.role,
            entry_point="inference.py",
            source_dir="training",
            framework_version="2.0.1",
            py_version="py310",
        )
        return model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )


def main() -> int:
    """Example usage of AgentDistillationJob (does not launch unless AWS is configured)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    job_manager = AgentDistillationJob(region=settings.aws_region)
    job_name = f"mangomas-distillation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    train_data_s3 = os.getenv(
        "MANGOMAS_DISTILL_TRAIN_S3",
        f"s3://{settings.training_data_bucket}/mangomas/training-data/train.jsonl",
    )
    eval_data_s3 = os.getenv(
        "MANGOMAS_DISTILL_EVAL_S3",
        f"s3://{settings.training_data_bucket}/mangomas/training-data/eval.jsonl",
    )
    hyperparameters = {
        "teacher_model_name": settings.teacher_model,
        "student_model_name": settings.student_model,
        "num_train_epochs": 3,
        "distillation_alpha": 0.5,
        "temperature": 2.0,
        "use_lora": True,
        "use_fp16": True,
        "trust_remote_code": str(settings.trust_remote_code).lower(),
    }
    try:
        created = job_manager.create_training_job(
            job_name=job_name,
            teacher_model=settings.teacher_model,
            student_model=settings.student_model,
            train_data_s3=train_data_s3,
            eval_data_s3=eval_data_s3,
            instance_type=settings.gpu_instance_type,
            hyperparameters=hyperparameters,
            use_spot_instances=True,
            max_wait_time=settings.max_run_seconds,
        )
        print(f"Training job created: {created}")
        job_manager.monitor_training_job(created)
        return 0
    except Exception:
        logger.exception("Error creating training job")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
