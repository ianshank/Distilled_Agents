#!/usr/bin/env python3
"""SageMaker-compatible distillation entrypoint (facade)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from scripts.training.distill.trainer import AgentDistillationTrainer, DistillationTrainer
except ImportError:
    # SageMaker copies source_dir (scripts/training) onto PYTHONPATH.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from distill.trainer import AgentDistillationTrainer, DistillationTrainer


def main():
    parser = argparse.ArgumentParser(description="MangoMAS Agent Distillation Training")
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default=os.getenv("MANGOMAS_TEACHER_MODEL", "mistralai/Mistral-7B-v0.1"),
    )
    parser.add_argument(
        "--student_model_name",
        type=str,
        default=os.getenv("MANGOMAS_STUDENT_MODEL", "microsoft/DialoGPT-medium"),
    )
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--distillation_alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--use_lora", type=str, default="True")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--use_fp16", type=str, default="False")
    parser.add_argument("--use_device_map", type=str, default="False")
    parser.add_argument(
        "--trust_remote_code",
        type=str,
        default=os.getenv("MANGOMAS_TRUST_REMOTE_CODE", "False"),
    )
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--use_wandb", type=str, default="False")
    parser.add_argument("--adapter_type", type=str, default="default_alora")
    parser.add_argument("--agent_name", type=str, default="default-agent")
    parser.add_argument("--agent_role", type=str, default="Default")
    parser.add_argument("--agent_specialization", type=str, default="Default")
    parser.add_argument("--capabilities", type=str, default="")
    args = parser.parse_args()
    args.use_fp16 = args.use_fp16.lower() == "true"
    args.use_device_map = args.use_device_map.lower() == "true"
    args.use_wandb = args.use_wandb.lower() == "true"
    args.use_lora = args.use_lora.lower() == "true"
    args.trust_remote_code = args.trust_remote_code.lower() == "true"
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Starting distillation for agent=%s", args.agent_name)
    AgentDistillationTrainer(args).train()


if __name__ == "__main__":
    main()


__all__ = ["AgentDistillationTrainer", "DistillationTrainer", "main"]
