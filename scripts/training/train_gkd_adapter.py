#!/usr/bin/env python3
"""Trains a LoRA adapter using TRL Generalized Knowledge Distillation (GKD) on trajectories."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from enhanced_system.ops.settings import get_settings

from scripts.training.distill.alpha import parse_trajectory_distill_alpha

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Train a Distilled Agent using TRL Generalized Knowledge Distillation."
    )
    parser.add_argument(
        "--student_model_name_or_path",
        type=str,
        default=os.getenv("MANGOMAS_STUDENT_MODEL", settings.student_model),
        help="Base student model path or Hub ID",
    )
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        default=os.getenv("MANGOMAS_TEACHER_MODEL", settings.teacher_model),
        help="Teacher model path or Hub ID for knowledge transfer",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to trajectory JSONL dataset (or MANGOMAS_TRAIN_FILE / train.jsonl)",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to evaluation JSONL dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gkd_adapter",
        help="Output adapter directory",
    )
    parser.add_argument(
        "--distillation_alpha",
        type=float,
        default=None,
        help="Teacher distillation loss weight in [0.0, 1.0] (defaults to MANGOMAS_TRAJECTORY_DISTILL_ALPHA / 0.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=settings.gkd_beta,
        help="Generalized JSD interpolation beta (0.0=KL, 0.5=JSD, 1.0=InvKL)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=settings.gkd_temperature,
        help="Softmax temperature for distillation",
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        default=settings.gkd_lmbda,
        help="Student data fraction for on-policy sampling",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=settings.harness_max_length,
        help="Maximum sequence length",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--use_device_map", action="store_true", help="Use auto device map")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code (subject to SDLC guardrails)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    settings = get_settings()

    from scripts.training.distill.gkd_adapter import (
        AgentGKDTrainer,
        GKDTrainingConfig,
        validate_vocab_alignment,
        verify_trl_version_pin,
    )

    # 1. Guard exact TRL pin
    try:
        verify_trl_version_pin(strict=False)
    except (ImportError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import TrainingArguments

        from scripts.training.distill.model_load import load_causal_lm, load_tokenizer
        from scripts.training.distill.sagemaker_io import resolve_train_file
        from scripts.training.distill.trainer import resolve_target_modules_for_model
        from scripts.training.distill.trajectory_collator import (
            TrajectoryDataCollator,
            has_supervised_tokens,
        )
    except ImportError as exc:
        logger.error("GKD training dependencies missing: %s", exc)
        return 1

    # 2. Parse alpha (defaults to 0.0)
    if args.distillation_alpha is not None:
        alpha = args.distillation_alpha
    else:
        raw_alpha = os.getenv(
            "MANGOMAS_TRAJECTORY_DISTILL_ALPHA",
            str(settings.trajectory_distill_alpha),
        )
        try:
            alpha = parse_trajectory_distill_alpha(raw_alpha)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

    # 3. Load tokenizers and check vocabulary alignment
    logger.info("Loading student tokenizer from %s...", args.student_model_name_or_path)
    student_tokenizer = load_tokenizer(args.student_model_name_or_path, args)
    teacher_model = None

    if alpha > 0.0:
        logger.info("Loading teacher tokenizer from %s...", args.teacher_model_name_or_path)
        teacher_tokenizer = load_tokenizer(args.teacher_model_name_or_path, args)
        student_vocab = len(student_tokenizer)
        teacher_vocab = len(teacher_tokenizer)
        try:
            validate_vocab_alignment(student_vocab, teacher_vocab, strict=True)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

        logger.info("Loading teacher model from %s...", args.teacher_model_name_or_path)
        teacher_model = load_causal_lm(args.teacher_model_name_or_path, args)
        for param in teacher_model.parameters():
            param.requires_grad = False

    # 4. Load student model with LoRA
    logger.info("Loading student model from %s...", args.student_model_name_or_path)
    student_model = load_causal_lm(args.student_model_name_or_path, args)

    target_modules = (
        [item.strip() for item in settings.lora_target_modules.split(",") if item.strip()]
        if settings.lora_target_modules
        else resolve_target_modules_for_model(args.student_model_name_or_path)
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_dora=settings.use_dora,
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # 5. Prepare trajectory dataset
    train_file = resolve_train_file(train_file=args.dataset_path)
    if not Path(train_file).is_file():
        logger.error("Dataset not found: %s", train_file)
        return 1

    raw_dataset = load_dataset("json", data_files={"train": train_file})  # nosec: B615
    train_dataset = raw_dataset["train"].filter(
        lambda row: has_supervised_tokens(student_tokenizer, row, args.max_length)
    )
    logger.info(
        "Trajectory dataset prepared: %d valid supervised rows from %d total",
        len(train_dataset),
        len(raw_dataset["train"]),
    )

    data_collator = TrajectoryDataCollator(student_tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=args.use_fp16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
    )

    gkd_config = GKDTrainingConfig(
        enabled=True,
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        distillation_alpha=alpha,
    )

    logger.info("Initializing AgentGKDTrainer (alpha=%.2f, beta=%.2f)...", alpha, args.beta)
    trainer = AgentGKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=student_tokenizer,
        data_collator=data_collator,
        distillation_alpha=alpha,
        beta=args.beta,
        temperature=args.temperature,
        lmbda=args.lmbda,
        gkd_config=gkd_config,
    )

    logger.info("Starting GKD on-policy training...")
    trainer.train()

    logger.info("Saving adapter to %s...", args.output_dir)
    trainer.save_model(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
