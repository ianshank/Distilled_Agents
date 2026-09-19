#!/usr/bin/env python3
"""Trains a LoRA adapter using Direct Preference Optimization (DPO)."""

import argparse
import logging
import sys
from pathlib import Path

from datasets import load_dataset
from enhanced_system.ops.settings import get_settings

from scripts.training.distill.dpo_collator import format_dpo_example
from scripts.training.distill.model_load import load_causal_lm, load_tokenizer

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Distilled Agent using DPO.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model path or Hub ID")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to preference JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./dpo_adapter", help="Output adapter directory")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--use_device_map", action="store_true", help="Use auto device map")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code (subject to SDLC guardrails)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    return parser.parse_args()

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    settings = get_settings()

    try:
        from peft import LoraConfig, get_peft_model
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        logger.error("DPO requires 'trl' and 'peft'. Install with: pip install trl peft")
        return 1

    logger.info("Loading tokenizer and model...")
    tokenizer = load_tokenizer(args.model_name_or_path, args)

    # We load the reference model and the active model.
    # DPOTrainer can optionally accept a separate ref_model, or it can create one
    # automatically if we pass peft_config and no ref_model.
    model = load_causal_lm(args.model_name_or_path, args)

    # Setup LoRA
    # Dynamically resolve target modules based on Phase 1 auto-detection if empty
    from scripts.training.train_distilled_adapter import resolve_target_modules_for_model
    target_modules = settings.lora_target_modules or resolve_target_modules_for_model(model)

    peft_config = LoraConfig(
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=settings.use_dora,
    )

    # Load preference dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_file():
        logger.error("Dataset not found: %s", dataset_path)
        return 1

    logger.info("Formatting dataset...")
    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def process(example):
        return format_dpo_example(example, tokenizer)

    formatted_dataset = raw_dataset.map(process, remove_columns=raw_dataset.column_names)

    dpo_config = DPOConfig(
        beta=settings.dpo_beta,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    logger.info("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # trl creates reference model automatically from peft_config
        args=dpo_config,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    logger.info("Saving adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    return 0

if __name__ == "__main__":
    sys.exit(main())
