"""Distillation trainer classes (extracted from the SageMaker entrypoint)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .model_load import load_causal_lm, load_tokenizer
from .sagemaker_io import resolve_train_file, texts_from_examples

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None  # type: ignore[misc,assignment]
    TaskType = None  # type: ignore[misc,assignment]
    get_peft_model = None  # type: ignore[misc,assignment]

try:
    import wandb
except ImportError:
    wandb = None


class AgentDistillationTrainer:
    """Advanced trainer for agent distillation with knowledge transfer."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_wandb()

    def setup_wandb(self):
        if self.args.use_wandb and wandb is not None:
            wandb.init(
                project="mangomas-agent-distillation",
                name=f"distillation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(self.args),
            )

    def load_teacher_model(self):
        logger.info("Loading teacher model: %s", self.args.teacher_model_name)
        teacher_model = load_causal_lm(self.args.teacher_model_name, self.args)
        for param in teacher_model.parameters():
            param.requires_grad = False
        return teacher_model

    def load_student_model(self):
        logger.info("Loading student model: %s", self.args.student_model_name)
        return load_causal_lm(self.args.student_model_name, self.args)

    def setup_lora_config(self):
        if LoraConfig is None:
            raise ImportError("PEFT is required for LoRA training")
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=self.args.lora_target_modules.split(","),
        )

    def prepare_dataset(self) -> Dataset:
        train_file = resolve_train_file(train_file=getattr(self.args, "train_file", None))
        dataset = load_dataset("json", data_files={"train": train_file})  # nosec B615
        if getattr(self.args, "trajectory_mode", False):
            from .trajectory_collator import has_supervised_tokens

            logger.info("Trajectory mode: keeping raw columns for masked collator")
            tokenizer = load_tokenizer(self.args.student_model_name, self.args)
            filtered = dataset["train"].filter(
                lambda row: has_supervised_tokens(tokenizer, row, self.args.max_length)
            )
            logger.info("Trajectory mode: filtered %s -> %s rows", len(dataset["train"]), len(filtered))
            return filtered
        tokenizer = load_tokenizer(self.args.student_model_name, self.args)

        def tokenize_function(examples):
            texts = texts_from_examples(examples)
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.args.max_length,
                return_tensors="pt",
            )

        tokenized_dataset = dataset["train"].map(
            tokenize_function, batched=True, remove_columns=dataset["train"].column_names
        )
        logger.info("Dataset prepared: %s samples", len(tokenized_dataset))
        return tokenized_dataset

    def train(self):
        logger.info("Starting agent distillation training")
        teacher_model = None
        if self.args.distillation_alpha > 0 and not getattr(self.args, "trajectory_mode", False):
            teacher_model = self.load_teacher_model()
        student_model = self.load_student_model()
        if self.args.use_lora:
            student_model = get_peft_model(student_model, self.setup_lora_config())
            student_model.print_trainable_parameters()
        train_dataset = self.prepare_dataset()
        tokenizer = load_tokenizer(self.args.student_model_name, self.args)
        if getattr(self.args, "trajectory_mode", False):
            from .trajectory_collator import TrajectoryDataCollator

            data_collator = TrajectoryDataCollator(tokenizer, max_length=self.args.max_length)
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_steps=self.args.warmup_steps,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            evaluation_strategy="steps" if self.args.eval_file else "no",
            eval_steps=self.args.eval_steps if self.args.eval_file else None,
            load_best_model_at_end=bool(self.args.eval_file),
            fp16=self.args.use_fp16,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if self.args.use_wandb else None,
            logging_dir=f"{self.args.output_dir}/logs",
        )
        trainer = DistillationTrainer(
            model=student_model,
            teacher_model=teacher_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=self.prepare_eval_dataset() if self.args.eval_file else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            distillation_alpha=self.args.distillation_alpha,
            temperature=self.args.temperature,
        )
        trainer.train()
        final_output_dir = os.path.join(self.args.output_dir, "final")
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        self.save_training_metadata(final_output_dir)
        if self.args.use_wandb and wandb is not None:
            wandb.finish()

    def prepare_eval_dataset(self) -> Optional[Dataset]:
        if not self.args.eval_file:
            return None
        dataset = load_dataset("json", data_files={"eval": self.args.eval_file})  # nosec B615
        if getattr(self.args, "trajectory_mode", False):
            from .trajectory_collator import has_supervised_tokens

            tokenizer = load_tokenizer(self.args.student_model_name, self.args)
            filtered = dataset["eval"].filter(
                lambda row: has_supervised_tokens(tokenizer, row, self.args.max_length)
            )
            logger.info("Trajectory eval: filtered %s -> %s rows", len(dataset["eval"]), len(filtered))
            return filtered
        tokenizer = load_tokenizer(self.args.student_model_name, self.args)

        def tokenize_function(examples):
            return tokenizer(
                texts_from_examples(examples),
                truncation=True,
                padding="max_length",
                max_length=self.args.max_length,
                return_tensors="pt",
            )

        return dataset["eval"].map(
            tokenize_function, batched=True, remove_columns=dataset["eval"].column_names
        )

    def save_training_metadata(self, output_dir: str):
        metadata = {
            "training_config": vars(self.args),
            "completed_at": datetime.now().isoformat(),
            "device": str(self.device),
        }
        with open(
            os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2)


class DistillationTrainer(Trainer):
    """Custom trainer with knowledge distillation capabilities."""

    def __init__(self, teacher_model, distillation_alpha, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_alpha = distillation_alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        student_outputs = model(**inputs)
        teacher_outputs = None
        if self.teacher_model is not None and self.distillation_alpha > 0:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
        loss = self.create_distillation_loss(student_outputs, teacher_outputs, inputs.get("labels"))
        return (loss, student_outputs) if return_outputs else loss

    def create_distillation_loss(self, student_outputs, teacher_outputs, labels):
        task_loss = torch.nn.functional.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        if self.distillation_alpha > 0 and teacher_outputs is not None:
            student_logits = student_outputs.logits / self.temperature
            teacher_logits = teacher_outputs.logits / self.temperature
            distillation_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            ) * (self.temperature**2)
            return (
                1 - self.distillation_alpha
            ) * task_loss + self.distillation_alpha * distillation_loss
        return task_loss
