"""Distillation trainer classes (extracted from the SageMaker entrypoint)."""

from __future__ import annotations

import inspect
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

# Architecture-specific LoRA target module mappings.
# Key = model_type from config.json; value = list of linear module names.
_LORA_MODULE_MAP: dict[str, list[str]] = {
    # GPT-2 / DialoGPT family
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    # LLaMA / Mistral / Qwen / Yi / DeepSeek family
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
_DEFAULT_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def resolve_target_modules_for_model(model_name: str, revision: Optional[str] = None) -> list[str]:
    """Detect LoRA target modules from model config's ``model_type``.

    Returns architecture-appropriate module names. Falls back to the
    LLaMA-style default if the model type is unrecognized.
    """
    try:
        from transformers import AutoConfig

        # Use explicit revision pin or environment fallback; nosec B615 justified
        # by revision pinning and safe trust_remote_code=False.
        rev = revision if revision is not None else os.getenv("MANGOMAS_MODEL_REVISION")
        config = AutoConfig.from_pretrained(  # nosec B615
            model_name,
            trust_remote_code=False,
            revision=rev,
        )
        model_type = getattr(config, "model_type", "").lower()
        modules = _LORA_MODULE_MAP.get(model_type, _DEFAULT_MODULES)
        logger.info(
            "Auto-detected LoRA modules for %s (type=%s): %s", model_name, model_type, modules
        )
        return list(modules)
    except Exception:
        logger.warning("Could not auto-detect model type for %s, using default modules", model_name)
        return list(_DEFAULT_MODULES)


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

    def _validate_distillation_compatibility(self) -> None:
        if self.args.distillation_alpha <= 0:
            return
        student_tokenizer = load_tokenizer(self.args.student_model_name, self.args)
        teacher_tokenizer = load_tokenizer(self.args.teacher_model_name, self.args)
        student_vocab = len(student_tokenizer)
        teacher_vocab = len(teacher_tokenizer)
        if student_vocab != teacher_vocab:
            raise ValueError(
                "Incompatible teacher/student tokenizers for KL distillation: "
                f"student_vocab={student_vocab}, teacher_vocab={teacher_vocab}. "
                "Set distillation_alpha=0 or use tokenizer-compatible model pairs."
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
        target_modules = self._resolve_target_modules()
        lora_kwargs = {
            "task_type": TaskType.CAUSAL_LM,
            "inference_mode": False,
            "r": self.args.lora_r,
            "lora_alpha": self.args.lora_alpha,
            "lora_dropout": self.args.lora_dropout,
            "target_modules": target_modules,
        }
        # DoRA support (peft >= 0.14.0)
        if getattr(self.args, "use_dora", False):
            lora_kwargs["use_dora"] = True
            logger.info("DoRA enabled for LoRA config")
        return LoraConfig(**lora_kwargs)

    def _resolve_target_modules(self) -> list[str]:
        """Auto-detect LoRA target modules based on model architecture.

        Falls back to explicit --lora_target_modules if provided.
        """
        explicit = getattr(self.args, "lora_target_modules", "")
        if explicit:
            return explicit.split(",")
        rev = getattr(self.args, "model_revision", None)
        return resolve_target_modules_for_model(self.args.student_model_name, revision=rev)

    def _filter_supervised_trajectory_rows(self, split, split_name: str):
        from .trajectory_collator import has_supervised_tokens

        tokenizer = load_tokenizer(self.args.student_model_name, self.args)
        filtered = split.filter(lambda row: has_supervised_tokens(tokenizer, row, self.args.max_length))
        logger.info("Trajectory %s: filtered %s -> %s rows", split_name, len(split), len(filtered))
        if len(filtered) == 0:
            raise ValueError(
                f"Trajectory {split_name} dataset filtered to 0 rows. "
                "Ensure the input JSONL contains valid trajectory 'turns' instead of bare prompts."
            )
        return filtered

    def prepare_dataset(self) -> Dataset:
        train_file = resolve_train_file(train_file=getattr(self.args, "train_file", None))
        # Local JSON file loading; Hub is not contacted (green-trunk-ci)
        dataset = load_dataset("json", data_files={"train": train_file})  # nosec: B615
        if getattr(self.args, "trajectory_mode", False):
            logger.info("Trajectory mode: keeping raw columns for masked collator")
            return self._filter_supervised_trajectory_rows(dataset["train"], "train")
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
        if self.args.distillation_alpha > 0:
            self._validate_distillation_compatibility()
        teacher_model = None
        if self.args.distillation_alpha > 0:
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
        eval_strategy_key = (
            "eval_strategy"
            if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
            else "evaluation_strategy"
        )
        extra_eval_args = {eval_strategy_key: "steps" if self.args.eval_file else "no"}
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
            eval_steps=self.args.eval_steps if self.args.eval_file else None,
            load_best_model_at_end=bool(self.args.eval_file),
            fp16=self.args.use_fp16,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if self.args.use_wandb else None,
            logging_dir=f"{self.args.output_dir}/logs",
            **extra_eval_args,
        )
        use_gkd = getattr(self.args, "use_gkd", False)
        if use_gkd:
            from .gkd_adapter import AgentGKDTrainer, GKDTrainingConfig, verify_trl_version_pin

            verify_trl_version_pin(strict=False)
            gkd_config = GKDTrainingConfig.from_settings(
                distillation_alpha=self.args.distillation_alpha,
                enabled=True,
            )
            trainer = AgentGKDTrainer(
                model=student_model,
                teacher_model=teacher_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=self.prepare_eval_dataset() if self.args.eval_file else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                distillation_alpha=self.args.distillation_alpha,
                beta=getattr(self.args, "gkd_beta", None) or gkd_config.beta,
                temperature=getattr(self.args, "temperature", None) or gkd_config.temperature,
                lmbda=getattr(self.args, "gkd_lmbda", None) or gkd_config.lmbda,
                seq_kd=getattr(self.args, "gkd_seq_kd", None) or gkd_config.seq_kd,
                gkd_config=gkd_config,
            )
        else:
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
        # Local JSON file loading; Hub is not contacted (green-trunk-ci)
        dataset = load_dataset("json", data_files={"eval": self.args.eval_file})  # nosec: B615
        if getattr(self.args, "trajectory_mode", False):
            return self._filter_supervised_trajectory_rows(dataset["eval"], "eval")
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        student_outputs = model(**inputs)
        teacher_outputs = None
        if self.teacher_model is not None and self.distillation_alpha > 0:
            try:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
            except (RuntimeError, IndexError) as exc:
                raise ValueError(
                    "Teacher forward pass failed with student-tokenized inputs. "
                    "Use tokenizer-compatible teacher/student models or disable distillation."
                ) from exc
        loss = self.create_distillation_loss(student_outputs, teacher_outputs, inputs.get("labels"))
        return (loss, student_outputs) if return_outputs else loss

    def create_distillation_loss(self, student_outputs, teacher_outputs, labels):
        shifted_logits = student_outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        task_loss = torch.nn.functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
        )
        # Avoid NaN if all labels are masked (e.g. during certain trajectory segments)
        if torch.isnan(task_loss):
            task_loss = (shifted_logits * 0.0).sum()

        if self.distillation_alpha > 0 and teacher_outputs is not None:
            student_logits = student_outputs.logits[:, :-1, :] / self.temperature
            teacher_logits = teacher_outputs.logits[:, :-1, :] / self.temperature

            s_vocab = student_logits.size(-1)
            t_vocab = teacher_logits.size(-1)
            if s_vocab != t_vocab:
                import warnings

                warnings.warn(
                    f"Vocab mismatch: Student({s_vocab}) vs Teacher({t_vocab}). "
                    "Cannot safely compute KL divergence across disparate token spaces. "
                    "Falling back to task loss.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return task_loss

            # Compute KL divergence
            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction="none",
            ).sum(dim=-1)  # shape: (batch, seq_len)

            # Apply label mask: only compute KL on supervised positions
            label_mask = shifted_labels != -100
            if label_mask.any():
                kl_loss = (kl_loss * label_mask.float()).sum() / label_mask.float().sum()
            else:
                kl_loss = (kl_loss * 0.0).sum()

            distillation_loss = kl_loss * (self.temperature**2)
            return (
                1 - self.distillation_alpha
            ) * task_loss + self.distillation_alpha * distillation_loss
        return task_loss
