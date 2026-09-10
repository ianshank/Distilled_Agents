"""
Adaptive Training Configuration Module
Provides adaptive training configurations with early stopping, LR scheduling, etc.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration"""

    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.001
    monitor: str = "val_loss"
    mode: str = "min"  # 'min' or 'max'


@dataclass
class LearningRateSchedule:
    """Learning rate schedule configuration"""

    strategy: str = "cosine_annealing"  # cosine_annealing, linear, exponential, constant
    warmup_steps: int = 1000
    min_lr: float = 1e-7
    max_lr: float = 5e-5
    cycle_length: Optional[int] = None


@dataclass
class GradientConfig:
    """Gradient configuration"""

    accumulation_steps: int = 4
    clip_value: float = 1.0
    clip_norm: bool = True


@dataclass
class MixedPrecisionConfig:
    """Mixed precision training configuration"""

    enabled: bool = True
    precision: str = "bf16"  # bf16, fp16, fp32
    loss_scale: str = "dynamic"


@dataclass
class CheckpointingConfig:
    """Checkpointing strategy configuration"""

    strategy: str = "best_and_periodic"  # best_only, periodic_only, best_and_periodic
    save_every_n_steps: int = 1000
    keep_best_n: int = 3
    save_optimizer_state: bool = True
    save_on_train_end: bool = True


@dataclass
class AdaptiveTrainingConfig:
    """
    Comprehensive adaptive training configuration

    Features:
    - Early stopping
    - Learning rate scheduling
    - Gradient accumulation and clipping
    - Mixed precision training
    - Smart checkpointing
    """

    # Basic training parameters
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01

    # Adaptive configurations
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    lr_schedule: LearningRateSchedule = field(default_factory=LearningRateSchedule)
    gradient_config: GradientConfig = field(default_factory=GradientConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)

    # Additional options
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    seed: int = 42
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AdaptiveTrainingConfig":
        """
        Create configuration from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            AdaptiveTrainingConfig object
        """
        # Extract nested configs
        early_stopping_dict = config_dict.pop("early_stopping", {})
        lr_schedule_dict = config_dict.pop("lr_schedule", {})
        gradient_dict = config_dict.pop("gradient_config", {})
        mixed_precision_dict = config_dict.pop("mixed_precision", {})
        checkpointing_dict = config_dict.pop("checkpointing", {})

        # Create nested config objects
        early_stopping = EarlyStoppingConfig(**early_stopping_dict)
        lr_schedule = LearningRateSchedule(**lr_schedule_dict)
        gradient_config = GradientConfig(**gradient_dict)
        mixed_precision = MixedPrecisionConfig(**mixed_precision_dict)
        checkpointing = CheckpointingConfig(**checkpointing_dict)

        # Create main config
        return cls(
            early_stopping=early_stopping,
            lr_schedule=lr_schedule,
            gradient_config=gradient_config,
            mixed_precision=mixed_precision,
            checkpointing=checkpointing,
            **config_dict,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "early_stopping": {
                "enabled": self.early_stopping.enabled,
                "patience": self.early_stopping.patience,
                "min_delta": self.early_stopping.min_delta,
                "monitor": self.early_stopping.monitor,
                "mode": self.early_stopping.mode,
            },
            "lr_schedule": {
                "strategy": self.lr_schedule.strategy,
                "warmup_steps": self.lr_schedule.warmup_steps,
                "min_lr": self.lr_schedule.min_lr,
                "max_lr": self.lr_schedule.max_lr,
                "cycle_length": self.lr_schedule.cycle_length,
            },
            "gradient_config": {
                "accumulation_steps": self.gradient_config.accumulation_steps,
                "clip_value": self.gradient_config.clip_value,
                "clip_norm": self.gradient_config.clip_norm,
            },
            "mixed_precision": {
                "enabled": self.mixed_precision.enabled,
                "precision": self.mixed_precision.precision,
                "loss_scale": self.mixed_precision.loss_scale,
            },
            "checkpointing": {
                "strategy": self.checkpointing.strategy,
                "save_every_n_steps": self.checkpointing.save_every_n_steps,
                "keep_best_n": self.checkpointing.keep_best_n,
                "save_optimizer_state": self.checkpointing.save_optimizer_state,
                "save_on_train_end": self.checkpointing.save_on_train_end,
            },
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "pin_memory": self.pin_memory,
        }

    def get_effective_batch_size(self) -> int:
        """Get effective batch size with gradient accumulation"""
        return self.batch_size * self.gradient_config.accumulation_steps

    def calculate_lr(self, step: int, total_steps: int) -> float:
        """
        Calculate learning rate for given step

        Args:
            step: Current training step
            total_steps: Total training steps

        Returns:
            Learning rate
        """
        if self.lr_schedule.strategy == "constant":
            return self.learning_rate

        # Warmup phase
        if step < self.lr_schedule.warmup_steps:
            warmup_factor = step / max(1, self.lr_schedule.warmup_steps)
            return self.lr_schedule.max_lr * warmup_factor

        # After warmup
        if self.lr_schedule.strategy == "cosine_annealing":
            import math

            progress = (step - self.lr_schedule.warmup_steps) / max(
                1, total_steps - self.lr_schedule.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = (
                self.lr_schedule.min_lr
                + (self.lr_schedule.max_lr - self.lr_schedule.min_lr) * cosine_decay
            )
            return lr

        elif self.lr_schedule.strategy == "linear":
            progress = (step - self.lr_schedule.warmup_steps) / max(
                1, total_steps - self.lr_schedule.warmup_steps
            )
            lr = (
                self.lr_schedule.max_lr
                - (self.lr_schedule.max_lr - self.lr_schedule.min_lr) * progress
            )
            return max(lr, self.lr_schedule.min_lr)

        elif self.lr_schedule.strategy == "exponential":
            decay_steps = total_steps - self.lr_schedule.warmup_steps
            decay_rate = (self.lr_schedule.min_lr / self.lr_schedule.max_lr) ** (1.0 / decay_steps)
            lr = self.lr_schedule.max_lr * (decay_rate ** (step - self.lr_schedule.warmup_steps))
            return max(lr, self.lr_schedule.min_lr)

        return self.learning_rate

    def should_save_checkpoint(self, step: int) -> bool:
        """
        Check if checkpoint should be saved at this step

        Args:
            step: Current training step

        Returns:
            True if should save checkpoint
        """
        if self.checkpointing.strategy in ["periodic_only", "best_and_periodic"]:
            return step % self.checkpointing.save_every_n_steps == 0

        return False

    def log_config(self):
        """Log configuration details"""
        logger.info("=== Adaptive Training Configuration ===")
        logger.info(f"Batch size: {self.batch_size} (effective: {self.get_effective_batch_size()})")
        logger.info(f"Learning rate: {self.learning_rate} (schedule: {self.lr_schedule.strategy})")
        logger.info(
            f"Early stopping: {self.early_stopping.enabled} (patience: {self.early_stopping.patience})"
        )
        logger.info(
            f"Mixed precision: {self.mixed_precision.enabled} ({self.mixed_precision.precision})"
        )
        logger.info(f"Gradient accumulation: {self.gradient_config.accumulation_steps} steps")
        logger.info(f"Checkpointing: {self.checkpointing.strategy}")
        logger.info("=" * 40)


def create_default_config() -> AdaptiveTrainingConfig:
    """Create default training configuration"""
    return AdaptiveTrainingConfig()


def create_fast_config() -> AdaptiveTrainingConfig:
    """Create configuration optimized for speed"""
    config = AdaptiveTrainingConfig()
    config.batch_size = 32
    config.gradient_config.accumulation_steps = 2
    config.mixed_precision.enabled = True
    config.mixed_precision.precision = "bf16"
    config.checkpointing.save_every_n_steps = 2000
    return config


def create_quality_config() -> AdaptiveTrainingConfig:
    """Create configuration optimized for quality"""
    config = AdaptiveTrainingConfig()
    config.batch_size = 8
    config.num_epochs = 10
    config.gradient_config.accumulation_steps = 8
    config.early_stopping.patience = 10
    config.lr_schedule.warmup_steps = 2000
    config.checkpointing.keep_best_n = 5
    return config


def create_memory_efficient_config() -> AdaptiveTrainingConfig:
    """Create configuration optimized for memory efficiency"""
    config = AdaptiveTrainingConfig()
    config.batch_size = 4
    config.gradient_config.accumulation_steps = 16
    config.mixed_precision.enabled = True
    config.mixed_precision.precision = "fp16"
    config.dataloader_num_workers = 2
    config.checkpointing.save_optimizer_state = False
    return config
