"""TRL GKD / on-policy distillation adapter module.

Integrates Generalized Knowledge Distillation (Agarwal et al., 2023) and on-policy
distillation APIs from TRL (pinned exact version trl==0.15.2) with MangoMAS
trajectory datasets, ensuring label-masked divergence and strict vocabulary alignment.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from enhanced_system.ops.settings import MangoMASSettings, get_settings

logger = logging.getLogger(__name__)

EXPECTED_TRL_VERSION = "0.15.2"


def is_trl_available() -> bool:
    """Check if TRL is installed in the current environment."""
    try:
        import trl  # noqa: F401

        return True
    except ImportError:
        return False


def get_trl_version() -> Optional[str]:
    """Return the installed TRL version, or None if not installed."""
    try:
        import trl

        return getattr(trl, "__version__", None)
    except ImportError:
        return None


def verify_trl_version_pin(strict: bool = False) -> str:
    """Verify that installed TRL matches the pinned exact minor version (0.15.2).

    Raises:
        ImportError: If TRL is not installed.
        ValueError: If installed TRL differs from EXPECTED_TRL_VERSION and strict is True.
    """
    version = get_trl_version()
    if version is None:
        raise ImportError(
            f"TRL is required for GKD training. Expected trl=={EXPECTED_TRL_VERSION}. "
            "Install with: pip install trl==0.15.2"
        )
    if version != EXPECTED_TRL_VERSION:
        msg = (
            f"Installed TRL version '{version}' differs from pinned '{EXPECTED_TRL_VERSION}'. "
            "On-policy GKD safety guarantees are validated against trl==0.15.2."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    return version


def validate_vocab_alignment(
    student_vocab_size: int,
    teacher_vocab_size: int,
    strict: bool = True,
) -> bool:
    """Validate that student and teacher vocabularies are identical.

    Cross-vocabulary KL / JSD divergence computation across different token spaces
    is mathematically invalid and unsafe.

    Args:
        student_vocab_size: Vocabulary size of student model or tokenizer.
        teacher_vocab_size: Vocabulary size of teacher model or tokenizer.
        strict: If True, raise ValueError on mismatch. If False, issue warning.

    Returns:
        True if vocabularies match, False otherwise.

    Raises:
        ValueError: If vocabs differ and strict=True.
    """
    if student_vocab_size != teacher_vocab_size:
        msg = (
            f"Incompatible teacher/student vocabularies for GKD distillation: "
            f"student_vocab={student_vocab_size}, teacher_vocab={teacher_vocab_size}. "
            "Cannot safely compute cross-vocab KL/JSD across disparate token spaces."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(
            f"{msg} Falling back to task loss without cross-vocab divergence.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


@dataclass
class GKDTrainingConfig:
    """Configuration for TRL Generalized Knowledge Distillation on trajectories."""

    enabled: bool = False
    lmbda: float = 0.5
    beta: float = 0.5
    temperature: float = 0.9
    max_new_tokens: int = 128
    seq_kd: bool = False
    distillation_alpha: float = 0.0
    expected_trl_version: str = EXPECTED_TRL_VERSION

    @classmethod
    def from_settings(
        cls,
        settings: Optional[MangoMASSettings] = None,
        distillation_alpha: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> GKDTrainingConfig:
        """Create GKD configuration from operational settings."""
        cfg_settings = settings or get_settings()
        alpha = (
            distillation_alpha
            if distillation_alpha is not None
            else cfg_settings.trajectory_distill_alpha
        )
        is_enabled = enabled if enabled is not None else cfg_settings.gkd_enabled
        return cls(
            enabled=is_enabled,
            lmbda=cfg_settings.gkd_lmbda,
            beta=cfg_settings.gkd_beta,
            temperature=cfg_settings.gkd_temperature,
            max_new_tokens=cfg_settings.gkd_max_new_tokens,
            seq_kd=cfg_settings.gkd_seq_kd,
            distillation_alpha=alpha,
        )


def compute_label_masked_gkd_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> Any:
    """Compute Generalized JSD or KL divergence strictly masked to supervised label positions.

    Invariants:
    1. If student and teacher vocabulary sizes mismatch, issues a RuntimeWarning
       and returns pure task cross-entropy loss without computing cross-vocab divergence.
    2. Divergence is computed ONLY on positions where labels != -100 (assistant spans).
    3. If all labels are masked (-100), returns zero loss.

    Args:
        student_logits: Tensor of shape (batch_size, seq_len, student_vocab)
        teacher_logits: Tensor of shape (batch_size, seq_len, teacher_vocab)
        labels: Tensor of shape (batch_size, seq_len) with -100 for non-supervised tokens
        beta: Interpolation coefficient: 0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL
        temperature: Softmax temperature

    Returns:
        Scalar loss tensor.
    """
    import torch
    import torch.nn.functional as F

    s_vocab = student_logits.size(-1)
    t_vocab = teacher_logits.size(-1)
    if s_vocab != t_vocab:
        warnings.warn(
            f"Vocab mismatch: Student({s_vocab}) vs Teacher({t_vocab}). "
            "Cannot safely compute cross-vocab divergence. Falling back to task loss.",
            RuntimeWarning,
            stacklevel=2,
        )
        task_loss = F.cross_entropy(
            student_logits.view(-1, s_vocab),
            labels.view(-1),
            ignore_index=-100,
        )
        if torch.isnan(task_loss):
            task_loss = (student_logits * 0.0).sum()
        return task_loss

    scaled_s_logits = student_logits / temperature
    scaled_t_logits = teacher_logits / temperature

    s_log_probs = F.log_softmax(scaled_s_logits, dim=-1)
    t_log_probs = F.log_softmax(scaled_t_logits, dim=-1)

    label_mask = labels != -100
    if not label_mask.any():
        return (student_logits * 0.0).sum()

    if beta == 0.0:
        # Forward KL: KL(teacher || student)
        kl = F.kl_div(s_log_probs, t_log_probs, reduction="none", log_target=True).sum(dim=-1)
        masked_loss = (kl * label_mask.float()).sum() / label_mask.float().sum()
        return masked_loss * (temperature**2)
    elif beta == 1.0:
        # Reverse KL: KL(student || teacher)
        kl = F.kl_div(t_log_probs, s_log_probs, reduction="none", log_target=True).sum(dim=-1)
        masked_loss = (kl * label_mask.float()).sum() / label_mask.float().sum()
        return masked_loss * (temperature**2)
    else:
        # Generalized Jensen-Shannon Divergence
        beta_tensor = torch.tensor(beta, dtype=s_log_probs.dtype, device=s_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    s_log_probs + torch.log(beta_tensor),
                    t_log_probs + torch.log(1.0 - beta_tensor),
                ]
            ),
            dim=0,
        )
        kl_t = F.kl_div(mixture_log_probs, t_log_probs, reduction="none", log_target=True).sum(
            dim=-1
        )
        kl_s = F.kl_div(mixture_log_probs, s_log_probs, reduction="none", log_target=True).sum(
            dim=-1
        )
        jsd = beta * kl_t + (1.0 - beta) * kl_s
        masked_loss = (jsd * label_mask.float()).sum() / label_mask.float().sum()
        return masked_loss * (temperature**2)


def create_agent_gkd_trainer(
    student_model: Any,
    teacher_model: Any,
    training_args: Any,
    train_dataset: Any,
    tokenizer: Any,
    data_collator: Any,
    gkd_config: GKDTrainingConfig,
    eval_dataset: Optional[Any] = None,
) -> Any:
    """Factory creating an AgentGKDTrainer instance.

    Guards on TRL availability and enforces vocabulary safety.
    """
    if not is_trl_available():
        raise ImportError(
            f"TRL is required to create AgentGKDTrainer. Expected trl=={EXPECTED_TRL_VERSION}. "
            "Install with: pip install trl==0.15.2"
        )

    from scripts.training.distill.gkd_adapter import AgentGKDTrainer

    return AgentGKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        gkd_config=gkd_config,
    )
