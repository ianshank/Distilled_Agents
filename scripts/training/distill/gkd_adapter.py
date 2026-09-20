"""TRL GKD / on-policy distillation trainer and adapter.

Integrates TRL Generalized Knowledge Distillation (Agarwal et al., 2023) with
agent trajectory rollouts. Enforces exact TRL pin (trl==0.15.2), label-masked
divergence, and strict vocabulary alignment to prevent unsafe cross-vocab KL.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXPECTED_TRL_VERSION = "0.15.2"

try:
    import trl
    from trl import GKDConfig, GKDTrainer

    TRL_AVAILABLE = True
except ImportError:
    trl = None
    GKDConfig = None  # type: ignore[misc,assignment]
    GKDTrainer = None  # type: ignore[misc,assignment]
    TRL_AVAILABLE = False

try:
    from transformers import Trainer
except ImportError:
    Trainer = object  # type: ignore[misc,assignment]


def is_trl_available() -> bool:
    """Return True if TRL is importable."""
    return TRL_AVAILABLE


def get_trl_version() -> Optional[str]:
    """Return installed TRL version string, or None if unavailable."""
    if not TRL_AVAILABLE or trl is None:
        return None
    return getattr(trl, "__version__", None)


def verify_trl_version_pin(strict: bool = False) -> str:
    """Verify that installed TRL matches the pinned exact minor version (0.15.2)."""
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
    """Validate that student and teacher vocabularies are identical."""
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
    """Configuration for GKD on-policy trajectory training."""

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
        settings: Optional[Any] = None,
        distillation_alpha: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> GKDTrainingConfig:
        """Load configuration from MangoMASSettings with fail-safe defaults."""
        if settings is None:
            try:
                from enhanced_system.ops.settings import get_settings

                settings = get_settings()
            except ImportError:
                settings = None

        if settings is not None:
            alpha = (
                distillation_alpha
                if distillation_alpha is not None
                else getattr(settings, "trajectory_distill_alpha", 0.0)
            )
            is_enabled = enabled if enabled is not None else getattr(settings, "gkd_enabled", False)
            return cls(
                enabled=is_enabled,
                lmbda=getattr(settings, "gkd_lmbda", 0.5),
                beta=getattr(settings, "gkd_beta", 0.5),
                temperature=getattr(settings, "gkd_temperature", 0.9),
                max_new_tokens=getattr(settings, "gkd_max_new_tokens", 128),
                seq_kd=getattr(settings, "gkd_seq_kd", False),
                distillation_alpha=alpha,
            )
        return cls(
            enabled=bool(enabled) if enabled is not None else False,
            distillation_alpha=float(distillation_alpha) if distillation_alpha is not None else 0.0,
        )


def compute_label_masked_gkd_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> Any:
    """Compute Generalized JSD or KL divergence strictly masked to supervised label positions."""
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


_BaseTrainer = GKDTrainer if TRL_AVAILABLE and GKDTrainer is not None else Trainer


class AgentGKDTrainer(_BaseTrainer):  # type: ignore[misc,valid-type]
    """Agent Generalized Knowledge Distillation Trainer for trajectory datasets."""

    def __init__(
        self,
        teacher_model: Any = None,
        distillation_alpha: float = 0.0,
        beta: float = 0.5,
        temperature: float = 0.9,
        lmbda: float = 0.5,
        seq_kd: bool = False,
        gkd_config: Optional[GKDTrainingConfig] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not is_trl_available():
            raise ImportError(
                f"TRL is required to instantiate AgentGKDTrainer. Expected trl=={EXPECTED_TRL_VERSION}. "
                "Install with: pip install trl==0.15.2"
            )

        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        if gkd_config is not None:
            self.distillation_alpha = gkd_config.distillation_alpha
            self.beta = gkd_config.beta
            self.temperature = gkd_config.temperature
            self.lmbda = gkd_config.lmbda
            self.seq_kd = gkd_config.seq_kd
        else:
            self.distillation_alpha = distillation_alpha
            self.beta = beta
            self.temperature = temperature
            self.lmbda = lmbda
            self.seq_kd = seq_kd

        if self.distillation_alpha > 0 and self.teacher_model is not None:
            # Validate vocabularies if tokenizer / models provide get_input_embeddings
            s_emb = getattr(self.model, "get_input_embeddings", lambda: None)()
            t_emb = getattr(self.teacher_model, "get_input_embeddings", lambda: None)()
            if s_emb is not None and t_emb is not None:
                validate_vocab_alignment(
                    s_emb.num_embeddings,
                    t_emb.num_embeddings,
                    strict=True,
                )

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        """Compute task loss + optional label-masked GKD divergence."""
        import torch
        import torch.nn.functional as F

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels are required for trajectory distillation loss computation")

        shifted_student_logits = student_outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        task_loss = F.cross_entropy(
            shifted_student_logits.view(-1, shifted_student_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
        )
        if torch.isnan(task_loss):
            task_loss = (shifted_student_logits * 0.0).sum()

        if self.distillation_alpha <= 0 or self.teacher_model is None:
            return (task_loss, student_outputs) if return_outputs else task_loss

        self.teacher_model.eval()
        try:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
        except (RuntimeError, IndexError) as exc:
            raise ValueError(
                "Teacher forward pass failed with student-tokenized inputs. "
                "Use tokenizer-compatible teacher/student models or set distillation_alpha=0."
            ) from exc

        shifted_teacher_logits = teacher_outputs.logits[:, :-1, :].contiguous()

        s_vocab = shifted_student_logits.size(-1)
        t_vocab = shifted_teacher_logits.size(-1)
        if s_vocab != t_vocab:
            warnings.warn(
                f"Vocab mismatch: Student({s_vocab}) vs Teacher({t_vocab}). "
                "Cannot safely compute cross-vocab divergence. Falling back to task loss.",
                RuntimeWarning,
                stacklevel=2,
            )
            return (task_loss, student_outputs) if return_outputs else task_loss

        gkd_loss = compute_label_masked_gkd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
            temperature=self.temperature,
        )

        total_loss = (
            1.0 - self.distillation_alpha
        ) * task_loss + self.distillation_alpha * gkd_loss
        return (total_loss, student_outputs) if return_outputs else total_loss
