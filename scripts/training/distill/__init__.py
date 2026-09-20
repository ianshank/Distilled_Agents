"""Distillation package (SageMaker source_dir and installed-package compatible)."""

from .model_load import load_causal_lm, load_tokenizer
from .sagemaker_io import resolve_train_file, texts_from_examples

__all__ = [
    "load_causal_lm",
    "load_tokenizer",
    "resolve_train_file",
    "texts_from_examples",
    "AgentDistillationTrainer",
    "DistillationTrainer",
    "AgentGKDTrainer",
    "GKDTrainingConfig",
    "compute_label_masked_gkd_loss",
    "validate_vocab_alignment",
    "verify_trl_version_pin",
    "EXPECTED_TRL_VERSION",
]


def __getattr__(name: str):
    # Trainer imports torch; keep that optional for lightweight unit tests.
    if name in {"AgentDistillationTrainer", "DistillationTrainer"}:
        from .trainer import AgentDistillationTrainer, DistillationTrainer

        return (
            AgentDistillationTrainer if name == "AgentDistillationTrainer" else DistillationTrainer
        )
    if name in {
        "AgentGKDTrainer",
        "GKDTrainingConfig",
        "compute_label_masked_gkd_loss",
        "validate_vocab_alignment",
        "verify_trl_version_pin",
        "EXPECTED_TRL_VERSION",
    }:
        from .gkd_adapter import (
            EXPECTED_TRL_VERSION,
            AgentGKDTrainer,
            GKDTrainingConfig,
            compute_label_masked_gkd_loss,
            validate_vocab_alignment,
            verify_trl_version_pin,
        )

        _map = {
            "AgentGKDTrainer": AgentGKDTrainer,
            "GKDTrainingConfig": GKDTrainingConfig,
            "compute_label_masked_gkd_loss": compute_label_masked_gkd_loss,
            "validate_vocab_alignment": validate_vocab_alignment,
            "verify_trl_version_pin": verify_trl_version_pin,
            "EXPECTED_TRL_VERSION": EXPECTED_TRL_VERSION,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
