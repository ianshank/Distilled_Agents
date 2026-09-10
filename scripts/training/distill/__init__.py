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
]


def __getattr__(name: str):
    # Trainer imports torch; keep that optional for lightweight unit tests.
    if name in {"AgentDistillationTrainer", "DistillationTrainer"}:
        from .trainer import AgentDistillationTrainer, DistillationTrainer

        return (
            AgentDistillationTrainer if name == "AgentDistillationTrainer" else DistillationTrainer
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
