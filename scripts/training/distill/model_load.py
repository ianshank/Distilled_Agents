"""Model loading helpers for distillation."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _optional_settings():
    """SageMaker images may not have enhanced_system on PYTHONPATH."""
    try:
        from enhanced_system.ops.settings import get_settings

        return get_settings()
    except Exception:
        return None


def resolve_trust_remote_code(args: Any) -> bool:
    if hasattr(args, "trust_remote_code") and args.trust_remote_code is not None:
        return bool(args.trust_remote_code)
    settings = _optional_settings()
    if settings is not None:
        return bool(settings.trust_remote_code)
    return os.getenv("MANGOMAS_TRUST_REMOTE_CODE", "false").lower() == "true"


def resolve_model_revision(args: Any) -> Optional[str]:
    """Optional Hub revision pin (CLI, settings, or env); None keeps library default."""
    explicit = getattr(args, "model_revision", None)
    if explicit:
        return str(explicit)
    settings = _optional_settings()
    if settings is not None and settings.model_revision:
        return settings.model_revision
    env_rev = os.getenv("MANGOMAS_MODEL_REVISION")
    return env_rev or None


def load_causal_lm(model_name: str, args: Any):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
        device_map="auto" if args.use_device_map else None,
        trust_remote_code=resolve_trust_remote_code(args),
        revision=resolve_model_revision(args),
    )


def load_tokenizer(model_name: str, args: Any):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=resolve_trust_remote_code(args),
        revision=resolve_model_revision(args),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
