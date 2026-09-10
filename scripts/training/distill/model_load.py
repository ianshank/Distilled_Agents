"""Model loading helpers for distillation."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_trust_remote_code(args: Any) -> bool:
    return bool(getattr(args, "trust_remote_code", False))


def resolve_model_revision(args: Any) -> Optional[str]:
    """Optional Hub revision pin (env or CLI); None keeps the library default."""
    explicit = getattr(args, "model_revision", None)
    if explicit:
        return str(explicit)
    env_rev = os.getenv("MANGOMAS_MODEL_REVISION")
    return env_rev or None


def load_causal_lm(model_name: str, args: Any):
    import torch
    from transformers import AutoModelForCausalLM

    trust = resolve_trust_remote_code(args)
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
        device_map="auto" if args.use_device_map else None,
        trust_remote_code=trust,
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
