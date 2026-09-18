"""Model loading helpers for distillation."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_trust_remote_code(args: Any) -> bool:
    """Resolves trust_remote_code strictly.

    SDLC DevSecOps Guardrail: Remote code execution is disabled by default.
    It can ONLY be enabled if MANGOMAS_OVERRIDE_TRUST_REMOTE_CODE=I_KNOW_THIS_IS_UNSAFE
    is set in the environment, overriding args.
    """
    arg_trust = bool(getattr(args, "trust_remote_code", False))
    escape_hatch = os.getenv("MANGOMAS_OVERRIDE_TRUST_REMOTE_CODE") == "I_KNOW_THIS_IS_UNSAFE"

    if arg_trust or escape_hatch:
        if not escape_hatch:
            logger.warning(
                "SECURITY: trust_remote_code was requested via args, but is blocked by SDLC guardrails. "
                "Set MANGOMAS_OVERRIDE_TRUST_REMOTE_CODE=I_KNOW_THIS_IS_UNSAFE to override."
            )
            return False

        logger.warning(
            "SECURITY [DANGER]: trust_remote_code is ENABLED via override. "
            "Arbitrary code from model repos will be executed on this machine!"
        )
        return True

    return False


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
    return AutoModelForCausalLM.from_pretrained(  # nosec B615
        model_name,
        torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
        device_map="auto" if args.use_device_map else None,
        trust_remote_code=trust,
        revision=resolve_model_revision(args),
    )


def load_tokenizer(model_name: str, args: Any):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        model_name,
        trust_remote_code=resolve_trust_remote_code(args),
        revision=resolve_model_revision(args),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "is_fast", True) is False:
        logger.warning(
            "slow tokenizer %s has no offset mapping; trajectory SFT needs a fast tokenizer",
            model_name,
        )
    return tokenizer
