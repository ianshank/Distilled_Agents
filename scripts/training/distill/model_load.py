"""Model loading helpers for distillation."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_trust_remote_code(args: Any) -> bool:
    """Resolves trust_remote_code strictly.
    SDLC DevSecOps Guardrail: Remote code execution is disabled by default.
    It adheres to AGENTS.md contract via MANGOMAS_TRUST_REMOTE_CODE=true.
    """
    from enhanced_system.ops.settings import get_settings

    arg_trust = bool(getattr(args, "trust_remote_code", False))
    settings_trust = get_settings().trust_remote_code

    # Settings from MANGOMAS_TRUST_REMOTE_CODE environment variable take precedence
    # over CLI arguments to ensure operators have deterministic global control.
    effective_trust = settings_trust or arg_trust

    if effective_trust:
        if not settings_trust:
            logger.warning(
                "SECURITY: trust_remote_code was requested via args. "
                "Ensure MANGOMAS_TRUST_REMOTE_CODE=true is set if execution fails."
            )
        else:
            logger.warning(
                "SECURITY [DANGER]: trust_remote_code is ENABLED via MANGOMAS_TRUST_REMOTE_CODE. "
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
