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
    if arg_trust and not settings_trust:
        logger.warning(
            "Ignoring --trust_remote_code because MANGOMAS_TRUST_REMOTE_CODE is not enabled."
        )
    if settings_trust:
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
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16 if args.use_fp16 else torch.float32,
        "device_map": "auto" if args.use_device_map else None,
        "trust_remote_code": trust,
        "revision": resolve_model_revision(args),
    }
    if getattr(args, "quantize_4bit", False):
        from transformers import BitsAndBytesConfig

        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "quantize_4bit=True requires bitsandbytes. Install with the quantization extra."
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if args.use_fp16 else torch.float32,
        )
        kwargs["device_map"] = "auto"
        logger.info("4-bit quantization enabled via BitsAndBytesConfig")
    return AutoModelForCausalLM.from_pretrained(  # nosec B615
        model_name,
        **kwargs,
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
