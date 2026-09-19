"""Data collator and dataset formatting for Direct Preference Optimization (DPO)."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def format_dpo_example(example: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """Formats a raw preference pair into a DPO-compatible dict.

    Expects `example` to have:
    - prompt (or history)
    - chosen (the accepted assistant trajectory)
    - rejected (the failed or suboptimal trajectory)

    Returns a dict with 'prompt', 'chosen', and 'rejected' strings.
    """
    # Extract common context
    context_msgs = example.get("history", [])
    if "prompt" in example and isinstance(example["prompt"], str):
        # Allow simple string prompts
        context_msgs.append({"role": "user", "content": example["prompt"]})

    if not context_msgs:
        raise ValueError("DPO example must contain 'history' or 'prompt' context.")

    try:
        # Format the shared context as the 'prompt' field for DPOTrainer
        prompt_str = tokenizer.apply_chat_template(
            context_msgs,
            tokenize=False,
            add_generation_prompt=True
        )

        # Format the chosen and rejected continuations
        # DPO requires the continuations to NOT contain the prompt.
        # We wrap them in temporary messages to get the correct role formatting.

        chosen_msg = example.get("chosen", "")
        if isinstance(chosen_msg, str):
            chosen_msg = [{"role": "assistant", "content": chosen_msg}]
        elif isinstance(chosen_msg, dict):
            chosen_msg = [chosen_msg]

        rejected_msg = example.get("rejected", "")
        if isinstance(rejected_msg, str):
            rejected_msg = [{"role": "assistant", "content": rejected_msg}]
        elif isinstance(rejected_msg, dict):
            rejected_msg = [rejected_msg]

        chosen_str = tokenizer.apply_chat_template(chosen_msg, tokenize=False)
        rejected_str = tokenizer.apply_chat_template(rejected_msg, tokenize=False)

        # apply_chat_template might add BOS tokens to the continuations. We strip them
        # if they duplicate what the prompt already ends with, but trl DPOTrainer
        # is fairly robust to this if the tokenizer's chat template is standard.

        return {
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str
        }
    except Exception as e:
        logger.error("Failed to format DPO example: %s", e)
        raise
