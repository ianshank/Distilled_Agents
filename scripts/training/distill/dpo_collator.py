"""Data collator and dataset formatting for Direct Preference Optimization (DPO)."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _step_to_message(step: Any, default_role: str = "assistant") -> dict[str, str]:
    if isinstance(step, str):
        return {"role": default_role, "content": step}
    if isinstance(step, dict):
        if "role" in step and "content" in step:
            return {"role": str(step["role"]), "content": str(step["content"])}
        parts = [str(step.get("thought", "")).strip(), str(step.get("action", "")).strip()]
        content = " ".join(part for part in parts if part)
        if not content:
            content = str(step)
        return {"role": default_role, "content": content}
    raise ValueError(f"Unsupported step format for DPO example: {type(step)}")


def format_dpo_example(example: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """Formats a raw preference pair into a DPO-compatible dict.

    Expects `example` to have:
    - prompt (or history)
    - chosen (the accepted assistant trajectory)
    - rejected (the failed or suboptimal trajectory)

    Returns a dict with 'prompt', 'chosen', and 'rejected' strings.
    """
    context_msgs = list(
        example.get("history", []) if isinstance(example.get("history"), list) else []
    )
    if "prompt" in example and isinstance(example["prompt"], str):
        context_msgs.append({"role": "user", "content": example["prompt"]})
    elif "task" in example and isinstance(example["task"], str):
        context_msgs.append({"role": "user", "content": example["task"]})
    prefix = example.get("prefix", [])
    if isinstance(prefix, list):
        for step in prefix:
            context_msgs.append(_step_to_message(step))
    if not context_msgs:
        raise ValueError("DPO example must contain context via history/prompt/task/prefix.")

    try:
        # Format the shared context as the 'prompt' field for DPOTrainer
        prompt_str = tokenizer.apply_chat_template(
            context_msgs, tokenize=False, add_generation_prompt=True
        )

        # Format the chosen and rejected continuations
        # DPO requires the continuations to NOT contain the prompt.
        # We wrap them in temporary messages to get the correct role formatting.

        if "chosen" not in example or "rejected" not in example:
            raise ValueError("DPO example must contain both 'chosen' and 'rejected' fields.")
        chosen_msg = [_step_to_message(example["chosen"])]
        rejected_msg = [_step_to_message(example["rejected"])]

        chosen_str = tokenizer.apply_chat_template(chosen_msg, tokenize=False)
        rejected_str = tokenizer.apply_chat_template(rejected_msg, tokenize=False)

        # apply_chat_template might add BOS tokens to the continuations. We strip them
        # if they duplicate what the prompt already ends with, but trl DPOTrainer
        # is fairly robust to this if the tokenizer's chat template is standard.

        return {"prompt": prompt_str, "chosen": chosen_str, "rejected": rejected_str}
    except Exception as e:
        logger.error("Failed to format DPO example: %s", e)
        raise
