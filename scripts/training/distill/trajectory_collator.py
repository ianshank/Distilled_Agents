"""Masked LM collator for agent trajectories (SageMaker source_dir safe)."""

from __future__ import annotations

from typing import Any

from .prompt_render import supervised_spans


def _encode(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids) if ids is not None else []


def encode_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    """Build input_ids and labels; user/obs/fault turns unlabeled, assistant labeled."""
    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or "")
    trajectory = row.get("trajectory") or {}
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    instruction = str(row.get("instruction") or "")
    if isinstance(trajectory, dict) and not instruction:
        instruction = str(trajectory.get("instruction") or "")
    input_ids: list[int] = []
    labels: list[int] = []
    if steps:
        for text, supervised in supervised_spans(prompt, steps, instruction=instruction):
            token_ids = _encode(tokenizer, text)
            input_ids.extend(token_ids)
            labels.extend(token_ids if supervised else [-100] * len(token_ids))
    else:
        for text, supervised in supervised_spans(
            prompt,
            [{"thought": "", "action": completion, "observation": "", "tool_id": "final_answer"}],
            instruction=instruction,
        ):
            token_ids = _encode(tokenizer, text)
            input_ids.extend(token_ids)
            labels.extend(token_ids if supervised else [-100] * len(token_ids))
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def has_supervised_tokens(tokenizer: Any, row: dict[str, Any], max_length: int) -> bool:
    encoded = encode_row(tokenizer, row, max_length)
    return any(label != -100 for label in encoded["labels"])


class TrajectoryDataCollator:
    """Preserve observation masks; do not clone labels from input_ids."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        pad_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = 0 if pad_id is None else pad_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = [encode_row(self.tokenizer, row, self.max_length) for row in features]
        max_len = max(len(item["input_ids"]) for item in encoded) if encoded else 0
        input_ids = []
        labels = []
        attention = []
        for item in encoded:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        try:
            import torch
        except ImportError:  # pragma: no cover
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention,
            }
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
