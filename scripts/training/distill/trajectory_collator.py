"""Masked LM collator for agent trajectories (SageMaker source_dir safe)."""

from __future__ import annotations

from typing import Any


def _encode(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids) if ids is not None else []


def encode_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    """Build input_ids and labels; observations and prompt are unlabeled."""
    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or "")
    trajectory = row.get("trajectory") or {}
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    input_ids = _encode(tokenizer, prompt)
    labels = [-100] * len(input_ids)
    if steps:
        for step in steps:
            thought = str(step.get("thought") or "")
            action = str(step.get("action") or "")
            labeled = " ".join(part for part in (thought, action) if part).strip()
            labeled_ids = _encode(tokenizer, labeled)
            input_ids.extend(labeled_ids)
            labels.extend(labeled_ids)
            observation = str(step.get("observation") or "")
            obs_ids = _encode(tokenizer, observation)
            input_ids.extend(obs_ids)
            labels.extend([-100] * len(obs_ids))
    else:
        labeled_ids = _encode(tokenizer, completion)
        input_ids.extend(labeled_ids)
        labels.extend(labeled_ids)
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
