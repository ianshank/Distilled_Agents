"""Masked LM collator for agent trajectories (SageMaker source_dir safe)."""

from __future__ import annotations

from typing import Any, Iterable

from .prompt_render import supervised_spans


def _encode(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids) if ids is not None else []


def encode_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    """Build input_ids and labels from one rendered body (not per-chunk encode)."""
    spans = _row_spans(row)
    body = "".join(text for text, _supervised in spans)
    input_ids, offsets = _tokenize_body(tokenizer, body, max_length)
    labels = _labels_from_spans(input_ids, body, spans, offsets, max_length)
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


def _row_spans(row: dict[str, Any]) -> list[tuple[str, bool]]:
    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or "")
    trajectory = row.get("trajectory") or {}
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    instruction = str(row.get("instruction") or "")
    if isinstance(trajectory, dict) and not instruction:
        instruction = str(trajectory.get("instruction") or "")
    if steps:
        return supervised_spans(prompt, steps, instruction=instruction)
    stub = [{"thought": "", "action": completion, "observation": "", "tool_id": "final_answer"}]
    return supervised_spans(prompt, stub, instruction=instruction)


def _tokenize_body(
    tokenizer: Any,
    body: str,
    max_length: int,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    if not body:
        return [], []
    try:
        encoded = tokenizer(
            body,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    except TypeError:
        return _encode(tokenizer, body)[:max_length], None
    ids = list(encoded["input_ids"])[:max_length]
    raw_offsets = None
    if hasattr(encoded, "get"):
        raw_offsets = encoded.get("offset_mapping")
    if raw_offsets is None:
        raw_offsets = getattr(encoded, "offset_mapping", None)
    if not raw_offsets:
        return ids, None
    offsets = [(int(start), int(end)) for start, end in list(raw_offsets)[: len(ids)]]
    return ids, offsets


def _labels_from_spans(
    input_ids: list[int],
    body: str,
    spans: Iterable[tuple[str, bool]],
    offsets: list[tuple[int, int]] | None,
    max_length: int,
) -> list[int]:
    mask = _supervised_chars(body, spans)
    if offsets is not None:
        labels: list[int] = []
        for token_id, (start, end) in zip(input_ids, offsets):
            if start >= end or not any(mask[start:end]):
                labels.append(-100)
            else:
                labels.append(token_id)
        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        return labels
    if len(input_ids) <= len(mask) and len(input_ids) == min(len(body), max_length):
        return [token_id if mask[index] else -100 for index, token_id in enumerate(input_ids)]
    return [-100] * len(input_ids)


def _supervised_chars(body: str, spans: Iterable[tuple[str, bool]]) -> list[bool]:
    mask: list[bool] = []
    for text, supervised in spans:
        mask.extend([supervised] * len(text))
    if len(mask) < len(body):
        mask.extend([False] * (len(body) - len(mask)))
    return mask[: len(body)]
