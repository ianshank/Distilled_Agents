"""SageMaker channel and dataset I/O."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CHANNEL_ENV_VARS = ("SM_CHANNEL_TRAINING", "SM_CHANNEL_TRAIN")
_DEFAULT_CHANNEL_DIRS = (
    "/opt/ml/input/data/training",
    "/opt/ml/input/data/train",
)


def _channel_directories() -> list[str]:
    dirs: list[str] = []
    for env_name in _CHANNEL_ENV_VARS:
        value = os.environ.get(env_name)
        if value:
            dirs.append(value)
    dirs.extend(_DEFAULT_CHANNEL_DIRS)
    seen: set[str] = set()
    unique: list[str] = []
    for path in dirs:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def resolve_train_file(
    train_dir: Optional[str] = None,
    train_file: Optional[str] = None,
) -> str:
    """Resolve a JSONL training file from an explicit path or SageMaker channel."""
    if train_file and os.path.isfile(train_file):
        logger.info("Loading dataset from: %s", train_file)
        return train_file

    directories = [train_dir] if train_dir else []
    directories.extend(_channel_directories())
    basename = os.path.basename(train_file) if train_file else None

    for directory in directories:
        if not directory:
            continue
        if basename:
            candidate = os.path.join(directory, basename)
            if os.path.isfile(candidate):
                logger.info("Loading dataset from: %s", candidate)
                return candidate
        jsonl_files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
        if jsonl_files:
            selected = jsonl_files[0]
            logger.info("Loading dataset from: %s", selected)
            return selected

    raise FileNotFoundError(
        f"No .jsonl files found in {train_dir or 'SageMaker train/training channels'}"
    )


def _pair_prompt_completion(prompts: Any, completions: Any) -> Any:
    if isinstance(prompts, list) and isinstance(completions, list):
        return [f"{prompt}\n{completion}" for prompt, completion in zip(prompts, completions)]
    return f"{prompts}\n{completions}"


def texts_from_examples(examples: dict[str, Any]) -> Any:
    """Map JSONL rows to LM text, pairing prompt+completion when both exist."""
    if "prompt" in examples and "completion" in examples:
        return _pair_prompt_completion(examples["prompt"], examples["completion"])
    if "prompt" in examples:
        return examples["prompt"]
    if "text" in examples:
        return examples["text"]
    if "input" in examples:
        return examples["input"]
    return list(examples.values())[0]
