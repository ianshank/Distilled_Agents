"""SageMaker channel and dataset I/O."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_train_file(train_dir: Optional[str] = None) -> str:
    directory = train_dir or os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {directory}")
    train_file = jsonl_files[0]
    logger.info("Loading dataset from: %s", train_file)
    return train_file


def texts_from_examples(examples: dict[str, Any]) -> Any:
    if "prompt" in examples:
        return examples["prompt"]
    if "text" in examples:
        return examples["text"]
    if "input" in examples:
        return examples["input"]
    return list(examples.values())[0]
