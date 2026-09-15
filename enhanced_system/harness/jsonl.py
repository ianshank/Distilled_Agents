"""JSONL helpers for harness CLIs (not SageMaker ``source_dir``)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonlRowError(ValueError):
    """A JSONL row failed validation in strict mode."""


def iter_jsonl_dicts(
    path: Path,
    *,
    require_prompt: bool = False,
    strict: bool = False,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_no, mapping)`` for non-blank JSON object rows.

    Blank lines are ignored. Invalid JSON, non-object rows, and (when
    ``require_prompt``) empty ``prompt`` fields are skipped with a warning,
    or raised as :class:`JsonlRowError` when ``strict`` is true.
    """
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            row = _parse_mapping_row(
                text,
                line_no,
                require_prompt=require_prompt,
                strict=strict,
            )
            if row is not None:
                yield line_no, row


def _parse_mapping_row(
    text: str,
    line_no: int,
    *,
    require_prompt: bool,
    strict: bool,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        _reject(line_no, str(exc), strict=strict)
        return None
    if not isinstance(payload, dict):
        _reject(line_no, "row is not a JSON object", strict=strict)
        return None
    if require_prompt and not str(payload.get("prompt") or "").strip():
        _reject(line_no, "empty prompt", strict=strict)
        return None
    return payload


def _reject(line_no: int, reason: str, *, strict: bool) -> None:
    logger.warning("skipping line %s: %s", line_no, reason)
    if strict:
        raise JsonlRowError(f"skipping line {line_no}: {reason}")
