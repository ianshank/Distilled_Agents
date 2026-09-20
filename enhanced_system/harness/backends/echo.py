"""Echo backend for tests and offline collection."""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence


class EchoBackend:
    """Scripted completions; last prefix is recorded for FTP assertions."""

    def __init__(self, scripted: Optional[list[str] | dict[str, Any]] = None) -> None:
        self.last_prefix: Optional[str] = None
        self.last_messages: list[dict[str, str]] = []
        self.call_count = 0
        self._initial_scripted = scripted
        self._mapping: dict[str, Any] = {}
        self._queue: list[str] = []
        self.reset()

    def reset(self) -> None:
        """Reset queue and mapping to initial state for subsequent evaluation trials."""
        if isinstance(self._initial_scripted, dict):
            self._mapping = {
                k: list(v) if isinstance(v, list) else v for k, v in self._initial_scripted.items()
            }
            self._queue = []
        elif isinstance(self._initial_scripted, list):
            self._mapping = {}
            self._queue = list(self._initial_scripted)
        else:
            self._mapping = {}
            self._queue = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        prefix: Optional[str] = None,
        n: int = 1,
        temperature: Optional[float] = None,
    ) -> list[str]:
        self.call_count += 1
        self.last_prefix = prefix
        self.last_messages = list(messages)
        count = n if n and n > 0 else 1
        item = '{"tool": "final_answer", "args": {"text": "done"}}'
        if self._mapping:
            user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
            full_user = " ".join(user_texts)
            matched_val = None
            for key, val in self._mapping.items():
                if key in full_user or any(key in ut for ut in user_texts):
                    matched_val = val
                    break
            if matched_val is None and "default" in self._mapping:
                matched_val = self._mapping["default"]
            if matched_val is not None:
                if isinstance(matched_val, list):
                    if matched_val:
                        elem = matched_val.pop(0)
                        item = json.dumps(elem) if isinstance(elem, dict) else str(elem)
                elif isinstance(matched_val, dict):
                    item = json.dumps(matched_val)
                else:
                    item = str(matched_val)
        elif self._queue:
            item = self._queue.pop(0)
        return [item] * count
