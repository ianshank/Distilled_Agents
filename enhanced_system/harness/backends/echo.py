"""Echo backend for tests and offline collection."""

from __future__ import annotations

from typing import Optional, Sequence


class EchoBackend:
    """Scripted completions; last prefix is recorded for FTP assertions."""

    def __init__(self, scripted: Optional[list[str]] = None) -> None:
        self._queue = list(scripted or [])
        self.last_prefix: Optional[str] = None
        self.last_messages: list[dict[str, str]] = []
        self.call_count = 0

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
        if self._queue:
            item = self._queue.pop(0)
        else:
            item = '{"tool": "final_answer", "args": {"text": "done"}}'
        count = n if n and n > 0 else 1
        return [item] * count
