"""Optional Transformers backend; import is deferred so tests stay CPU-only."""

from __future__ import annotations

from typing import Optional, Sequence


class TransformersBackend:
    """Thin wrapper around a generate callable (injected for tests)."""

    def __init__(self, generate_fn=None) -> None:
        self._generate_fn = generate_fn

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        prefix: Optional[str] = None,
        n: int = 1,
        temperature: Optional[float] = None,
    ) -> list[str]:
        if self._generate_fn is None:
            raise ImportError("transformers backend requires an injected generate_fn")
        return self._generate_fn(list(messages), prefix=prefix, n=n, temperature=temperature)
