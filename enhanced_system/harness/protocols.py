"""Protocols for model backends, tools, and trace storage."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from enhanced_system.harness.types import Trajectory


@runtime_checkable
class ModelBackend(Protocol):
    """Generates candidate action strings."""

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        prefix: Optional[str] = None,
        n: int = 1,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """Return n completions for the given chat messages."""


@runtime_checkable
class Tool(Protocol):
    """Deterministic tool invoked by AST/JSON dispatch."""

    tool_id: str

    def run(self, payload: dict[str, Any]) -> str:
        """Execute the tool with literal arguments."""


@runtime_checkable
class TraceStore(Protocol):
    """Append-only trajectory store."""

    def append(self, trajectory: Trajectory) -> None:
        """Persist one trajectory."""
