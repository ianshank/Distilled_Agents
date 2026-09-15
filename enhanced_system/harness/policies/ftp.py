"""First-thought prefix (teacher-only)."""

from __future__ import annotations

from typing import Optional

from enhanced_system.harness.protocols import ModelBackend


def maybe_prefix(
    backend: ModelBackend,
    task: str,
    *,
    enabled: bool,
    teacher: bool,
    instruction: str = "",
) -> Optional[str]:
    """Ask the teacher for a CoT seed; students skip this call."""
    if not enabled or not teacher:
        return None
    messages: list[dict[str, str]] = []
    text = instruction.strip()
    if text:
        messages.append({"role": "system", "content": text})
    messages.append({"role": "user", "content": task})
    seeds = backend.generate(
        messages,
        prefix=None,
        n=1,
        temperature=None,
    )
    if not seeds:
        return None
    return seeds[0]
