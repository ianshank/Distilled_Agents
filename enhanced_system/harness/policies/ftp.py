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
) -> Optional[str]:
    """Ask the teacher for a CoT seed; students skip this call."""
    if not enabled or not teacher:
        return None
    seeds = backend.generate(
        [{"role": "user", "content": task}],
        prefix=None,
        n=1,
        temperature=None,
    )
    if not seeds:
        return None
    return seeds[0]
