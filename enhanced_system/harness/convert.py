"""Convert trajectories to legacy prompt/completion JSONL."""

from __future__ import annotations

import logging
from typing import Any

from enhanced_system.harness.types import Trajectory
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def serialize_thought_action(trajectory: Trajectory) -> str:
    """Join thought+action for each step (observations excluded)."""
    parts = []
    for step in trajectory.steps:
        chunk = " ".join(part for part in (step.thought, step.action) if part).strip()
        if chunk:
            parts.append(chunk)
    return "\n".join(parts)


def trajectory_to_legacy(trajectory: Trajectory) -> dict[str, Any]:
    """Always emit prompt/completion; attach trajectory when present."""
    row = {
        "prompt": trajectory.task,
        "completion": serialize_thought_action(trajectory) or trajectory.final_answer,
        "trajectory": trajectory.model_dump(),
    }
    max_length = get_settings().harness_max_length
    serialized_len = len(row["prompt"]) + len(row["completion"])
    if max_length and serialized_len > max_length:
        logger.warning(
            "serialized trajectory length %s exceeds harness_max_length=%s",
            serialized_len,
            max_length,
        )
    return row


def legacy_to_stub_trajectory(row: dict[str, Any]) -> Trajectory:
    """Build a one-step stub from a prompt/completion row."""
    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or "")
    existing = row.get("trajectory")
    if isinstance(existing, dict):
        return Trajectory.model_validate(existing)
    return Trajectory(
        task=prompt,
        final_answer=completion,
        steps=[],
    )
