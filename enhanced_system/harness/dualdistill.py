"""DualDistill trajectory composition (same task, two teachers, grader)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from enhanced_system.harness.convert import serialize_thought_action
from enhanced_system.harness.critic import CriticRejectCode, CriticTelemetry
from enhanced_system.harness.score import answers_match
from enhanced_system.harness.types import Step, Trajectory
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)

TRANSITION_FIX = "Wait, that approach failed. Let us try the other strategy."
TRANSITION_BOTH = "Alternatively, another valid strategy is as follows."


def final_text(row: dict[str, Any]) -> str:
    """Read final_answer from a collect/legacy JSONL row."""
    trajectory = row.get("trajectory")
    if isinstance(trajectory, dict) and trajectory.get("final_answer"):
        return str(trajectory.get("final_answer") or "")
    return str(row.get("completion") or "")


def expected_text(row: dict[str, Any], fallback: str = "") -> str:
    """Reference answer from the row or a shared fallback."""
    value = row.get("expected")
    if value is None or str(value).strip() == "":
        return fallback
    return str(value)


def compose_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    expected: str,
    telemetry: Optional[CriticTelemetry] = None,
    critic_enabled: Optional[bool] = None,
) -> Optional[dict[str, Any]]:
    """Compose y1 and y2 using DualDistill's (g1, g2) table. Drop (0, 0)."""
    grade_first = answers_match(final_text(first), expected)
    grade_second = answers_match(final_text(second), expected)
    prompt = str(first.get("prompt") or second.get("prompt") or "")
    if not grade_first and not grade_second:
        logger.info(
            "critic_reject_code: %s prompt=%r",
            CriticRejectCode.DUALDISTILL_DROP_0_0.value,
            prompt,
        )
        should_record = (telemetry is not None and telemetry.enabled) or (
            telemetry is None
            and (critic_enabled or (critic_enabled is None and get_settings().critic_enabled))
        )
        metadata = {
            "first_final": final_text(first),
            "second_final": final_text(second),
            "rule": "dual_teacher_both_failed_0_0",
        }
        if telemetry is not None:
            telemetry.record_reject(
                code=CriticRejectCode.DUALDISTILL_DROP_0_0,
                prompt=prompt,
                expected=expected,
                metadata=metadata,
            )
        elif should_record:
            settings = get_settings()
            sink_telemetry = CriticTelemetry(
                sink_path=settings.critic_reject_log,
                enabled=True,
            )
            sink_telemetry.record_reject(
                code=CriticRejectCode.DUALDISTILL_DROP_0_0,
                prompt=prompt,
                expected=expected,
                metadata=metadata,
            )
        return None

    if telemetry is not None:
        telemetry.record_kept(prompt=prompt)

    if grade_first and not grade_second:
        kept = dict(first)
        kept["expected"] = expected
        return kept
    if not grade_first and grade_second:
        return _stitch(first, second, TRANSITION_FIX, expected)
    return _stitch(first, second, TRANSITION_BOTH, expected)


def _stitch(
    first: dict[str, Any], second: dict[str, Any], transition: str, expected: str
) -> dict[str, Any]:
    left = _as_trajectory(first)
    right = _as_trajectory(second)
    steps = list(left.steps)
    steps.append(Step(thought=transition, action=""))
    steps.extend(right.steps)
    merged = Trajectory(
        schema_version=left.schema_version,
        harness_id=left.harness_id or right.harness_id,
        task=left.task or right.task,
        instruction=left.instruction or right.instruction,
        steps=steps,
        final_answer=right.final_answer or left.final_answer,
        faults=list(left.faults) + list(right.faults),
    )
    return {
        "prompt": merged.task,
        "completion": serialize_thought_action(merged) or merged.final_answer,
        "expected": expected,
        "trajectory": merged.model_dump(),
    }


def _as_trajectory(row: dict[str, Any]) -> Trajectory:
    payload = row.get("trajectory")
    if isinstance(payload, dict):
        return Trajectory.model_validate(payload)
    return Trajectory(
        task=str(row.get("prompt") or ""),
        final_answer=str(row.get("completion") or ""),
        steps=[Step(thought="", action=str(row.get("completion") or ""))],
    )
