"""SCoRe-SFT helpers: earliest-error index and preference pairs (no GRPO)."""

from __future__ import annotations

import json
from typing import Any, Optional

from enhanced_system.harness.types import Step, Trajectory


def answers_match(actual: str, expected: str) -> bool:
    """Exact-match grader used for outcome filter and DualDistill."""
    return actual.strip() == expected.strip()


def earliest_error_index(trajectory: Trajectory, expected: Optional[str] = None) -> Optional[int]:
    """Index of the action to replace. Outcome success is not a fault bag.

    Recovered parse/tool errors are kept when the final answer matches. A wrong
    (or missing) answer prefers the last non-fault action over DispatchError.
    """
    if expected is not None:
        if answers_match(trajectory.final_answer, expected):
            return None
    elif trajectory.final_answer and "loop" not in trajectory.faults:
        return None
    for index in range(len(trajectory.steps) - 1, -1, -1):
        step = trajectory.steps[index]
        if step.action and not step.fault:
            return index
    for index, step in enumerate(trajectory.steps):
        if step.fault:
            return index
    return None


def review_prompt(trajectory: Trajectory, expected: Optional[str] = None) -> list[dict[str, str]]:
    """Teacher review of the full student chain (SCoRe Appendix E, no GRPO)."""
    expected_line = (
        f"Expected final answer: {expected}" if expected is not None else "No gold answer given."
    )
    payload = json.dumps([step.model_dump() for step in trajectory.steps], ensure_ascii=True)
    return [
        {
            "role": "system",
            "content": (
                "Review the student trajectory. Identify the first semantic error "
                "(wrong tool or wrong final answer), not a recovered parse glitch. "
                "Reply with only the replacement JSON or Call action for that step."
            ),
        },
        {
            "role": "user",
            "content": f"Task: {trajectory.task}\n{expected_line}\nSteps: {payload}",
        },
    ]


def preference_pair(
    student: Trajectory, corrected: Trajectory, index: int
) -> Optional[dict[str, Any]]:
    """Same-prefix chosen/rejected pair for later DPO/GRPO (not trained here)."""
    if index >= len(student.steps) or index >= len(corrected.steps):
        return None
    rejected = student.steps[index]
    chosen = corrected.steps[index]
    prefix = [step.model_dump() for step in student.steps[:index]]
    return {
        "task": student.task,
        "harness_id": student.harness_id,
        "index": index,
        "prefix": prefix,
        "rejected": rejected.model_dump(),
        "chosen": chosen.model_dump(),
    }


def prefix_steps(trajectory: Trajectory, index: int) -> list[Step]:
    """Verified prefix preceding the earliest error."""
    return list(trajectory.steps[:index])
