"""SCoRe-SFT helpers: earliest-error index and preference pairs (no GRPO)."""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from enhanced_system.harness.types import Step, Trajectory


def answers_match(actual: str, expected: str) -> bool:
    """Exact-match grader used for outcome filter and DualDistill."""
    return actual.strip() == expected.strip()


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate Chen et al. unbiased pass@k estimator for a single problem.

    Formula (arXiv:2107.03374):
        pass@k = 1 - comb(n - c, k) / comb(n, k)
    If n - c < k, comb(n - c, k) == 0, so pass@k = 1.0.
    """
    if n < k:
        raise ValueError(f"Total samples n={n} must be >= k={k}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if c < 0 or c > n:
        raise ValueError(f"Correct samples c={c} must be between 0 and n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - float(math.comb(n - c, k)) / float(math.comb(n, k))


def calculate_pass_at_k(problem_sample_counts: list[tuple[int, int]], k: int) -> float:
    """Calculate average unbiased pass@k across multiple problems.

    Each element of problem_sample_counts is (n, c): total samples and correct samples.
    """
    if not problem_sample_counts:
        return 0.0
    total = sum(estimate_pass_at_k(n, c, k) for n, c in problem_sample_counts)
    return total / len(problem_sample_counts)


def semantic_match(actual: str, expected: str) -> bool:
    """Basic semantic matching without an LLM: normalize whitespace, casing, and basic punctuation."""
    import re

    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    act_norm = normalize(actual)
    exp_norm = normalize(expected)
    return act_norm == exp_norm or exp_norm in act_norm


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
