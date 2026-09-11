"""Trajectory-mode distillation_alpha parsing (no torch / enhanced_system)."""

from __future__ import annotations

import math


def parse_trajectory_distill_alpha(raw: str) -> float:
    """Parse MANGOMAS_TRAJECTORY_DISTILL_ALPHA (finite, inclusive [0.0, 1.0])."""
    try:
        alpha = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"MANGOMAS_TRAJECTORY_DISTILL_ALPHA must be a number, got {raw!r}"
        ) from exc
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"MANGOMAS_TRAJECTORY_DISTILL_ALPHA must be in [0.0, 1.0], got {alpha}")
    return alpha
