"""Deterministic critic cascade, validation rules, and reject telemetry."""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from enhanced_system.harness.score import answers_match
from enhanced_system.harness.types import Step, Trajectory

logger = logging.getLogger(__name__)


class CriticRejectCode(str, Enum):
    """Authoritative rejection codes conforming to blocked-reject-codes.md."""

    # Upstream pipeline codes
    OUTCOME_MISMATCH = "OUTCOME_MISMATCH"
    DUALDISTILL_DROP_0_0 = "DUALDISTILL_DROP_0_0"

    # Shared teacher-rule / solver reject codes
    CYCLE_DETECTED = "CYCLE_DETECTED"
    UNSAT = "UNSAT"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    SYNTAX_INVALID = "SYNTAX_INVALID"
    UNSUPPORTED_THEORY = "UNSUPPORTED_THEORY"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"

    # Allowlist / tool validation code
    TOOL_NOT_ALLOWED = "TOOL_NOT_ALLOWED"


def _extract_trajectory_components(
    trajectory: Trajectory | dict[str, Any],
) -> tuple[list[tuple[str, str]], list[str], str]:
    """Extract (tool_id, fault) pairs, faults, and final_answer from Trajectory or dict."""
    if isinstance(trajectory, Trajectory):
        steps_source = trajectory.steps
        faults = list(trajectory.faults)
        final_answer = trajectory.final_answer
    else:
        payload_raw = trajectory.get("trajectory")
        payload = payload_raw if isinstance(payload_raw, dict) else trajectory
        steps_source = payload.get("steps") or []
        faults = [str(f) for f in (payload.get("faults") or [])]
        final_answer = str(payload.get("final_answer") or payload.get("completion") or "")

    extracted_steps: list[tuple[str, str]] = []
    for step in steps_source:
        tool_id = step.tool_id if isinstance(step, Step) else str(step.get("tool_id") or "")
        obs = step.observation if isinstance(step, Step) else str(step.get("observation") or "")
        action = step.action if isinstance(step, Step) else str(step.get("action") or "")
        fault = step.fault if isinstance(step, Step) else str(step.get("fault") or "")

        if not tool_id and " not allowed (must be one of " in obs:
            try:
                tool_id = obs.split("tool ", 1)[1].split(" not allowed", 1)[0].strip("'\"")
            except Exception:
                tool_id = ""
        elif not tool_id and action and "(" in action:
            tool_id = action.split("(", 1)[0].strip()

        extracted_steps.append((tool_id, fault))

    return extracted_steps, faults, final_answer


def check_outcome(
    final_answer: str,
    expected: Optional[str],
) -> tuple[bool, Optional[CriticRejectCode], Optional[str]]:
    """Verify that final_answer matches expected using answers_match exact comparison."""
    if expected is None or not str(expected).strip():
        return True, None, None
    if answers_match(final_answer, str(expected)):
        return True, None, None
    reason = f"Outcome mismatch: final_answer={final_answer!r} expected={expected!r}"
    return False, CriticRejectCode.OUTCOME_MISMATCH, reason


def check_tool_allowlist(
    trajectory: Trajectory | dict[str, Any],
    allowed_tools: Iterable[str],
) -> tuple[bool, Optional[CriticRejectCode], Optional[str]]:
    """Verify all steps in trajectory only invoke tools from allowed_tools."""
    allowed_set = set(allowed_tools)
    steps, _, _ = _extract_trajectory_components(trajectory)
    for tool_id, _ in steps:
        if tool_id and tool_id not in allowed_set:
            reason = f"Tool '{tool_id}' not in allowed tools: {sorted(allowed_set)}"
            return False, CriticRejectCode.SCHEMA_VIOLATION, reason
    return True, None, None


def check_expected_tools(
    trajectory: Trajectory | dict[str, Any],
    expected_tools: Optional[Iterable[str]],
) -> tuple[bool, Optional[CriticRejectCode], Optional[str]]:
    """Verify that all expected_tools were called in the trajectory."""
    if expected_tools is None:
        return True, None, None
    expected_set = set(expected_tools)
    if not expected_set:
        return True, None, None
    steps, _, _ = _extract_trajectory_components(trajectory)
    called_tools = {tool_id for tool_id, _ in steps if tool_id}
    missing = expected_set - called_tools
    if missing:
        reason = f"Missing expected tools: {sorted(missing)}"
        return False, CriticRejectCode.SCHEMA_VIOLATION, reason
    return True, None, None


def is_recovery_trace(
    trajectory: Trajectory | dict[str, Any],
    expected: Optional[str] = None,
) -> bool:
    """Return True if trajectory had intermediate faults but reached matching final_answer."""
    steps, faults, final_answer = _extract_trajectory_components(trajectory)
    outcome_ok, _, _ = check_outcome(final_answer, expected)
    if not outcome_ok:
        return False
    if faults:
        return True
    if any(fault for _, fault in steps if fault):
        return True
    return False


class CriticTelemetry:
    """Telemetry sink and counter accumulator for critic rejections and recoveries."""

    def __init__(
        self,
        sink_path: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.sink_path = Path(sink_path) if sink_path else None
        self.counters: dict[str, int] = {}

    def record_reject(
        self,
        code: CriticRejectCode | str,
        prompt: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
        expected: Optional[str] = None,
        final_answer: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record a rejected row or pair to counters, log, and JSONL sink."""
        code_str = code.value if isinstance(code, CriticRejectCode) else str(code)
        counter_key = f"critic_rejected_{code_str.lower()}"
        self.counters[counter_key] = self.counters.get(counter_key, 0) + 1
        self.counters["critic_rejected_total"] = self.counters.get("critic_rejected_total", 0) + 1

        logger.warning("critic_reject_code=%s prompt=%r", code_str, prompt)

        entry: dict[str, Any] = {
            "critic_reject_code": code_str,
            "prompt": prompt,
            "metadata": metadata or {},
        }
        if expected is not None:
            entry["expected"] = str(expected)
        if final_answer is not None:
            entry["final_answer"] = str(final_answer)

        if self.enabled and self.sink_path:
            self._append_to_sink(entry)

        return entry

    def record_recovery(
        self,
        prompt: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Increment recovery counter for successfully recovered trajectories."""
        self.counters["critic_kept_recovery"] = self.counters.get("critic_kept_recovery", 0) + 1
        logger.info("critic_kept_recovery prompt=%r", prompt)

    def record_kept(
        self,
        prompt: str,
        *,
        is_recovery: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a kept trajectory, incrementing recovery if applicable."""
        self.counters["critic_kept_total"] = self.counters.get("critic_kept_total", 0) + 1
        if is_recovery:
            self.record_recovery(prompt, metadata=metadata)

    def emit_summary(self) -> dict[str, int]:
        """Log and return snapshot of current critic counters."""
        logger.info("critic summary: %s", self.counters)
        return dict(self.counters)

    def _append_to_sink(self, entry: dict[str, Any]) -> None:
        if not self.sink_path:
            return
        self.sink_path.parent.mkdir(parents=True, exist_ok=True)
        with self.sink_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
