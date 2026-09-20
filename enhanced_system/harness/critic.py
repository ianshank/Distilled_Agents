"""Pure critic helpers, reject codes, and telemetry sink for Phase P2 critic-cascade."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from enhanced_system.harness.score import answers_match, semantic_match
from enhanced_system.harness.types import Step, Trajectory

logger = logging.getLogger(__name__)


class CriticRejectCode(str, Enum):
    """Canonical reject codes per openspec/changes/_shared/blocked-reject-codes.md."""

    # Deterministic solver / teacher-rule codes
    CYCLE_DETECTED = "CYCLE_DETECTED"
    UNSAT = "UNSAT"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    SYNTAX_INVALID = "SYNTAX_INVALID"
    UNSUPPORTED_THEORY = "UNSUPPORTED_THEORY"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"

    # Upstream pipeline codes
    OUTCOME_MISMATCH = "OUTCOME_MISMATCH"
    DUALDISTILL_DROP_0_0 = "DUALDISTILL_DROP_0_0"


VALID_REJECT_CODES: set[str] = {code.value for code in CriticRejectCode}


@dataclass(frozen=True)
class CriticDecision:
    """Outcome of a critic check."""

    passed: bool
    reject_code: Optional[str] = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


def _normalize_steps_and_final(
    data: Trajectory | dict[str, Any] | list[Any] | str,
) -> tuple[list[Step], str, list[str]]:
    """Extract steps, final_answer, and faults from heterogeneous inputs."""
    if isinstance(data, Trajectory):
        return list(data.steps), data.final_answer, list(data.faults)

    if isinstance(data, dict):
        traj = data.get("trajectory")
        if isinstance(traj, dict):
            steps_raw = traj.get("steps") or []
            final_ans = str(traj.get("final_answer") or data.get("completion") or "")
            faults = list(traj.get("faults") or [])
        elif isinstance(traj, Trajectory):
            return list(traj.steps), traj.final_answer, list(traj.faults)
        else:
            steps_raw = data.get("steps") or []
            final_ans = str(data.get("final_answer") or data.get("completion") or "")
            faults = list(data.get("faults") or [])

        steps: list[Step] = []
        for item in steps_raw:
            if isinstance(item, Step):
                steps.append(item)
            elif isinstance(item, dict):
                steps.append(
                    Step(
                        thought=str(item.get("thought") or ""),
                        action=str(item.get("action") or ""),
                        observation=str(item.get("observation") or ""),
                        tool_id=str(item.get("tool_id") or ""),
                        fault=str(item.get("fault") or ""),
                    )
                )
        return steps, final_ans, faults

    if isinstance(data, list):
        steps = []
        for item in data:
            if isinstance(item, Step):
                steps.append(item)
            elif isinstance(item, dict):
                steps.append(
                    Step(
                        thought=str(item.get("thought") or ""),
                        action=str(item.get("action") or ""),
                        observation=str(item.get("observation") or ""),
                        tool_id=str(item.get("tool_id") or ""),
                        fault=str(item.get("fault") or ""),
                    )
                )
        return steps, "", []

    return [], str(data), []


def check_tool_allowlist(
    trajectory: Trajectory | dict[str, Any] | list[Any],
    allowed_tools: Iterable[str],
    *,
    reject_code: str = CriticRejectCode.SCHEMA_VIOLATION.value,
) -> CriticDecision:
    """Validate that all tool invocations in trajectory are in the allowlist.

    Pure function: checks tool_id and action against allowed_tools.
    Returns CriticDecision(passed=True) if valid, CriticDecision(passed=False) with reject_code otherwise.
    """
    allowed_set = set(allowed_tools)
    steps, _final_ans, _faults = _normalize_steps_and_final(trajectory)

    for idx, step in enumerate(steps):
        # 1. Direct tool_id attribute
        tid = (step.tool_id or "").strip()
        if tid:
            if tid not in allowed_set:
                return CriticDecision(
                    passed=False,
                    reject_code=reject_code,
                    reason=f"step {idx}: tool '{tid}' not in allowed tools: {sorted(allowed_set)}",
                )

        # 2. Check observation for dispatch/allowlist failure markers
        obs = step.observation or ""
        if "unknown tool id:" in obs or "tool not enabled for this harness:" in obs:
            return CriticDecision(
                passed=False,
                reject_code=reject_code,
                reason=f"step {idx}: allowlist error in observation: {obs}",
            )

        # 3. Check action string if present
        action = (step.action or "").strip()
        if action and not step.fault:
            if action.startswith("{"):
                try:
                    payload = json.loads(action)
                    if isinstance(payload, dict):
                        act_tool = payload.get("tool")
                        if isinstance(act_tool, str) and act_tool and act_tool not in allowed_set:
                            return CriticDecision(
                                passed=False,
                                reject_code=reject_code,
                                reason=f"step {idx}: action tool '{act_tool}' not in allowed tools",
                            )
                except json.JSONDecodeError:
                    pass
            elif "(" in action:
                call_name = action.split("(", 1)[0].strip()
                if call_name.isidentifier() and call_name not in allowed_set:
                    return CriticDecision(
                        passed=False,
                        reject_code=reject_code,
                        reason=f"step {idx}: action call '{call_name}' not in allowed tools",
                    )

    return CriticDecision(passed=True)


def check_outcome(
    outcome: str | Trajectory | dict[str, Any],
    expected: Optional[str],
    *,
    allow_semantic: bool = False,
    reject_code: str = CriticRejectCode.OUTCOME_MISMATCH.value,
) -> CriticDecision:
    """Grade trajectory final answer against expected string.

    If expected is None or blank, check passes (unlabeled).
    Otherwise verifies exact match (or semantic match if allow_semantic=True).
    """
    if expected is None or not str(expected).strip():
        return CriticDecision(passed=True)

    expected_str = str(expected).strip()
    if isinstance(outcome, str):
        final_answer = outcome
    else:
        _steps, final_answer, _faults = _normalize_steps_and_final(outcome)

    if answers_match(final_answer, expected_str):
        return CriticDecision(passed=True)

    if allow_semantic and semantic_match(final_answer, expected_str):
        return CriticDecision(passed=True)

    return CriticDecision(
        passed=False,
        reject_code=reject_code,
        reason=f"outcome mismatch: final_answer={final_answer!r} != expected={expected_str!r}",
    )


def check_expected_tools(
    trajectory: Trajectory | dict[str, Any] | list[Any],
    expected_tools: Optional[list[str]],
    *,
    ordered: bool = False,
    reject_code: str = CriticRejectCode.SCHEMA_VIOLATION.value,
) -> CriticDecision:
    """Check that trajectory executed the expected tool sequence/set."""
    if expected_tools is None:
        return CriticDecision(passed=True)

    steps, _final_ans, _faults = _normalize_steps_and_final(trajectory)
    actual_tools = [
        step.tool_id for step in steps if step.tool_id and not step.fault and step.tool_id != ""
    ]

    if ordered:
        if actual_tools != expected_tools:
            return CriticDecision(
                passed=False,
                reject_code=reject_code,
                reason=f"expected tool sequence {expected_tools} != actual sequence {actual_tools}",
            )
    else:
        missing = [tool for tool in expected_tools if tool not in actual_tools]
        if missing:
            return CriticDecision(
                passed=False,
                reject_code=reject_code,
                reason=f"missing expected tools: {missing} (actual: {actual_tools})",
            )

    return CriticDecision(passed=True)


def is_recovery_trace(
    trajectory: Trajectory | dict[str, Any],
    expected: Optional[str] = None,
) -> bool:
    """Return True if trajectory had intermediate parse/tool faults but achieved the expected outcome."""
    steps, final_ans, faults = _normalize_steps_and_final(trajectory)
    has_fault = any(step.fault in ("parse_error", "tool_error") for step in steps) or any(
        f in ("parse_error", "tool_error") for f in faults
    )
    if not has_fault:
        return False

    if expected is not None and str(expected).strip():
        return answers_match(final_ans, str(expected))

    return False


def evaluate_trace(
    trajectory: Trajectory | dict[str, Any],
    *,
    allowed_tools: Optional[Iterable[str]] = None,
    expected: Optional[str] = None,
    expected_tools: Optional[list[str]] = None,
    allow_semantic: bool = False,
) -> CriticDecision:
    """Run full critic cascade on a trajectory."""
    if allowed_tools is not None:
        allowlist_dec = check_tool_allowlist(trajectory, allowed_tools)
        if not allowlist_dec:
            return allowlist_dec

    if expected_tools is not None:
        tools_dec = check_expected_tools(trajectory, expected_tools)
        if not tools_dec:
            return tools_dec

    if expected is not None:
        outcome_dec = check_outcome(trajectory, expected, allow_semantic=allow_semantic)
        if not outcome_dec:
            return outcome_dec

    return CriticDecision(passed=True)


class RejectSink:
    """Structured reject logger appending JSONL records to a file."""

    def __init__(self, path: str | Path | None = "artifacts/critic_rejects.jsonl") -> None:
        self.path: Optional[Path] = Path(path) if path else None
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        reject_code: str,
        prompt: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Record and persist a structured reject row."""
        entry: dict[str, Any] = {
            "critic_reject_code": reject_code,
            "prompt": prompt,
            "metadata": metadata or {},
        }
        self.records.append(entry)

        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
            except OSError as exc:
                logger.warning("failed to write reject record to %s: %s", self.path, exc)

        return entry


@dataclass
class CriticMetrics:
    """Aggregated counters for critic rejection and recovery events."""

    counters: dict[str, int] = field(
        default_factory=lambda: {
            "critic_kept_recovery": 0,
            "critic_rejected_outcome_mismatch": 0,
            "critic_rejected_allowlist": 0,
            "critic_rejected_expected_tools": 0,
            "critic_rejected_dualdistill_0_0": 0,
        }
    )

    def record_reject(self, code: str) -> str:
        code_norm = code.lower().replace("-", "_")
        key = f"critic_rejected_{code_norm}"
        self.counters[key] = self.counters.get(key, 0) + 1
        return key

    def record_recovery(self) -> None:
        self.counters["critic_kept_recovery"] = self.counters.get("critic_kept_recovery", 0) + 1

    @property
    def critic_kept_recovery(self) -> int:
        return self.counters.get("critic_kept_recovery", 0)

    @property
    def critic_rejected_outcome_mismatch(self) -> int:
        return self.counters.get("critic_rejected_outcome_mismatch", 0)

    @property
    def critic_rejected_allowlist(self) -> int:
        return self.counters.get("critic_rejected_allowlist", 0)

    @property
    def critic_rejected_expected_tools(self) -> int:
        return self.counters.get("critic_rejected_expected_tools", 0)

    @property
    def critic_rejected_dualdistill_0_0(self) -> int:
        return self.counters.get("critic_rejected_dualdistill_0_0", 0)

    def __getattr__(self, name: str) -> int:
        if name in self.counters:
            return self.counters[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def as_dict(self) -> dict[str, int]:
        return dict(self.counters)
