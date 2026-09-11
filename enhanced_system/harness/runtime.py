"""Synchronous agent runtime (user-task validation only)."""

from __future__ import annotations

import logging
from typing import Optional

from enhanced_system.core.input_validator import InputValidator
from enhanced_system.harness.dispatch import DispatchError, allowed_tools, parse_action
from enhanced_system.harness.policies.ftp import maybe_prefix
from enhanced_system.harness.policies.sag import generate_action
from enhanced_system.harness.protocols import ModelBackend, TraceStore
from enhanced_system.harness.registry import load_spec
from enhanced_system.harness.tools.registry import get_tool
from enhanced_system.harness.types import HarnessRunResult, HarnessSpec, Step, Trajectory
from enhanced_system.ops.settings import MangoMASSettings, get_settings

logger = logging.getLogger(__name__)


class AgentRuntime:
    """ReAct-style loop with AST/JSON tool dispatch."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        spec: Optional[HarnessSpec] = None,
        settings: Optional[MangoMASSettings] = None,
        validator: Optional[InputValidator] = None,
        store: Optional[TraceStore] = None,
        teacher: Optional[bool] = None,
    ) -> None:
        self.backend = backend
        self.spec = spec
        self.settings = settings or get_settings()
        self.validator = validator or InputValidator(
            {
                "max_length": self.settings.harness_max_length,
                "enable_pii_detection": False,
                "enable_injection_detection": True,
            }
        )
        self.store = store
        self.teacher = teacher

    def run(self, task: str, *, harness_id: Optional[str] = None) -> HarnessRunResult:
        """Validate the user task, then loop until final_answer or max_steps."""
        validation = self.validator.validate_task_input(task)
        if not validation.is_valid:
            raise ValueError(validation.error_message or "invalid task")
        # Keep original text when PII is off so CodeAct collect does not flatten newlines.
        recorded = (
            validation.sanitized_input or task if self.validator.enable_pii_detection else task
        )
        spec_id = harness_id or self.settings.harness_id
        spec = self.spec or load_spec(spec_id, self.settings)
        if spec_id and spec.id != spec_id:
            spec = load_spec(spec_id, self.settings)
        allowed = allowed_tools(spec.action.tool_ids)
        teacher = self.teacher if self.teacher is not None else spec.policy.teacher
        prefix = maybe_prefix(
            self.backend,
            recorded,
            enabled=spec.planning.first_thought_prefix,
            teacher=teacher,
        )
        trajectory = Trajectory(harness_id=spec.id, task=recorded)
        max_steps = spec.planning.max_steps or self.settings.harness_max_steps
        window = spec.memory.window_turns or self.settings.harness_memory_window
        messages = [{"role": "user", "content": recorded}]
        truncated = True
        final_answer = ""
        logger.info(
            "harness start harness_id=%s seed=%s max_steps=%s",
            spec.id,
            self.settings.harness_seed,
            max_steps,
        )
        for step_idx in range(max_steps):
            try:
                raw, _tool_name = generate_action(
                    self.backend,
                    messages,
                    allowed,
                    samples=spec.policy.sag_samples or 1,
                    temperature=spec.policy.sag_temperature,
                    prefix=prefix,
                )
                thought = prefix or ""
                tool_id, args = parse_action(raw, allowed)
                observation = get_tool(tool_id).run(args)
                step = Step(
                    thought=thought,
                    action=raw,
                    observation=observation if spec.memory.write_observations else "",
                    tool_id=tool_id,
                )
                trajectory.steps.append(step)
                logger.info(
                    "harness step=%s harness_id=%s tool=%s",
                    step_idx,
                    spec.id,
                    tool_id,
                )
                if tool_id == "final_answer":
                    final_answer = observation
                    truncated = False
                    break
                if spec.memory.write_observations:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": observation})
                    messages = _trim_window(messages, window)
            except (DispatchError, ValueError, KeyError) as exc:
                fault = "parse_error" if isinstance(exc, DispatchError) else "tool_error"
                trajectory.faults.append(fault)
                trajectory.steps.append(Step(action="", observation=str(exc), fault=fault))
                logger.warning("harness step=%s harness_id=%s fault=%s", step_idx, spec.id, fault)
                messages.append({"role": "user", "content": f"{fault}: {exc}"})
            finally:
                prefix = None
        if truncated:
            trajectory.faults.append("loop")
        result = HarnessRunResult(
            final_answer=final_answer,
            trajectory=trajectory,
            harness_id=spec.id,
            truncated=truncated,
            metadata={"harness_seed": self.settings.harness_seed},
        )
        logger.info(
            "harness done harness_id=%s truncated=%s steps=%s",
            spec.id,
            truncated,
            len(trajectory.steps),
        )
        if self.store is not None:
            self.store.append(trajectory)
        return result


def _trim_window(messages: list[dict[str, str]], window_turns: int) -> list[dict[str, str]]:
    if window_turns <= 0:
        return messages
    head = messages[:1]
    tail = messages[1:]
    keep = window_turns * 2
    return head + tail[-keep:]
