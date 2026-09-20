"""Synchronous agent runtime (user-task validation only)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from enhanced_system.core.input_validator import InputValidator
from enhanced_system.harness.dispatch import (
    DispatchError,
    allowed_tools,
    parse_action,
    split_thought_action,
)
from enhanced_system.harness.memory_bank import MemoryBank, function_hint, workflow_hint
from enhanced_system.harness.policies.ftp import maybe_prefix
from enhanced_system.harness.policies.sag import generate_action
from enhanced_system.harness.prompt_render import messages_from_steps
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
        memory_bank: Optional[MemoryBank] = None,
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
        self.memory_bank = memory_bank

    def reset(self) -> None:
        """Reset backend state between evaluation trials/samples."""
        if hasattr(self.backend, "reset"):
            self.backend.reset()

    def run(
        self,
        task: str,
        *,
        harness_id: Optional[str] = None,
        resume_steps: Optional[list[Step]] = None,
        inject_action: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> HarnessRunResult:
        """Validate the user task, then loop until final_answer or max_steps."""
        recorded, spec, teacher = self._prepare(task, harness_id=harness_id)
        bank = self._load_bank(spec)
        instruction = self._instruction(spec, recorded, teacher, bank)
        allowed = allowed_tools(spec.action.tool_ids)
        fresh = resume_steps is None and inject_action is None
        prefix = maybe_prefix(
            self.backend,
            recorded,
            enabled=spec.planning.first_thought_prefix and fresh,
            teacher=teacher,
            instruction=instruction,
        )
        resumed = list(resume_steps or [])
        trajectory = Trajectory(
            harness_id=spec.id,
            task=recorded,
            instruction=instruction,
            steps=resumed,
            faults=[step.fault for step in resumed if step.fault],
        )
        max_steps = spec.planning.max_steps or self.settings.harness_max_steps
        window = (
            spec.memory.window_turns
            if spec.memory.window_turns is not None
            else self.settings.harness_memory_window
        )
        pending = inject_action
        truncated = True
        final_answer = ""
        logger.info(
            "harness start harness_id=%s seed=%s max_steps=%s resume_steps=%s inject=%s",
            spec.id,
            self.settings.harness_seed,
            max_steps,
            0 if resume_steps is None else len(resume_steps),
            bool(inject_action),
        )
        while len(trajectory.steps) < max_steps:
            messages = _trim_window(
                messages_from_steps(
                    recorded,
                    [step.model_dump() for step in trajectory.steps],
                    instruction=instruction,
                ),
                window,
            )
            tool_id = ""
            try:
                raw = pending
                pending = None
                if raw is None:
                    effective_temperature = (
                        temperature if temperature is not None else spec.policy.sag_temperature
                    )
                    raw, _name = generate_action(
                        self.backend,
                        messages,
                        allowed,
                        samples=spec.policy.sag_samples or 1,
                        temperature=effective_temperature,
                        prefix=prefix,
                    )
                thought, action, tool_id, args = self._parse_generation(raw, prefix, allowed)
                live_obs = get_tool(tool_id).run(args)
                stored = live_obs if spec.memory.write_observations else ""
                step = Step(
                    thought=thought,
                    action=action,
                    observation=stored,
                    tool_id=tool_id,
                )
                trajectory.steps.append(step)
                logger.info(
                    "harness step=%s harness_id=%s tool=%s",
                    len(trajectory.steps) - 1,
                    spec.id,
                    step.tool_id,
                )
                if step.tool_id == "final_answer":
                    final_answer = live_obs
                    truncated = False
                    break
            except (DispatchError, ValueError, KeyError) as exc:
                fault = "parse_error" if isinstance(exc, DispatchError) else "tool_error"
                observation = str(exc)
                if fault == "tool_error" and not teacher:
                    hint = function_hint(bank, tool_id)
                    if hint:
                        observation = f"{observation}\n{hint}"
                trajectory.faults.append(fault)
                trajectory.steps.append(
                    Step(
                        action="",
                        observation=observation,
                        fault=fault,
                        tool_id=tool_id if fault == "tool_error" else "",
                    )
                )
                logger.warning(
                    "harness step=%s harness_id=%s fault=%s",
                    len(trajectory.steps) - 1,
                    spec.id,
                    fault,
                )
            finally:
                prefix = None
        if truncated:
            trajectory.faults.append("loop")
        if final_answer and final_answer.startswith("BLOCKED:"):
            trajectory.faults.append("blocked")
        trajectory.final_answer = final_answer
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

    def _prepare(self, task: str, *, harness_id: Optional[str]) -> tuple[str, HarnessSpec, bool]:
        validation = self.validator.validate_task_input(task)
        if not validation.is_valid:
            raise ValueError(validation.error_message or "invalid task")
        recorded = (
            validation.sanitized_input or task if self.validator.enable_pii_detection else task
        )
        spec_id = harness_id or self.settings.harness_id
        spec = self.spec or load_spec(spec_id, self.settings)
        if spec_id and spec.id != spec_id:
            spec = load_spec(spec_id, self.settings)
        teacher = self.teacher if self.teacher is not None else spec.policy.teacher
        return recorded, spec, teacher

    def _instruction(
        self,
        spec: HarnessSpec,
        task: str,
        teacher: bool,
        bank: Optional[MemoryBank],
    ) -> str:
        parts = [spec.planning.instruction.strip()]
        if bank is not None and not teacher:
            hint = workflow_hint(bank, task)
            if hint:
                parts.append(f"Workflow: {hint}")
        return "\n".join(part for part in parts if part)

    def _load_bank(self, spec: HarnessSpec) -> Optional[MemoryBank]:
        if self.memory_bank is not None:
            return self.memory_bank
        path = spec.memory.bank_path or self.settings.harness_memory_bank
        if not path:
            return None
        try:
            return MemoryBank.load(path)
        except (OSError, ValueError) as exc:
            logger.warning("memory bank skipped: %s", exc)
            return None

    def _parse_generation(
        self,
        raw: str,
        prefix: Optional[str],
        allowed: set[str],
    ) -> tuple[str, str, str, dict[str, Any]]:
        thought_tail, action = split_thought_action(raw, allowed)
        thought = " ".join(part for part in ((prefix or "").strip(), thought_tail) if part)
        tool_id, args = parse_action(action, allowed)
        return thought, action, tool_id, args


def _trim_window(messages: list[dict[str, str]], window_turns: int) -> list[dict[str, str]]:
    if window_turns <= 0:
        return messages
    head: list[dict[str, str]] = []
    rest = list(messages)
    while rest and rest[0].get("role") == "system":
        head.append(rest.pop(0))
    if rest:
        head.append(rest.pop(0))
    keep = window_turns * 2
    return head + rest[-keep:]
