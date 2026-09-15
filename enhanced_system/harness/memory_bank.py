"""Training-free workflow/function memory distilled from teacher traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from enhanced_system.harness.types import Trajectory


class MemoryBank(BaseModel):
    """Hierarchical teacher memory: workflow plans and per-tool call guides."""

    workflows: list[dict[str, str]] = Field(default_factory=list)
    functions: dict[str, list[str]] = Field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MemoryBank:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("memory bank must be a JSON object")
        return cls.model_validate(payload)


def build_bank(trajectories: list[Trajectory]) -> MemoryBank:
    """Keep successful traces; do not drop tools (opposite of HarnessTailor)."""
    bank = MemoryBank()
    for traj in trajectories:
        if not traj.final_answer or "loop" in traj.faults:
            continue
        strategy = _first_strategy(traj)
        if strategy:
            bank.workflows.append(
                {"task": traj.task, "strategy": strategy, "harness_id": traj.harness_id}
            )
        _add_function_tips(bank, traj)
    return bank


def workflow_hint(bank: Optional[MemoryBank], task: str, *, limit: int = 1) -> str:
    """Keyword overlap retrieval of teacher workflow text."""
    if bank is None or not bank.workflows:
        return ""
    task_tokens = _tokens(task)
    scored: list[tuple[int, str]] = []
    for item in bank.workflows:
        overlap = len(task_tokens & _tokens(item.get("task", "")))
        strategy = item.get("strategy", "")
        if overlap <= 0 or not strategy:
            continue
        scored.append((overlap, strategy))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return "\n".join(strategy for _overlap, strategy in scored[:limit])


def function_hint(bank: Optional[MemoryBank], tool_id: str, *, limit: int = 4) -> str:
    """Return calling conventions for a tool after a tool_error."""
    if bank is None or not tool_id:
        return ""
    tips = bank.functions.get(tool_id) or []
    if not tips:
        return ""
    return "Function memory:\n" + "\n".join(tips[:limit])


def _first_strategy(traj: Trajectory) -> str:
    for step in traj.steps:
        if step.thought.strip():
            return step.thought.strip()
        if step.action.strip() and not step.fault:
            return step.action.strip()
    return ""


def _add_function_tips(bank: MemoryBank, traj: Trajectory) -> None:
    for step in traj.steps:
        if not step.tool_id:
            continue
        tips = bank.functions.setdefault(step.tool_id, [])
        if step.fault == "tool_error" and step.observation:
            tip = f"avoid: {step.observation}"
            if tip not in tips:
                tips.append(tip)
            continue
        if step.action and not step.fault:
            tip = step.action.strip()
            if tip not in tips:
                tips.append(tip)


def _tokens(text: str) -> set[str]:
    return {part.lower() for part in text.replace(",", " ").split() if len(part) > 2}
