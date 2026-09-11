"""Adapters over existing public APIs (no god-file edits)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from enhanced_system.core.adaptive_router import AdaptiveRouter
from enhanced_system.harness.runtime import AgentRuntime
from enhanced_system.ops.settings import get_settings


def _profiles_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    packaged = Path(__file__).resolve().parents[1] / "config" / "agent_profiles.json"
    repo = Path(__file__).resolve().parents[2] / "configs" / "agent_profiles.json"
    if repo.is_file():
        return repo
    return packaged


def harness_id_for_agent(agent_id: str, profiles_path: Optional[str] = None) -> str:
    """Read harness_id from profiles JSON without mutating AgentProfile."""
    path = _profiles_path(profiles_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    for agent in payload.get("agents", []):
        if agent.get("agent_id") == agent_id:
            found = agent.get("harness_id") or ""
            return str(found)
    return ""


class RouterAdapter:
    """Map AdaptiveRouter.selected_agents to a harness id."""

    def __init__(self, profiles_path: Optional[str] = None) -> None:
        path = _profiles_path(profiles_path)
        self.profiles_path = str(path)
        self.router = AdaptiveRouter({"enabled": True, "agent_profiles_path": self.profiles_path})

    def resolve(self, task: str) -> str:
        decision = self.router.route_task(task)
        if not decision.selected_agents:
            return get_settings().harness_id
        return harness_id_for_agent(decision.selected_agents[0], self.profiles_path)


class SkillEvalAdapter:
    """SkillEvaluator-compatible callable returning final_answer only."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    def __call__(self, prompt: str) -> str:
        result = self.runtime.run(prompt)
        return result.final_answer
