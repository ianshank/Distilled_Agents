"""Factory for AgentRuntime (separate from core.factories)."""

from __future__ import annotations

from typing import Any, Optional

from enhanced_system.harness.backends.echo import EchoBackend
from enhanced_system.harness.registry import load_spec
from enhanced_system.harness.runtime import AgentRuntime
from enhanced_system.ops.settings import get_settings


class HarnessFactory:
    """Create an AgentRuntime from a config mapping."""

    @staticmethod
    def create(config: Optional[dict[str, Any]] = None) -> AgentRuntime:
        payload = config or {}
        settings = get_settings()
        backend = payload.get("backend") or EchoBackend(payload.get("scripted"))
        harness_id = payload.get("harness_id") or settings.harness_id
        spec = payload.get("spec")
        if spec is None and harness_id:
            spec = load_spec(harness_id, settings)
        return AgentRuntime(
            backend,
            spec=spec,
            settings=settings,
            teacher=bool(payload.get("teacher", False)),
            store=payload.get("store"),
        )
