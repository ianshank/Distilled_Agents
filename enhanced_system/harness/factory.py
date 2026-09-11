"""Factory for AgentRuntime (separate from core.factories)."""

from __future__ import annotations

from typing import Any, Optional

from enhanced_system.core.input_validator import InputValidator
from enhanced_system.harness.backends.echo import EchoBackend
from enhanced_system.harness.backends.transformers import TransformersBackend
from enhanced_system.harness.registry import load_spec
from enhanced_system.harness.runtime import AgentRuntime
from enhanced_system.harness.types import HarnessSpec
from enhanced_system.ops.settings import get_settings


def _resolve_backend(payload: dict[str, Any], settings, spec: Optional[HarnessSpec] = None) -> Any:
    backend = payload.get("backend")
    if backend is not None:
        return backend
    scripted = payload.get("scripted")
    if scripted is not None:
        return EchoBackend(scripted)
    teacher = payload.get("teacher")
    if teacher is None and spec is not None:
        teacher = spec.policy.teacher
    model_name = payload.get("model_name") or (
        settings.teacher_model if teacher else settings.student_model
    )
    return TransformersBackend(
        model_name=model_name,
        trust_remote_code=settings.trust_remote_code,
        model_revision=settings.model_revision or None,
        max_input_length=settings.harness_max_length,
    )


class HarnessFactory:
    """Create an AgentRuntime from a config mapping."""

    @staticmethod
    def create(config: Optional[dict[str, Any]] = None) -> AgentRuntime:
        payload = config or {}
        settings = get_settings()
        harness_id = payload.get("harness_id") or settings.harness_id
        spec = payload.get("spec")
        if spec is None and harness_id:
            spec = load_spec(harness_id, settings)
        validator = payload.get("validator")
        if validator is None:
            validator = InputValidator(
                {
                    "max_length": settings.harness_max_length,
                    "enable_pii_detection": False,
                    "enable_injection_detection": bool(payload.get("strict_injection", True)),
                }
            )
        backend = _resolve_backend(payload, settings, spec)
        return AgentRuntime(
            backend,
            spec=spec,
            settings=settings,
            validator=validator,
            teacher=payload.get("teacher"),
            store=payload.get("store"),
        )
