"""Deterministic YAML tailor: clamp steps and drop repeatedly failing tools."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import yaml

from enhanced_system.harness.registry import apply_settings
from enhanced_system.harness.types import HarnessSpec, Trajectory
from enhanced_system.ops.settings import MangoMASSettings, get_settings


class HarnessTailor:
    """Rule-based harness edits; never writes Python and never uses an LLM."""

    def propose(self, spec: HarnessSpec, trajectories: list[Trajectory]) -> HarnessSpec:
        settings = get_settings()
        planning = spec.planning.model_copy()
        action = spec.action.model_copy()
        if any("loop" in traj.faults for traj in trajectories):
            cap = settings.harness_max_steps
            planning.max_steps = cap
        tool_errors: Counter[str] = Counter()
        for traj in trajectories:
            for step in traj.steps:
                if step.fault == "tool_error" and step.tool_id:
                    tool_errors[step.tool_id] += 1
        droppable = [tool_id for tool_id, count in tool_errors.items() if count >= 2]
        remaining = [tid for tid in action.tool_ids if tid not in droppable]
        if remaining:
            action.tool_ids = remaining
        return apply_settings(
            spec.model_copy(update={"planning": planning, "action": action}),
            settings,
        )

    def persist(
        self,
        spec: HarnessSpec,
        *,
        archive_dir: Path,
        apply_patches: Optional[bool] = None,
        live_path: Optional[Path] = None,
        settings: Optional[MangoMASSettings] = None,
    ) -> Path:
        """Write archive YAML; copy to live_path only when apply_patches is true."""
        settings = settings or get_settings()
        enabled = settings.harness_apply_patches if apply_patches is None else apply_patches
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{spec.id}.yaml"
        payload = spec.model_dump(exclude_none=True)
        archive_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        if enabled and live_path is not None:
            live_path.parent.mkdir(parents=True, exist_ok=True)
            live_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
        return archive_path
