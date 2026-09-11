"""Pydantic types for harness specifications and traces."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PlanConfig(BaseModel):
    """Planning slice of a harness (H.P)."""

    style: str = "react"
    max_steps: Optional[int] = None
    first_thought_prefix: bool = False


class ActionConfig(BaseModel):
    """Action slice of a harness (H.A)."""

    tool_ids: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    """Memory slice of a harness (H.M)."""

    window_turns: Optional[int] = None
    write_observations: bool = True


class PolicyConfig(BaseModel):
    """Runtime policy knobs (SAG / teacher FTP)."""

    sag_samples: Optional[int] = None
    sag_temperature: Optional[float] = None
    teacher: bool = False


class HarnessSpec(BaseModel):
    """YAML-backed harness specification."""

    id: str
    schema_version: str = "1"
    planning: PlanConfig = Field(default_factory=PlanConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


class Step(BaseModel):
    """One reason-act-observe step."""

    thought: str = ""
    action: str = ""
    observation: str = ""
    tool_id: str = ""
    fault: str = ""


class Trajectory(BaseModel):
    """Full run trace, serializable to JSONL."""

    schema_version: str = "1"
    harness_id: str = ""
    task: str = ""
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    faults: list[str] = Field(default_factory=list)


class HarnessRunResult(BaseModel):
    """Runtime result passed to evaluators and stores."""

    final_answer: str = ""
    trajectory: Trajectory = Field(default_factory=Trajectory)
    harness_id: str = ""
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
