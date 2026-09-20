"""Pydantic types for harness specifications and traces."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlanConfig(BaseModel):
    """Planning slice of a harness (H.P)."""

    style: str = "react"
    max_steps: Optional[int] = None
    first_thought_prefix: bool = False
    instruction: str = ""


class ActionConfig(BaseModel):
    """Action slice of a harness (H.A)."""

    tool_ids: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    """Memory slice of a harness (H.M)."""

    window_turns: Optional[int] = None
    write_observations: bool = True
    bank_path: str = ""


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
    instruction: str = ""
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


class GoldenRow(BaseModel):
    """Schema for golden evaluation dataset rows (core and hard slices)."""

    model_config = ConfigDict(extra="ignore")

    prompt: str
    expected: Optional[str] = None
    id: Optional[str] = None
    harness_id: Optional[str] = None
    expected_tools: Optional[list[str]] = None
    slice: Literal["core", "hard"] = "core"
    allow_semantic: bool = False

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("GoldenRow prompt must not be empty")
        return str(v).strip()

    @field_validator("slice", mode="before")
    @classmethod
    def normalize_slice(cls, v: Any) -> str:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in ("core", "hard"):
                return v_lower
        return "core" if v is None else str(v)

    def validate_for_hard_slice(self) -> None:
        """Validate hard-slice invariant: non-empty id, non-empty expected, slice == 'hard'."""
        if self.slice != "hard":
            raise ValueError(f"Row {self.id or '<unnamed>'} must have slice='hard'")
        if not self.id or not str(self.id).strip():
            raise ValueError(f"Hard slice row missing stable id (prompt: {self.prompt!r})")
        if self.expected is None or not str(self.expected).strip():
            raise ValueError(f"Hard slice row {self.id} must have non-empty expected answer")
