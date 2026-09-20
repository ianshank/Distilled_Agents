"""MangoMAS harness data types and schemas."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlanConfig(BaseModel):
    """Planning configuration for a harness spec."""

    style: str = "react"
    instruction: str = ""
    max_steps: Optional[int] = None
    first_thought_prefix: bool = False


class ActionConfig(BaseModel):
    """Action space configuration for a harness spec."""

    tool_ids: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    """Memory configuration for a harness spec."""

    window_turns: Optional[int] = None
    write_observations: bool = True
    bank_path: str = ""


class PolicyConfig(BaseModel):
    """Policy execution configuration for a harness spec."""

    teacher: bool = False
    sag_samples: Optional[int] = None
    sag_temperature: Optional[float] = None


class HarnessSpec(BaseModel):
    """Full declarative specification of an agent harness."""

    schema_version: str = "1"
    id: str
    planning: PlanConfig = Field(default_factory=PlanConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


class Step(BaseModel):
    """Single step in an agent trajectory."""

    thought: str = ""
    action: str = ""
    observation: str = ""
    fault: Optional[str] = None
    tool_id: str = ""


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


CANONICAL_REJECT_CODES = (
    "CYCLE_DETECTED",
    "UNSAT",
    "SCHEMA_VIOLATION",
    "SYNTAX_INVALID",
    "UNSUPPORTED_THEORY",
    "RESOURCE_LIMIT",
)

CANONICAL_REFUSAL_TOKENS = tuple(f"BLOCKED:{code}" for code in CANONICAL_REJECT_CODES)


class GoldenRow(BaseModel):
    """Schema for golden evaluation dataset rows (core, hard, and ood slices)."""

    model_config = ConfigDict(extra="ignore")

    prompt: str
    expected: Optional[str] = None
    id: Optional[str] = None
    harness_id: Optional[str] = None
    expected_tools: Optional[list[str]] = None
    slice: Literal["core", "easy", "hard", "ood"] = "core"
    grader: str = "exact"
    tags: Optional[list[str]] = None
    solver_fixture: Optional[str] = None
    ood: bool = False
    bucket: Optional[str] = None
    notes: Optional[str] = None
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
            if v_lower in ("core", "easy", "hard", "ood"):
                return v_lower
        return "core" if v is None else str(v)

    @property
    def is_ood(self) -> bool:
        """A row is OOD iff slice == 'ood' OR ood == true."""
        return self.slice == "ood" or bool(self.ood)

    @property
    def is_hard_or_ood(self) -> bool:
        """Check if row belongs to hard or ood slice."""
        return self.slice == "hard" or self.is_ood

    def validate_for_hard_slice(self) -> None:
        """Validate hard-slice invariant: non-empty id, non-empty expected, slice == 'hard'."""
        if self.slice != "hard":
            raise ValueError(f"Row {self.id or '<unnamed>'} must have slice='hard'")
        if not self.id or not str(self.id).strip():
            raise ValueError(f"Hard slice row missing stable id (prompt: {self.prompt!r})")
        if self.expected is None or not str(self.expected).strip():
            raise ValueError(f"Hard slice row {self.id} must have non-empty expected answer")

    def validate_for_hard_or_ood(self) -> None:
        """Validate invariants for hard or OOD slice rows in Pass@K gate."""
        if not self.is_hard_or_ood:
            raise ValueError(
                f"Row {self.id or '<unnamed>'} must have slice in {{'hard', 'ood'}} or ood=true"
            )
        if not self.id or not str(self.id).strip():
            raise ValueError(f"Row missing stable id (prompt: {self.prompt!r})")
        if self.expected is None or not str(self.expected).strip():
            raise ValueError(f"Row {self.id} must have non-empty expected answer")
        if self.grader != "exact":
            raise ValueError(f"Row {self.id} must have grader='exact', got {self.grader!r}")
        if self.allow_semantic:
            raise ValueError(f"Row {self.id} must have allow_semantic=False")
        if self.is_ood and self.expected not in CANONICAL_REFUSAL_TOKENS:
            raise ValueError(
                f"OOD row {self.id} expected must be canonical refusal in {CANONICAL_REFUSAL_TOKENS}, "
                f"got {self.expected!r}"
            )
