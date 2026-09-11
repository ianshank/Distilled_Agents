"""Load harness YAML; omitted numeric fields inherit MangoMASSettings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from enhanced_system.harness.dispatch import allowed_tools
from enhanced_system.harness.types import HarnessSpec
from enhanced_system.ops.settings import MangoMASSettings, get_settings


def harness_search_dirs(settings: Optional[MangoMASSettings] = None) -> list[Path]:
    """Search MANGOMAS_HARNESS_DIR, repo configs/harnesses, then packaged YAML."""
    settings = settings or get_settings()
    dirs: list[Path] = []
    explicit = settings.harness_dir or os.getenv("MANGOMAS_HARNESS_DIR")
    if explicit:
        dirs.append(Path(explicit))
    package_dir = Path(__file__).resolve().parents[1] / "config" / "harnesses"
    repo_dir = Path(__file__).resolve().parents[2] / "configs" / "harnesses"
    dirs.append(repo_dir)
    dirs.append(package_dir)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in dirs:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def resolve_spec_path(harness_id: str, settings: Optional[MangoMASSettings] = None) -> Path:
    """Return the first `{harness_id}.yaml` on the search path."""
    settings = settings or get_settings()
    if not harness_id:
        raise FileNotFoundError("harness id is required")
    filename = f"{harness_id}.yaml"
    searched: list[str] = []
    for directory in harness_search_dirs(settings):
        candidate = directory / filename
        searched.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"harness not found: {filename} (searched {searched})")


def load_spec(harness_id: str, settings: Optional[MangoMASSettings] = None) -> HarnessSpec:
    """Load `{harness_id}.yaml` and fill defaults from settings."""
    settings = settings or get_settings()
    resolved = resolve_spec_path(harness_id, settings)
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"harness YAML must be a mapping: {resolved}")
    validate_payload(raw, load_schema())
    spec = HarnessSpec.model_validate(raw)
    return apply_settings(spec, settings)


def validate_payload(raw: dict[str, Any], schema: dict[str, Any], path: str = "$") -> None:
    """Validate a YAML mapping against the checked-in JSON Schema (no jsonschema dep)."""
    _check_schema(raw, schema, path)


def apply_settings(spec: HarnessSpec, settings: Optional[MangoMASSettings] = None) -> HarnessSpec:
    """Fill omitted max_steps / window / SAG from settings and cap max_steps."""
    settings = settings or get_settings()
    planning = spec.planning.model_copy()
    memory = spec.memory.model_copy()
    policy = spec.policy.model_copy()
    if planning.max_steps is None:
        planning.max_steps = settings.harness_max_steps
    else:
        planning.max_steps = min(planning.max_steps, settings.harness_max_steps)
    if memory.window_turns is None:
        memory.window_turns = settings.harness_memory_window
    if policy.sag_samples is None:
        policy.sag_samples = settings.harness_sag_samples
    if policy.sag_temperature is None:
        policy.sag_temperature = settings.harness_sag_temperature
    filled = spec.model_copy(update={"planning": planning, "memory": memory, "policy": policy})
    allowed_tools(filled.action.tool_ids)
    return filled


def load_schema() -> dict:
    """Return the checked-in JSON Schema mapping."""
    for directory in harness_search_dirs():
        candidate = directory / "schema.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError("harness schema.json not found")


def _check_schema(value: Any, schema: dict[str, Any], path: str) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    expected = schema.get("type")
    if expected == "object" or "properties" in schema:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        for key in schema.get("required") or []:
            if key not in value:
                raise ValueError(f"{path} missing required key {key}")
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"{path} unknown keys: {extra}")
        for key, child in value.items():
            if key in properties:
                _check_schema(child, properties[key], f"{path}.{key}")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            raise ValueError(f"{path} must have at least {min_items} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _check_schema(item, item_schema, f"{path}[{index}]")
        return
    if expected == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < min_length:
            raise ValueError(f"{path} is shorter than minLength {min_length}")
        return
    if expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path} must be an integer")
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise ValueError(f"{path} is below minimum {minimum}")
        return
    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be a number")
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise ValueError(f"{path} is below minimum {minimum}")
        return
    if expected == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
