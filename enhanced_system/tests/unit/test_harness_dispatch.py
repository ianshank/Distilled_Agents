"""Dispatch, tools, and JSON Schema unit tests."""

from __future__ import annotations

import json

import pytest
from enhanced_system.harness.dispatch import DispatchError, parse_action
from enhanced_system.harness.registry import (
    apply_settings,
    load_schema,
    load_spec,
    resolve_spec_path,
    validate_payload,
)
from enhanced_system.harness.tools.registry import get_tool
from enhanced_system.harness.types import HarnessSpec
from enhanced_system.ops.settings import get_settings


@pytest.mark.unit
@pytest.mark.harness
def test_unknown_tool_id_in_spec_rejected():
    spec = HarnessSpec(id="tmp", action={"tool_ids": ["not_a_tool"]})
    with pytest.raises(DispatchError):
        apply_settings(spec)


@pytest.mark.unit
@pytest.mark.harness
def test_json_and_ast_dispatch():
    allowed = {"json_schema", "final_answer"}
    tool_id, args = parse_action(
        '{"tool": "json_schema", "args": {"required": ["a"], "document": {"a": 1}}}',
        allowed,
    )
    assert tool_id == "json_schema"
    tool_id, args = parse_action('final_answer(text="ok")', allowed)
    assert tool_id == "final_answer"
    assert args["text"] == "ok"


@pytest.mark.unit
@pytest.mark.harness
def test_dispatch_rejects_exec_trees():
    allowed = {"final_answer"}
    with pytest.raises(DispatchError):
        parse_action("import os", allowed)
    with pytest.raises(DispatchError):
        parse_action("__import__('os').system('x')", allowed)
    with pytest.raises(DispatchError):
        parse_action("final_answer(text=open('x'))", allowed)


@pytest.mark.unit
@pytest.mark.harness
def test_dispatch_json_and_ast_error_paths():
    allowed = {"json_schema", "final_answer"}
    with pytest.raises(DispatchError):
        parse_action("   ", allowed)
    with pytest.raises(DispatchError):
        parse_action("{", allowed)
    with pytest.raises(DispatchError):
        parse_action("[]", allowed)
    with pytest.raises(DispatchError):
        parse_action('{"args": {}}', allowed)
    with pytest.raises(DispatchError):
        parse_action('{"tool": "final_answer", "args": []}', allowed)
    with pytest.raises(DispatchError):
        parse_action('{"tool": "not_a_tool", "args": {}}', allowed)
    with pytest.raises(DispatchError):
        parse_action('pytest_runner(suite=["a"])', allowed)
    tool_id, args = parse_action(
        'json_schema(required=["a"], document={"a": 1, "nested": [1, 2]})',
        allowed,
    )
    assert tool_id == "json_schema"
    assert args["document"]["nested"] == [1, 2]
    with pytest.raises(DispatchError):
        parse_action("final_answer(*args)", allowed)
    with pytest.raises(DispatchError):
        parse_action("final_answer(text=unknown)", allowed)


@pytest.mark.unit
@pytest.mark.harness
def test_tools_valid_and_invalid():
    assert json.loads(get_tool("json_schema").run({"required": ["a"], "document": {"a": 1}}))["ok"]
    with pytest.raises(ValueError):
        get_tool("json_schema").run({"required": ["a"], "document": {}})
    assert json.loads(get_tool("pytest_runner").run({"suite": ["test_a"]}))["count"] == 1
    with pytest.raises(ValueError):
        get_tool("pytest_runner").run({"suite": []})
    assert json.loads(get_tool("sqe_checklist").run({"assertions": ["assert status == 200"]}))["ok"]
    with pytest.raises(ValueError):
        get_tool("sqe_checklist").run({"assertions": ["nope"]})
    assert (
        json.loads(
            get_tool("architect_structure").run({"nodes": ["api"], "edges": [["api", "db"]]})
        )["nodes"]
        == 1
    )
    assert json.loads(get_tool("pm_acceptance").run({"criteria": ["REQ-1: login"]}))["ids"] == [
        "REQ-1"
    ]
    with pytest.raises(ValueError):
        get_tool("final_answer").run({"text": "  "})


@pytest.mark.unit
@pytest.mark.harness
def test_tools_string_document_and_invalid_payloads():
    assert json.loads(get_tool("json_schema").run({"required": ["a"], "document": '{"a": 1}'}))[
        "ok"
    ]
    with pytest.raises(ValueError):
        get_tool("json_schema").run({"required": [], "document": "{"})
    with pytest.raises(ValueError):
        get_tool("json_schema").run({"required": [], "document": [1]})
    with pytest.raises(ValueError):
        get_tool("architect_structure").run({"nodes": [], "edges": []})
    with pytest.raises(ValueError):
        get_tool("architect_structure").run({"nodes": ["api"], "edges": "nope"})
    with pytest.raises(ValueError):
        get_tool("pm_acceptance").run({"criteria": []})
    with pytest.raises(ValueError):
        get_tool("pm_acceptance").run({"criteria": [""]})
    with pytest.raises(KeyError):
        get_tool("missing")


@pytest.mark.unit
@pytest.mark.harness
def test_schema_lists_required_keys():
    schema = load_schema()
    assert "id" in schema["required"]
    assert "schema_version" in schema["required"]


@pytest.mark.unit
@pytest.mark.harness
def test_schema_rejects_extra_and_empty_id():
    schema = load_schema()
    with pytest.raises(ValueError, match="unknown keys"):
        validate_payload({"id": "x", "schema_version": "1", "nope": 1}, schema)
    with pytest.raises(ValueError, match="minLength"):
        validate_payload({"id": "", "schema_version": "1"}, schema)
    with pytest.raises(ValueError, match="missing required"):
        validate_payload({"schema_version": "1"}, schema)
    with pytest.raises(ValueError, match="one of"):
        validate_payload(
            {"id": "x", "schema_version": "1", "planning": {"style": "nope"}},
            schema,
        )


@pytest.mark.unit
@pytest.mark.harness
def test_load_spec_rejects_non_mapping_and_prefers_explicit_dir(tmp_path, monkeypatch):
    (tmp_path / "bad.yaml").write_text("- not a mapping\n", encoding="utf-8")
    monkeypatch.setenv("MANGOMAS_HARNESS_DIR", str(tmp_path))
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="mapping"):
        load_spec("bad")
    (tmp_path / "base_react.yaml").write_text(
        'id: base_react\nschema_version: "1"\naction:\n  tool_ids: [final_answer]\n',
        encoding="utf-8",
    )
    spec = load_spec("base_react")
    assert spec.action.tool_ids == ["final_answer"]
    assert resolve_spec_path("base_react").parent == tmp_path


@pytest.mark.unit
@pytest.mark.harness
def test_load_spec_empty_yaml_file(tmp_path, monkeypatch):
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("MANGOMAS_HARNESS_DIR", str(tmp_path))
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="missing required"):
        load_spec("empty")


@pytest.mark.unit
@pytest.mark.harness
def test_load_spec_schema_failure_from_yaml(tmp_path, monkeypatch):
    (tmp_path / "broken.yaml").write_text(
        "id: broken\naction:\n  tool_ids: [final_answer]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MANGOMAS_HARNESS_DIR", str(tmp_path))
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="missing required"):
        load_spec("broken")


@pytest.mark.unit
@pytest.mark.harness
def test_load_schema_missing(monkeypatch):
    monkeypatch.setattr(
        "enhanced_system.harness.registry.harness_search_dirs",
        lambda settings=None: [],
    )
    with pytest.raises(FileNotFoundError, match="schema.json"):
        load_schema()
    with pytest.raises(FileNotFoundError, match="harness id is required"):
        resolve_spec_path("")


@pytest.mark.unit
@pytest.mark.harness
def test_schema_type_branches():
    schema = {
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "count": {"type": "integer", "minimum": 1},
            "temp": {"type": "number", "minimum": 0},
            "items": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
    }
    validate_payload({"flag": True, "count": 1, "temp": 0.5, "items": ["a"]}, schema)
    with pytest.raises(ValueError):
        validate_payload("nope", {"type": "object"})
    with pytest.raises(ValueError):
        validate_payload(
            {"flag": "yes"}, {"type": "object", "properties": {"flag": {"type": "boolean"}}}
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"count": True}, {"type": "object", "properties": {"count": {"type": "integer"}}}
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"count": 0},
            {"type": "object", "properties": {"count": {"type": "integer", "minimum": 1}}},
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"temp": True}, {"type": "object", "properties": {"temp": {"type": "number"}}}
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"temp": -1},
            {"type": "object", "properties": {"temp": {"type": "number", "minimum": 0}}},
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"items": "a"}, {"type": "object", "properties": {"items": {"type": "array"}}}
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"items": []},
            {"type": "object", "properties": {"items": {"type": "array", "minItems": 1}}},
        )
    with pytest.raises(ValueError):
        validate_payload({"id": 1}, {"type": "object", "properties": {"id": {"type": "string"}}})
