"""Frozen tool implementations keyed by id (no YAML class_path imports)."""

from __future__ import annotations

import json
from typing import Any


class JsonSchemaTool:
    """Validate that payload JSON includes required keys."""

    tool_id = "json_schema"

    def run(self, payload: dict[str, Any]) -> str:
        required = payload.get("required") or []
        document = payload.get("document")
        if isinstance(document, str):
            try:
                document = json.loads(document)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json document: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError("document must be an object")
        missing = [key for key in required if key not in document]
        if missing:
            raise ValueError(f"missing keys: {missing}")
        return json.dumps({"ok": True, "keys": sorted(document.keys())})


class PytestRunnerTool:
    """Checklist that recorded test ids are present (no subprocess)."""

    tool_id = "pytest_runner"

    def run(self, payload: dict[str, Any]) -> str:
        suite = payload.get("suite") or []
        if not isinstance(suite, list) or not suite:
            raise ValueError("suite must be a non-empty list of test ids")
        missing = [item for item in suite if not isinstance(item, str) or not item.strip()]
        if missing:
            raise ValueError("suite entries must be non-empty strings")
        return json.dumps({"ok": True, "count": len(suite)})


class SqeChecklistTool:
    """Validate assertion-style checklist items."""

    tool_id = "sqe_checklist"

    def run(self, payload: dict[str, Any]) -> str:
        items = payload.get("assertions") or []
        if not isinstance(items, list) or not items:
            raise ValueError("assertions must be a non-empty list")
        for item in items:
            if not isinstance(item, str) or "assert" not in item.lower():
                raise ValueError("each assertion must mention assert")
        return json.dumps({"ok": True, "count": len(items)})


class ArchitectStructureTool:
    """C4-lite structure check: node and edge counts from payload."""

    tool_id = "architect_structure"

    def run(self, payload: dict[str, Any]) -> str:
        nodes = payload.get("nodes") or []
        edges = payload.get("edges") or []
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("nodes must be a non-empty list")
        if not isinstance(edges, list):
            raise ValueError("edges must be a list")
        return json.dumps({"ok": True, "nodes": len(nodes), "edges": len(edges)})


class PmAcceptanceTool:
    """Extract requirement ids from an acceptance-criteria list."""

    tool_id = "pm_acceptance"

    def run(self, payload: dict[str, Any]) -> str:
        items = payload.get("criteria") or []
        if not isinstance(items, list) or not items:
            raise ValueError("criteria must be a non-empty list")
        ids = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("criteria entries must be non-empty strings")
            ids.append(item.split(":", 1)[0].strip())
        return json.dumps({"ok": True, "ids": ids})


class FinalAnswerTool:
    """Terminate the loop with a user-visible answer."""

    tool_id = "final_answer"

    def run(self, payload: dict[str, Any]) -> str:
        text = payload.get("text")
        if text is None:
            text = payload.get("answer") or ""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("final_answer requires text")
        return text.strip()


TOOL_REGISTRY: dict[str, type[Any]] = {
    JsonSchemaTool.tool_id: JsonSchemaTool,
    PytestRunnerTool.tool_id: PytestRunnerTool,
    SqeChecklistTool.tool_id: SqeChecklistTool,
    ArchitectStructureTool.tool_id: ArchitectStructureTool,
    PmAcceptanceTool.tool_id: PmAcceptanceTool,
    FinalAnswerTool.tool_id: FinalAnswerTool,
}


def get_tool(tool_id: str) -> Any:
    """Instantiate a registered tool or raise KeyError."""
    cls = TOOL_REGISTRY.get(tool_id)
    if cls is None:
        raise KeyError(f"unknown tool id: {tool_id}")
    return cls()
