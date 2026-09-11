"""AST/JSON action dispatch with literal arguments only (no exec/eval)."""

from __future__ import annotations

import ast
import json
from typing import Any

from enhanced_system.harness.tools.registry import TOOL_REGISTRY

_FINAL = "final_answer"


class DispatchError(ValueError):
    """Raised when an action cannot be parsed or is not allowlisted."""


def parse_action(text: str, allowed: set[str]) -> tuple[str, dict[str, Any]]:
    """Parse JSON `{tool, args}` or a single allowlisted Call with literals."""
    stripped = (text or "").strip()
    if not stripped:
        raise DispatchError("empty action")
    if stripped.startswith("{"):
        return _parse_json(stripped, allowed)
    return _parse_ast_call(stripped, allowed)


def _parse_json(text: str, allowed: set[str]) -> tuple[str, dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DispatchError(f"invalid json: {exc}") from exc
    if not isinstance(payload, dict):
        raise DispatchError("json action must be an object")
    tool_id = payload.get("tool")
    raw_args: Any = payload["args"] if "args" in payload else {}
    if raw_args is None:
        raw_args = {}
    if not isinstance(tool_id, str) or not tool_id:
        raise DispatchError("json action missing tool")
    if not isinstance(raw_args, dict):
        raise DispatchError("args must be an object")
    args: dict[str, Any] = {str(key): value for key, value in raw_args.items()}
    _ensure_allowed(tool_id, allowed)
    return tool_id, args


def _parse_ast_call(text: str, allowed: set[str]) -> tuple[str, dict[str, Any]]:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise DispatchError(f"invalid action syntax: {exc}") from exc
    body = tree.body
    if not isinstance(body, ast.Call) or not isinstance(body.func, ast.Name):
        raise DispatchError("only a single Name(...) call is allowed")
    if body.args:
        raise DispatchError("positional args are not allowed")
    tool_id = body.func.id
    _ensure_allowed(tool_id, allowed)
    kwargs: dict[str, Any] = {}
    for keyword in body.keywords:
        if keyword.arg is None:
            raise DispatchError("star kwargs are not allowed")
        kwargs[keyword.arg] = _literal(keyword.value)
    return tool_id, kwargs


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        mapping: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise DispatchError("only literal arguments are allowed")
            mapping[_literal(key)] = _literal(value)
        return mapping
    raise DispatchError("only literal arguments are allowed")


def _ensure_allowed(tool_id: str, allowed: set[str]) -> None:
    if tool_id not in TOOL_REGISTRY:
        raise DispatchError(f"unknown tool id: {tool_id}")
    if tool_id not in allowed and tool_id != _FINAL:
        raise DispatchError(f"tool not enabled for this harness: {tool_id}")


def allowed_tools(tool_ids: list[str]) -> set[str]:
    """Union of spec tool ids and final_answer."""
    unknown = [item for item in tool_ids if item not in TOOL_REGISTRY]
    if unknown:
        raise DispatchError(f"unknown tool ids in spec: {unknown}")
    allowed = set(tool_ids)
    allowed.add(_FINAL)
    return allowed
