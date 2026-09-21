#!/usr/bin/env python3
"""Antigravity PreToolUse gate. Exit 2 to block."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import command_denied, load_contract, path_allowed, project_root, read_stdin_json


def main() -> None:
    payload = read_stdin_json()
    root = project_root()
    contract = load_contract(root)
    tool = payload.get("tool_name") or payload.get("tool") or payload.get("name") or ""
    tool_input = (
        payload.get("tool_input") or payload.get("arguments") or payload.get("params") or {}
    )

    if tool in {"run_command", "Bash"}:
        command = tool_input.get("command") or tool_input.get("cmd") or ""
        reason = command_denied(contract, command)
        if reason:
            print(reason, file=sys.stderr)
            sys.exit(2)

    if tool in {"write_to_file", "replace_file_content", "Write", "Edit"}:
        target = (
            tool_input.get("TargetFile")
            or tool_input.get("target_file")
            or tool_input.get("file_path")
            or tool_input.get("path")
            or ""
        )
        if target and not path_allowed(root, contract, target):
            print(f"write outside allowed prefixes: {target}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
