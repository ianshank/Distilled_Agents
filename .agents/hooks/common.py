#!/usr/bin/env python3
"""Shared gate logic for Claude Code and Antigravity hook scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR") or os.environ.get("AGY_WORKSPACE") or os.getcwd()
    return Path(env).resolve()


def load_contract(root: Path) -> dict:
    for candidate in (
        root / "config" / "workflow-contract.json",
        root / ".claude" / "workflow-contract.json",
        root / ".agents" / "workflow-contract.json",
    ):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {
        "stages": [],
        "allowed_write_prefixes": ["artifacts/", "openspec/changes/"],
        "allowed_commands": ["pytest", "ruff", "mypy", "planlint"],
        "denied_command_patterns": ["rm -rf /", "sudo"],
        "ship_requires_decision": "ship",
    }


def read_stdin_json() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


def current_stage(root: Path) -> str:
    pointer = root / "artifacts" / "stage.json"
    if pointer.exists():
        try:
            return json.loads(pointer.read_text(encoding="utf-8")).get("stage", "discover")
        except json.JSONDecodeError:
            return "discover"
    return "discover"


def artifact_exists(root: Path, name: str) -> bool:
    return (root / "artifacts" / name).exists() or (root / "openspec" / "changes").joinpath(
        name
    ).exists()


def missing_for_stage(root: Path, contract: dict, stage: str) -> list[str]:
    required: list[str] = []
    for item in contract.get("stages", []):
        if item.get("id") == stage:
            required = list(item.get("required", []))
            break
    missing = []
    for name in required:
        path = root / "artifacts" / name
        if name in {"proposal.md", "design.md", "tasks.md"}:
            if path.exists() or any((root / "openspec" / "changes").glob(f"**/{name}")):
                continue
            missing.append(name)
            continue
        if not path.exists():
            missing.append(name)
    return missing


def eval_decision(root: Path) -> str | None:
    path = root / "artifacts" / "eval_decision.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return str(data.get("decision", "")).lower() or None


def path_allowed(root: Path, contract: dict, target: str) -> bool:
    if not target:
        return True
    try:
        rel = os.path.relpath(target, root)
    except ValueError:
        rel = target
    rel = rel.replace("\\", "/")
    if rel.startswith("../") or rel.startswith("/"):
        return False
    stage = current_stage(root)
    if stage in {"build", "evaluate", "ship"}:
        build_prefixes = [
            "enhanced_system/",
            "scripts/",
            "tests/",
            "configs/",
            ".agents/",
            "Makefile",
            "pyproject.toml",
            "README.md",
            "CHANGELOG.md",
            "AGENTS.md",
            "GEMINI.md",
        ]
        if any(rel.startswith(prefix) for prefix in build_prefixes):
            return True
    prefixes = contract.get("allowed_write_prefixes", ["artifacts/"])
    return any(rel.startswith(prefix) for prefix in prefixes)


def command_denied(contract: dict, command: str) -> str | None:
    text = (command or "").strip()
    if not text:
        return None
    for pattern in contract.get("denied_command_patterns", []):
        if pattern in text:
            return f"denied pattern: {pattern}"
    head = text.split()[0]
    head = os.path.basename(head)
    allowed = set(contract.get("allowed_commands", []))
    # git subcommands: allow status/diff/add/commit; deny push --force and reset --hard
    if head == "git":
        if "push --force" in text or "reset --hard" in text or "clean -fd" in text:
            return "destructive git command"
        return None
    if allowed and head not in allowed and head not in {"./" + a for a in allowed}:
        return f"command not allowlisted: {head}"
    return None
