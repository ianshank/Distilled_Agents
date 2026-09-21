#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def main() -> None:
    required = [
        ROOT / "GEMINI.md",
        ROOT / "AGENTS.md",
        ROOT / "config" / "workflow-contract.json",
        ROOT / ".agents" / "hooks.json",
        ROOT / ".agents" / "plugin.json",
        ROOT / ".agents" / "hooks" / "common.py",
        ROOT / ".agents" / "hooks" / "pre_tool_use.py",
        ROOT / ".agents" / "hooks" / "stop_gate.py",
        ROOT / ".agents" / "agents" / "delivery.md",
        ROOT / ".agents" / "agents" / "researcher.md",
        ROOT / ".agents" / "skills" / "product-sdlc" / "SKILL.md",
        ROOT / ".agents" / "rules" / "workflow-invariants.md",
    ]
    for path in required:
        if not path.exists():
            fail(f"missing {path.relative_to(ROOT)}")
    json.loads((ROOT / "config" / "workflow-contract.json").read_text())
    json.loads((ROOT / ".agents" / "hooks.json").read_text())
    json.loads((ROOT / ".agents" / "plugin.json").read_text())
    for agent in (ROOT / ".agents" / "agents").glob("*.md"):
        if not agent.read_text(encoding="utf-8").startswith("---"):
            fail(f"{agent.name} missing frontmatter")
    print("OK antigravity package")


if __name__ == "__main__":
    main()
