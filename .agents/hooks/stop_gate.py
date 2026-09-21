#!/usr/bin/env python3
"""Antigravity Stop hook: exit 2 when required artifacts are missing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    current_stage,
    eval_decision,
    load_contract,
    missing_for_stage,
    project_root,
    read_stdin_json,
)


def main() -> None:
    payload = read_stdin_json()
    if payload.get("stop_hook_active"):
        sys.exit(0)
    root = project_root()
    contract = load_contract(root)
    stage = current_stage(root)
    missing = missing_for_stage(root, contract, stage)
    if missing:
        print(f"stage {stage} missing artifacts: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)
    if stage == "ship" and eval_decision(root) != contract.get("ship_requires_decision", "ship"):
        print("ship blocked: eval_decision.json is not ship", file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
