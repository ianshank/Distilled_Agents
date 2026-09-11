#!/usr/bin/env python3
"""Collect teacher/student traces from prompt/completion JSONL (local)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enhanced_system.harness.convert import trajectory_to_legacy
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.traces import JsonlTraceStore
from enhanced_system.ops.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Collect harness trajectories")
    parser.add_argument("--input", required=True, help="Source JSONL with prompt field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--scripted", default="")
    args = parser.parse_args(argv)
    scripted = json.loads(args.scripted) if args.scripted else None
    out_path = Path(args.output)
    store = JsonlTraceStore(out_path.with_suffix(".raw.jsonl"))
    runtime = HarnessFactory.create(
        {
            "harness_id": args.harness_id,
            "scripted": scripted,
            "teacher": True,
            "store": store,
        }
    )
    rows = []
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt") or ""
            result = runtime.run(prompt, harness_id=args.harness_id)
            rows.append(trajectory_to_legacy(result.trajectory))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
