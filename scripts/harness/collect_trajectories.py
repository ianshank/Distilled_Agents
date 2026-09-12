#!/usr/bin/env python3
"""Collect teacher/student traces from prompt/completion JSONL (local)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from enhanced_system.harness.convert import trajectory_to_legacy
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.traces import JsonlTraceStore
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def raw_store_path(out_path: Path) -> Path:
    """Return a distinct raw-trace path so collect never truncates its own store."""
    if out_path.name.endswith(".raw.jsonl"):
        stem = out_path.name[: -len(".raw.jsonl")]
        return out_path.with_name(f"{stem}.raw.traces.jsonl")
    return out_path.with_suffix(".raw.jsonl")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Collect harness trajectories")
    parser.add_argument("--input", required=True, help="Source JSONL with prompt field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--scripted", default="")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on the first row error (default: skip and continue)",
    )
    args = parser.parse_args(argv)
    try:
        scripted = json.loads(args.scripted) if args.scripted else None
    except json.JSONDecodeError as exc:
        logger.error("invalid --scripted JSON: %s", exc)
        return 1
    out_path = Path(args.output)
    try:
        store = JsonlTraceStore(raw_store_path(out_path))
        runtime = HarnessFactory.create(
            {
                "harness_id": args.harness_id,
                "scripted": scripted,
                "teacher": True,
                "store": store,
                "strict_injection": False,
            }
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    rows = []
    try:
        handle = Path(args.input).open(encoding="utf-8")
    except OSError as exc:
        logger.error("%s", exc)
        return 1
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"row must be a JSON object, got {type(payload).__name__}")
                prompt = str(payload.get("prompt") or "")
                if not prompt.strip():
                    logger.warning("skipping empty prompt at line %s", line_no)
                    if args.strict:
                        return 1
                    continue
                result = runtime.run(prompt, harness_id=args.harness_id)
                rows.append(trajectory_to_legacy(result.trajectory))
            except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
                logger.warning("skipping line %s: %s", line_no, exc)
                if args.strict:
                    return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    logger.info("wrote %s rows to %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
