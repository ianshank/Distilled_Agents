#!/usr/bin/env python3
"""Collect teacher/student traces from prompt/completion JSONL (local)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from enhanced_system.harness.convert import trajectory_to_legacy
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts
from enhanced_system.harness.score import answers_match
from enhanced_system.harness.traces import JsonlTraceStore, raw_store_path
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Collect harness trajectories")
    parser.add_argument("--input", required=True, help="Source JSONL with prompt field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--scripted", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--teacher",
        dest="student",
        action="store_false",
        help="Collect with the teacher model (default)",
    )
    mode.add_argument(
        "--student",
        dest="student",
        action="store_true",
        help="Collect with the student model",
    )
    parser.set_defaults(student=False)
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
                "teacher": not args.student,
                "store": store,
                "strict_injection": False,
            }
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    try:
        rows = _collect_rows(runtime, Path(args.input), args.harness_id, args.strict)
    except JsonlRowError as exc:
        logger.error("%s", exc)
        return 1
    except OSError as exc:
        logger.error("%s", exc)
        return 1
    if rows is None:
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    logger.info("wrote %s rows to %s", len(rows), out_path)
    return 0


def _collect_rows(
    runtime: object,
    path: Path,
    harness_id: str,
    strict: bool,
) -> list[dict[str, object]] | None:
    rows: list[dict[str, object]] = []
    for line_no, payload in iter_jsonl_dicts(path, require_prompt=True, strict=strict):
        try:
            prompt = str(payload["prompt"])
            result = runtime.run(prompt, harness_id=harness_id)  # type: ignore[attr-defined]
            expected = payload.get("expected")
            if expected is not None and str(expected).strip():
                if not answers_match(result.final_answer, str(expected)):
                    logger.warning("skipping outcome mismatch at line %s", line_no)
                    if strict:
                        return None
                    continue
            rows.append(trajectory_to_legacy(result.trajectory, expected=expected))
        except (ValueError, FileNotFoundError, OSError) as exc:
            logger.warning("skipping line %s: %s", line_no, exc)
            if strict:
                return None
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
