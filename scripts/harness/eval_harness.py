#!/usr/bin/env python3
"""Evaluate AgentRuntime on JSONL (not evaluate_agent_skill fixture counting)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts
from enhanced_system.harness.score import answers_match
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate a local harness agent")
    parser.add_argument("--input", required=True, help="JSONL with prompt and optional expected")
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--scripted", default="")
    parser.add_argument("--student", action="store_true", default=False)
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.evaluation_threshold,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="fail on malformed JSONL rows instead of skipping",
    )
    args = parser.parse_args(argv)
    try:
        scripted = json.loads(args.scripted) if args.scripted else None
        runtime = HarnessFactory.create(
            {
                "harness_id": args.harness_id,
                "scripted": scripted,
                "teacher": not args.student,
                "strict_injection": False,
            }
        )
    except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    if not Path(args.input).is_file():
        logger.error("input not found: %s", args.input)
        return 1
    try:
        summary = _evaluate_file(runtime, Path(args.input), args.harness_id, strict=args.strict)
    except (JsonlRowError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    json.dump(summary, sys.stdout)
    sys.stdout.write("\n")
    if summary["total"] and summary["pass_rate"] < args.threshold:
        logger.error(
            "pass_rate %s below threshold %s",
            summary["pass_rate"],
            args.threshold,
        )
        return 1
    return 0


def _evaluate_file(
    runtime: object,
    path: Path,
    harness_id: str,
    *,
    strict: bool,
) -> dict[str, float | int]:
    total = 0
    with_expected = 0
    exact = 0
    truncated = 0
    with_faults = 0
    valid_tools = 0
    tool_steps = 0
    successes = 0
    rows = iter_jsonl_dicts(path, require_prompt=True, strict=strict)
    for _line_no, payload in rows:
        result = runtime.run(str(payload["prompt"]), harness_id=harness_id)  # type: ignore[attr-defined]
        total += 1
        if result.truncated:
            truncated += 1
        if result.trajectory.faults:
            with_faults += 1
        for step in result.trajectory.steps:
            if not step.action or step.fault:
                continue
            tool_steps += 1
            if step.tool_id:
                valid_tools += 1
        expected = payload.get("expected")
        labeled = expected is not None and bool(str(expected).strip())
        if labeled:
            with_expected += 1
            matched = answers_match(result.final_answer, str(expected))
            if matched:
                exact += 1
                successes += 1
        elif not result.truncated:
            successes += 1
    pass_rate = 100.0 * successes / total if total else 0.0
    return {
        "total": total,
        "with_expected": with_expected,
        "exact_match": exact,
        "truncated": truncated,
        "with_faults": with_faults,
        "valid_tool_steps": valid_tools,
        "tool_steps": tool_steps,
        "pass_rate": pass_rate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
