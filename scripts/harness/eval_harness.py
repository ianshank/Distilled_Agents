#!/usr/bin/env python3
"""Evaluate AgentRuntime on JSONL (not evaluate_agent_skill fixture counting)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts
from enhanced_system.harness.score import answers_match
from enhanced_system.harness.types import GoldenRow
from enhanced_system.ops.settings import get_settings
from pydantic import ValidationError

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
        help="Fail on the first row error (default: skip and continue)",
    )
    parser.add_argument(
        "--scan-security",
        action="store_true",
        default=False,
        help="Run bandit/security scans on generated agent code. Fails the row if issues found.",
    )
    args = parser.parse_args(argv)
    try:
        scripted = json.loads(args.scripted) if args.scripted else None
    except json.JSONDecodeError as exc:
        logger.error("invalid --scripted JSON: %s", exc)
        return 1
    scanner = None
    if args.scan_security:
        try:
            from enhanced_system.harness.security import SecurityScanner

            scanner = SecurityScanner(fail_on_high=True)
            logger.info("Security scanning (Bandit) enabled on agent outputs")
        except ImportError as exc:
            logger.error("Cannot enable security scanning: %s", exc)
            return 1
    try:
        runtime = HarnessFactory.create(
            {
                "harness_id": args.harness_id,
                "scripted": scripted,
                "teacher": not args.student,
                "strict_injection": False,
                "security_scanner": scanner,
            }
        )
    except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    if not Path(args.input).is_file():
        logger.error("input not found: %s", args.input)
        return 1
    try:
        summary = _evaluate_file(
            runtime, Path(args.input), args.harness_id, strict=args.strict, scanner=scanner
        )
    except (JsonlRowError, OSError, ValueError, RuntimeError) as exc:
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
    scanner: Any | None = None,
) -> dict[str, float | int]:
    total = 0
    with_expected = 0
    exact = 0
    semantic_matches = 0
    tool_sequence_exact = 0
    truncated = 0
    with_faults = 0
    valid_tools = 0
    tool_steps = 0
    successes = 0
    rows = iter_jsonl_dicts(path, require_prompt=True, strict=strict)
    for line_no, payload in rows:
        try:
            row = GoldenRow.model_validate(payload)
        except (TypeError, ValidationError, ValueError) as exc:
            logger.warning("skipping line %s: %s", line_no, exc)
            if strict:
                raise
            continue
        row_harness_id = row.harness_id or harness_id
        try:
            result = runtime.run(  # type: ignore[attr-defined]
                str(row.prompt),
                harness_id=row_harness_id,
            )
        except ValueError as exc:
            logger.warning("skipping line %s: %s", line_no, exc)
            if strict:
                raise
            continue

        total += 1
        security_failed = False

        # Security scan
        if scanner is not None:
            findings = scanner.scan_trajectory(result.trajectory)
            if findings:
                logger.error("Line %s failed security scan: %s", line_no, findings)
                if strict:
                    raise ValueError(f"Security vulnerability generated at line {line_no}")
                # Count as a fault if not strict
                result.trajectory.faults.append(f"Security finding: {len(findings)} issues")
                security_failed = True

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
        expected = row.expected
        expected_tools = row.expected_tools
        labeled = expected is not None and bool(str(expected).strip())

        # Tool sequence accuracy
        if expected_tools is not None and isinstance(expected_tools, list):
            actual_tools = [
                step.tool_id
                for step in result.trajectory.steps
                if step.tool_id and not step.fault and step.action
            ]
            if actual_tools == expected_tools:
                tool_sequence_exact += 1

        if labeled:
            with_expected += 1
            from enhanced_system.harness.score import semantic_match

            matched = answers_match(result.final_answer, str(expected))
            sem_matched = semantic_match(result.final_answer, str(expected))
            if matched and not security_failed:
                exact += 1
                successes += 1
            elif (
                row.allow_semantic
                and row.grader != "exact"
                and not row.is_hard_or_ood
                and sem_matched
                and not security_failed
            ):
                semantic_matches += 1
                successes += 1
        elif not result.truncated and not security_failed:
            successes += 1
    pass_rate = 100.0 * successes / total if total else 0.0
    return {
        "total": total,
        "with_expected": with_expected,
        "exact_match": exact,
        "semantic_match": semantic_matches,
        "tool_sequence_exact": tool_sequence_exact,
        "truncated": truncated,
        "with_faults": with_faults,
        "valid_tool_steps": valid_tools,
        "tool_steps": tool_steps,
        "pass_rate": pass_rate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
