#!/usr/bin/env python3
"""Evaluate AgentRuntime on JSONL with multi-trial pass@1 and pass@k metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts
from enhanced_system.harness.score import answers_match, semantic_match
from enhanced_system.harness.types import GoldenRow
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Evaluate a local harness agent with multi-trial pass@k"
    )
    parser.add_argument("--input", required=True, help="JSONL with prompt and optional expected")
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--scripted", default="")
    parser.add_argument("--student", action="store_true", default=False)
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.evaluation_threshold,
        help="Fail if pass_at_k (0-100) is below this threshold",
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
    parser.add_argument(
        "--pass-k",
        type=int,
        default=settings.eval_pass_k,
        help="Number of independent trials per prompt for pass@k evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for multi-trial evaluation (default from settings)",
    )
    parser.add_argument(
        "--require-hard",
        action="store_true",
        default=False,
        help="Enforce that every row satisfies hard-slice requirements (slice=hard, id, expected)",
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

    pass_k = max(1, args.pass_k)
    eval_temp = (
        args.temperature
        if args.temperature is not None
        else (settings.eval_pass_k_temperature if pass_k > 1 else 0.0)
    )

    try:
        summary = _evaluate_file(
            runtime,
            Path(args.input),
            args.harness_id,
            strict=args.strict,
            scanner=scanner,
            pass_k=pass_k,
            temperature=eval_temp,
            require_hard=args.require_hard,
        )
    except (JsonlRowError, OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    json.dump(summary, sys.stdout)
    sys.stdout.write("\n")
    if summary["total"] and summary["pass_at_k"] < args.threshold:
        logger.error(
            "pass_at_k %s below threshold %s",
            summary["pass_at_k"],
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
    pass_k: int = 1,
    temperature: float | None = None,
    require_hard: bool = False,
) -> dict[str, Any]:
    rows = iter_jsonl_dicts(path, require_prompt=True, strict=strict)
    parsed_rows: list[tuple[int, GoldenRow]] = []
    for line_no, payload in rows:
        try:
            row = GoldenRow.model_validate(payload)
            if require_hard:
                row.validate_for_hard_slice()
            parsed_rows.append((line_no, row))
        except (ValueError, TypeError) as exc:
            logger.warning("skipping line %s: %s", line_no, exc)
            if strict:
                raise JsonlRowError(f"line {line_no} validation error: {exc}") from exc
            continue

    if not parsed_rows:
        return {
            "total": 0,
            "k": pass_k,
            "pass_at_1": 0.0,
            "pass_at_k": 0.0,
            "pass_rate": 0.0,
            "with_expected": 0,
            "exact_match": 0,
            "semantic_match": 0,
            "tool_sequence_exact": 0,
            "truncated": 0,
            "with_faults": 0,
            "valid_tool_steps": 0,
            "tool_steps": 0,
            "core_total": 0,
            "core_pass_at_1": 0.0,
            "core_pass_at_k": 0.0,
            "hard_total": 0,
            "hard_pass_at_1": 0.0,
            "hard_pass_at_k": 0.0,
        }

    row_trial_successes: list[list[bool]] = [[] for _ in range(len(parsed_rows))]
    row_has_exact: list[bool] = [False] * len(parsed_rows)
    row_has_semantic: list[bool] = [False] * len(parsed_rows)
    row_has_tool_seq_exact: list[bool] = [False] * len(parsed_rows)
    row_has_truncated: list[bool] = [False] * len(parsed_rows)
    row_has_faults: list[bool] = [False] * len(parsed_rows)
    row_skipped_due_to_runtime_error: set[int] = set()

    total_valid_tool_steps = 0
    total_tool_steps = 0

    for trial_idx in range(pass_k):
        backend = getattr(runtime, "backend", None)
        if backend is not None and hasattr(backend, "reset"):
            backend.reset()

        for idx, (line_no, row) in enumerate(parsed_rows):
            if idx in row_skipped_due_to_runtime_error:
                continue

            try:
                result = runtime.run(  # type: ignore[attr-defined]
                    row.prompt,
                    harness_id=row.harness_id or harness_id,
                    temperature=temperature,
                )
            except ValueError as exc:
                logger.warning("skipping line %s: %s", line_no, exc)
                if strict:
                    raise
                row_skipped_due_to_runtime_error.add(idx)
                continue

            security_failed = False
            if scanner is not None:
                findings = scanner.scan_trajectory(result.trajectory)
                if findings:
                    logger.error(
                        "Line %s trial %s failed security scan: %s",
                        line_no,
                        trial_idx + 1,
                        findings,
                    )
                    if strict:
                        raise ValueError(f"Security vulnerability generated at line {line_no}")
                    result.trajectory.faults.append(f"Security finding: {len(findings)} issues")
                    security_failed = True

            if result.truncated:
                row_has_truncated[idx] = True
            if result.trajectory.faults:
                row_has_faults[idx] = True

            for step in result.trajectory.steps:
                if not step.action or step.fault:
                    continue
                total_tool_steps += 1
                if step.tool_id:
                    total_valid_tool_steps += 1

            if row.expected_tools is not None and isinstance(row.expected_tools, list):
                actual_tools = [
                    step.tool_id
                    for step in result.trajectory.steps
                    if step.tool_id and not step.fault and step.action
                ]
                if actual_tools == row.expected_tools:
                    row_has_tool_seq_exact[idx] = True

            expected = row.expected
            labeled = expected is not None and bool(str(expected).strip())
            trial_success = False

            if labeled:
                matched = answers_match(result.final_answer, str(expected))
                sem_matched = semantic_match(result.final_answer, str(expected))
                if matched:
                    row_has_exact[idx] = True
                if sem_matched:
                    row_has_semantic[idx] = True

                if row.slice == "hard":
                    if matched and not security_failed:
                        trial_success = True
                else:
                    if matched and not security_failed:
                        trial_success = True
                    elif row.allow_semantic and sem_matched and not security_failed:
                        trial_success = True
            elif not result.truncated and not security_failed:
                trial_success = True

            row_trial_successes[idx].append(trial_success)

    evaluated_indices = [
        i for i in range(len(parsed_rows)) if i not in row_skipped_due_to_runtime_error
    ]
    total_evaluated = len(evaluated_indices)

    with_expected_count = 0
    exact_count = 0
    semantic_count = 0
    tool_sequence_count = 0
    truncated_count = 0
    with_faults_count = 0

    pass_at_1_count = 0
    pass_at_k_count = 0

    core_total = 0
    core_pass_at_1_count = 0
    core_pass_at_k_count = 0

    hard_total = 0
    hard_pass_at_1_count = 0
    hard_pass_at_k_count = 0

    for idx in evaluated_indices:
        line_no, row = parsed_rows[idx]
        is_hard = row.slice == "hard"
        if is_hard:
            hard_total += 1
        else:
            core_total += 1

        labeled = row.expected is not None and bool(str(row.expected).strip())
        if labeled:
            with_expected_count += 1

        if row_has_exact[idx]:
            exact_count += 1
        if row_has_semantic[idx]:
            semantic_count += 1
        if row_has_tool_seq_exact[idx]:
            tool_sequence_count += 1
        if row_has_truncated[idx]:
            truncated_count += 1
        if row_has_faults[idx]:
            with_faults_count += 1

        trials = row_trial_successes[idx]
        p_1 = bool(trials and trials[0])
        p_k = any(trials)

        if p_1:
            pass_at_1_count += 1
            if is_hard:
                hard_pass_at_1_count += 1
            else:
                core_pass_at_1_count += 1

        if p_k:
            pass_at_k_count += 1
            if is_hard:
                hard_pass_at_k_count += 1
            else:
                core_pass_at_k_count += 1

        logger.info(
            "eval row line=%s id=%s slice=%s trials=%s pass_at_1=%s pass_at_k=%s",
            line_no,
            row.id or f"row-{line_no}",
            row.slice,
            len(trials),
            p_1,
            p_k,
        )

    pass_at_1_rate = 100.0 * pass_at_1_count / total_evaluated if total_evaluated else 0.0
    pass_at_k_rate = 100.0 * pass_at_k_count / total_evaluated if total_evaluated else 0.0

    core_pass_at_1_rate = 100.0 * core_pass_at_1_count / core_total if core_total else 0.0
    core_pass_at_k_rate = 100.0 * core_pass_at_k_count / core_total if core_total else 0.0

    hard_pass_at_1_rate = 100.0 * hard_pass_at_1_count / hard_total if hard_total else 0.0
    hard_pass_at_k_rate = 100.0 * hard_pass_at_k_count / hard_total if hard_total else 0.0

    logger.info(
        "eval summary: total=%s k=%s pass_at_1=%.2f%% pass_at_k=%.2f%% core_pass_at_k=%.2f%% hard_pass_at_k=%.2f%%",
        total_evaluated,
        pass_k,
        pass_at_1_rate,
        pass_at_k_rate,
        core_pass_at_k_rate,
        hard_pass_at_k_rate,
    )

    return {
        "total": total_evaluated,
        "k": pass_k,
        "pass_at_1": pass_at_1_rate,
        "pass_at_k": pass_at_k_rate,
        "pass_rate": pass_at_k_rate,
        "with_expected": with_expected_count,
        "exact_match": exact_count,
        "semantic_match": semantic_count,
        "tool_sequence_exact": tool_sequence_count,
        "truncated": truncated_count,
        "with_faults": with_faults_count,
        "valid_tool_steps": total_valid_tool_steps,
        "tool_steps": total_tool_steps,
        "core_total": core_total,
        "core_pass_at_1": core_pass_at_1_rate,
        "core_pass_at_k": core_pass_at_k_rate,
        "hard_total": hard_total,
        "hard_pass_at_1": hard_pass_at_1_rate,
        "hard_pass_at_k": hard_pass_at_k_rate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
