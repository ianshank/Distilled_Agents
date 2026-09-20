#!/usr/bin/env python3
"""Run Chen et al. unbiased Pass@K evaluation gate for hard and OOD golden sets.

Sole CLI entrypoint for Pass@K hard/OOD gating per Phase 0 architecture.
Formula: Chen et al. (arXiv:2107.03374)
    pass@k = E[1 - comb(n - c, k) / comb(n, k)]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.golden import validate_golden_file
from enhanced_system.harness.score import answers_match, calculate_pass_at_k
from enhanced_system.harness.types import GoldenRow
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def evaluate_pass_at_k(
    runtime: Any,
    rows: list[GoldenRow],
    *,
    n: int,
    k: int,
    harness_id: str,
    temperature: float | None = None,
    strict: bool = False,
    scanner: Any | None = None,
) -> dict[str, Any]:
    """Execute n samples per row and compute unbiased pass@1 and pass@k."""
    ood_synthetic_success_violations = 0
    problem_counts: list[tuple[int, int]] = []
    total_problems = len(rows)

    for idx, row in enumerate(rows, 1):
        c = 0
        is_ood = row.is_ood
        expected = row.expected or ""

        for sample_idx in range(n):
            security_failed = False
            row_harness_id = row.harness_id or harness_id
            try:
                result = runtime.run(
                    str(row.prompt),
                    harness_id=row_harness_id,
                    temperature=temperature,
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Error executing prompt %s (sample %d/%d): %s",
                    row.id or f"row_{idx}",
                    sample_idx + 1,
                    n,
                    exc,
                )
                if strict:
                    raise
                continue

            if scanner is not None:
                findings = scanner.scan_trajectory(result.trajectory)
                if findings:
                    logger.warning(
                        "Sample %d on row %s failed security scan: %s",
                        sample_idx + 1,
                        row.id,
                        findings,
                    )
                    security_failed = True

            actual_answer = result.final_answer
            is_match = answers_match(actual_answer, expected) and not security_failed
            answer_mismatch = not answers_match(actual_answer, expected)

            if is_match:
                c += 1
            elif is_ood and answer_mismatch:
                # OOD refusal check: non-empty non-matching answer is an OOD synthetic success violation
                if bool(actual_answer and str(actual_answer).strip()):
                    ood_synthetic_success_violations += 1
                    logger.error(
                        "OOD synthetic-success violation on %s (sample %d/%d): "
                        "emitted %r instead of expected refusal %r",
                        row.id,
                        sample_idx + 1,
                        n,
                        actual_answer,
                        expected,
                    )

        problem_counts.append((n, c))

    pass_at_1 = calculate_pass_at_k(problem_counts, 1)
    pass_at_k_val = calculate_pass_at_k(problem_counts, k)

    return {
        "pass_at_k": {
            "1": pass_at_1,
            str(k): pass_at_k_val,
        },
        "exact_only": True,
        "semantic_counted": False,
        "ood_synthetic_success_violations": ood_synthetic_success_violations,
        "n": n,
        "k": k,
        "total_problems": total_problems,
        "problem_counts": problem_counts,
    }


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Run Chen et al. unbiased Pass@K evaluation gate on hard/OOD golden set"
    )
    parser.add_argument(
        "--golden-set",
        default="configs/golden_sets/sqe_hard_ood.jsonl",
        help="Path to hard/OOD golden set JSONL",
    )
    parser.add_argument(
        "--harness-id",
        default=settings.harness_id or "base_react",
        help="Harness spec identifier",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Target k for Pass@K evaluation (default: MANGOMAS_EVAL_PASS_K)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Samples per problem for Chen estimator (default: MANGOMAS_EVAL_PASS_N)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: MANGOMAS_EVAL_PASS_K_TEMPERATURE)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum required pass@k threshold in range [0, 1.0] (default: 1.0 for Phase 0)",
    )
    parser.add_argument(
        "--scripted",
        type=str,
        help="Path to scripted responses JSON file or JSON string for deterministic testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aqa-passk-summary.json",
        help="Path to write summary JSON artifact (default: aqa-passk-summary.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail fast on execution errors",
    )
    parser.add_argument(
        "--scan-security",
        action="store_true",
        default=False,
        help="Run bandit/security scans on generated trajectories",
    )
    parser.add_argument(
        "--student",
        action="store_true",
        default=False,
        help="Run in student mode (default: teacher)",
    )
    args = parser.parse_args(argv)

    k = args.k if args.k is not None else settings.eval_pass_k
    n = args.n if args.n is not None else settings.eval_pass_n
    temperature = (
        args.temperature
        if args.temperature is not None
        else (settings.eval_pass_k_temperature if k > 1 else 0.2)
    )

    threshold = args.threshold
    if threshold > 1.0 and threshold <= 100.0:
        threshold = threshold / 100.0
    if threshold < 0.0 or threshold > 1.0:
        logger.error("threshold must be in [0, 1] or [0, 100], got %s", args.threshold)
        return 1

    if n < k:
        logger.error("Sample count n=%d must be >= k=%d", n, k)
        return 1
    if k <= 0:
        logger.error("k must be positive, got %d", k)
        return 1

    golden_path = Path(args.golden_set)
    if not golden_path.is_file():
        logger.error("Golden set not found: %s", golden_path)
        return 1

    # Validate golden set bucket minima and invariants
    try:
        validation_result = validate_golden_file(golden_path)
        logger.info(
            "Golden set validation passed (%d rows, %d buckets)",
            validation_result["total"],
            len(validation_result["bucket_counts"]),
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Golden set validation failed for %s: %s", golden_path, exc)
        return 1

    # Load rows as GoldenRow
    rows: list[GoldenRow] = []
    with golden_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rows.append(GoldenRow.model_validate(json.loads(line_str)))

    # Parse scripted fixtures if provided
    scripted_payload = None
    if args.scripted:
        scripted_path = Path(args.scripted)
        if scripted_path.is_file():
            try:
                scripted_payload = json.loads(scripted_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("Invalid scripted responses file %s: %s", scripted_path, exc)
                return 1
        else:
            try:
                scripted_payload = json.loads(args.scripted)
            except json.JSONDecodeError as exc:
                logger.error("Invalid --scripted JSON string: %s", exc)
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
                "scripted": scripted_payload,
                "teacher": not args.student,
                "strict_injection": False,
                "security_scanner": scanner,
            }
        )
    except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Failed to initialize AgentRuntime: %s", exc)
        return 1

    logger.info(
        "Running Pass@K Gate on %s (n=%d, k=%d, threshold=%.2f, temp=%.2f)...",
        golden_path.name,
        n,
        k,
        threshold,
        temperature,
    )

    result_dict = evaluate_pass_at_k(
        runtime,
        rows,
        n=n,
        k=k,
        harness_id=args.harness_id,
        temperature=temperature,
        strict=args.strict,
        scanner=scanner,
    )

    summary: dict[str, Any] = {
        "pass_at_k": result_dict["pass_at_k"],
        "exact_only": True,
        "semantic_counted": False,
        "ood_synthetic_success_violations": result_dict["ood_synthetic_success_violations"],
        "n": n,
        "k": k,
        "golden_path": str(golden_path),
        "golden path": str(golden_path),
        "harness_id": args.harness_id,
        "harness id": args.harness_id,
        "total_problems": result_dict["total_problems"],
        "threshold": threshold,
    }

    # Dump summary to stdout
    summary_json = json.dumps(summary, indent=2)
    sys.stdout.write(summary_json + "\n")

    # Write summary artifact
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary_json, encoding="utf-8")
        logger.info("Summary artifact written to %s", out_path)

    # Gate decision
    target_pass_k = result_dict["pass_at_k"].get(str(k), 0.0)
    violations = result_dict["ood_synthetic_success_violations"]

    failed = False
    if violations > 0:
        logger.error("AQA Pass@K Gate FAILED: %d OOD synthetic-success violation(s)", violations)
        failed = True
    if target_pass_k < threshold:
        logger.error(
            "AQA Pass@K Gate FAILED: pass@%d = %.4f below threshold %.4f",
            k,
            target_pass_k,
            threshold,
        )
        failed = True

    if failed:
        return 1

    logger.info(
        "AQA Pass@K Gate PASSED (pass@1=%.4f, pass@%d=%.4f, violations=%d).",
        result_dict["pass_at_k"]["1"],
        k,
        target_pass_k,
        violations,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
