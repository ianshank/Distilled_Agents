#!/usr/bin/env python3
"""AQA Regression Gate: Multi-trial evaluation of golden sets enforcing pass@1, pass@k, and hard-slice thresholds."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess  # nosec: B404
import sys
from pathlib import Path
from typing import Any

from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def run_gate(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="AQA Regression Gate enforcing multi-trial pass@1, pass@k, and hard-slice thresholds"
    )
    parser.add_argument("--golden-set", required=True, help="Path to golden set JSONL")
    parser.add_argument(
        "--threshold",
        type=float,
        default=75.0,
        help="Minimum overall pass@k percentage threshold [0-100]",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=None,
        help="Minimum pass@k percentage threshold for hard-slice rows (defaults to --threshold)",
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
        help="Sampling temperature for multi-trial evaluation",
    )
    parser.add_argument(
        "--scripted",
        type=str,
        help="Path to scripted responses JSON file for deterministic CI tests",
    )
    parser.add_argument(
        "--scan-security",
        action="store_true",
        default=False,
        help="Run bandit/security scans on generated agent code",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on the first row error (default: skip and continue)",
    )
    parser.add_argument(
        "--require-hard",
        action="store_true",
        default=False,
        help="Enforce that all rows in the golden set satisfy hard-slice requirements",
    )
    args = parser.parse_args(argv)

    golden_set_path = Path(args.golden_set)
    if not golden_set_path.is_file():
        logger.error("Golden set not found: %s", golden_set_path)
        return 1

    threshold = args.threshold * 100.0 if 0.0 <= args.threshold <= 1.0 else args.threshold
    if threshold < 0.0 or threshold > 100.0:
        logger.error("Threshold must be in range [0, 100] or [0, 1].")
        return 1

    if args.hard_threshold is not None:
        hard_thresh = (
            args.hard_threshold * 100.0
            if 0.0 <= args.hard_threshold <= 1.0
            else args.hard_threshold
        )
        if hard_thresh < 0.0 or hard_thresh > 100.0:
            logger.error("Hard threshold must be in range [0, 100] or [0, 1].")
            return 1
    else:
        hard_thresh = threshold

    pass_k = max(1, args.pass_k)

    cmd = [
        sys.executable,
        "scripts/harness/eval_harness.py",
        "--input",
        str(golden_set_path),
        "--threshold",
        str(threshold),
        "--pass-k",
        str(pass_k),
    ]
    if args.scan_security:
        cmd.append("--scan-security")
    if args.strict:
        cmd.append("--strict")
    if args.require_hard:
        cmd.append("--require-hard")
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])

    if args.scripted:
        scripted_path = Path(args.scripted)
        if not scripted_path.is_file():
            logger.error("Scripted responses file not found: %s", scripted_path)
            return 1
        try:
            scripted_payload = json.loads(scripted_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Invalid scripted responses file %s: %s", scripted_path, exc)
            return 1
        cmd.extend(["--scripted", json.dumps(scripted_payload)])

    logger.info(
        "Running AQA Regression Gate against %s (pass@%s, threshold=%.1f%%, hard_threshold=%.1f%%)...",
        golden_set_path.name,
        pass_k,
        threshold,
        hard_thresh,
    )
    # nosec: B603 - Controlled list args invoking sys.executable with shell=False for regression gate (green-trunk-ci)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # nosec: B603

    summary: dict[str, Any] = {}
    if result.stdout:
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                summary = json.loads(line)
                if isinstance(summary, dict) and "pass_at_k" in summary:
                    break
            except json.JSONDecodeError:
                continue
        print(result.stdout.strip())

    if not summary or "pass_at_1" not in summary or "pass_at_k" not in summary:
        logger.error("AQA Gate FAILED: summary JSON missing required pass_at_1 or pass_at_k keys")
        if result.stderr:
            logger.error("Error details:\n%s", result.stderr.strip())
        return 1

    if result.returncode != 0:
        logger.error("AQA Gate FAILED (exit code %s)", result.returncode)
        if result.stderr:
            logger.error("Error details:\n%s", result.stderr.strip())
        return 1

    overall_pass_k = float(summary.get("pass_at_k", 0.0))
    if summary.get("total", 0) > 0 and overall_pass_k < threshold:
        logger.error(
            "AQA Gate FAILED: overall pass_at_k %.2f%% is below threshold %.2f%%",
            overall_pass_k,
            threshold,
        )
        return 1

    hard_total = int(summary.get("hard_total", 0))
    hard_pass_k = float(summary.get("hard_pass_at_k", 0.0))
    if hard_total > 0 and hard_pass_k < hard_thresh:
        logger.error(
            "AQA Gate FAILED: hard-slice pass_at_k %.2f%% is below hard threshold %.2f%%",
            hard_pass_k,
            hard_thresh,
        )
        return 1

    logger.info(
        "AQA Gate PASSED (pass@1: %.2f%%, pass@%s: %.2f%%, hard_pass@%s: %.2f%%).",
        float(summary.get("pass_at_1", 0.0)),
        pass_k,
        overall_pass_k,
        pass_k,
        hard_pass_k,
    )
    return 0


def main() -> int:
    return run_gate()


if __name__ == "__main__":
    sys.exit(main())
