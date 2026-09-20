#!/usr/bin/env python3
"""AQA Regression Gate: Runs Golden Sets and enforces Pass@K thresholds."""

import argparse
import json
import logging
import subprocess  # nosec: B404
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="AQA Regression Gate")
    parser.add_argument("--golden-set", required=True, help="Path to golden set JSONL")
    parser.add_argument("--threshold", type=float, default=75.0, help="Minimum pass rate (0-100)")
    parser.add_argument(
        "--scripted",
        type=str,
        help="Path to scripted responses JSON file for deterministic CI tests",
    )
    args = parser.parse_args()

    golden_set_path = Path(args.golden_set)
    if not golden_set_path.is_file():
        logger.error("Golden set not found: %s", golden_set_path)
        return 1

    threshold = args.threshold * 100.0 if 0.0 <= args.threshold <= 1.0 else args.threshold
    if threshold < 0.0 or threshold > 100.0:
        logger.error("Threshold must be in range [0, 100] or [0, 1].")
        return 1

    cmd = [
        sys.executable,
        "scripts/harness/eval_harness.py",
        "--input",
        str(golden_set_path),
        "--threshold",
        str(threshold),
        "--scan-security",
        "--strict",
    ]
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

    logger.info("Running AQA Regression Gate against %s...", golden_set_path.name)
    # nosec: B603 - Controlled list args invoking sys.executable with shell=False for regression gate (green-trunk-ci)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # nosec: B603

    # We only print stdout if it's there (eval_harness dumps JSON at the end)
    if result.stdout:
        print(result.stdout.strip())

    if result.returncode != 0:
        logger.error("AQA Gate FAILED (exit code %s)", result.returncode)
        if result.stderr:
            logger.error("Error details:\n%s", result.stderr.strip())
        return 1

    logger.info("AQA Gate PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
