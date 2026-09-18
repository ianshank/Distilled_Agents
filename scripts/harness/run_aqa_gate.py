#!/usr/bin/env python3
"""AQA Regression Gate: Runs Golden Sets and enforces Pass@K thresholds."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="AQA Regression Gate")
    parser.add_argument("--golden-set", required=True, help="Path to golden set JSONL")
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum pass rate")
    args = parser.parse_args()

    golden_set_path = Path(args.golden_set)
    if not golden_set_path.is_file():
        logger.error("Golden set not found: %s", golden_set_path)
        return 1

    cmd = [
        sys.executable,
        "scripts/harness/eval_harness.py",
        "--input", str(golden_set_path),
        "--threshold", str(args.threshold),
        "--scan-security"
    ]
    
    logger.info("Running AQA Regression Gate against %s...", golden_set_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
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
