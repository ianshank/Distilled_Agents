#!/usr/bin/env python3
"""Verify that every hard golden row ID is mapped in configs/rule_traceability/matrix.yaml."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from enhanced_system.harness.jsonl import iter_jsonl_dicts

logger = logging.getLogger(__name__)

REQUIRED_MATRIX_FIELDS = ("rule_id", "source_trace", "harness_id", "golden_id", "solver_status")


def verify_rule_matrix(
    matrix_path: Path | str,
    golden_path: Path | str,
) -> list[str]:
    """Check that every hard golden row ID appears in the rule matrix.

    Returns a list of missing hard golden IDs (empty list on complete coverage).
    Raises ValueError on schema or file loading errors.
    """
    matrix_file = Path(matrix_path)
    if not matrix_file.is_file():
        raise ValueError(f"Matrix file not found: {matrix_file}")

    try:
        matrix_data = yaml.safe_load(matrix_file.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        raise ValueError(f"Failed to read matrix YAML {matrix_file}: {exc}") from exc

    if not isinstance(matrix_data, dict) or "rules" not in matrix_data:
        raise ValueError(f"Matrix YAML must be an object with a 'rules' list: {matrix_file}")

    rules = matrix_data["rules"]
    if not isinstance(rules, list):
        raise ValueError(f"Matrix YAML 'rules' key must be a list: {matrix_file}")

    matrix_golden_ids: set[str] = set()
    for idx, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Rule item {idx} is not a dictionary")
        missing_fields = [f for f in REQUIRED_MATRIX_FIELDS if not rule.get(f)]
        if missing_fields:
            raise ValueError(f"Rule item {idx} missing required fields: {missing_fields}")
        matrix_golden_ids.add(str(rule["golden_id"]).strip())

    golden_file = Path(golden_path)
    if not golden_file.is_file():
        raise ValueError(f"Golden file not found: {golden_file}")

    hard_golden_ids: set[str] = set()
    for line_no, payload in iter_jsonl_dicts(golden_file, require_prompt=True, strict=True):
        row_slice = str(payload.get("slice", "core")).strip().lower()
        if row_slice == "hard":
            row_id = str(payload.get("id") or "").strip()
            if not row_id:
                raise ValueError(f"Hard slice row at line {line_no} is missing a stable 'id'")
            hard_golden_ids.add(row_id)

    missing = sorted(hard_golden_ids - matrix_golden_ids)
    return missing


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    parser = argparse.ArgumentParser(
        description="Verify rule traceability matrix coverage for hard golden sets"
    )
    parser.add_argument(
        "--matrix",
        default="configs/rule_traceability/matrix.yaml",
        help="Path to rule traceability matrix YAML",
    )
    parser.add_argument(
        "--golden",
        default="configs/golden_sets/hard_sdlc.jsonl",
        help="Path to hard golden set JSONL",
    )
    args = parser.parse_args(argv)

    try:
        missing = verify_rule_matrix(args.matrix, args.golden)
    except ValueError as exc:
        logger.error("Rule matrix verification failed: %s", exc)
        return 1

    if missing:
        logger.error(
            "Hard golden IDs missing from rule traceability matrix %s: %s",
            args.matrix,
            missing,
        )
        return 1

    logger.info("Rule traceability matrix check PASSED (%s covered)", args.golden)
    return 0


if __name__ == "__main__":
    sys.exit(main())
