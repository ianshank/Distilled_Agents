#!/usr/bin/env python3
"""Check that configs/rule_traceability/matrix.yaml covers all hard golden IDs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REQUIRED_MATRIX_FIELDS = ("rule_id", "source_trace", "harness_id", "golden_id", "solver_status")
VALID_SOLVER_STATUSES = frozenset({"fixture", "active", "pending"})


def verify_rule_matrix(
    matrix_path: Path | str,
    golden_path: Path | str,
) -> dict[str, Any]:
    """Verify that every slice=='hard' golden ID is mapped in matrix.yaml."""
    matrix_file = Path(matrix_path)
    golden_file = Path(golden_path)

    if not matrix_file.is_file():
        raise FileNotFoundError(f"Matrix file not found: {matrix_file}")
    if not golden_file.is_file():
        raise FileNotFoundError(f"Golden set not found: {golden_file}")

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
        status = str(rule.get("solver_status", "")).strip().lower()
        if status not in VALID_SOLVER_STATUSES:
            raise ValueError(
                f"Rule item {idx} has invalid solver_status '{status}'; "
                f"must be one of {sorted(VALID_SOLVER_STATUSES)} and cannot be 'none'"
            )
        matrix_golden_ids.add(str(rule["golden_id"]).strip())

    hard_golden_ids: set[str] = set()
    with golden_file.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            slice_val = str(item.get("slice", "")).lower()
            if slice_val == "hard":
                row_id = item.get("id")
                if row_id:
                    hard_golden_ids.add(row_id)

    missing = hard_golden_ids - matrix_golden_ids
    if missing:
        raise ValueError(
            f"Rule matrix {matrix_file} missing coverage for hard golden ids: {sorted(missing)}"
        )

    return {
        "covered_hard_ids": len(hard_golden_ids),
        "total_rules": len(rules),
        "matrix_golden_ids": sorted(matrix_golden_ids),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Check rule traceability matrix coverage")
    parser.add_argument(
        "--matrix",
        default="configs/rule_traceability/matrix.yaml",
        help="Path to rule traceability matrix YAML",
    )
    parser.add_argument(
        "--golden-set",
        "--golden",
        dest="golden_set",
        default="configs/golden_sets/sqe_hard_ood.jsonl",
        help="Path to hard/OOD golden set JSONL",
    )
    args = parser.parse_args(argv)

    try:
        result = verify_rule_matrix(Path(args.matrix), Path(args.golden_set))
        logger.info(
            "Rule matrix verification passed (%d hard IDs covered across %d rules).",
            result["covered_hard_ids"],
            result["total_rules"],
        )
        return 0
    except Exception as exc:
        logger.error("Rule matrix check failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
