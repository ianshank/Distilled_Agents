"""Golden evaluation dataset schema and bucket-minima validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from enhanced_system.harness.types import (
    CANONICAL_REFUSAL_TOKENS,
    CANONICAL_REJECT_CODES,
    GoldenRow,
)

BUCKET_MINIMA: Mapping[str, int] = {
    "dag": 6,
    "condition_tree": 6,
    "mixed": 4,
    "cycle": 2,
    "unsat": 2,
    "schema_syntax": 2,
    "unknown_theory": 2,
}

TOTAL_MINIMUM: int = 24

_BUCKET_ALIASES: Mapping[str, str] = {
    "dag": "dag",
    "hard_dag": "dag",
    "hard-dag": "dag",
    "condition_tree": "condition_tree",
    "tree": "condition_tree",
    "condition-tree": "condition_tree",
    "hard_tree": "condition_tree",
    "mixed": "mixed",
    "hard_mixed": "mixed",
    "hard-mixed": "mixed",
    "cycle": "cycle",
    "ood_cycle": "cycle",
    "ood-cycle": "cycle",
    "unsat": "unsat",
    "ood_unsat": "unsat",
    "ood-unsat": "unsat",
    "schema": "schema_syntax",
    "syntax": "schema_syntax",
    "schema_syntax": "schema_syntax",
    "schema-syntax": "schema_syntax",
    "ood_schema": "schema_syntax",
    "ood_syntax": "schema_syntax",
    "theory": "unknown_theory",
    "unknown_theory": "unknown_theory",
    "unsupported_theory": "unknown_theory",
    "ood_theory": "unknown_theory",
}


def normalize_bucket_name(name: str) -> str:
    """Normalize bucket identifier using canonical names and common aliases."""
    normalized = name.strip().lower().replace(" ", "_")
    return _BUCKET_ALIASES.get(normalized, normalized)


def infer_bucket_from_row(row: GoldenRow | dict[str, Any]) -> str:
    """Infer bucket name from explicit bucket field, tags, or row ID."""
    bucket_raw = getattr(row, "bucket", None) if isinstance(row, GoldenRow) else row.get("bucket")
    if bucket_raw:
        return normalize_bucket_name(str(bucket_raw))

    tags = getattr(row, "tags", None) if isinstance(row, GoldenRow) else row.get("tags")
    if tags and isinstance(tags, (list, tuple)):
        for tag in tags:
            norm = normalize_bucket_name(str(tag))
            if norm in BUCKET_MINIMA:
                return norm

    row_id = getattr(row, "id", "") if isinstance(row, GoldenRow) else str(row.get("id", ""))
    row_id_lower = str(row_id).lower()
    for key in BUCKET_MINIMA:
        if key in row_id_lower:
            return key
    if "tree" in row_id_lower:
        return "condition_tree"
    if "schema" in row_id_lower or "syntax" in row_id_lower:
        return "schema_syntax"
    if "theory" in row_id_lower:
        return "unknown_theory"

    return "unknown"


def validate_golden_rows(
    rows: Sequence[GoldenRow | dict[str, Any]],
    *,
    min_total: int = TOTAL_MINIMUM,
    minima: Mapping[str, int] = BUCKET_MINIMA,
) -> dict[str, Any]:
    """Validate that golden rows meet schema invariants and per-bucket minima."""
    if len(rows) < min_total:
        raise ValueError(f"Golden set has {len(rows)} rows, minimum required is {min_total}")

    bucket_counts: dict[str, int] = {bucket: 0 for bucket in minima}
    seen_ids: set[str] = set()

    for idx, raw in enumerate(rows):
        row = raw if isinstance(raw, GoldenRow) else GoldenRow.model_validate(raw)
        row.validate_for_hard_or_ood()

        if row.id in seen_ids:
            raise ValueError(f"Duplicate row id detected at index {idx}: {row.id}")
        if row.id:
            seen_ids.add(row.id)

        bucket = infer_bucket_from_row(row)
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
        else:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    # Check per-bucket minima
    deficient: dict[str, str] = {}
    for bucket, min_count in minima.items():
        actual = bucket_counts.get(bucket, 0)
        if actual < min_count:
            deficient[bucket] = f"found {actual}, expected >= {min_count}"

    if deficient:
        details = ", ".join(f"{b} ({msg})" for b, msg in deficient.items())
        raise ValueError(f"Golden set failed bucket minima validation: {details}")

    return {
        "valid": True,
        "total": len(rows),
        "bucket_counts": bucket_counts,
    }


def validate_golden_file(
    path: str | Path,
    *,
    min_total: int = TOTAL_MINIMUM,
    minima: Mapping[str, int] = BUCKET_MINIMA,
) -> dict[str, Any]:
    """Load JSONL file and validate rows against golden schema and bucket minima."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Golden set file not found: {file_path}")

    rows: list[GoldenRow] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {file_path}: {exc}") from exc
            rows.append(GoldenRow.model_validate(data))

    return validate_golden_rows(rows, min_total=min_total, minima=minima)


__all__ = [
    "BUCKET_MINIMA",
    "CANONICAL_REFUSAL_TOKENS",
    "CANONICAL_REJECT_CODES",
    "TOTAL_MINIMUM",
    "infer_bucket_from_row",
    "normalize_bucket_name",
    "validate_golden_file",
    "validate_golden_rows",
]
