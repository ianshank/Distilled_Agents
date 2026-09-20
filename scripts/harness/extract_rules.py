#!/usr/bin/env python3
"""Rule extraction stub: extracts candidate rules from teacher traces into matrix candidates.

Candidates are written to a candidate file or stdout for operator review;
they are strictly NOT auto-promoted into configs/rule_traceability/matrix.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from enhanced_system.harness.jsonl import iter_jsonl_dicts

logger = logging.getLogger(__name__)


def extract_candidate_rules(
    input_path: Path | str,
    *,
    default_harness_id: str = "qc_constraints",
    default_status: str = "pending",
) -> list[dict[str, str]]:
    """Inspect input trace/prompt JSONL and construct candidate matrix rule entries."""
    source_file = Path(input_path)
    if not source_file.is_file():
        raise FileNotFoundError(f"Input file not found: {source_file}")

    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for _line_no, payload in iter_jsonl_dicts(source_file, require_prompt=False, strict=False):
        prompt = str(payload.get("prompt") or payload.get("task") or "").strip()
        golden_id = str(payload.get("id") or payload.get("golden_id") or "").strip()
        harness_id = str(payload.get("harness_id") or default_harness_id).strip()

        if not prompt and not golden_id:
            continue

        if not golden_id:
            # Deterministic hash ID based on prompt
            digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
            golden_id = f"cand-{digest}"

        if golden_id in seen_ids:
            continue
        seen_ids.add(golden_id)

        digest = hashlib.sha256(f"{golden_id}:{prompt}".encode()).hexdigest()[:6].upper()
        rule_id = f"CAND-RULE-{digest}"

        source_trace = str(payload.get("source_trace") or source_file.name)

        candidates.append(
            {
                "rule_id": rule_id,
                "source_trace": source_trace,
                "harness_id": harness_id,
                "golden_id": golden_id,
                "solver_status": default_status,
            }
        )

    logger.info("Extracted %d rule candidates from %s", len(candidates), source_file.name)
    return candidates


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    parser = argparse.ArgumentParser(
        description="Extract rule candidates from teacher traces for manual matrix curation (no auto-promote)"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input trace/golden JSONL file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output candidate YAML file (prints to stdout if omitted)",
    )
    parser.add_argument(
        "--harness-id",
        default="qc_constraints",
        help="Default harness ID to tag candidate rules with",
    )
    parser.add_argument(
        "--status",
        default="pending",
        choices=["pending", "fixture"],
        help="Candidate rule status (strictly not active; default: pending)",
    )
    args = parser.parse_args(argv)

    try:
        candidates = extract_candidate_rules(
            args.input,
            default_harness_id=args.harness_id,
            default_status=args.status,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to extract candidate rules: %s", exc)
        return 1

    payload: dict[str, Any] = {
        "schema_version": "1",
        "candidates": candidates,
    }
    rendered = yaml.safe_dump(payload, sort_keys=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        logger.info("Wrote %d candidates to %s", len(candidates), out_path)
    else:
        print(rendered.strip())

    return 0


if __name__ == "__main__":
    sys.exit(main())
