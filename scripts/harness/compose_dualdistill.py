#!/usr/bin/env python3
"""Compose DualDistill JSONL from two same-task teacher files (needs expected)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from enhanced_system.harness.dualdistill import compose_pair, expected_text

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Compose dual-teacher trajectories")
    parser.add_argument("--first", required=True, help="Teacher A JSONL")
    parser.add_argument("--second", required=True, help="Teacher B JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        first_rows = _index_rows(Path(args.first))
        second_rows = _index_rows(Path(args.second))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    written = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for prompt, left in first_rows.items():
            right = second_rows.get(prompt)
            if right is None:
                continue
            expected = expected_text(left, expected_text(right))
            if not expected.strip():
                logger.warning("skipping unlabeled prompt")
                continue
            composed = compose_pair(left, right, expected=expected)
            if composed is None:
                continue
            handle.write(json.dumps(composed, ensure_ascii=True) + "\n")
            written += 1
    logger.info("wrote %s composed rows to %s", written, out_path)
    return 0


def _index_rows(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("row must be a JSON object")
            prompt = str(payload.get("prompt") or "")
            if not prompt.strip():
                continue
            indexed[prompt] = payload
    return indexed


if __name__ == "__main__":
    raise SystemExit(main())
