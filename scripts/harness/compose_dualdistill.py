#!/usr/bin/env python3
"""Compose DualDistill JSONL from two same-task teacher files (needs expected)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from enhanced_system.harness.critic import CriticMetrics, RejectSink
from enhanced_system.harness.dualdistill import compose_pair, expected_text
from enhanced_system.harness.jsonl import iter_jsonl_dicts
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Compose dual-teacher trajectories")
    parser.add_argument("--first", required=True, help="Teacher A JSONL")
    parser.add_argument("--second", required=True, help="Teacher B JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reject-log",
        default=None,
        help="Path to JSONL reject log (default: MANGOMAS_CRITIC_REJECT_LOG or artifacts/critic_rejects.jsonl)",
    )
    args = parser.parse_args(argv)
    try:
        first_rows = _index_rows(Path(args.first))
        second_rows = _index_rows(Path(args.second))
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    reject_path = args.reject_log or settings.critic_reject_log or "artifacts/critic_rejects.jsonl"
    reject_sink = RejectSink(reject_path)
    metrics = CriticMetrics()

    written = 0
    unmatched = len(set(second_rows) - set(first_rows))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for prompt, left in first_rows.items():
            right = second_rows.get(prompt)
            if right is None:
                unmatched += 1
                continue
            expected = expected_text(left, expected_text(right))
            if not expected.strip():
                logger.warning("skipping unlabeled prompt")
                continue
            composed = compose_pair(left, right, expected=expected, reject_sink=reject_sink)
            if composed is None:
                metrics.record_reject("DUALDISTILL_DROP_0_0")
                continue
            handle.write(json.dumps(composed, ensure_ascii=True) + "\n")
            written += 1
    if unmatched:
        logger.warning("%s unmatched prompts across teacher files", unmatched)
    logger.info(
        "wrote %s composed rows to %s (critic_rejected_dualdistill_0_0=%s)",
        written,
        out_path,
        metrics.critic_rejected_dualdistill_0_0,
    )
    return 0


def _index_rows(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for _line_no, payload in iter_jsonl_dicts(path, require_prompt=True, strict=False):
        prompt = str(payload.get("prompt") or "")
        if prompt in indexed:
            logger.warning("duplicate prompt in %s; last row wins", path)
        indexed[prompt] = payload
    return indexed


if __name__ == "__main__":
    raise SystemExit(main())
