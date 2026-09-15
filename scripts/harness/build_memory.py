#!/usr/bin/env python3
"""Build AMD-lite workflow/function memory from successful teacher traces."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts
from enhanced_system.harness.memory_bank import build_bank
from enhanced_system.harness.types import Trajectory

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build hierarchical teacher memory")
    parser.add_argument("--traces", required=True, help="JSONL of Trajectory objects")
    parser.add_argument("--output", required=True, help="Memory bank JSON path")
    args = parser.parse_args(argv)
    try:
        trajectories = _load_traces(Path(args.traces))
    except (OSError, JsonlRowError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    bank = build_bank(trajectories)
    try:
        bank.save(args.output)
    except OSError as exc:
        logger.error("%s", exc)
        return 1
    logger.info(
        "wrote %s workflows=%s tools=%s",
        args.output,
        len(bank.workflows),
        len(bank.functions),
    )
    return 0


def _load_traces(path: Path) -> list[Trajectory]:
    traces: list[Trajectory] = []
    for _line_no, payload in iter_jsonl_dicts(path, require_prompt=False, strict=True):
        if "trajectory" in payload and isinstance(payload["trajectory"], dict):
            payload = payload["trajectory"]
        traces.append(Trajectory.model_validate(payload))
    return traces


if __name__ == "__main__":
    raise SystemExit(main())
