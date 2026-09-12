#!/usr/bin/env python3
"""Dry-run rule-based harness tailor (APPLY_PATCHES defaults false)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from enhanced_system.harness.registry import load_spec, resolve_spec_path
from enhanced_system.harness.tailor import HarnessTailor
from enhanced_system.harness.types import Trajectory
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Propose harness YAML patches")
    parser.add_argument("--harness-id", required=True)
    parser.add_argument("--traces", required=True, help="JSONL of Trajectory objects")
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write live YAML (default: archive only)",
    )
    parser.add_argument(
        "--live-path",
        default="",
        help="Live YAML path used only when --apply is set",
    )
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.harness_id, settings)
        traces: list[Trajectory] = []
        with Path(args.traces).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    traces.append(Trajectory.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.error("invalid trace at line %s: %s", line_no, exc)
                    return 1
        proposed = HarnessTailor().propose(spec, traces)
        apply_patches = args.apply or settings.harness_apply_patches
        live_path = None
        if apply_patches:
            live_path = (
                Path(args.live_path)
                if args.live_path
                else resolve_spec_path(args.harness_id, settings)
            )
        archive = HarnessTailor().persist(
            proposed,
            archive_dir=Path(args.archive_dir),
            apply_patches=apply_patches,
            live_path=live_path,
            settings=settings,
        )
    except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("archived %s traces=%s apply=%s", archive, len(traces), apply_patches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
