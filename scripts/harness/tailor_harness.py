#!/usr/bin/env python3
"""Dry-run rule-based harness tailor (APPLY_PATCHES defaults false)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enhanced_system.harness.registry import load_spec, resolve_spec_path
from enhanced_system.harness.tailor import HarnessTailor
from enhanced_system.harness.types import Trajectory
from enhanced_system.ops.settings import get_settings


def main(argv: list[str] | None = None) -> int:
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
    spec = load_spec(args.harness_id, settings)
    traces = []
    with Path(args.traces).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                traces.append(Trajectory.model_validate(json.loads(line)))
    proposed = HarnessTailor().propose(spec, traces)
    apply_patches = args.apply or settings.harness_apply_patches
    live_path = None
    if apply_patches:
        live_path = (
            Path(args.live_path) if args.live_path else resolve_spec_path(args.harness_id, settings)
        )
    HarnessTailor().persist(
        proposed,
        archive_dir=Path(args.archive_dir),
        apply_patches=apply_patches,
        live_path=live_path,
        settings=settings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
