#!/usr/bin/env python3
"""SCoRe-SFT collect: student rollout, teacher patches earliest error, resume prefix."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from enhanced_system.harness.convert import trajectory_to_legacy
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.score import (
    earliest_error_index,
    preference_pair,
    prefix_steps,
    review_prompt,
)
from enhanced_system.harness.traces import JsonlTraceStore, raw_store_path
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Collect SCoRe-SFT corrected traces")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--student-scripted", default="")
    parser.add_argument("--teacher-scripted", default="")
    parser.add_argument("--prefs", default="", help="Optional preference-pair JSONL")
    args = parser.parse_args(argv)
    try:
        student_scripted = json.loads(args.student_scripted) if args.student_scripted else None
        teacher_scripted = json.loads(args.teacher_scripted) if args.teacher_scripted else None
    except json.JSONDecodeError as exc:
        logger.error("invalid scripted JSON: %s", exc)
        return 1
    out_path = Path(args.output)
    try:
        store = JsonlTraceStore(raw_store_path(out_path))
        student = HarnessFactory.create(
            {
                "harness_id": args.harness_id,
                "scripted": student_scripted,
                "teacher": False,
                "store": store,
                "strict_injection": False,
            }
        )
        teacher = HarnessFactory.create(
            {
                "harness_id": args.harness_id,
                "scripted": teacher_scripted,
                "teacher": True,
                "strict_injection": False,
            }
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    rows: list[dict[str, object]] = []
    prefs: list[dict[str, object]] = []
    try:
        handle = Path(args.input).open(encoding="utf-8")
    except OSError as exc:
        logger.error("%s", exc)
        return 1
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                prompt = str(payload.get("prompt") or "")
                if not prompt.strip():
                    continue
                expected = payload.get("expected")
                expected_text = str(expected) if expected is not None else None
                student_result = student.run(prompt, harness_id=args.harness_id)
                index = earliest_error_index(student_result.trajectory, expected_text)
                if index is None:
                    rows.append(trajectory_to_legacy(student_result.trajectory))
                    continue
                prefix = prefix_steps(student_result.trajectory, index)
                messages = review_prompt(student_result.trajectory, expected_text)
                generated = teacher.backend.generate(messages, prefix=None, n=1)
                if not generated:
                    logger.warning("teacher produced no correction at line %s", line_no)
                    continue
                patched = student.run(
                    prompt,
                    harness_id=args.harness_id,
                    resume_steps=prefix,
                    inject_action=generated[0],
                )
                rows.append(trajectory_to_legacy(patched.trajectory))
                pair = preference_pair(student_result.trajectory, patched.trajectory, index)
                if pair is not None:
                    prefs.append(pair)
            except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
                logger.warning("skipping line %s: %s", line_no, exc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    if args.prefs and prefs:
        prefs_path = Path(args.prefs)
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
        with prefs_path.open("w", encoding="utf-8") as handle:
            for row in prefs:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    logger.info("wrote %s rows to %s prefs=%s", len(rows), out_path, len(prefs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
