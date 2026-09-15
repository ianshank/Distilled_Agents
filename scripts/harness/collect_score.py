#!/usr/bin/env python3
"""SCoRe-SFT collect: student rollout, teacher patches earliest error, resume prefix."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from enhanced_system.harness.convert import trajectory_to_legacy
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts
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
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="fail on malformed JSONL rows instead of skipping",
    )
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
    try:
        collected = _collect_score_rows(
            student,
            teacher,
            Path(args.input),
            args.harness_id,
            strict=args.strict,
        )
    except JsonlRowError as exc:
        logger.error("%s", exc)
        return 1
    except OSError as exc:
        logger.error("%s", exc)
        return 1
    if collected is None:
        return 1
    rows, prefs = collected
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    if args.prefs:
        prefs_path = Path(args.prefs)
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
        with prefs_path.open("w", encoding="utf-8") as handle:
            for row in prefs:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    logger.info("wrote %s rows to %s prefs=%s", len(rows), out_path, len(prefs))
    return 0


def _collect_score_rows(
    student: object,
    teacher: object,
    path: Path,
    harness_id: str,
    *,
    strict: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    rows: list[dict[str, object]] = []
    prefs: list[dict[str, object]] = []
    for line_no, payload in iter_jsonl_dicts(path, require_prompt=True, strict=strict):
        try:
            _score_one_row(
                student,
                teacher,
                payload,
                harness_id,
                line_no,
                rows,
                prefs,
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            logger.warning("skipping line %s: %s", line_no, exc)
            if strict:
                return None
    return rows, prefs


def _score_one_row(
    student: object,
    teacher: object,
    payload: dict[str, Any],
    harness_id: str,
    line_no: int,
    rows: list[dict[str, object]],
    prefs: list[dict[str, object]],
) -> None:
    prompt = str(payload["prompt"])
    expected = payload.get("expected")
    expected_text = str(expected).strip() if expected is not None else ""
    expected_text = expected_text or None
    student_result = student.run(prompt, harness_id=harness_id)  # type: ignore[attr-defined]
    index = earliest_error_index(student_result.trajectory, expected_text)
    if index is None:
        rows.append(trajectory_to_legacy(student_result.trajectory, expected=expected))
        return
    prefix = prefix_steps(student_result.trajectory, index)
    messages = review_prompt(student_result.trajectory, expected_text)
    generated = teacher.backend.generate(messages, prefix=None, n=1)  # type: ignore[attr-defined]
    if not generated or not str(generated[0]).strip():
        logger.warning("teacher produced no correction at line %s", line_no)
        return
    patched = student.run(  # type: ignore[attr-defined]
        prompt,
        harness_id=harness_id,
        resume_steps=prefix,
        inject_action=generated[0],
    )
    if not _injected_action_ok(patched.trajectory, index):
        logger.warning("teacher correction failed at line %s", line_no)
        return
    rows.append(trajectory_to_legacy(patched.trajectory, expected=expected))
    pair = preference_pair(student_result.trajectory, patched.trajectory, index)
    if pair is not None:
        prefs.append(pair)


def _injected_action_ok(trajectory: object, index: int) -> bool:
    steps = getattr(trajectory, "steps", ())
    if index < 0 or index >= len(steps):
        return False
    step = steps[index]
    return bool(getattr(step, "action", "")) and not getattr(step, "fault", "")


if __name__ == "__main__":
    raise SystemExit(main())
