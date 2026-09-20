#!/usr/bin/env python3
"""Collect teacher/student traces from prompt/completion JSONL (local)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from enhanced_system.harness.convert import trajectory_to_legacy
from enhanced_system.harness.critic import (
    CriticRejectCode,
    CriticTelemetry,
    check_outcome,
    check_tool_allowlist,
    is_recovery_trace,
)
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts
from enhanced_system.harness.registry import load_spec
from enhanced_system.harness.traces import JsonlTraceStore, raw_store_path
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Collect harness trajectories")
    parser.add_argument("--input", required=True, help="Source JSONL with prompt field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--scripted", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--teacher",
        dest="student",
        action="store_false",
        help="Collect with the teacher model (default)",
    )
    mode.add_argument(
        "--student",
        dest="student",
        action="store_true",
        help="Collect with the student model",
    )
    parser.set_defaults(student=False)
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on the first row error (default: skip and continue)",
    )
    parser.add_argument(
        "--redact-pii",
        action="store_true",
        default=False,
        help="Use presidio to redact PII from the collected trajectories",
    )
    parser.add_argument(
        "--reject-log",
        default=settings.critic_reject_log,
        help="JSONL sink path for critic reject telemetry (default: artifacts/critic_rejects.jsonl)",
    )
    critic_group = parser.add_mutually_exclusive_group()
    critic_group.add_argument(
        "--critic",
        dest="critic_enabled",
        action="store_true",
        default=None,
        help="Enable critic filtering and rejection telemetry",
    )
    critic_group.add_argument(
        "--no-critic",
        dest="critic_enabled",
        action="store_false",
        default=None,
        help="Disable critic filtering and rejection telemetry",
    )
    args = parser.parse_args(argv)
    critic_enabled = (
        args.critic_enabled if args.critic_enabled is not None else settings.critic_enabled
    )
    telemetry = CriticTelemetry(sink_path=args.reject_log, enabled=critic_enabled)
    try:
        scripted = json.loads(args.scripted) if args.scripted else None
    except json.JSONDecodeError as exc:
        logger.error("invalid --scripted JSON: %s", exc)
        return 1
    out_path = Path(args.output)

    scrubber = None
    if args.redact_pii:
        try:
            from enhanced_system.harness.data_governance import PIIScrubber

            scrubber = PIIScrubber()
            logger.info("PII redaction enabled via Presidio")
        except (ImportError, RuntimeError) as exc:
            logger.error("Cannot enable PII redaction: %s", exc)
            return 1

    try:
        store = JsonlTraceStore(raw_store_path(out_path))
        runtime = HarnessFactory.create(
            {
                "harness_id": args.harness_id,
                "scripted": scripted,
                "teacher": not args.student,
                "store": store,
                "strict_injection": False,
            }
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    try:
        rows = _collect_rows(
            runtime,
            Path(args.input),
            args.harness_id,
            args.strict,
            scrubber=scrubber,
            telemetry=telemetry,
        )
    except JsonlRowError as exc:
        logger.error("%s", exc)
        telemetry.emit_summary()
        return 1
    except OSError as exc:
        logger.error("%s", exc)
        telemetry.emit_summary()
        return 1
    if rows is None:
        telemetry.emit_summary()
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    logger.info("wrote %s rows to %s", len(rows), out_path)
    telemetry.emit_summary()
    return 0


def _collect_rows(
    runtime: object,
    path: Path,
    harness_id: str,
    strict: bool,
    scrubber: object | None = None,
    telemetry: CriticTelemetry | None = None,
) -> list[dict[str, object]] | None:
    rows: list[dict[str, object]] = []
    allowed_tools: set[str] | None = None
    try:
        allowed_tools = set(load_spec(harness_id).action.tool_ids)
    except Exception:
        allowed_tools = None

    for line_no, payload in iter_jsonl_dicts(path, require_prompt=True, strict=strict):
        try:
            prompt = str(payload["prompt"])
            result = runtime.run(prompt, harness_id=harness_id)  # type: ignore[attr-defined]
            expected = payload.get("expected")

            # Critic check: Tool allowlist
            if telemetry is not None and telemetry.enabled and allowed_tools is not None:
                valid_tools, code, reason = check_tool_allowlist(result.trajectory, allowed_tools)
                if not valid_tools:
                    logger.warning(
                        "skipping tool allowlist violation at line %s (critic_reject_code: %s): %s",
                        line_no,
                        code.value if code else CriticRejectCode.SCHEMA_VIOLATION.value,
                        reason,
                    )
                    telemetry.record_reject(
                        code=code or CriticRejectCode.SCHEMA_VIOLATION,
                        prompt=prompt,
                        expected=str(expected) if expected is not None else None,
                        final_answer=result.final_answer,
                        metadata={
                            "line_no": line_no,
                            "harness_id": harness_id,
                            "reason": reason,
                        },
                    )
                    if strict:
                        return None
                    continue

            # Outcome check
            if expected is not None and str(expected).strip():
                valid_outcome, code, reason = check_outcome(result.final_answer, str(expected))
                if not valid_outcome:
                    logger.warning(
                        "skipping outcome mismatch at line %s (critic_reject_code: %s)",
                        line_no,
                        CriticRejectCode.OUTCOME_MISMATCH.value,
                    )
                    if telemetry is not None:
                        telemetry.record_reject(
                            code=CriticRejectCode.OUTCOME_MISMATCH,
                            prompt=prompt,
                            expected=str(expected),
                            final_answer=result.final_answer,
                            metadata={
                                "line_no": line_no,
                                "harness_id": harness_id,
                                "reason": reason,
                            },
                        )
                    if strict:
                        return None
                    continue

            is_rec = is_recovery_trace(
                result.trajectory, expected=str(expected) if expected is not None else None
            )
            if telemetry is not None:
                telemetry.record_kept(
                    prompt=prompt,
                    is_recovery=is_rec,
                    metadata={"line_no": line_no, "faults": result.trajectory.faults},
                )

            row_dict = trajectory_to_legacy(result.trajectory, expected=expected)
            if scrubber is not None:
                row_dict = scrubber.redact_object(row_dict)  # type: ignore[attr-defined]
            rows.append(row_dict)
        except (ValueError, FileNotFoundError, OSError) as exc:
            logger.warning("skipping line %s: %s", line_no, exc)
            if strict:
                return None
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
