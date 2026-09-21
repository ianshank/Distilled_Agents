#!/usr/bin/env python3
"""Collect teacher/student traces from prompt/completion JSONL (local)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from enhanced_system.harness.convert import trajectory_to_legacy
from enhanced_system.harness.critic import (
    CriticMetrics,
    CriticRejectCode,
    RejectSink,
    check_expected_tools,
    check_outcome,
    check_tool_allowlist,
    is_recovery_trace,
)
from enhanced_system.harness.dispatch import allowed_tools
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts
from enhanced_system.harness.registry import load_spec
from enhanced_system.harness.score import answers_match
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
        "--critic",
        dest="critic_enabled",
        action="store_true",
        default=None,
        help="Enable trace critic cascade (default: from MANGOMAS_CRITIC_ENABLED)",
    )
    parser.add_argument(
        "--no-critic",
        dest="critic_enabled",
        action="store_false",
        help="Disable trace critic cascade",
    )
    parser.add_argument(
        "--reject-log",
        default=None,
        help="Path to JSONL reject log (default: MANGOMAS_CRITIC_REJECT_LOG or artifacts/critic_rejects.jsonl)",
    )
    args = parser.parse_args(argv)
    scripted = None
    if args.scripted:
        scripted_path = Path(args.scripted)
        if scripted_path.is_file():
            try:
                scripted = json.loads(scripted_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("invalid --scripted file %s JSON: %s", scripted_path, exc)
                return 1
        else:
            try:
                scripted = json.loads(args.scripted)
            except json.JSONDecodeError as exc:
                logger.error("invalid --scripted JSON: %s", exc)
                return 1
    out_path = Path(args.output)

    scrubber = None
    if args.redact_pii:
        try:
            from enhanced_system.harness.data_governance import PIIScrubber

            scrubber = PIIScrubber()
            if not getattr(scrubber, "_available", False):
                logger.error(
                    "Cannot enable PII redaction: Presidio libraries not found. "
                    "Install with: pip install 'mangomas[security]'"
                )
                return 1
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
        critic_enabled = (
            settings.critic_enabled if args.critic_enabled is None else args.critic_enabled
        )
        reject_path = (
            args.reject_log or settings.critic_reject_log or "artifacts/critic_rejects.jsonl"
        )
        reject_sink = RejectSink(reject_path) if critic_enabled or args.reject_log else None
        metrics = CriticMetrics()

        spec = getattr(runtime, "spec", None)
        if spec is None and args.harness_id:
            try:
                spec = load_spec(args.harness_id, settings)
            except (FileNotFoundError, OSError, ValueError):
                spec = None
        allowed_tool_ids = allowed_tools(spec.action.tool_ids) if spec is not None else None

        rows = _collect_rows(
            runtime,
            Path(args.input),
            args.harness_id,
            args.strict,
            scrubber=scrubber,
            critic_enabled=critic_enabled,
            reject_sink=reject_sink,
            metrics=metrics,
            allowed_tool_ids=allowed_tool_ids,
        )
    except JsonlRowError as exc:
        logger.error("%s", exc)
        return 1
    except OSError as exc:
        logger.error("%s", exc)
        return 1
    if rows is None:
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    logger.info("wrote %s rows to %s", len(rows), out_path)
    if critic_enabled:
        logger.info(
            "critic summary: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(metrics.as_dict().items())),
            extra={"critic_metrics": metrics.as_dict()},
        )
    return 0


def _collect_rows(
    runtime: object,
    path: Path,
    harness_id: str,
    strict: bool,
    scrubber: object | None = None,
    *,
    critic_enabled: bool = False,
    reject_sink: RejectSink | None = None,
    metrics: CriticMetrics | None = None,
    allowed_tool_ids: set[str] | None = None,
) -> list[dict[str, object]] | None:
    rows: list[dict[str, object]] = []
    for line_no, payload in iter_jsonl_dicts(path, require_prompt=True, strict=strict):
        try:
            prompt = str(payload["prompt"])
            result = runtime.run(prompt, harness_id=harness_id)  # type: ignore[attr-defined]
            expected = payload.get("expected")
            expected_str = (
                str(expected).strip() if expected is not None and str(expected).strip() else None
            )
            expected_tools = payload.get("expected_tools")

            if critic_enabled:
                # 1. Allowlist check
                if allowed_tool_ids is not None:
                    allow_dec = check_tool_allowlist(result.trajectory, allowed_tool_ids)
                    if not allow_dec:
                        code = allow_dec.reject_code or CriticRejectCode.SCHEMA_VIOLATION.value
                        logger.warning(
                            "critic rejected allowlist at line %s: critic_reject_code: %s (%s)",
                            line_no,
                            code,
                            allow_dec.reason,
                            extra={"critic_reject_code": code},
                        )
                        if reject_sink is not None:
                            reject_sink.record(
                                code,
                                prompt,
                                metadata={
                                    "line_no": line_no,
                                    "reason": allow_dec.reason,
                                    "harness_id": harness_id,
                                    "expected": expected,
                                },
                            )
                        if metrics is not None:
                            metrics.record_reject(code)
                            metrics.record_reject("allowlist")
                        if strict:
                            return None
                        continue

                # 2. Expected tools check (if row specifies expected_tools)
                if expected_tools is not None and isinstance(expected_tools, list):
                    tools_dec = check_expected_tools(result.trajectory, expected_tools)
                    if not tools_dec:
                        code = tools_dec.reject_code or CriticRejectCode.SCHEMA_VIOLATION.value
                        logger.warning(
                            "critic rejected expected_tools at line %s: critic_reject_code: %s (%s)",
                            line_no,
                            code,
                            tools_dec.reason,
                            extra={"critic_reject_code": code},
                        )
                        if reject_sink is not None:
                            reject_sink.record(
                                code,
                                prompt,
                                metadata={
                                    "line_no": line_no,
                                    "reason": tools_dec.reason,
                                    "expected_tools": expected_tools,
                                },
                            )
                        if metrics is not None:
                            metrics.record_reject(code)
                            metrics.record_reject("expected_tools")
                        if strict:
                            return None
                        continue

                # 3. Outcome check when expected is present
                if expected_str is not None:
                    outcome_dec = check_outcome(result.final_answer, expected_str)
                    if not outcome_dec:
                        code = outcome_dec.reject_code or CriticRejectCode.OUTCOME_MISMATCH.value
                        logger.warning(
                            "skipping outcome mismatch at line %s: critic_reject_code: %s",
                            line_no,
                            code,
                            extra={"critic_reject_code": code},
                        )
                        if reject_sink is not None:
                            reject_sink.record(
                                code,
                                prompt,
                                metadata={
                                    "line_no": line_no,
                                    "final_answer": result.final_answer,
                                    "expected": expected_str,
                                },
                            )
                        if metrics is not None:
                            metrics.record_reject(code)
                        if strict:
                            return None
                        continue

                # 4. Check for recovery trace when outcome matches
                if expected_str is not None and is_recovery_trace(result.trajectory, expected_str):
                    if metrics is not None:
                        metrics.record_recovery()
                    logger.info(
                        "critic_kept_recovery at line %s for prompt: %s",
                        line_no,
                        prompt,
                        extra={"critic_kept_recovery": True, "line_no": line_no},
                    )

            else:
                # Backwards compatible path: simple outcome match check
                if expected_str is not None:
                    if not answers_match(result.final_answer, expected_str):
                        logger.warning("skipping outcome mismatch at line %s", line_no)
                        if strict:
                            return None
                        continue

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
