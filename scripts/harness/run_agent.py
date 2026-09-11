#!/usr/bin/env python3
"""Run one task through AgentRuntime (local; not SageMaker predict_fn)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run a MangoMAS harness agent")
    parser.add_argument("--task", required=True)
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--scripted", default="", help="JSON list of scripted backend outputs")
    args = parser.parse_args(argv)
    try:
        scripted = json.loads(args.scripted) if args.scripted else None
        runtime = HarnessFactory.create(
            {
                "harness_id": args.harness_id,
                "scripted": scripted,
                "teacher": False,
                "strict_injection": True,
            }
        )
        result = runtime.run(args.task, harness_id=args.harness_id)
    except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    json.dump(
        {
            "final_answer": result.final_answer,
            "harness_id": result.harness_id,
            "truncated": result.truncated,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
