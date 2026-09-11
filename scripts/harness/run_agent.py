#!/usr/bin/env python3
"""Run one task through AgentRuntime (local; not SageMaker predict_fn)."""

from __future__ import annotations

import argparse
import json
import sys

from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.ops.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run a MangoMAS harness agent")
    parser.add_argument("--task", required=True)
    parser.add_argument("--harness-id", default=settings.harness_id or "base_react")
    parser.add_argument("--scripted", default="", help="JSON list of scripted backend outputs")
    args = parser.parse_args(argv)
    scripted = json.loads(args.scripted) if args.scripted else None
    runtime = HarnessFactory.create(
        {"harness_id": args.harness_id, "scripted": scripted, "teacher": False}
    )
    result = runtime.run(args.task, harness_id=args.harness_id)
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
