#!/usr/bin/env python3
"""Launch SageMaker training jobs for all MangoMAS agents.

Uses :class:`~enhanced_system.ops.MangoMASSageMakerLauncher` from the ops
package (not the thin CLI wrapper in ``launch_all_agents_sagemaker``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from enhanced_system.ops import MangoMASSageMakerLauncher
from enhanced_system.ops.settings import get_settings

logger = logging.getLogger(__name__)


async def _run(args: argparse.Namespace) -> int:
    launcher = MangoMASSageMakerLauncher(region=args.region)

    if args.dry_run:
        logger.info("Dry run: validating training data and printing job specs")
        if not launcher.validate_training_data():
            logger.error("Training data validation failed")
            return 1
        specs = [launcher.create_job_spec(config) for config in launcher.agent_configs]
        print(json.dumps(specs, indent=2))
        return 0

    results = await launcher.launch_all_jobs(max_concurrent=args.max_concurrent)
    payload: dict = {
        "started_jobs": results,
        "failed_jobs": launcher.failed_jobs,
        "summary": launcher.generate_summary_report(),
    }
    if args.wait and results:
        names = [item["job_name"] for item in results if item.get("job_name")]
        payload["statuses"] = launcher.monitor_training_jobs(names)
    print(json.dumps(payload, indent=2, default=str))
    if not results or launcher.failed_jobs:
        logger.error(
            "SageMaker launch incomplete: started=%s failed=%s",
            len(results),
            len(launcher.failed_jobs),
        )
        return 1
    return 0


def main() -> int:
    """Launch SageMaker jobs for MangoMAS agents."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Launch SageMaker training jobs for all MangoMAS agents"
    )
    parser.add_argument(
        "--region",
        default=settings.aws_region,
        help="AWS region (default: MANGOMAS_AWS_REGION / AWS_DEFAULT_REGION)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print job specs without launching",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent jobs (default: MANGOMAS_MAX_CONCURRENT_JOBS)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="After launch, poll SageMaker for job statuses",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
