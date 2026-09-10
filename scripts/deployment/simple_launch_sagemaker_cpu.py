#!/usr/bin/env python3
"""CPU SageMaker launcher (compatibility wrapper)."""

from __future__ import annotations

import sys


def launch_sagemaker_training_jobs(*_args, **_kwargs):
    """Always launch the CPU/quota-friendly job set (legacy zero-arg contract)."""
    try:
        from scripts.deployment.simple_launch_sagemaker import (
            launch_sagemaker_training_jobs as launch_jobs,
        )
    except ImportError:
        from simple_launch_sagemaker import launch_sagemaker_training_jobs as launch_jobs

    return launch_jobs(use_cpu=True)


if __name__ == "__main__":
    jobs = launch_sagemaker_training_jobs()
    print(f"Launched {len(jobs)} CPU jobs")
    sys.exit(0 if jobs else 1)
