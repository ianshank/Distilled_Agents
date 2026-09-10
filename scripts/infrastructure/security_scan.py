#!/usr/bin/env python3
"""Dependency and secret hygiene scan for CI (does not claim 'secure' by default)."""

from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone
from pathlib import Path

import click


def _run(command: list[str]) -> dict:
    try:
        completed = subprocess.run(  # nosec B603
            command, check=False, capture_output=True, text=True, timeout=120
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-2000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": 1, "error": str(exc)}


@click.command()
@click.option("--output", default="security_report.json")
def security_scan(output):
    checks = [
        _run(
            [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                "enhanced_system",
                "scripts",
                "-x",
                "tests,enhanced_system/tests,enhanced_system/examples",
                "-c",
                "pyproject.toml",
                "-q",
                "-f",
                "json",
            ]
        ),
        _run([sys.executable, "-m", "pip_audit", "-f", "json"]),
    ]
    failed = [item for item in checks if item.get("returncode", 1) not in (0, None)]
    report = {
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": "issues_found" if failed else "checks_passed",
        "checks": checks,
        "repo_root": str(Path.cwd()),
    }
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output} status={report['overall_status']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    security_scan()
