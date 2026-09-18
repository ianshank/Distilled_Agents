"""Security scanning for generated agent outputs."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Dict, List

logger = logging.getLogger(__name__)


class SecurityScanner:
    """Scans generated agent outputs for security vulnerabilities using Bandit.

    This is designed to catch unsafe code generation (e.g. hardcoded secrets,
    command injection risks) before they are fully evaluated or executed.
    """

    def __init__(self, fail_on_high: bool = True):
        self.fail_on_high = fail_on_high

    def scan_python_code(self, code: str) -> List[Dict[str, str]]:
        """Scans a python snippet using bandit.

        Returns a list of finding dictionaries.
        """
        if not code.strip():
            return []

        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", encoding="utf-8"
        ) as tf:
            tf.write(code)
            temp_path = tf.name

        try:
            # Run bandit with JSON output format
            cmd = ["bandit", "-f", "json", temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            # Bandit returns 0 if no issues, 1 if issues found
            try:
                import json

                report = json.loads(result.stdout)
                findings = report.get("results", [])

                if self.fail_on_high:
                    high_sev = [f for f in findings if f.get("issue_severity") == "HIGH"]
                    if high_sev:
                        logger.error(
                            "High severity security issues found in generated code: %s",
                            [f.get("issue_text") for f in high_sev],
                        )
                return findings
            except json.JSONDecodeError:
                logger.error("Failed to parse bandit output: %s", result.stdout)
                return []
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def scan_trajectory(self, trajectory: object) -> List[Dict[str, str]]:
        """Scan all tool action arguments in a trajectory that look like code."""
        findings = []
        if not hasattr(trajectory, "steps"):
            return findings

        for step in trajectory.steps:
            if step.action and "code" in step.action:
                code = step.action["code"]
                step_findings = self.scan_python_code(str(code))
                if step_findings:
                    findings.extend(step_findings)
        return findings
