"""Security scanning for generated agent outputs."""

from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec: B404
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
            import sys

            cmd = [sys.executable, "-m", "bandit", "-f", "json", temp_path]
            # nosec: B603 - Controlled list args invoking sys.executable with shell=False on internal temp path (green-trunk-ci)
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # nosec: B603
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"Bandit failed with exit code {result.returncode}: {result.stderr}"
                )

            # Bandit returns 0 if no issues, 1 if issues found
            try:
                report = json.loads(result.stdout)
            except json.JSONDecodeError:
                if self.fail_on_high:
                    raise RuntimeError(
                        "Failed to parse bandit JSON output while fail_on_high=True"
                    ) from None
                logger.warning(
                    "Failed to parse bandit JSON output (possible stderr contamination), "
                    "returning empty findings. Raw stdout: %s",
                    result.stdout[:500],
                )
                return []
            findings: List[Dict[str, str]] = report.get("results", [])

            if self.fail_on_high:
                high_sev = [f for f in findings if f.get("issue_severity") == "HIGH"]
                if high_sev:
                    logger.error(
                        "High severity security issues found in generated code: %s",
                        [f.get("issue_text") for f in high_sev],
                    )
            return findings
        except (FileNotFoundError, OSError, subprocess.SubprocessError, RuntimeError) as e:
            logger.error("Failed to run bandit: %s", e)
            raise RuntimeError(f"Security scanner failed: {e}") from e
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def scan_trajectory(self, trajectory: object) -> List[Dict[str, str]]:
        """Scan all tool action arguments in a trajectory that look like code."""
        findings: List[Dict[str, str]] = []
        if not hasattr(trajectory, "steps"):
            return findings

        for step in trajectory.steps:
            action_payload = None
            if isinstance(step.action, str) and step.action:
                try:
                    action_payload = json.loads(step.action)
                except json.JSONDecodeError:
                    action_payload = None
            elif isinstance(step.action, dict):
                action_payload = step.action
            code = None
            if isinstance(action_payload, dict):
                if "code" in action_payload:
                    code = action_payload["code"]
                elif isinstance(action_payload.get("args"), dict):
                    code = action_payload["args"].get("code")
            if code is not None:
                step_findings = self.scan_python_code(str(code))
                if step_findings:
                    findings.extend(step_findings)
        return findings
