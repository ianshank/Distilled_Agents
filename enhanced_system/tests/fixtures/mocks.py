"""
Mock Objects
============

Provides mock objects for testing external dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "mock_agent"):
        self.name = name
        self.call_count = 0
        self.last_task = None

    async def infer(self, task: str, **kwargs) -> dict[str, Any]:
        """Mock inference method."""
        self.call_count += 1
        self.last_task = task

        return {
            "response": f"{self.name} processed: {task[:30]}",
            "confidence": 0.85,
            "agent": self.name,
        }


class MockValidator:
    """Mock validator for testing."""

    def __init__(self, should_pass: bool = True):
        self.should_pass = should_pass
        self.validation_count = 0

    def validate_task_input(self, task: str):
        """Mock validation method."""
        self.validation_count += 1

        from enhanced_system.core.input_validator import ValidationResult

        if self.should_pass:
            return ValidationResult(is_valid=True, sanitized_input=task)
        else:
            return ValidationResult(is_valid=False, error_message="Mock validation failure")


class MockCache:
    """Mock cache for testing."""

    def __init__(self):
        self.data = {}
        self.get_count = 0
        self.set_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Mock get method."""
        self.get_count += 1
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Mock set method."""
        self.set_count += 1
        self.data[key] = value

    def clear(self) -> None:
        """Mock clear method."""
        self.data.clear()


class MockMetricsCollector:
    """Mock metrics collector for testing."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record_request(self, agent: str, status: str, latency: float):
        """Record request metric."""
        self.metrics["requests"].append({"agent": agent, "status": status, "latency": latency})

    def record_error(self, agent: str, error_type: str):
        """Record error metric."""
        self.metrics["errors"].append({"agent": agent, "error_type": error_type})

    def get_metrics(self) -> dict[str, list]:
        """Get all recorded metrics."""
        return dict(self.metrics)


class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.logs = []

    def info(self, msg: str):
        self.logs.append(("INFO", msg))

    def warning(self, msg: str):
        self.logs.append(("WARNING", msg))

    def error(self, msg: str):
        self.logs.append(("ERROR", msg))

    def debug(self, msg: str):
        self.logs.append(("DEBUG", msg))
