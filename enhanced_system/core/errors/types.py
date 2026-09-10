"""Error types and exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from enhanced_system.core.enums import ErrorType

__all__ = [
    "ErrorType",
    "TemporaryError",
    "PermanentError",
    "ErrorRecord",
    "AllStrategiesFailedError",
]


class TemporaryError(Exception):
    """Exception for temporary/retryable errors."""


class PermanentError(Exception):
    """Exception for permanent/non-retryable errors."""


class AllStrategiesFailedError(Exception):
    """Exception when all fallback strategies fail."""


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_type: str
    error_message: str
    agent: str
    task: str
    timestamp: datetime
    retry_count: int
    resolved: bool
    resolution_method: Optional[str] = None
