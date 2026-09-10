"""Facade for error handling (backward-compatible import path)."""

from enhanced_system.core.errors import (
    AllStrategiesFailedError,
    ErrorLearner,
    ErrorRecord,
    ErrorType,
    FallbackManager,
    IntelligentRetryHandler,
    PermanentError,
    TemporaryError,
)

__all__ = [
    "ErrorType",
    "TemporaryError",
    "PermanentError",
    "ErrorRecord",
    "AllStrategiesFailedError",
    "ErrorLearner",
    "IntelligentRetryHandler",
    "FallbackManager",
]
