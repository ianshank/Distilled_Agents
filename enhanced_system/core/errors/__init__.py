"""Error handling package public API."""

from enhanced_system.core.enums import ErrorType
from enhanced_system.core.errors.fallback import FallbackManager
from enhanced_system.core.errors.learner import ErrorLearner
from enhanced_system.core.errors.retry import IntelligentRetryHandler
from enhanced_system.core.errors.types import (
    AllStrategiesFailedError,
    ErrorRecord,
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
