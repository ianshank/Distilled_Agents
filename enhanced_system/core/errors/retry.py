"""Intelligent retry with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable

from enhanced_system.core.enums import ErrorType
from enhanced_system.core.errors.learner import ErrorLearner
from enhanced_system.core.errors.types import ErrorRecord, PermanentError, TemporaryError

logger = logging.getLogger(__name__)


class IntelligentRetryHandler:
    """Retry with classification and exponential backoff."""

    def __init__(self, config: dict[str, Any]):
        self.max_retries = config.get("max_retries", 3)
        self.base_delay = config.get("base_delay", 1.0)
        self.max_delay = config.get("max_delay", 32.0)
        self.exponential_base = config.get("exponential_base", 2)
        self.error_learner = None
        if config.get("enable_error_learning", True):
            try:
                self.error_learner = ErrorLearner(config.get("error_db_path", "./data/errors.db"))
            except Exception as exc:
                logger.warning("Failed to initialize error learner: %s", exc)

    def classify_error(self, error: Exception) -> ErrorType:
        if isinstance(error, TemporaryError):
            return ErrorType.TEMPORARY
        if isinstance(error, PermanentError):
            return ErrorType.PERMANENT

        error_str = str(error).lower()
        error_name = type(error).__name__.lower()
        combined = f"{error_name} {error_str}"

        temporary_indicators = [
            "timeout",
            "connection",
            "network",
            "rate limit",
            "throttle",
            "busy",
            "unavailable",
            "retry",
        ]
        for indicator in temporary_indicators:
            if indicator in combined:
                return ErrorType.TEMPORARY

        permanent_indicators = [
            "invalid",
            "forbidden",
            "unauthorized",
            "not found",
            "permission",
            "authentication",
            "malformed",
            "valueerror",
            "autherror",
        ]
        for indicator in permanent_indicators:
            if indicator in combined:
                return ErrorType.PERMANENT

        return ErrorType.UNKNOWN

    def calculate_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        agent: str = "unknown",
        task: str = "",
        **kwargs,
    ) -> Any:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                if attempt > 0 and self.error_learner:
                    self.error_learner.record_error(
                        ErrorRecord(
                            error_type="resolved",
                            error_message=str(last_error),
                            agent=agent,
                            task=task,
                            timestamp=datetime.now(),
                            retry_count=attempt,
                            resolved=True,
                            resolution_method="retry",
                        )
                    )
                return result
            except Exception as error:
                last_error = error
                error_type = self.classify_error(error)
                logger.warning(
                    "Attempt %s/%s failed: %s (type: %s)",
                    attempt + 1,
                    self.max_retries + 1,
                    error,
                    error_type.value,
                )
                if error_type == ErrorType.PERMANENT:
                    if self.error_learner:
                        self.error_learner.record_error(
                            ErrorRecord(
                                error_type=error_type.value,
                                error_message=str(error),
                                agent=agent,
                                task=task,
                                timestamp=datetime.now(),
                                retry_count=attempt,
                                resolved=False,
                            )
                        )
                    raise
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    if self.error_learner:
                        self.error_learner.record_error(
                            ErrorRecord(
                                error_type=error_type.value,
                                error_message=str(error),
                                agent=agent,
                                task=task,
                                timestamp=datetime.now(),
                                retry_count=attempt + 1,
                                resolved=False,
                            )
                        )
                    raise
        raise last_error  # pragma: no cover
