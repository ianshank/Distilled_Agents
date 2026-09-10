"""Fallback strategy chain."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Optional

from enhanced_system.core.errors.learner import ErrorLearner
from enhanced_system.core.errors.types import ErrorRecord

logger = logging.getLogger(__name__)


class FallbackManager:
    """Execute a primary function, then registered fallbacks."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enable_fallback = config.get("enable_fallback", True)
        self.fallback_chain: list[dict[str, Any]] = []
        self.error_learner = None
        if config.get("enable_error_learning", True):
            try:
                self.error_learner = ErrorLearner(
                    config.get("error_db_path", "./data/errors.db")
                )
            except Exception as exc:
                logger.warning("Failed to initialize error learner: %s", exc)

    def register_fallback(
        self,
        strategy: Callable,
        priority: int = 100,
        name: str = "unnamed",
    ) -> None:
        self.fallback_chain.append(
            {"strategy": strategy, "priority": priority, "name": name}
        )
        self.fallback_chain.sort(key=lambda item: item["priority"])
        logger.info("Registered fallback strategy: %s (priority: %s)", name, priority)

    async def execute_with_fallback(
        self,
        primary_func: Callable,
        *args,
        agent: str = "unknown",
        task: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        async def _call(func: Callable) -> Any:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        if not self.enable_fallback:
            result = await _call(primary_func)
            return {"result": result, "method": "primary", "success": True}

        primary_error: Optional[Exception] = None
        try:
            result = await _call(primary_func)
            return {"result": result, "method": "primary", "success": True}
        except Exception as exc:
            primary_error = exc
            logger.warning("Primary execution failed: %s", exc)

        for fallback in self.fallback_chain:
            strategy_name = fallback["name"]
            strategy_func = fallback["strategy"]
            try:
                result = await _call(strategy_func)
                if self._is_acceptable_result(result):
                    if self.error_learner:
                        self.error_learner.record_error(
                            ErrorRecord(
                                error_type="resolved",
                                error_message=str(primary_error),
                                agent=agent,
                                task=task,
                                timestamp=datetime.now(),
                                retry_count=0,
                                resolved=True,
                                resolution_method=strategy_name,
                            )
                        )
                    return {
                        "result": result,
                        "method": strategy_name,
                        "success": True,
                        "fallback": True,
                    }
            except Exception as fallback_error:
                logger.warning(
                    "Fallback strategy %s failed: %s", strategy_name, fallback_error
                )
                continue

        if self.error_learner:
            self.error_learner.record_error(
                ErrorRecord(
                    error_type="all_failed",
                    error_message=str(primary_error),
                    agent=agent,
                    task=task,
                    timestamp=datetime.now(),
                    retry_count=len(self.fallback_chain),
                    resolved=False,
                )
            )
        raise Exception(f"All strategies failed. Primary error: {primary_error}")

    def _is_acceptable_result(self, result: Any) -> bool:
        if result is None:
            return False
        if isinstance(result, dict) and "success" in result:
            return bool(result["success"])
        if isinstance(result, str):
            return len(result.strip()) > 0
        return True
