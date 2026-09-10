"""Unit tests for retry, fallback, and error learning APIs."""

from __future__ import annotations

from datetime import datetime

import pytest

from enhanced_system.core.enums import ErrorType
from enhanced_system.core.error_handler import (
    ErrorLearner,
    ErrorRecord,
    FallbackManager,
    IntelligentRetryHandler,
)
from enhanced_system.core.constants import DEFAULT_BASE_DELAY, DEFAULT_MAX_RETRIES


@pytest.mark.unit
class TestIntelligentRetryHandler:
    def test_initialization(self, test_config):
        config = test_config.get("error_handling", {})
        handler = IntelligentRetryHandler(config)
        assert handler.max_retries == config.get("max_retries", DEFAULT_MAX_RETRIES)
        assert handler.base_delay == config.get("base_delay", DEFAULT_BASE_DELAY)

    @pytest.mark.asyncio
    async def test_success_first_attempt(self, test_config):
        handler = IntelligentRetryHandler(test_config.get("error_handling", {}))

        async def success_func():
            return {"result": "success"}

        result = await handler.execute_with_retry(success_func)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_success_after_transient_failures(self, test_config):
        config = dict(test_config.get("error_handling", {}))
        config["max_retries"] = 3
        config["base_delay"] = 0.01
        config["enable_error_learning"] = False
        handler = IntelligentRetryHandler(config)
        attempt_count = 0

        async def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network timeout")
            return {"result": "success"}

        result = await handler.execute_with_retry(flaky_func)
        assert result == {"result": "success"}
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, test_config):
        config = dict(test_config.get("error_handling", {}))
        config["max_retries"] = 2
        config["base_delay"] = 0.01
        config["enable_error_learning"] = False
        handler = IntelligentRetryHandler(config)

        async def always_fail():
            raise ConnectionError("Network timeout")

        with pytest.raises(ConnectionError, match="Network timeout"):
            await handler.execute_with_retry(always_fail)

    def test_classify_error_temporary(self, test_config):
        handler = IntelligentRetryHandler(test_config.get("error_handling", {}))
        assert handler.classify_error(ConnectionError("Network timeout")) == ErrorType.TEMPORARY
        assert handler.classify_error(TimeoutError("Request timeout")) == ErrorType.TEMPORARY

    def test_classify_error_permanent(self, test_config):
        handler = IntelligentRetryHandler(test_config.get("error_handling", {}))
        assert handler.classify_error(ValueError("Invalid input")) == ErrorType.PERMANENT

        class AuthError(Exception):
            pass

        assert handler.classify_error(AuthError("Authentication failed")) == ErrorType.PERMANENT

    @pytest.mark.asyncio
    async def test_no_retry_for_permanent_errors(self, test_config):
        config = dict(test_config.get("error_handling", {}))
        config["max_retries"] = 5
        config["enable_error_learning"] = False
        handler = IntelligentRetryHandler(config)
        attempt_count = 0

        async def permanent_error_func():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            await handler.execute_with_retry(permanent_error_func)
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_kwargs(self, test_config):
        handler = IntelligentRetryHandler(
            {**test_config.get("error_handling", {}), "enable_error_learning": False}
        )

        async def func_with_context(task_id: str):
            return {"task_id": task_id, "result": "success"}

        result = await handler.execute_with_retry(func_with_context, task_id="test_123")
        assert result["task_id"] == "test_123"

    def test_max_delay_enforcement(self, test_config):
        config = dict(test_config.get("error_handling", {}))
        config["base_delay"] = 1.0
        config["max_delay"] = 2.0
        handler = IntelligentRetryHandler(config)
        assert handler.calculate_delay(5) <= 2.0


@pytest.mark.unit
class TestFallbackManager:
    def test_initialization(self, test_config):
        manager = FallbackManager(
            {**test_config.get("error_handling", {}), "enable_error_learning": False}
        )
        assert manager.fallback_chain is not None

    @pytest.mark.asyncio
    async def test_primary_agent_success(self, test_config, mock_agent):
        manager = FallbackManager(
            {**test_config.get("error_handling", {}), "enable_error_learning": False}
        )
        result = await manager.execute_with_fallback(mock_agent, "Test task")
        assert result["success"] is True
        assert "response" in result["result"]

    @pytest.mark.asyncio
    async def test_fallback_to_secondary(self, test_config):
        manager = FallbackManager(
            {**test_config.get("error_handling", {}), "enable_error_learning": False}
        )

        async def primary_agent(task):
            raise Exception("Primary failed")

        async def secondary_agent(task):
            return {"response": "Secondary success", "confidence": 0.7}

        manager.register_fallback(secondary_agent, name="secondary")
        result = await manager.execute_with_fallback(primary_agent, "Test task")
        assert result["success"] is True
        assert "Secondary success" in result["result"]["response"]

    @pytest.mark.asyncio
    async def test_all_fallbacks_exhausted(self, test_config):
        manager = FallbackManager(
            {**test_config.get("error_handling", {}), "enable_error_learning": False}
        )

        async def failing_agent(task):
            raise Exception("Always fails")

        manager.register_fallback(failing_agent, name="fallback")
        with pytest.raises(Exception, match="All strategies failed"):
            await manager.execute_with_fallback(failing_agent, "Test task")

    def test_register_fallback(self, test_config):
        manager = FallbackManager(
            {**test_config.get("error_handling", {}), "enable_error_learning": False}
        )

        async def fallback_agent(task):
            return {"response": "Fallback"}

        manager.register_fallback(fallback_agent, name="test_fallback")
        names = [item["name"] for item in manager.fallback_chain]
        assert "test_fallback" in names


@pytest.mark.unit
class TestErrorLearner:
    def test_initialization(self, temp_db_path):
        learner = ErrorLearner(db_path=temp_db_path, enabled=True)
        assert learner.db_path == temp_db_path

    def test_record_and_stats(self, temp_db_path):
        learner = ErrorLearner(db_path=temp_db_path, enabled=True)
        learner.record_error(
            ErrorRecord(
                error_type="ValueError",
                error_message="Invalid input",
                agent="test_agent",
                task="Test task",
                timestamp=datetime.now(),
                retry_count=0,
                resolved=False,
            )
        )
        stats = learner.get_error_stats()
        assert stats["total_errors"] == 1
        assert stats["by_type"]["ValueError"] == 1

    def test_disabled_error_learning(self, temp_db_path):
        learner = ErrorLearner(db_path=temp_db_path, enabled=False)
        learner.record_error(
            ErrorRecord(
                error_type="Error",
                error_message="x",
                agent="a",
                task="t",
                timestamp=datetime.now(),
                retry_count=0,
                resolved=False,
            )
        )
        assert learner.get_error_stats()["total_errors"] == 0
