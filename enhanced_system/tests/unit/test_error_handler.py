"""
Unit Tests for Error Handler
==============================

Comprehensive tests for IntelligentRetryHandler, FallbackManager, and ErrorLearner.
Target: 80%+ coverage
"""

from __future__ import annotations

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time

from enhanced_system.core.error_handler import (
    IntelligentRetryHandler,
    FallbackManager,
    ErrorLearner
)
from enhanced_system.core.constants import *
from enhanced_system.core.enums import ErrorType


@pytest.mark.unit
class TestIntelligentRetryHandler:
    """Test IntelligentRetryHandler class."""
    
    def test_initialization(self, test_config):
        """Test retry handler initialization."""
        config = test_config.get('error_handling', {})
        handler = IntelligentRetryHandler(config)
        
        assert handler.max_retries == config.get('max_retries', DEFAULT_MAX_RETRIES)
        assert handler.base_delay == config.get('base_delay', DEFAULT_BASE_DELAY)
    
    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self, test_config):
        """Test successful execution on first attempt."""
        handler = IntelligentRetryHandler(test_config.get('error_handling', {}))
        
        async def success_func():
            return {"result": "success"}
        
        result = await handler.retry(success_func)
        assert result == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, test_config):
        """Test successful execution after transient failures."""
        config = test_config.get('error_handling', {})
        config['max_retries'] = 3
        handler = IntelligentRetryHandler(config)
        
        attempt_count = 0
        
        async def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Transient error")
            return {"result": "success"}
        
        result = await handler.retry(flaky_func)
        assert result == {"result": "success"}
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self, test_config):
        """Test retry exhaustion after max attempts."""
        config = test_config.get('error_handling', {})
        config['max_retries'] = 2
        handler = IntelligentRetryHandler(config)
        
        async def always_fail():
            raise Exception("Permanent error")
        
        with pytest.raises(Exception, match="Permanent error"):
            await handler.retry(always_fail)
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, test_config):
        """Test exponential backoff timing."""
        config = test_config.get('error_handling', {})
        config['base_delay'] = 0.1
        config['max_retries'] = 3
        handler = IntelligentRetryHandler(config)
        
        attempt_times = []
        
        async def failing_func():
            attempt_times.append(time.time())
            raise Exception("Error")
        
        try:
            await handler.retry(failing_func)
        except Exception:
            pass
        
        # Verify delays are exponential
        if len(attempt_times) >= 3:
            delay1 = attempt_times[1] - attempt_times[0]
            delay2 = attempt_times[2] - attempt_times[1]
            # Second delay should be roughly 2x first delay
            assert delay2 > delay1 * 1.5
    
    @pytest.mark.asyncio
    async def test_classify_error_temporary(self, test_config):
        """Test classification of temporary errors."""
        handler = IntelligentRetryHandler(test_config.get('error_handling', {}))
        
        # Network errors should be temporary
        error_type = handler._classify_error(ConnectionError("Network timeout"))
        assert error_type == ErrorType.TEMPORARY
        
        # Timeout errors should be temporary
        error_type = handler._classify_error(TimeoutError("Request timeout"))
        assert error_type == ErrorType.TEMPORARY
    
    @pytest.mark.asyncio
    async def test_classify_error_permanent(self, test_config):
        """Test classification of permanent errors."""
        handler = IntelligentRetryHandler(test_config.get('error_handling', {}))
        
        # ValueError should be permanent
        error_type = handler._classify_error(ValueError("Invalid input"))
        assert error_type == ErrorType.PERMANENT
        
        # AuthError should be permanent
        class AuthError(Exception):
            pass
        
        error_type = handler._classify_error(AuthError("Authentication failed"))
        assert error_type == ErrorType.PERMANENT
    
    @pytest.mark.asyncio
    async def test_no_retry_for_permanent_errors(self, test_config):
        """Test that permanent errors are not retried."""
        config = test_config.get('error_handling', {})
        config['max_retries'] = 5
        handler = IntelligentRetryHandler(config)
        
        attempt_count = 0
        
        async def permanent_error_func():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Invalid input")
        
        with pytest.raises(ValueError):
            await handler.retry(permanent_error_func)
        
        # Should fail immediately, no retries
        assert attempt_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_with_context(self, test_config):
        """Test retry with context information."""
        handler = IntelligentRetryHandler(test_config.get('error_handling', {}))
        
        async def func_with_context(task_id: str):
            return {"task_id": task_id, "result": "success"}
        
        result = await handler.retry(func_with_context, task_id="test_123")
        assert result["task_id"] == "test_123"
    
    @pytest.mark.asyncio
    async def test_max_delay_enforcement(self, test_config):
        """Test that delay doesn't exceed max_delay."""
        config = test_config.get('error_handling', {})
        config['base_delay'] = 1.0
        config['max_delay'] = 2.0
        config['max_retries'] = 10
        handler = IntelligentRetryHandler(config)
        
        # Calculate delay for high retry count
        delay = handler._calculate_delay(5)
        
        # Should be capped at max_delay
        assert delay <= 2.0


@pytest.mark.unit
class TestFallbackManager:
    """Test FallbackManager class."""
    
    def test_initialization(self, test_config):
        """Test fallback manager initialization."""
        manager = FallbackManager(test_config.get('error_handling', {}))
        assert manager.fallback_chain is not None
    
    @pytest.mark.asyncio
    async def test_primary_agent_success(self, test_config, mock_agent):
        """Test successful primary agent execution."""
        manager = FallbackManager(test_config.get('error_handling', {}))
        
        result = await manager.execute_with_fallback(mock_agent, "Test task")
        
        assert result is not None
        assert 'response' in result
    
    @pytest.mark.asyncio
    async def test_fallback_to_secondary(self, test_config):
        """Test fallback to secondary agent."""
        manager = FallbackManager(test_config.get('error_handling', {}))
        
        # Primary agent fails
        async def primary_agent(task):
            raise Exception("Primary failed")
        
        # Secondary agent succeeds
        async def secondary_agent(task):
            return {"response": "Secondary success", "confidence": 0.7}
        
        manager.register_fallback("primary", secondary_agent)
        
        result = await manager.execute_with_fallback(primary_agent, "Test task")
        
        assert result is not None
        assert "Secondary success" in result["response"]
    
    @pytest.mark.asyncio
    async def test_multi_level_fallback(self, test_config):
        """Test multi-level fallback chain."""
        manager = FallbackManager(test_config.get('error_handling', {}))
        
        # All agents fail except last
        async def agent1(task):
            raise Exception("Agent 1 failed")
        
        async def agent2(task):
            raise Exception("Agent 2 failed")
        
        async def agent3(task):
            return {"response": "Agent 3 success", "confidence": 0.5}
        
        manager.register_fallback("level1", agent2)
        manager.register_fallback("level2", agent3)
        
        # Should eventually succeed with agent3
        result = await manager.execute_with_fallback(agent1, "Test task")
        
        assert result is not None
        assert "Agent 3" in result["response"]
    
    @pytest.mark.asyncio
    async def test_all_fallbacks_exhausted(self, test_config):
        """Test when all fallbacks are exhausted."""
        manager = FallbackManager(test_config.get('error_handling', {}))
        
        async def failing_agent(task):
            raise Exception("Always fails")
        
        manager.register_fallback("fallback", failing_agent)
        
        with pytest.raises(Exception):
            await manager.execute_with_fallback(failing_agent, "Test task")
    
    @pytest.mark.asyncio
    async def test_fallback_with_degraded_quality(self, test_config):
        """Test fallback with quality degradation warning."""
        manager = FallbackManager(test_config.get('error_handling', {}))
        
        async def high_quality_agent(task):
            raise Exception("High quality agent failed")
        
        async def low_quality_agent(task):
            return {"response": "Low quality", "confidence": 0.4}
        
        manager.register_fallback("degraded", low_quality_agent, quality_degradation=True)
        
        result = await manager.execute_with_fallback(high_quality_agent, "Test task")
        
        assert result is not None
        assert result["confidence"] < 0.5
    
    def test_register_fallback(self, test_config):
        """Test registering fallback agents."""
        manager = FallbackManager(test_config.get('error_handling', {}))
        
        async def fallback_agent(task):
            return {"response": "Fallback"}
        
        manager.register_fallback("test_fallback", fallback_agent)
        
        assert "test_fallback" in manager.fallback_chain


@pytest.mark.unit
class TestErrorLearner:
    """Test ErrorLearner class."""
    
    def test_initialization(self, temp_db_path, test_config):
        """Test error learner initialization."""
        config = test_config.get('error_handling', {})
        config['error_db_path'] = temp_db_path
        
        learner = ErrorLearner(config)
        assert learner.config == config
    
    @pytest.mark.asyncio
    async def test_log_error(self, temp_db_path, test_config):
        """Test logging an error."""
        config = test_config.get('error_handling', {})
        config['error_db_path'] = temp_db_path
        config['enable_error_learning'] = True
        
        learner = ErrorLearner(config)
        
        error_info = {
            'task': 'Test task',
            'error_type': 'ValueError',
            'error_message': 'Invalid input',
            'agent': 'test_agent',
            'timestamp': time.time()
        }
        
        learner.log_error(error_info)
        
        # Verify error was logged
        patterns = learner.analyze_error_patterns()
        assert len(patterns) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_error_patterns(self, temp_db_path, test_config):
        """Test error pattern analysis."""
        config = test_config.get('error_handling', {})
        config['error_db_path'] = temp_db_path
        config['enable_error_learning'] = True
        
        learner = ErrorLearner(config)
        
        # Log multiple similar errors
        for i in range(5):
            learner.log_error({
                'task': f'Task {i}',
                'error_type': 'ValueError',
                'error_message': 'Invalid input format',
                'agent': 'test_agent'
            })
        
        patterns = learner.analyze_error_patterns()
        
        # Should identify ValueError as common pattern
        assert any(p['error_type'] == 'ValueError' for p in patterns)
    
    @pytest.mark.asyncio
    async def test_generate_training_data(self, temp_db_path, test_config):
        """Test generating training data from errors."""
        config = test_config.get('error_handling', {})
        config['error_db_path'] = temp_db_path
        config['enable_error_learning'] = True
        
        learner = ErrorLearner(config)
        
        # Log errors
        learner.log_error({
            'task': 'Malformed task',
            'error_type': 'ValidationError',
            'error_message': 'Invalid format',
            'agent': 'test_agent'
        })
        
        training_data = learner.generate_training_data()
        
        assert len(training_data) > 0
        assert 'prompt' in training_data[0]
        assert 'completion' in training_data[0]
    
    def test_get_error_stats(self, temp_db_path, test_config):
        """Test getting error statistics."""
        config = test_config.get('error_handling', {})
        config['error_db_path'] = temp_db_path
        config['enable_error_learning'] = True
        
        learner = ErrorLearner(config)
        
        # Log various errors
        for error_type in ['ValueError', 'TypeError', 'ValueError']:
            learner.log_error({
                'task': 'Test',
                'error_type': error_type,
                'error_message': 'Test error'
            })
        
        stats = learner.get_error_stats()
        
        assert stats['total_errors'] == 3
        assert stats['by_type']['ValueError'] == 2
        assert stats['by_type']['TypeError'] == 1
    
    def test_disabled_error_learning(self, temp_db_path, test_config):
        """Test that error learning can be disabled."""
        config = test_config.get('error_handling', {})
        config['error_db_path'] = temp_db_path
        config['enable_error_learning'] = False
        
        learner = ErrorLearner(config)
        
        # Should not log when disabled
        learner.log_error({'task': 'Test', 'error_type': 'Error'})
        
        stats = learner.get_error_stats()
        assert stats['total_errors'] == 0

