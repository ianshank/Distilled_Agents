"""
Test Factories
==============

Provides factory methods for creating test objects.
"""

from __future__ import annotations

from typing import Any, Optional
import asyncio


def create_mock_agent(
    response_text: str = "Test response",
    confidence: float = 0.85,
    latency_ms: float = 10.0,
    should_fail: bool = False
):
    """
    Create a mock agent function.
    
    Args:
        response_text: Response text to return.
        confidence: Confidence score to return.
        latency_ms: Simulated latency in milliseconds.
        should_fail: Whether the agent should raise an error.
    
    Returns:
        Mock agent function.
    """
    async def agent_func(task: str, **kwargs) -> dict[str, Any]:
        await asyncio.sleep(latency_ms / 1000.0)
        
        if should_fail:
            raise Exception("Mock agent failure")
        
        return {
            'response': f"{response_text}: {task[:30]}",
            'confidence': confidence,
            'token_count': len(task.split()),
            'success': True,
            'latency_ms': latency_ms
        }
    
    return agent_func


def create_mock_validator(always_valid: bool = True):
    """
    Create a mock validator.
    
    Args:
        always_valid: Whether validator always returns valid.
    
    Returns:
        Mock validator object.
    """
    from enhanced_system.core.input_validator import ValidationResult
    
    class MockValidator:
        def validate_task_input(self, task: str) -> ValidationResult:
            if always_valid:
                return ValidationResult(
                    is_valid=True,
                    sanitized_input=task,
                    warnings=[]
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message="Validation failed"
                )
    
    return MockValidator()


def create_mock_cache(initial_data: Optional[dict[str, Any]] = None):
    """
    Create a mock cache.
    
    Args:
        initial_data: Initial cache data.
    
    Returns:
        Mock cache object.
    """
    class MockCache:
        def __init__(self):
            self.data = initial_data or {}
            self.hits = 0
            self.misses = 0
        
        def get(self, key: str) -> Optional[Any]:
            if key in self.data:
                self.hits += 1
                return self.data[key]
            self.misses += 1
            return None
        
        def set(self, key: str, value: Any) -> None:
            self.data[key] = value
        
        def delete(self, key: str) -> bool:
            if key in self.data:
                del self.data[key]
                return True
            return False
        
        def clear(self) -> None:
            self.data.clear()
        
        def get_stats(self) -> dict[str, Any]:
            total = self.hits + self.misses
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.hits / total if total > 0 else 0,
                'size': len(self.data)
            }
    
    return MockCache()

