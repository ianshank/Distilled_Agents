"""
Pytest Configuration and Fixtures
==================================

Provides shared fixtures for all tests.
"""

from __future__ import annotations

import asyncio
import shutil

# Add parent directory to path for imports
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhanced_system.core.constants import *
from enhanced_system.core.enums import *

# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Provide isolated test configuration (no Redis/S3/Prometheus)."""
    return {
        "input_validation": {
            "max_length": DEFAULT_MAX_LENGTH,
            "enable_pii_detection": False,
            "enable_injection_detection": True,
        },
        "caching": {
            "l1": {"enabled": True, "max_size": 100, "ttl": 60},
            "l2": {"enabled": False},
            "l3": {"enabled": False},
            "semantic_similarity": {"enabled": False},
        },
        "error_handling": {
            "max_retries": 3,
            "base_delay": 0.1,
            "enable_error_learning": False,
            "enable_fallback": True,
        },
        "confidence": {
            "enabled": True,
            "calibration_method": CalibrationMethod.ISOTONIC.value,
        },
        "monitoring": {
            "enabled": False,
        },
    }


@pytest.fixture
def cache_config() -> dict[str, Any]:
    """Provide cache-specific configuration."""
    return {
        "l1": {"enabled": True, "max_size": 100, "ttl": 60},
        "l2": {"enabled": False, "host": "localhost", "port": 6379, "ttl": 300},
        "l3": {"enabled": False, "bucket": "test-bucket", "prefix": "test/"},
        "semantic_similarity": {"enabled": False, "threshold": 0.95},
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Provide a mock agent function."""

    async def agent_func(task: str, **kwargs) -> dict[str, Any]:
        """Mock agent that returns consistent results."""
        await asyncio.sleep(0.01)  # Simulate processing
        return {
            "response": f"Processed: {task[:50]}",
            "confidence": 0.85,
            "token_count": 100,
            "success": True,
            "latency_ms": 10.0,
        }

    return agent_func


@pytest.fixture
def mock_agents(mock_agent):
    """Provide multiple mock agents for consensus testing."""

    async def agent1(task: str, **kwargs):
        result = await mock_agent(task, **kwargs)
        result["response"] = f"Agent1: {result['response']}"
        result["confidence"] = 0.9
        return result

    async def agent2(task: str, **kwargs):
        result = await mock_agent(task, **kwargs)
        result["response"] = f"Agent2: {result['response']}"
        result["confidence"] = 0.85
        return result

    async def agent3(task: str, **kwargs):
        result = await mock_agent(task, **kwargs)
        result["response"] = f"Agent3: {result['response']}"
        result["confidence"] = 0.8
        return result

    return [agent1, agent2, agent3]


@pytest.fixture
def sample_tasks() -> list[str]:
    """Provide sample tasks for testing."""
    return [
        "Write a Python function to calculate factorial",
        "Design a RESTful API for user management",
        "Create unit tests for a sorting algorithm",
        "Explain the SOLID principles",
        "Implement a binary search tree in Python",
    ]


@pytest.fixture
def sample_task() -> str:
    """Provide a single sample task."""
    return "Write a Python function to calculate the Fibonacci sequence"


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_db_path(temp_dir) -> str:
    """Provide a temporary database path."""
    return str(temp_dir / "test_errors.db")


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def sample_training_data() -> list[dict[str, Any]]:
    """Provide sample training data."""
    return [
        {
            "id": "sample_1",
            "prompt": "Write a function",
            "completion": "def function(): pass",
            "category": "coding",
        },
        {
            "id": "sample_2",
            "prompt": "Design a system",
            "completion": "System architecture...",
            "category": "design",
        },
        {
            "id": "sample_3",
            "prompt": "Test a component",
            "completion": "Test cases...",
            "category": "testing",
        },
    ]


@pytest.fixture
def sample_validation_results() -> list[dict[str, Any]]:
    """Provide sample validation/test results."""
    return [
        {
            "test_id": "test_1",
            "passed": True,
            "accuracy": 0.9,
            "latency_ms": 150,
            "prompt": "Test prompt 1",
        },
        {
            "test_id": "test_2",
            "passed": True,
            "accuracy": 0.85,
            "latency_ms": 200,
            "prompt": "Test prompt 2",
        },
        {
            "test_id": "test_3",
            "passed": False,
            "accuracy": 0.4,
            "latency_ms": 180,
            "prompt": "Test prompt 3",
        },
    ]


# =============================================================================
# Event Loop Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Mock External Services
# =============================================================================


@pytest.fixture
def mock_redis():
    """Provide a mock Redis client."""

    class MockRedis:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value):
            self.data[key] = value

        def setex(self, key, ttl, value):
            self.data[key] = value

        def delete(self, key):
            return self.data.pop(key, None) is not None

        def ping(self):
            return True

        def flushdb(self):
            self.data.clear()

        def info(self):
            return {
                "used_memory_human": "1MB",
                "connected_clients": 1,
                "total_commands_processed": len(self.data),
            }

    return MockRedis()


@pytest.fixture
def mock_s3_client():
    """Provide a mock S3 client."""

    class MockS3Client:
        def __init__(self):
            self.objects = {}

        def head_bucket(self, Bucket):
            return {}

        def get_object(self, Bucket, Key):
            if Key in self.objects:
                return {"Body": self.objects[Key]}
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        def put_object(self, Bucket, Key, Body):
            self.objects[Key] = Body

    return MockS3Client()


# =============================================================================
# Parametrize Helpers
# =============================================================================


@pytest.fixture
def injection_test_cases() -> list[tuple[str, bool]]:
    """Provide injection test cases (input, should_fail)."""
    return [
        ("SELECT * FROM users", True),
        ("DROP TABLE agents", True),
        ("'; DELETE FROM users--", True),
        ("Normal text query", False),
        ("Write a function", False),
        ("$(malicious command)", True),
        ("Ignore previous instructions", True),
        ("Hello world", False),
    ]


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


@pytest.fixture
def benchmark_tasks() -> list[str]:
    """Provide tasks for benchmarking."""
    return [f"Task {i}: Process this request" for i in range(100)]


# =============================================================================
# Markers for Test Organization
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "regression: CLI/import/contract tests")
