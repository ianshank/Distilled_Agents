"""Integration tests for the inference pipeline using public APIs."""

from __future__ import annotations

import asyncio

import pytest

from enhanced_system.core.adaptive_router import AdaptiveRouter
from enhanced_system.core.cache_manager import IntelligentCacheManager
from enhanced_system.core.confidence_calibrator import ConfidenceCalibrator
from enhanced_system.core.input_validator import InputValidator
from enhanced_system.core.monitoring import AgentMonitor


@pytest.mark.integration
class TestFullInferencePipeline:
    @pytest.mark.asyncio
    async def test_complete_inference_flow(self, test_config, mock_agent):
        validator = InputValidator(test_config.get("input_validation", {}))
        cache = IntelligentCacheManager(test_config.get("caching", {}))
        router = AdaptiveRouter({"enabled": True})

        task = "Write a Python function to calculate factorial"
        validation_result = validator.validate_task_input(task)
        assert validation_result.is_valid
        sanitized_task = validation_result.sanitized_input

        cache_key = cache.compute_cache_key(sanitized_task, "mock_agent")

        async def compute():
            routing_result = router.route_task(sanitized_task)
            assert routing_result.selected_agents
            return await mock_agent(sanitized_task)

        final_result = await cache.get_or_compute(cache_key, compute)
        assert final_result is not None
        assert "response" in final_result

    @pytest.mark.asyncio
    async def test_cache_hit_path(self, test_config, mock_agent):
        cache = IntelligentCacheManager(test_config.get("caching", {}))
        validator = InputValidator(test_config.get("input_validation", {}))
        task = "Calculate two plus two"
        validation = validator.validate_task_input(task)
        cache_key = cache.compute_cache_key(validation.sanitized_input, "mock_agent")

        result1 = cache.l1_cache.get(cache_key)
        assert result1 is None

        agent_result = await mock_agent(validation.sanitized_input)
        cache.l1_cache.set(cache_key, agent_result)

        result2 = cache.l1_cache.get(cache_key)
        assert result2 == agent_result
        stats = cache.get_stats()
        assert stats["l1"]["hits"] >= 1

    @pytest.mark.asyncio
    async def test_validation_failure_path(self, test_config):
        validator = InputValidator(
            {"max_length": 4096, "enable_injection_detection": True, "enable_pii_detection": False}
        )
        malicious_task = "SELECT * FROM users WHERE id = 1"
        validation_result = validator.validate_task_input(malicious_task)
        assert not validation_result.is_valid
        assert validation_result.error_message is not None

    @pytest.mark.asyncio
    async def test_confidence_calibration_flow(self, test_config, mock_agent):
        calibrator = ConfidenceCalibrator(test_config.get("confidence", {}))
        result = await mock_agent("Implement binary search")
        calibrated = calibrator.calibrate_confidence(
            result.get("confidence", 0.5),
            agent="mock_agent",
            task_type="coding",
        )
        assert calibrated.confidence is not None
        assert calibrated.reliability_band is not None
        assert 0 <= calibrated.confidence <= 1

    @pytest.mark.asyncio
    async def test_multi_request_workflow(self, test_config, mock_agent):
        validator = InputValidator(test_config.get("input_validation", {}))
        cache = IntelligentCacheManager(test_config.get("caching", {}))
        router = AdaptiveRouter({"enabled": True})
        tasks = [
            "Task 1: Write function",
            "Task 2: Design system architecture",
            "Task 3: Create tests",
        ]
        results = []
        for task in tasks:
            validation = validator.validate_task_input(task)
            assert validation.is_valid
            cache_key = cache.compute_cache_key(task, "mock_agent")
            cached = cache.l1_cache.get(cache_key)
            if cached is None:
                router.route_task(task)
                result = await mock_agent(task)
                cache.l1_cache.set(cache_key, result)
                results.append(result)
            else:
                results.append(cached)
        assert len(results) == 3
        assert all("response" in item for item in results)

    @pytest.mark.asyncio
    async def test_error_propagation(self, test_config):
        validator = InputValidator(test_config.get("input_validation", {}))

        async def failing_agent(task):
            raise ValueError("Agent execution failed")

        task = "Normal task"
        validation = validator.validate_task_input(task)
        assert validation.is_valid
        with pytest.raises(ValueError, match="Agent execution failed"):
            await failing_agent(validation.sanitized_input)

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, test_config, mock_agent):
        monitor = AgentMonitor({"enabled": False})
        result = await mock_agent("Test task")
        monitor.track_inference("mock_agent", "Test task", result)
        metrics = monitor.get_agent_stats("mock_agent")
        assert metrics["total_requests"] >= 1

    @pytest.mark.asyncio
    async def test_end_to_end_with_all_components(self, test_config, mock_agent):
        validator = InputValidator(test_config.get("input_validation", {}))
        cache = IntelligentCacheManager(test_config.get("caching", {}))
        router = AdaptiveRouter({"enabled": True})
        calibrator = ConfidenceCalibrator(test_config.get("confidence", {}))
        monitor = AgentMonitor({"enabled": False})
        task = "Create a REST API endpoint for user authentication"
        validation = validator.validate_task_input(task)
        assert validation.is_valid
        sanitized_task = validation.sanitized_input
        cache_key = cache.compute_cache_key(sanitized_task, "mock_agent")
        routing = router.route_task(sanitized_task)
        result = await mock_agent(sanitized_task)
        calibrated = calibrator.calibrate_confidence(
            result.get("confidence", 0.5),
            agent=routing.selected_agents[0],
            task_type="coding",
        )
        result["calibrated_confidence"] = calibrated.confidence
        cache.l1_cache.set(cache_key, result)
        monitor.track_inference(routing.selected_agents[0], sanitized_task, result)
        assert result["response"]
        assert calibrated.confidence >= 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_config, mock_agent):
        validator = InputValidator(test_config.get("input_validation", {}))
        cache = IntelligentCacheManager(test_config.get("caching", {}))

        async def process_task(task: str):
            validation = validator.validate_task_input(task)
            if not validation.is_valid:
                return None
            cache_key = cache.compute_cache_key(task, "mock_agent")
            cached = cache.l1_cache.get(cache_key)
            if cached is None:
                result = await mock_agent(task)
                cache.l1_cache.set(cache_key, result)
                return result
            return cached

        tasks = [f"Task {i}: Process request" for i in range(10)]
        results = await asyncio.gather(*[process_task(item) for item in tasks])
        assert len(results) == 10
        assert all(item is not None and "response" in item for item in results)
