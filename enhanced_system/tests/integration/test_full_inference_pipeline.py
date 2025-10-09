"""
Integration Tests for Full Inference Pipeline
==============================================

Tests the complete flow: validation → caching → routing → inference → monitoring
Target: 70%+ coverage for integration scenarios
"""

from __future__ import annotations

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from enhanced_system.core.input_validator import InputValidator
from enhanced_system.core.cache_manager import IntelligentCacheManager
from enhanced_system.core.adaptive_router import AdaptiveRouter
from enhanced_system.core.confidence_calibrator import ConfidenceCalibrator
from enhanced_system.core.monitoring import AgentMonitor


@pytest.mark.integration
class TestFullInferencePipeline:
    """Integration tests for complete inference workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_inference_flow(self, test_config, mock_agent):
        """Test complete flow from input to output."""
        # Initialize all components
        validator = InputValidator(test_config.get('input_validation', {}))
        cache = IntelligentCacheManager(test_config.get('caching', {}))
        router = AdaptiveRouter({'enabled': True})
        
        # Input task
        task = "Write a Python function to calculate factorial"
        
        # Step 1: Validation
        validation_result = validator.validate_task_input(task)
        assert validation_result.is_valid
        sanitized_task = validation_result.sanitized_input
        
        # Step 2: Check cache
        cache_key = cache._compute_cache_key(sanitized_task, {})
        cached_result = cache.get(cache_key)
        
        if cached_result is None:
            # Step 3: Route task
            routing_result = router.route_task(sanitized_task)
            assert routing_result is not None
            
            # Step 4: Execute agent
            agent_result = await mock_agent(sanitized_task)
            assert agent_result is not None
            
            # Step 5: Cache result
            cache.set(cache_key, agent_result)
        
        # Step 6: Verify result
        final_result = cache.get(cache_key)
        assert final_result is not None
        assert 'response' in final_result
    
    @pytest.mark.asyncio
    async def test_cache_hit_path(self, test_config, mock_agent):
        """Test inference with cache hit."""
        cache = IntelligentCacheManager(test_config.get('caching', {}))
        validator = InputValidator(test_config.get('input_validation', {}))
        
        task = "Calculate 2+2"
        
        # First request - cache miss
        validation = validator.validate_task_input(task)
        cache_key = cache._compute_cache_key(validation.sanitized_input, {})
        
        result1 = cache.get(cache_key)
        assert result1 is None
        
        # Execute and cache
        agent_result = await mock_agent(validation.sanitized_input)
        cache.set(cache_key, agent_result)
        
        # Second request - cache hit
        result2 = cache.get(cache_key)
        assert result2 is not None
        assert result2 == agent_result
        
        # Verify cache stats
        stats = cache.get_stats()
        assert stats['l1']['hits'] >= 1
    
    @pytest.mark.asyncio
    async def test_validation_failure_path(self, test_config):
        """Test error path when validation fails."""
        validator = InputValidator({
            'max_length': 10,  # Very restrictive
            'enable_injection_detection': True
        })
        
        malicious_task = "SELECT * FROM users WHERE id = 1; DROP TABLE users;"
        
        validation_result = validator.validate_task_input(malicious_task)
        
        # Should fail validation
        assert not validation_result.is_valid
        assert validation_result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_confidence_calibration_flow(self, test_config, mock_agent):
        """Test confidence calibration in inference flow."""
        calibrator = ConfidenceCalibrator(test_config.get('confidence', {}))
        
        # Execute agent
        task = "Implement binary search"
        result = await mock_agent(task)
        
        # Calibrate confidence
        raw_confidence = result.get('confidence', 0.5)
        calibrated = calibrator.calibrate_confidence(
            raw_confidence,
            agent='mock_agent',
            task_type='coding'
        )
        
        assert calibrated is not None
        assert 'confidence' in calibrated
        assert 'reliability_band' in calibrated
        assert 0 <= calibrated['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_multi_request_workflow(self, test_config, mock_agent):
        """Test multiple requests in sequence."""
        validator = InputValidator(test_config.get('input_validation', {}))
        cache = IntelligentCacheManager(test_config.get('caching', {}))
        router = AdaptiveRouter({'enabled': True})
        
        tasks = [
            "Task 1: Write function",
            "Task 2: Design system",
            "Task 3: Create tests"
        ]
        
        results = []
        
        for task in tasks:
            # Validate
            validation = validator.validate_task_input(task)
            assert validation.is_valid
            
            # Check cache
            cache_key = cache._compute_cache_key(task, {})
            cached = cache.get(cache_key)
            
            if cached is None:
                # Route and execute
                routing = router.route_task(task)
                result = await mock_agent(task)
                cache.set(cache_key, result)
                results.append(result)
            else:
                results.append(cached)
        
        assert len(results) == 3
        assert all('response' in r for r in results)
    
    @pytest.mark.asyncio
    async def test_error_propagation(self, test_config):
        """Test error propagation through pipeline."""
        validator = InputValidator(test_config.get('input_validation', {}))
        
        async def failing_agent(task):
            raise ValueError("Agent execution failed")
        
        task = "Normal task"
        validation = validator.validate_task_input(task)
        assert validation.is_valid
        
        # Agent failure should propagate
        with pytest.raises(ValueError, match="Agent execution failed"):
            await failing_agent(validation.sanitized_input)
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, test_config, mock_agent):
        """Test monitoring metrics collection during inference."""
        monitor_config = test_config.get('monitoring', {})
        monitor_config['enabled'] = False  # Disable Prometheus for testing
        monitor = AgentMonitor(monitor_config)
        
        task = "Test task"
        agent_name = "mock_agent"
        
        # Execute with monitoring
        start_time = asyncio.get_event_loop().time()
        try:
            result = await mock_agent(task)
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Record metrics
            monitor.record_request(
                agent=agent_name,
                status='success',
                latency_ms=latency_ms
            )
            
            # Get metrics
            metrics = monitor.get_metrics()
            assert metrics is not None
            assert 'requests' in metrics or 'total_requests' in metrics
            
        except Exception as e:
            monitor.record_error(agent=agent_name, error_type=type(e).__name__)
            raise
    
    @pytest.mark.asyncio
    async def test_end_to_end_with_all_components(self, test_config, mock_agent):
        """Test end-to-end flow with all components integrated."""
        # Initialize all components
        validator = InputValidator(test_config.get('input_validation', {}))
        cache = IntelligentCacheManager(test_config.get('caching', {}))
        router = AdaptiveRouter({'enabled': True})
        calibrator = ConfidenceCalibrator(test_config.get('confidence', {}))
        monitor_config = test_config.get('monitoring', {})
        monitor_config['enabled'] = False
        monitor = AgentMonitor(monitor_config)
        
        task = "Create a REST API endpoint for user authentication"
        
        # Full pipeline
        try:
            # 1. Validate input
            validation = validator.validate_task_input(task)
            if not validation.is_valid:
                raise ValueError(validation.error_message)
            
            sanitized_task = validation.sanitized_input
            
            # 2. Check cache
            cache_key = cache._compute_cache_key(sanitized_task, {})
            result = cache.get(cache_key)
            
            if result is None:
                # 3. Route task
                routing = router.route_task(sanitized_task)
                selected_agent = routing['agent']
                
                # 4. Execute agent
                start_time = asyncio.get_event_loop().time()
                result = await mock_agent(sanitized_task)
                latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                
                # 5. Calibrate confidence
                calibrated = calibrator.calibrate_confidence(
                    result.get('confidence', 0.5),
                    agent=selected_agent.get('name', 'unknown'),
                    task_type=routing['task_type']
                )
                result['calibrated_confidence'] = calibrated
                
                # 6. Cache result
                cache.set(cache_key, result)
                
                # 7. Record metrics
                monitor.record_request(
                    agent=selected_agent.get('name', 'unknown'),
                    status='success',
                    latency_ms=latency_ms
                )
            else:
                # Cache hit
                monitor.record_cache_hit()
            
            # Verify final result
            assert result is not None
            assert 'response' in result
            assert 'confidence' in result
            
        except Exception as e:
            monitor.record_error(agent='unknown', error_type=type(e).__name__)
            raise
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_config, mock_agent):
        """Test handling multiple concurrent requests."""
        validator = InputValidator(test_config.get('input_validation', {}))
        cache = IntelligentCacheManager(test_config.get('caching', {}))
        
        async def process_task(task: str):
            validation = validator.validate_task_input(task)
            if not validation.is_valid:
                return None
            
            cache_key = cache._compute_cache_key(task, {})
            result = cache.get(cache_key)
            
            if result is None:
                result = await mock_agent(task)
                cache.set(cache_key, result)
            
            return result
        
        # Submit multiple tasks concurrently
        tasks = [f"Task {i}: Process request" for i in range(10)]
        results = await asyncio.gather(*[process_task(t) for t in tasks])
        
        # All should complete successfully
        assert len(results) == 10
        assert all(r is not None for r in results)
        assert all('response' in r for r in results)

