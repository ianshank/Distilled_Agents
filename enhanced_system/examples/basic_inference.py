"""
Basic Inference Example
Demonstrates using the enhanced agent system for inference
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhanced_system.config import load_config
from enhanced_system.core import (
    AgentMonitor,
    ConfidenceCalibrator,
    InputValidator,
    IntelligentCacheManager,
    IntelligentRetryHandler,
)


# Mock agent function for demonstration
async def mock_agent_inference(task: str) -> dict:
    """Mock agent inference function"""
    await asyncio.sleep(0.1)  # Simulate processing

    return {
        "response": f"Processed: {task[:50]}...",
        "confidence": 0.85,
        "token_count": 100,
        "success": True,
    }


async def main():
    """Main demonstration function"""
    print("=" * 60)
    print("Enhanced Agent System - Basic Inference Example")
    print("=" * 60)
    print()

    # Load configuration
    print("1. Loading configuration...")
    config = load_config("development")
    print("   ✓ Configuration loaded\n")

    # Initialize components
    print("2. Initializing components...")
    validator = InputValidator(config.input_validation.dict())
    cache_manager = IntelligentCacheManager(config.caching.dict())
    retry_handler = IntelligentRetryHandler(config.error_handling.dict())
    calibrator = ConfidenceCalibrator(config.confidence.dict())
    monitor = AgentMonitor(config.monitoring.dict())
    print("   ✓ All components initialized\n")

    # Example task
    task = "Write a Python function to calculate the Fibonacci sequence"

    # Step 1: Validate input
    print("3. Validating input...")
    validation_result = validator.validate_task_input(task)

    if not validation_result.is_valid:
        print(f"   ✗ Validation failed: {validation_result.error_message}")
        return

    print("   ✓ Input validated")
    if validation_result.warnings:
        for warning in validation_result.warnings:
            print(f"     ⚠ {warning}")
    print()

    # Step 2: Check cache
    print("4. Checking cache...")
    cache_key = cache_manager.compute_cache_key(task, "mock_agent")
    print(f"   Cache key: {cache_key[:16]}...")

    # Try to get from cache (will miss first time)
    cached_result = await cache_manager.l1_cache.get(cache_key)
    if cached_result:
        print("   ✓ Cache hit!")
        result = cached_result
    else:
        print("   ○ Cache miss, executing agent...")

        # Step 3: Execute with retry
        print("\n5. Executing agent with retry handler...")
        import time

        start_time = time.time()

        result = await retry_handler.execute_with_retry(
            mock_agent_inference, task, agent="mock_agent", task=task
        )

        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = latency_ms

        print(f"   ✓ Agent executed successfully ({latency_ms:.0f}ms)")

        # Cache the result
        cache_manager.l1_cache.set(cache_key, result)
        print("   ✓ Result cached\n")

    # Step 4: Calibrate confidence
    print("6. Calibrating confidence...")
    confidence_result = calibrator.calibrate_confidence(
        raw_confidence=result["confidence"], agent="mock_agent", task_type="coding", result=result
    )

    print(f"   Raw confidence: {result['confidence']:.2%}")
    print(f"   Calibrated confidence: {confidence_result.confidence:.2%}")
    print(f"   Reliability band: {confidence_result.reliability_band.value}")
    print(f"   Explanation: {confidence_result.explanation}\n")

    # Step 5: Track metrics
    print("7. Tracking metrics...")
    monitor.track_inference(agent="mock_agent", task=task, result=result)

    # Display stats
    stats = monitor.get_agent_stats("mock_agent")
    print(f"   Total requests: {stats.get('total_requests', 0)}")
    print(f"   Success rate: {stats.get('success_rate', 0):.1%}")
    print(f"   Avg latency: {stats.get('avg_latency_ms', 0):.0f}ms")
    print()

    # Display final result
    print("8. Final Result:")
    print("-" * 60)
    print(f"Response: {result['response']}")
    print(
        f"Confidence: {confidence_result.confidence:.2%} ({confidence_result.reliability_band.value})"
    )
    print(f"Tokens: {result['token_count']}")
    print(f"Latency: {result.get('latency_ms', 0):.0f}ms")
    print("-" * 60)
    print()

    # Display cache stats
    print("9. Cache Statistics:")
    cache_stats = cache_manager.get_stats()
    print(f"   L1 Cache: {cache_stats['l1']['hit_rate']:.1%} hit rate")
    print(f"   L1 Size: {cache_stats['l1']['size']}/{cache_stats['l1']['max_size']}")
    print()

    print("=" * 60)
    print("✓ Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
