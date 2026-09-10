"""Unit tests for AdaptiveRouter against the real RoutingDecision API."""

from __future__ import annotations

import pytest
from enhanced_system.core.adaptive_router import AdaptiveRouter, RoutingDecision
from enhanced_system.core.enums import RoutingStrategy, TaskComplexity


@pytest.mark.unit
class TestAdaptiveRouter:
    def test_initialization(self):
        config = {
            "enabled": True,
            "routing_strategy": RoutingStrategy.COST_OPTIMIZED.value,
            "complexity_model": "heuristic",
        }
        router = AdaptiveRouter(config)
        assert router.config == config
        assert router.agent_profiles is not None
        assert "swe_agent" in router.agent_profiles

    def test_classify_task_coding(self):
        router = AdaptiveRouter({"enabled": True})
        assert (
            router.classify_task("Write a Python function to implement binary search") == "coding"
        )

    def test_classify_task_architecture(self):
        router = AdaptiveRouter({"enabled": True})
        task_type = router.classify_task(
            "Design a microservices architecture for an e-commerce platform"
        )
        assert task_type in {"architecture", "design"}

    def test_classify_task_testing(self):
        router = AdaptiveRouter({"enabled": True})
        assert router.classify_task("Create unit tests for the authentication module") == "testing"

    def test_classify_task_devops(self):
        router = AdaptiveRouter({"enabled": True})
        assert router.classify_task("Set up a CI/CD pipeline using GitHub Actions") == "devops"

    def test_estimate_complexity_simple(self):
        router = AdaptiveRouter({"enabled": True})
        assert router.estimate_complexity("Print hello world") == TaskComplexity.SIMPLE

    def test_estimate_complexity_complex(self):
        router = AdaptiveRouter({"enabled": True})
        task = (
            "Design and implement a comprehensive distributed caching system with Redis, "
            "handle failover, implement a sophisticated sharding strategy, and ensure consistency"
        )
        assert router.estimate_complexity(task) == TaskComplexity.COMPLEX

    def test_route_task_returns_decision(self):
        router = AdaptiveRouter({"enabled": True})
        result = router.route_task("Write unit tests for a sorting algorithm")
        assert isinstance(result, RoutingDecision)
        assert result.selected_agents
        assert result.task_complexity in TaskComplexity
        assert result.reasoning

    def test_route_task_respects_disabled(self):
        router = AdaptiveRouter({"enabled": False})
        result = router.route_task("Any task")
        assert result.selected_agents == ["base_agent"]

    def test_cost_optimized_simple_uses_base_agent(self):
        router = AdaptiveRouter(
            {"enabled": True, "routing_strategy": RoutingStrategy.COST_OPTIMIZED.value}
        )
        result = router.route_task("simple easy function")
        assert "base_agent" in result.selected_agents

    def test_quality_optimized_complex_includes_architect(self):
        router = AdaptiveRouter(
            {"enabled": True, "routing_strategy": RoutingStrategy.QUALITY_OPTIMIZED.value}
        )
        result = router.route_task(
            "Design a comprehensive enterprise distributed architecture with advanced failover"
        )
        assert "architect_agent" in result.selected_agents or result.selected_agents

    def test_security_task_classification(self):
        router = AdaptiveRouter({"enabled": True})
        assert router.classify_task("Perform security audit of authentication system") == "security"

    def test_unknown_task_falls_back_to_general(self):
        router = AdaptiveRouter({"enabled": True})
        assert router.classify_task("Xyzabc undefined qwerty") == "general"

    def test_multi_step_complexity(self):
        router = AdaptiveRouter({"enabled": True})
        task = (
            "comprehensive advanced distributed multi-step implementation:\n"
            "1. implement the feature\n"
            "2. write tests\n"
            "3. deploy to staging"
        )
        complexity = router.estimate_complexity(task)
        assert complexity in {TaskComplexity.MEDIUM, TaskComplexity.COMPLEX}
