"""Unit tests for evaluation modules."""

from __future__ import annotations

import pytest
from enhanced_system.evaluation.ab_testing import ABTestFramework
from enhanced_system.evaluation.skill_evaluator import SkillEvaluator


@pytest.mark.unit
def test_skill_evaluator_runs_suite():
    evaluator = SkillEvaluator({})

    def agent_func(prompt: str):
        return {"response": prompt, "confidence": 0.9, "success": True, "latency_ms": 5}

    result = evaluator.evaluate_agent(
        "swe_agent",
        [{"id": "t1", "prompt": "hello", "expected": "hello"}],
        agent_func,
    )
    assert result.agent_name == "swe_agent"
    assert 0 <= result.overall_score <= 1
    assert result.test_results


@pytest.mark.unit
def test_ab_framework_create_and_route():
    framework = ABTestFramework({"enabled": True, "default_traffic_split": 0.5})
    experiment_id = framework.create_experiment("exp", "control-model", "treat-model")
    assert experiment_id
    variant = framework.route_request(experiment_id, "user-1")
    assert variant in {"control", "treatment"}
