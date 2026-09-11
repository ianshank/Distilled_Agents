"""Integration: validator -> runtime -> convert (no torch required)."""

from __future__ import annotations

import pytest
from enhanced_system.core.input_validator import InputValidator
from enhanced_system.harness.convert import trajectory_to_legacy
from enhanced_system.harness.factory import HarnessFactory
from scripts.training.distill.sagemaker_io import texts_from_examples


@pytest.mark.integration
@pytest.mark.harness
def test_validator_runtime_convert_pipeline():
    validator = InputValidator(
        {"max_length": 4096, "enable_pii_detection": False, "enable_injection_detection": True}
    )
    task = "Write a short greeting for the user"
    assert validator.validate_task_input(task).is_valid
    runtime = HarnessFactory.create(
        {
            "harness_id": "base_react",
            "scripted": ['{"tool": "final_answer", "args": {"text": "hello user"}}'],
        }
    )
    result = runtime.run(task, harness_id="base_react")
    row = trajectory_to_legacy(result.trajectory)
    text = texts_from_examples(row)
    assert row["prompt"] == result.trajectory.task
    assert "hello user" in result.final_answer
    assert row["prompt"] in text


@pytest.mark.integration
@pytest.mark.harness
def test_tool_observation_quotes_do_not_abort():
    from enhanced_system.harness.backends.echo import EchoBackend
    from enhanced_system.harness.registry import load_spec
    from enhanced_system.harness.runtime import AgentRuntime

    spec = load_spec("swe_codeact")
    spec.policy.teacher = False
    spec.planning.first_thought_prefix = False
    runtime = AgentRuntime(
        EchoBackend(
            [
                '{"tool": "json_schema", "args": {"required": ["a"], "document": {"a": 1}}}',
                '{"tool": "final_answer", "args": {"text": "ok"}}',
            ]
        ),
        spec=spec,
    )
    result = runtime.run("Write a short greeting")
    assert result.final_answer == "ok"
    assert "{" in result.trajectory.steps[0].observation
