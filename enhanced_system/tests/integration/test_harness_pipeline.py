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
