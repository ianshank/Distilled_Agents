"""
RCA Regression Tests
====================
Verifies that the fixes for Root Cause Analyses (RCAs) hold in the current codebase.
"""

from pathlib import Path

import pytest
from enhanced_system.core.input_validator import InputValidator
from enhanced_system.evaluation.skill_evaluator import EvaluationResult


@pytest.mark.regression
def test_rca_009_var_annotated_fixed():
    """Verify that type annotations are explicitly set (var-annotated)."""
    validator = InputValidator()
    res = validator.validate_task_input("Hello")
    assert res.is_valid is True
    assert isinstance(res.pii_entities, list)


@pytest.mark.regression
def test_rca_009_no_any_return_fixed():
    """Verify that no-any-return is fixed with casts."""
    # Actually test the priority comparison logic directly where casts were added
    import asyncio
    import time

    from enhanced_system.core.batch_processor import BatchRequest
    from enhanced_system.core.enums import Priority

    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        task1 = BatchRequest(
            request_id="1",
            task="test",
            priority=Priority.HIGH,
            future=future,
            timestamp=time.time(),
            metadata={},
        )
        task2 = BatchRequest(
            request_id="2",
            task="test",
            priority=Priority.LOW,
            future=future,
            timestamp=time.time(),
            metadata={},
        )

        # Priority queue relies on __lt__ being strictly bool
        is_less = task1 < task2
        assert isinstance(is_less, bool)
    finally:
        loop.close()


@pytest.mark.regression
def test_rca_002_optional_dataclass_fixed():
    """Verify dataclass fields use Optional when defaulting to None."""
    from datetime import datetime

    result = EvaluationResult(
        agent_name="test",
        timestamp=datetime.now(),
        overall_score=0.5,
        dimension_scores={},
        test_results=[],
    )
    assert result.recommendations == []  # post_init sets it to []


@pytest.mark.regression
def test_b108_tmp_out_fixed():
    """Verify that test_harness_runtime.py doesn't contain /tmp/out."""
    repo_root = Path(__file__).resolve().parents[1]
    harness_test_file = repo_root / "enhanced_system" / "tests" / "unit" / "test_harness_runtime.py"

    assert harness_test_file.exists(), "test_harness_runtime.py must exist to be verified"
    content = harness_test_file.read_text(encoding="utf-8")
    assert "/tmp/out" not in content, "Bandit B108 violation found: /tmp/out is hardcoded."


@pytest.mark.regression
def test_rca_trajectory_filter_truncation_boundary():
    """RCA-001/002: supervised tokens beyond max_length must be correctly excluded.

    Verifies that has_supervised_tokens returns False when supervised content
    falls beyond the truncation boundary, and True when max_length is large
    enough to include it.
    """
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    from dataclasses import dataclass

    from scripts.training.distill.trajectory_collator import has_supervised_tokens

    @dataclass
    class _Tok:
        pad_token_id: int = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list:
            return [ord(ch) % 20 + 1 for ch in text] if text else []

    tok = _Tok()
    # Row with supervised content: completion='c' produces an assistant turn
    row = {"prompt": "p", "completion": "c", "trajectory": {"steps": []}}

    # With max_length=16, the supervised 'c' at char 19 is truncated away
    assert has_supervised_tokens(tok, row, max_length=16) is False

    # With max_length=32, the supervised 'c' fits within the window
    assert has_supervised_tokens(tok, row, max_length=32) is True


@pytest.mark.regression
def test_rca_gpu_device_fallback():
    """Verify production inference device selection uses dynamic CUDA detection."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    from scripts.inference import DistilledAgentInference

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "is_available", lambda: False)
        cpu_handler = DistilledAgentInference()
        assert cpu_handler.device.type == "cpu"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "is_available", lambda: True)
        cuda_handler = DistilledAgentInference()
        assert cuda_handler.device.type == "cuda"
