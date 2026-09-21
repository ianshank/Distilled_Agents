"""Local E2E tests validating GPU inference and AgentRuntime loop."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def _require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available on this system")
    yield
    gc.collect()
    torch.cuda.empty_cache()


def test_gpu_device_and_cuda_presence():
    """Verify CUDA device presence, compute capability, and memory."""
    import torch

    assert torch.cuda.is_available() is True
    device_count = torch.cuda.device_count()
    assert device_count >= 1

    device_name = torch.cuda.get_device_name(0)
    assert len(device_name) > 0

    props = torch.cuda.get_device_properties(0)
    assert props.total_memory > 1024 * 1024 * 1024  # At least 1GB VRAM


def test_transformers_backend_gpu_generation():
    """Verify TransformersBackend loads model onto CUDA and generates tokens."""
    from enhanced_system.harness.backends.transformers import TransformersBackend

    backend = TransformersBackend(
        model_name="microsoft/DialoGPT-medium",
        trust_remote_code=False,
        max_input_length=256,
        max_new_tokens=32,
    )
    completions = backend.generate(
        [{"role": "user", "content": "What is 2+2?"}],
        n=1,
        temperature=0.0,
    )
    assert isinstance(completions, list)
    assert len(completions) == 1
    assert isinstance(completions[0], str)
    assert str(backend._device).startswith("cuda")


def test_agent_runtime_gpu_loop():
    """Verify AgentRuntime executes multi-turn tool/reasoning loop with live GPU backend."""
    from enhanced_system.harness.backends.transformers import TransformersBackend
    from enhanced_system.harness.factory import HarnessFactory

    backend = TransformersBackend(
        model_name="microsoft/DialoGPT-medium",
        trust_remote_code=False,
        max_input_length=256,
        max_new_tokens=32,
    )
    runtime = HarnessFactory.create(
        {
            "harness_id": "base_react",
            "backend": backend,
            "teacher": False,
            "strict_injection": False,
        }
    )
    result = runtime.run("What is the capital of France?", harness_id="base_react")
    assert result is not None
    assert result.trajectory is not None
    assert len(result.trajectory.steps) >= 1


def test_run_e2e_gpu_cli_invocation(tmp_path: Path):
    """Verify scripts/harness/run_e2e_gpu.py runs end-to-end via CLI main()."""
    from scripts.harness.run_e2e_gpu import main

    out_file = tmp_path / "e2e_test_output.json"
    exit_code = main(
        [
            "--model",
            "microsoft/DialoGPT-medium",
            "--harness-id",
            "base_react",
            "--output",
            str(out_file),
            "--require-gpu",
        ]
    )
    assert exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "llm_load_and_generate" in content
    assert "llm_harness_loop" in content
