"""Shared test fixtures for the Distilled_Agents test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def mock_llm_endpoint():
    """Mock LLM endpoint for tests that don't need real inference."""
    return "http://localhost:0/v1"


@pytest.fixture
def live_llm_endpoint():
    """Real LLM endpoint from env. Skip if not available."""
    url = os.getenv("MANGOMAS_LLM_ENDPOINT", "http://localhost:1234/v1")
    try:
        import requests

        resp = requests.get(f"{url}/models", timeout=2)
        if resp.status_code != 200:
            pytest.skip(f"LLM endpoint at {url} not responding")
    except Exception:
        pytest.skip(f"LLM endpoint at {url} not reachable")
    return url


@pytest.fixture
def gpu_available():
    """Skip test if no CUDA GPU detected."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")
        return torch.cuda.get_device_name(0)
    except ImportError:
        pytest.skip("torch not installed")


@pytest.fixture
def tmp_artifacts(tmp_path):
    """Temporary artifacts directory for test isolation."""
    arts = tmp_path / "artifacts"
    arts.mkdir()
    (arts / "stage.json").write_text('{"stage": "discover"}')
    (arts / "templates").mkdir()
    return arts


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def mock_settings(monkeypatch):
    """Monkeypatch MANGOMAS_ env vars with test defaults."""
    test_env = {
        "MANGOMAS_AWS_REGION": "us-east-1",
        "MANGOMAS_BIND_HOST": "127.0.0.1",
        "MANGOMAS_BIND_PORT": "8080",
        "MANGOMAS_TRUST_REMOTE_CODE": "false",
        "MANGOMAS_MAX_PROMPT_BYTES": "1048576",
    }
    for key, val in test_env.items():
        monkeypatch.setenv(key, val)
    # Clear cached settings
    from enhanced_system.ops.settings import get_settings

    get_settings.cache_clear()
    yield test_env
    get_settings.cache_clear()
