"""CLI/import contract tests (no live AWS)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.regression
def test_run_sagemaker_training_imports_ops_launcher():
    import scripts.deployment.run_sagemaker_training as mod

    source = inspect.getsource(mod)
    assert "from enhanced_system.ops import MangoMASSageMakerLauncher" in source
    assert "from scripts.deployment.launch_all_agents_sagemaker" not in source
    assert "sys.path.append" not in source


@pytest.mark.regression
def test_distillation_job_uses_settings_and_safe_extract():
    source = (REPO / "scripts" / "deployment" / "sagemaker_distillation_job.py").read_text(
        encoding="utf-8"
    )
    assert "get_settings" in source
    assert "safe_extract_tar" in source
    assert 'region: str = "us-east-1"' not in source


@pytest.mark.regression
def test_generate_deployment_summary_region_default_from_settings():
    source = (REPO / "scripts" / "generate_deployment_summary.py").read_text(encoding="utf-8")
    assert "get_settings().aws_region" in source
    assert 'default="us-east-1"' not in source


@pytest.mark.regression
def test_train_cli_uses_skill_training_config():
    source = (REPO / "scripts" / "training" / "train_agent_skill.py").read_text(encoding="utf-8")
    assert "SkillTrainingConfig" in source
    assert "AgentTrainingConfig(" not in source
