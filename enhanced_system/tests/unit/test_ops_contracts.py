"""Regression tests for ops launcher, archive safety, and training-system contracts."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest
from enhanced_system.ops.archive import is_safe_tar_member, safe_extract_tar
from enhanced_system.ops.sagemaker_launcher import AgentTrainingConfig, MangoMASSageMakerLauncher
from enhanced_system.ops.training_system import (
    AgentTrainingConfig as DeprecatedSkillConfig,
)
from enhanced_system.ops.training_system import (
    AutomatedTrainingSystem,
    InfrastructureConfig,
    RegisteredSkill,
    SkillTrainingConfig,
)


@pytest.mark.unit
def test_validate_training_data_when_files_exist(tmp_path):
    launcher = MangoMASSageMakerLauncher(data_dir=tmp_path)
    for config in launcher.agent_configs:
        (tmp_path / config.training_file).write_text('{"prompt": "x"}\n', encoding="utf-8")
    assert launcher.validate_training_data() is True


@pytest.mark.unit
def test_training_source_dir_includes_distill():
    launcher = MangoMASSageMakerLauncher(data_dir=Path("."))
    source = launcher.training_source_dir()
    assert source.name == "training" or (source / "distill").exists()
    assert (source / "distill").is_dir()
    assert (source / "train_distilled_adapter.py").exists()
    assert (source / "requirements.txt").exists()


@pytest.mark.unit
def test_create_job_spec_disables_trust_remote_code(tmp_path):
    launcher = MangoMASSageMakerLauncher(data_dir=tmp_path)
    spec = launcher.create_job_spec(launcher.agent_configs[0])
    assert spec["hyperparameters"]["trust_remote_code"] == "false"
    assert spec["hyperparameters"]["teacher_model_name"]
    assert spec["hyperparameters"]["student_model_name"]
    assert "model_name_or_path" not in spec["hyperparameters"]


@pytest.mark.unit
def test_upload_training_data_failure_is_logged(tmp_path):
    launcher = MangoMASSageMakerLauncher(data_dir=tmp_path)
    assert launcher.upload_training_data_to_s3("missing-bucket") is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_launch_training_job_failure_shape(tmp_path):
    launcher = MangoMASSageMakerLauncher(data_dir=tmp_path)
    config = AgentTrainingConfig(agent_name="swe_agent", training_file="missing.jsonl")
    result = await launcher.launch_training_job(config)
    assert result["status"] == "failed"
    assert "error" in result
    assert result["agent_name"] == "swe_agent"


@pytest.mark.unit
def test_ops_exports_distinct_training_configs():
    from enhanced_system.ops import AgentTrainingConfig as LauncherConfig
    from enhanced_system.ops import SkillTrainingConfig as SkillConfig

    assert LauncherConfig is not SkillConfig
    launcher_cfg = LauncherConfig(agent_name="swe", training_file="swe.jsonl")
    skill_cfg = SkillConfig(role="swe", dataset_path="swe.jsonl", base_model="m")
    assert launcher_cfg.agent_name == "swe"
    assert skill_cfg.role == "swe"


@pytest.mark.unit
def test_deprecated_skill_alias_warns():
    with pytest.warns(DeprecationWarning, match="SkillTrainingConfig"):
        cfg = DeprecatedSkillConfig(role="swe", dataset_path="x.jsonl", base_model="m")
    assert isinstance(cfg, SkillTrainingConfig)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_launch_all_jobs_refuses_when_upload_fails(tmp_path, monkeypatch):
    launcher = MangoMASSageMakerLauncher(data_dir=tmp_path)
    for config in launcher.agent_configs:
        (tmp_path / config.training_file).write_text('{"prompt": "x"}\n', encoding="utf-8")
    monkeypatch.delenv("MANGOMAS_SKIP_UPLOAD", raising=False)
    monkeypatch.setattr(launcher, "upload_training_data_to_s3", lambda *_a, **_k: False)
    assert await launcher.launch_all_jobs() == []


@pytest.mark.unit
def test_create_job_spec_uses_cpu_models_when_set(tmp_path):
    launcher = MangoMASSageMakerLauncher(data_dir=tmp_path)
    spec = launcher.create_job_spec(
        AgentTrainingConfig(
            agent_name="swe_agent",
            training_file="swe.jsonl",
            model_name="distilgpt2",
            student_model="distilgpt2",
        )
    )
    assert spec["hyperparameters"]["teacher_model_name"] == "distilgpt2"
    assert spec["hyperparameters"]["student_model_name"] == "distilgpt2"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_automated_training_system_success_and_fail_shapes(tmp_path):
    system = AutomatedTrainingSystem(
        InfrastructureConfig(aws_region="us-west-2", s3_bucket="b", dynamodb_table="t")
    )
    system.aws_available = False
    ok = await system.train_agent_skill(
        SkillTrainingConfig(
            role="swe",
            dataset_path="data.jsonl",
            base_model="m",
            output_path="s3://mock/out",
            use_mock_training=True,
        )
    )
    assert ok["status"] == "success"
    assert ok["model_artifacts"] == "s3://mock/out"
    assert "training_time" in ok
    assert ok["mock"] is True

    missing = await system.train_agent_skill(
        SkillTrainingConfig(role="", dataset_path="", base_model="m", use_mock_training=True)
    )
    assert missing["status"] == "failed"
    assert "error" in missing

    setup = await system.setup_infrastructure()
    assert setup["status"] == "mock"
    assert setup["mode"] == "mock"
    assert setup["error"]

    missing_suite = await system.evaluate_agent_skill("swe", "missing-suite.json", 85)
    assert missing_suite.pass_rate == 0
    assert missing_suite.error

    registered = await system.register_skill(
        RegisteredSkill(role="swe", adapter_uri="s3://x", pass_rate=90, model="m")
    )
    assert registered["status"] == "failed"

    suite = tmp_path / "suite.json"
    suite.write_text(
        '{"tests": [{"passed": true}, {"passed": false}, {"expected": "ok", "actual": "ok"}]}',
        encoding="utf-8",
    )
    scored = await system.evaluate_agent_skill("swe", str(suite), 90)
    assert scored.total_tests == 3
    assert scored.passed_tests == 2
    assert scored.pass_rate == pytest.approx(200 / 3)
    assert scored.error

    onnx = await system.package_onnx("swe", "missing-adapter.bin", "out.onnx")
    assert onnx["status"] == "failed"

    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    aws_train = await system.train_agent_skill(
        SkillTrainingConfig(
            role="swe",
            dataset_path=str(dataset),
            base_model="m",
            use_mock_training=False,
        )
    )
    assert aws_train["status"] == "failed"


@pytest.mark.unit
def test_tar_member_traversal_is_blocked(tmp_path):
    dest = tmp_path / "extract"
    dest.mkdir()
    assert is_safe_tar_member(dest, "model.bin") is True
    assert is_safe_tar_member(dest, "../evil.bin") is False
    assert is_safe_tar_member(dest, "/etc/passwd") is False


@pytest.mark.unit
def test_safe_extract_tar_rejects_dotdot(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="../pwned.txt")
        info.size = 0
        tar.addfile(info)
    with pytest.raises(ValueError, match="path traversal"):
        safe_extract_tar(archive, dest)
    assert not (tmp_path / "pwned.txt").exists()
