"""Unit tests for factories, builders, cache serialize, and ops settings."""

from __future__ import annotations

import pytest
from enhanced_system.config.builders import ConfigBuilder
from enhanced_system.core.cache.serialize import dumps, loads
from enhanced_system.core.factories import CacheManagerFactory, ValidatorFactory
from enhanced_system.ops.sagemaker_launcher import MangoMASSageMakerLauncher
from enhanced_system.ops.settings import MangoMASSettings


@pytest.mark.unit
def test_json_serializer_roundtrip():
    payload = {"a": 1, "nested": ["x", "y"]}
    assert loads(dumps(payload)) == payload


@pytest.mark.unit
def test_json_serializer_rejects_unknown_types():
    with pytest.raises(TypeError):
        dumps(object())
    assert loads(None) is None


@pytest.mark.unit
def test_config_builder_returns_pydantic_config():
    config = (
        ConfigBuilder()
        .with_caching(l2_enabled=False, l3_enabled=False)
        .with_validation(max_length=128)
        .with_monitoring(enabled=False)
        .build()
    )
    assert config.input_validation.max_length == 128
    assert config.monitoring.enabled is False


@pytest.mark.unit
def test_validator_factory():
    validator = ValidatorFactory.create(
        {"max_length": 50, "enable_pii_detection": False, "enable_injection_detection": True}
    )
    result = validator.validate("Write a function")
    assert result.is_valid


@pytest.mark.unit
def test_cache_factory_l1_only():
    cache = CacheManagerFactory.create(
        {
            "l1": {"max_size": 10, "ttl": 30},
            "l2": {"enabled": False},
            "l3": {"enabled": False},
            "semantic_similarity": {"enabled": False},
        }
    )
    cache.l1_cache.set("k", {"v": 1})
    assert cache.l1_cache.get("k") == {"v": 1}


@pytest.mark.unit
def test_settings_role_arn():
    settings = MangoMASSettings(aws_region="us-west-2")
    assert "us-west-2" == settings.aws_region
    assert settings.role_arn("123").endswith(settings.execution_role_name)


@pytest.mark.unit
def test_launcher_validate_missing_files(tmp_path):
    launcher = MangoMASSageMakerLauncher(data_dir=tmp_path)
    assert launcher.validate_training_data() is False
    spec = launcher.create_job_spec(launcher.agent_configs[0])
    assert "hyperparameters" in spec
    assert spec["hyperparameters"]["trust_remote_code"] == "false"
    source = launcher.training_source_dir()
    assert (source / "train_distilled_adapter.py").exists()
    assert (source / "distill" / "trainer.py").exists()


@pytest.mark.unit
def test_load_config_default():
    from enhanced_system.config import load_config

    config = load_config("default")
    assert config.input_validation.max_length >= 1
    assert "agent_profiles" in config.routing.agent_profiles_path


@pytest.mark.unit
def test_distill_io_helpers(tmp_path):
    from scripts.training.distill.sagemaker_io import resolve_train_file, texts_from_examples

    sample = tmp_path / "sample.jsonl"
    sample.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    assert resolve_train_file(str(tmp_path)).endswith("sample.jsonl")
    assert texts_from_examples({"prompt": ["a"]}) == ["a"]
