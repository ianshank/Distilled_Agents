"""Distillation helper tests against real APIs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from enhanced_system.ops.settings import get_settings
from scripts.training.distill.model_load import resolve_model_revision, resolve_trust_remote_code
from scripts.training.distill.sagemaker_io import resolve_train_file, texts_from_examples


@pytest.mark.unit
def test_resolve_model_revision_none_without_env(monkeypatch):
    monkeypatch.delenv("MANGOMAS_MODEL_REVISION", raising=False)
    get_settings.cache_clear()
    args = SimpleNamespace(model_revision=None)
    assert resolve_model_revision(args) is None


@pytest.mark.unit
def test_resolve_model_revision_from_env(monkeypatch):
    monkeypatch.setenv("MANGOMAS_MODEL_REVISION", "rev-abc")
    get_settings.cache_clear()
    args = SimpleNamespace()
    assert resolve_model_revision(args) == "rev-abc"


@pytest.mark.unit
def test_resolve_model_revision_cli_wins(monkeypatch):
    monkeypatch.setenv("MANGOMAS_MODEL_REVISION", "from-env")
    get_settings.cache_clear()
    args = SimpleNamespace(model_revision="from-cli")
    assert resolve_model_revision(args) == "from-cli"


@pytest.mark.unit
def test_resolve_trust_remote_code_defaults_false(monkeypatch):
    monkeypatch.delenv("MANGOMAS_TRUST_REMOTE_CODE", raising=False)
    get_settings.cache_clear()
    args = SimpleNamespace(trust_remote_code=False)
    assert resolve_trust_remote_code(args) is False


@pytest.mark.unit
def test_resolve_train_file_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .jsonl files"):
        resolve_train_file(str(tmp_path))


@pytest.mark.unit
def test_resolve_train_file_training_channel(tmp_path, monkeypatch):
    channel = tmp_path / "training"
    channel.mkdir()
    target = channel / "job.jsonl"
    target.write_text('{"prompt": "x", "completion": "y"}\n', encoding="utf-8")
    monkeypatch.setenv("SM_CHANNEL_TRAINING", str(channel))
    monkeypatch.delenv("SM_CHANNEL_TRAIN", raising=False)
    assert resolve_train_file() == str(target)
    named = channel / "named.jsonl"
    named.write_text("{}\n", encoding="utf-8")
    assert resolve_train_file(train_file="named.jsonl").endswith("named.jsonl")


@pytest.mark.unit
def test_texts_from_examples_fallbacks():
    assert texts_from_examples({"prompt": ["a"]}) == ["a"]
    assert texts_from_examples({"prompt": ["p"], "completion": ["c"]}) == ["p\nc"]
    assert texts_from_examples({"text": ["b"]}) == ["b"]
    assert texts_from_examples({"input": ["c"]}) == ["c"]
    assert texts_from_examples({"other": ["d"]}) == ["d"]
