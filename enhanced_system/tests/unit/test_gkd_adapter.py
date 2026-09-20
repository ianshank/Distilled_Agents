"""Unit tests for enhanced_system.training.gkd_adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from enhanced_system.ops.settings import MangoMASSettings
from enhanced_system.training.gkd_adapter import (
    EXPECTED_TRL_VERSION,
    GKDTrainingConfig,
    compute_label_masked_gkd_loss,
    create_agent_gkd_trainer,
    get_trl_version,
    is_trl_available,
    validate_vocab_alignment,
    verify_trl_version_pin,
)

pytestmark = [pytest.mark.unit]


def test_is_trl_available_and_version():
    with patch("builtins.__import__", side_effect=ImportError("No module named 'trl'")):
        assert is_trl_available() is False
        assert get_trl_version() is None


def test_verify_trl_version_pin_matching():
    with patch("enhanced_system.training.gkd_adapter.get_trl_version", return_value="0.15.2"):
        assert verify_trl_version_pin(strict=True) == "0.15.2"
        assert verify_trl_version_pin(strict=False) == "0.15.2"


def test_verify_trl_version_pin_mismatch_warning():
    with patch("enhanced_system.training.gkd_adapter.get_trl_version", return_value="0.15.1"):
        # non-strict logs warning but returns version
        assert verify_trl_version_pin(strict=False) == "0.15.1"
        # strict raises ValueError
        with pytest.raises(ValueError, match="differs from pinned"):
            verify_trl_version_pin(strict=True)


def test_verify_trl_version_pin_missing():
    with patch("enhanced_system.training.gkd_adapter.get_trl_version", return_value=None):
        with pytest.raises(ImportError, match="TRL is required"):
            verify_trl_version_pin(strict=False)


def test_validate_vocab_alignment_success():
    assert validate_vocab_alignment(100, 100, strict=True) is True
    assert validate_vocab_alignment(100, 100, strict=False) is True


def test_validate_vocab_alignment_strict_raises():
    with pytest.raises(ValueError, match="Incompatible teacher/student vocabularies"):
        validate_vocab_alignment(100, 120, strict=True)


def test_validate_vocab_alignment_non_strict_warns():
    with pytest.warns(RuntimeWarning, match="Falling back to task loss"):
        res = validate_vocab_alignment(100, 120, strict=False)
    assert res is False


def test_gkd_training_config_from_settings():
    settings = MangoMASSettings()
    cfg = GKDTrainingConfig.from_settings(settings=settings)
    assert cfg.enabled is False
    assert cfg.distillation_alpha == 0.0
    assert cfg.lmbda == 0.5
    assert cfg.beta == 0.5
    assert cfg.temperature == 0.9
    assert cfg.max_new_tokens == 128
    assert cfg.seq_kd is False
    assert cfg.expected_trl_version == EXPECTED_TRL_VERSION

    # Explicit overrides
    cfg2 = GKDTrainingConfig.from_settings(settings=settings, distillation_alpha=0.4, enabled=True)
    assert cfg2.enabled is True
    assert cfg2.distillation_alpha == 0.4


def test_create_agent_gkd_trainer_guards_missing_trl():
    with patch("enhanced_system.training.gkd_adapter.is_trl_available", return_value=False):
        with pytest.raises(ImportError, match="TRL is required to create AgentGKDTrainer"):
            create_agent_gkd_trainer(
                student_model=MagicMock(),
                teacher_model=MagicMock(),
                training_args=MagicMock(),
                train_dataset=MagicMock(),
                tokenizer=MagicMock(),
                data_collator=MagicMock(),
                gkd_config=GKDTrainingConfig(enabled=True),
            )


def test_create_agent_gkd_trainer_success_when_trl_present():
    with patch("enhanced_system.training.gkd_adapter.is_trl_available", return_value=True):
        mock_trainer = MagicMock()
        with patch(
            "scripts.training.distill.gkd_adapter.AgentGKDTrainer", return_value=mock_trainer
        ):
            trainer = create_agent_gkd_trainer(
                student_model=MagicMock(),
                teacher_model=MagicMock(),
                training_args=MagicMock(),
                train_dataset=MagicMock(),
                tokenizer=MagicMock(),
                data_collator=MagicMock(),
                gkd_config=GKDTrainingConfig(enabled=True),
            )
            assert trainer is mock_trainer


def test_compute_label_masked_gkd_loss_torch():
    torch = pytest.importorskip("torch")

    # Mismatched vocabs
    s_logits = torch.randn(2, 4, 30)
    t_logits = torch.randn(2, 4, 20)
    labels = torch.randint(0, 30, (2, 4))
    with pytest.warns(RuntimeWarning, match="Vocab mismatch"):
        loss = compute_label_masked_gkd_loss(s_logits, t_logits, labels)
    assert torch.isfinite(loss)

    # Matched vocabs
    s_logits2 = torch.randn(2, 4, 25)
    t_logits2 = torch.randn(2, 4, 25)
    labels2 = torch.tensor([[-100, 5, -100, 10], [2, -100, -100, -100]])
    loss2 = compute_label_masked_gkd_loss(s_logits2, t_logits2, labels2, beta=0.5)
    assert torch.isfinite(loss2)
    assert loss2.item() >= 0.0

    # Forward KL (beta=0.0)
    loss_fwd = compute_label_masked_gkd_loss(s_logits2, t_logits2, labels2, beta=0.0)
    assert torch.isfinite(loss_fwd)

    # Reverse KL (beta=1.0)
    loss_rev = compute_label_masked_gkd_loss(s_logits2, t_logits2, labels2, beta=1.0)
    assert torch.isfinite(loss_rev)

    # All masked labels
    labels_all_masked = torch.full((2, 4), -100, dtype=torch.long)
    loss_zero = compute_label_masked_gkd_loss(s_logits2, t_logits2, labels_all_masked)
    assert loss_zero.item() == 0.0
