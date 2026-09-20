"""Tests for Phase P4: trl-gkd-usage, TRL pin, label-masking, and vocab safety."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from enhanced_system.ops.settings import MangoMASSettings

pytestmark = [pytest.mark.unit, pytest.mark.regression]


# ---------------------------------------------------------------------------
# 1. Exact TRL Pin and Policy Tests
# ---------------------------------------------------------------------------


class TestTRLVersionPin:
    """Validate exact TRL version pin and version verification."""

    def test_pyproject_contains_exact_trl_pin(self):
        """pyproject.toml must declare exact pin trl==0.15.2, not only an open range."""
        with open("pyproject.toml", encoding="utf-8") as f:
            content = f.read()

        assert "trl==0.15.2" in content
        assert "trl>=0.14.0" not in content

    def test_expected_trl_version_constant(self):
        from enhanced_system.training.gkd_adapter import EXPECTED_TRL_VERSION

        assert EXPECTED_TRL_VERSION == "0.15.2"

    def test_verify_trl_version_success_when_matching(self):
        from enhanced_system.training.gkd_adapter import verify_trl_version_pin

        with patch("enhanced_system.training.gkd_adapter.get_trl_version", return_value="0.15.2"):
            assert verify_trl_version_pin(strict=True) == "0.15.2"

    def test_verify_trl_version_strict_raises_on_mismatch(self):
        from enhanced_system.training.gkd_adapter import verify_trl_version_pin

        with patch("enhanced_system.training.gkd_adapter.get_trl_version", return_value="0.14.0"):
            with pytest.raises(ValueError, match="differs from pinned '0.15.2'"):
                verify_trl_version_pin(strict=True)

    def test_verify_trl_version_raises_import_error_when_missing(self):
        from enhanced_system.training.gkd_adapter import verify_trl_version_pin

        with patch("enhanced_system.training.gkd_adapter.get_trl_version", return_value=None):
            with pytest.raises(ImportError, match="TRL is required for GKD training"):
                verify_trl_version_pin(strict=True)


# ---------------------------------------------------------------------------
# 2. Fail-Closed Defaults Tests
# ---------------------------------------------------------------------------


class TestFailClosedDefaults:
    """Verify that defaults strictly fail closed (alpha=0.0, gkd_enabled=False)."""

    def test_default_trajectory_distill_alpha_zero(self):
        settings = MangoMASSettings()
        assert settings.trajectory_distill_alpha == 0.0

    def test_default_gkd_enabled_false(self):
        settings = MangoMASSettings()
        assert settings.gkd_enabled is False

    def test_parse_trajectory_distill_alpha_default_zero(self):
        from scripts.training.distill.alpha import parse_trajectory_distill_alpha

        assert parse_trajectory_distill_alpha("0.0") == 0.0

    def test_gkd_config_defaults_from_settings(self):
        from enhanced_system.training.gkd_adapter import GKDTrainingConfig

        cfg = GKDTrainingConfig.from_settings()
        assert cfg.enabled is False
        assert cfg.distillation_alpha == 0.0
        assert cfg.lmbda == 0.5
        assert cfg.beta == 0.5
        assert cfg.temperature == 0.9
        assert cfg.max_new_tokens == 128
        assert cfg.seq_kd is False


# ---------------------------------------------------------------------------
# 3. Vocabulary Alignment and Refusal Tests
# ---------------------------------------------------------------------------


class TestVocabAlignment:
    """Verify vocabulary alignment checks and refusal of cross-vocab divergence."""

    def test_identical_vocab_passes(self):
        from enhanced_system.training.gkd_adapter import validate_vocab_alignment

        assert validate_vocab_alignment(32000, 32000, strict=True) is True

    def test_mismatched_vocab_raises_in_strict_mode(self):
        from enhanced_system.training.gkd_adapter import validate_vocab_alignment

        with pytest.raises(ValueError, match="Incompatible teacher/student vocabularies"):
            validate_vocab_alignment(50000, 32000, strict=True)

    def test_mismatched_vocab_warns_and_returns_false_in_non_strict(self):
        from enhanced_system.training.gkd_adapter import validate_vocab_alignment

        with pytest.warns(RuntimeWarning, match="Falling back to task loss"):
            result = validate_vocab_alignment(50000, 32000, strict=False)
        assert result is False


# ---------------------------------------------------------------------------
# 4. GKD Loss & Label Masking Tests (Requires torch)
# ---------------------------------------------------------------------------


class TestLabelMaskedGKDLoss:
    """Tests for compute_label_masked_gkd_loss and falsifier contracts."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_falsifier_mismatched_vocabs_refuse_cross_vocab_kl(self):
        """FALSIFIER: A test with mismatched vocabs and alpha>0 MUST NOT compute cross-vocab KL.

        Instead, it must warn and safely fall back to supervised task cross-entropy.
        """
        import torch
        from enhanced_system.training.gkd_adapter import compute_label_masked_gkd_loss

        # Student vocab = 50, Teacher vocab = 32
        student_logits = torch.randn(2, 6, 50, requires_grad=True)
        teacher_logits = torch.randn(2, 6, 32)
        labels = torch.randint(0, 50, (2, 6))

        with pytest.warns(
            RuntimeWarning, match="Vocab mismatch: Student\\(50\\) vs Teacher\\(32\\)"
        ):
            loss = compute_label_masked_gkd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                beta=0.5,
                temperature=1.0,
            )

        assert loss.dim() == 0
        assert torch.isfinite(loss)
        # Verify gradient flows from task loss without tensor shape error
        loss.backward()
        assert student_logits.grad is not None

    def test_label_mask_restricts_loss_to_supervised_positions(self):
        """Divergence must strictly be computed on tokens where labels != -100."""
        import torch
        from enhanced_system.training.gkd_adapter import compute_label_masked_gkd_loss

        student_logits = torch.randn(1, 4, 10)
        teacher_logits = torch.randn(1, 4, 10)

        # Positions 0, 1 are unsupervised (-100); positions 2, 3 are supervised
        labels1 = torch.tensor([[-100, -100, 3, 4]])
        loss1 = compute_label_masked_gkd_loss(student_logits, teacher_logits, labels1)

        # Perturbing teacher logits at unsupervised position 0 must NOT affect loss1
        teacher_perturbed = teacher_logits.clone()
        teacher_perturbed[0, 0, :] += 100.0
        loss2 = compute_label_masked_gkd_loss(student_logits, teacher_perturbed, labels1)

        assert torch.isclose(loss1, loss2, atol=1e-5)

    def test_all_labels_masked_returns_zero(self):
        """When all labels are -100, loss must be safely 0 without NaN."""
        import torch
        from enhanced_system.training.gkd_adapter import compute_label_masked_gkd_loss

        student_logits = torch.randn(2, 5, 20)
        teacher_logits = torch.randn(2, 5, 20)
        labels = torch.full((2, 5), -100, dtype=torch.long)

        loss = compute_label_masked_gkd_loss(student_logits, teacher_logits, labels)
        assert loss.item() == 0.0

    def test_beta_variations(self):
        """Test beta=0.0 (forward KL), beta=0.5 (JSD), and beta=1.0 (reverse KL)."""
        import torch
        from enhanced_system.training.gkd_adapter import compute_label_masked_gkd_loss

        student_logits = torch.randn(1, 5, 16)
        teacher_logits = torch.randn(1, 5, 16)
        labels = torch.randint(0, 16, (1, 5))

        for beta in [0.0, 0.5, 1.0]:
            loss = compute_label_masked_gkd_loss(
                student_logits, teacher_logits, labels, beta=beta, temperature=1.5
            )
            assert torch.isfinite(loss)
            assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 5. AgentGKDTrainer Loss Computation Tests
# ---------------------------------------------------------------------------


class TestAgentGKDTrainerComputeLoss:
    """Test AgentGKDTrainer compute_loss behavior across alpha values."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def _make_trainer(self, alpha=0.0, beta=0.5, has_teacher=True):
        from scripts.training.distill.gkd_adapter import AgentGKDTrainer

        trainer = AgentGKDTrainer.__new__(AgentGKDTrainer)
        trainer.distillation_alpha = alpha
        trainer.beta = beta
        trainer.temperature = 1.0
        trainer.lmbda = 0.5
        trainer.seq_kd = False
        trainer.teacher_model = MagicMock() if has_teacher else None
        return trainer

    def test_task_loss_only_when_alpha_zero(self):
        """When distillation_alpha=0, teacher forward pass is not called."""
        import torch

        trainer = self._make_trainer(alpha=0.0, has_teacher=True)
        model = MagicMock()
        model_out = MagicMock()
        model_out.logits = torch.randn(2, 8, 30)
        model.return_value = model_out

        inputs = {
            "input_ids": torch.randint(0, 30, (2, 8)),
            "attention_mask": torch.ones(2, 8),
            "labels": torch.randint(0, 30, (2, 8)),
        }

        loss = trainer.compute_loss(model, inputs)
        assert torch.isfinite(loss)
        # Teacher model must NOT be called when alpha=0.0
        trainer.teacher_model.assert_not_called()

    def test_blended_loss_when_alpha_positive_and_vocabs_match(self):
        """When distillation_alpha > 0 and vocabs match, blends task loss and GKD loss."""
        import torch

        trainer = self._make_trainer(alpha=0.4, has_teacher=True)
        model = MagicMock()
        student_out = MagicMock()
        student_out.logits = torch.randn(2, 6, 20)
        model.return_value = student_out

        teacher_out = MagicMock()
        teacher_out.logits = torch.randn(2, 6, 20)
        trainer.teacher_model.return_value = teacher_out

        inputs = {
            "input_ids": torch.randint(0, 20, (2, 6)),
            "attention_mask": torch.ones(2, 6),
            "labels": torch.randint(0, 20, (2, 6)),
        }

        loss = trainer.compute_loss(model, inputs)
        assert torch.isfinite(loss)
        trainer.teacher_model.assert_called_once()

    def test_vocab_mismatch_in_trainer_falls_back_safely(self):
        """When student and teacher logit vocabs mismatch in compute_loss, falls back to task loss."""
        import torch

        trainer = self._make_trainer(alpha=0.5, has_teacher=True)
        model = MagicMock()
        student_out = MagicMock()
        student_out.logits = torch.randn(1, 5, 50)  # student vocab = 50
        model.return_value = student_out

        teacher_out = MagicMock()
        teacher_out.logits = torch.randn(1, 5, 32)  # teacher vocab = 32
        trainer.teacher_model.return_value = teacher_out

        inputs = {
            "input_ids": torch.randint(0, 50, (1, 5)),
            "labels": torch.randint(0, 50, (1, 5)),
        }

        with pytest.warns(
            RuntimeWarning, match="Vocab mismatch: Student\\(50\\) vs Teacher\\(32\\)"
        ):
            loss = trainer.compute_loss(model, inputs)

        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 6. CLI Integration Tests
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Validate CLI arguments and options for GKD trajectory training."""

    def test_train_distilled_adapter_has_use_gkd(self):
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        script = repo / "scripts" / "training" / "train_distilled_adapter.py"
        source = script.read_text(encoding="utf-8")
        assert "--use_gkd" in source
        assert "--gkd_beta" in source
        assert "--gkd_lmbda" in source
        assert "verify_trl_version_pin" in source

    def test_train_gkd_adapter_cli_parse_args(self):
        from scripts.training.train_gkd_adapter import parse_args

        with patch(
            "sys.argv", ["train_gkd_adapter.py", "--epochs", "2", "--distillation_alpha", "0.3"]
        ):
            args = parse_args()
            assert args.epochs == 2
            assert args.distillation_alpha == 0.3
            assert args.output_dir == "./gkd_adapter"
