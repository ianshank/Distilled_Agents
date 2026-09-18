"""Tests for Phase 1 distillation expansion: LoRA auto-detect, KL fixes, settings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from enhanced_system.ops.settings import MangoMASSettings

# ---------------------------------------------------------------------------
# LoRA Target Module Auto-Detection
# ---------------------------------------------------------------------------


class TestResolveTargetModulesForModel:
    """Tests for architecture-aware LoRA module detection."""

    def _import_resolve(self):
        from scripts.training.distill.trainer import resolve_target_modules_for_model

        return resolve_target_modules_for_model

    def test_gpt2_architecture(self):
        """GPT-2 / DialoGPT should return c_attn, c_proj, c_fc."""
        resolve = self._import_resolve()
        config = MagicMock()
        config.model_type = "gpt2"
        with patch("transformers.AutoConfig") as mock_ac:
            mock_ac.from_pretrained.return_value = config
            modules = resolve("microsoft/DialoGPT-medium")
        assert "c_attn" in modules
        assert "c_proj" in modules
        assert "c_fc" in modules
        assert "q_proj" not in modules

    def test_llama_architecture(self):
        """LLaMA family should return q/k/v/o_proj + gate/up/down_proj."""
        resolve = self._import_resolve()
        config = MagicMock()
        config.model_type = "llama"
        with patch("transformers.AutoConfig") as mock_ac:
            mock_ac.from_pretrained.return_value = config
            modules = resolve("meta-llama/Llama-3-8B")
        expected = {"q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert set(modules) == expected

    def test_mistral_architecture(self):
        """Mistral should return LLaMA-style modules."""
        resolve = self._import_resolve()
        config = MagicMock()
        config.model_type = "mistral"
        with patch("transformers.AutoConfig") as mock_ac:
            mock_ac.from_pretrained.return_value = config
            modules = resolve("mistralai/Mistral-7B-v0.1")
        assert "q_proj" in modules
        assert "gate_proj" in modules

    def test_qwen2_architecture(self):
        """Qwen2 should return LLaMA-style modules."""
        resolve = self._import_resolve()
        config = MagicMock()
        config.model_type = "qwen2"
        with patch("transformers.AutoConfig") as mock_ac:
            mock_ac.from_pretrained.return_value = config
            modules = resolve("Qwen/Qwen2.5-0.5B")
        assert "q_proj" in modules

    def test_phi3_architecture(self):
        """Phi-3 uses fused qkv_proj, different from LLaMA."""
        resolve = self._import_resolve()
        config = MagicMock()
        config.model_type = "phi3"
        with patch("transformers.AutoConfig") as mock_ac:
            mock_ac.from_pretrained.return_value = config
            modules = resolve("microsoft/Phi-3-mini-4k-instruct")
        assert "qkv_proj" in modules
        assert "q_proj" not in modules

    def test_unknown_architecture_falls_back(self):
        """Unknown model types should fall back to LLaMA-style defaults."""
        resolve = self._import_resolve()
        config = MagicMock()
        config.model_type = "totally_unknown_model"
        with patch("transformers.AutoConfig") as mock_ac:
            mock_ac.from_pretrained.return_value = config
            modules = resolve("some/unknown-model")
        assert "q_proj" in modules

    def test_config_load_failure_falls_back(self):
        """If AutoConfig.from_pretrained fails, should still return defaults."""
        resolve = self._import_resolve()
        with patch("transformers.AutoConfig") as mock_ac:
            mock_ac.from_pretrained.side_effect = OSError("Model not found")
            modules = resolve("nonexistent/model")
        assert "q_proj" in modules
        assert len(modules) == 7

    def test_returns_new_list_each_call(self):
        """Ensure returned lists are independent copies (no shared mutable state)."""
        resolve = self._import_resolve()
        config = MagicMock()
        config.model_type = "gpt2"
        with patch("transformers.AutoConfig") as mock_ac:
            mock_ac.from_pretrained.return_value = config
            modules1 = resolve("test/model")
            modules2 = resolve("test/model")
        assert modules1 is not modules2
        assert modules1 == modules2


# ---------------------------------------------------------------------------
# Vocab-Aligned KL Distillation Loss
# ---------------------------------------------------------------------------


class TestDistillationLoss:
    """Tests for the fixed KL distillation loss with vocab alignment."""

    def _make_trainer(self, alpha=0.5, temperature=2.0):
        from scripts.training.distill.trainer import DistillationTrainer

        # Minimal mock to instantiate without full HF Trainer init
        with patch.object(DistillationTrainer, "__init__", lambda self, *a, **kw: None):
            trainer = DistillationTrainer.__new__(DistillationTrainer)
        trainer.teacher_model = MagicMock() if alpha > 0 else None
        trainer.distillation_alpha = alpha
        trainer.temperature = temperature
        return trainer

    def test_task_loss_only_when_alpha_zero(self):
        """When alpha=0, should return pure cross-entropy (no KL)."""
        trainer = self._make_trainer(alpha=0.0)
        student_out = MagicMock()
        student_out.logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        loss = trainer.create_distillation_loss(student_out, None, labels)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_vocab_mismatch_handled(self):
        """Teacher vocab (32K) vs student vocab (50K) should not crash."""
        trainer = self._make_trainer(alpha=0.5)
        student_out = MagicMock()
        student_out.logits = torch.randn(1, 5, 50000)  # student: 50K vocab
        teacher_out = MagicMock()
        teacher_out.logits = torch.randn(1, 5, 32000)  # teacher: 32K vocab
        labels = torch.randint(0, 100, (1, 5))
        loss = trainer.create_distillation_loss(student_out, teacher_out, labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_label_mask_applied_to_kl(self):
        """KL loss should only be computed on supervised positions (labels != -100)."""
        trainer = self._make_trainer(alpha=0.5)
        # 1 batch, 6 tokens, 100 vocab
        student_out = MagicMock()
        student_out.logits = torch.randn(1, 6, 100)
        teacher_out = MagicMock()
        teacher_out.logits = torch.randn(1, 6, 100)
        # Only token positions 2,3 are supervised; rest are masked
        labels = torch.full((1, 6), -100, dtype=torch.long)
        labels[0, 2] = 5
        labels[0, 3] = 10
        loss = trainer.create_distillation_loss(student_out, teacher_out, labels)
        assert torch.isfinite(loss)

    def test_all_labels_masked(self):
        """When all labels are -100, KL should still compute without error."""
        trainer = self._make_trainer(alpha=0.5)
        student_out = MagicMock()
        student_out.logits = torch.randn(1, 4, 100)
        teacher_out = MagicMock()
        teacher_out.logits = torch.randn(1, 4, 100)
        labels = torch.full((1, 4), -100, dtype=torch.long)
        loss = trainer.create_distillation_loss(student_out, teacher_out, labels)
        assert torch.isfinite(loss)

    def test_equal_vocab_no_truncation(self):
        """When vocabs match, full vocab should be used for KL."""
        trainer = self._make_trainer(alpha=0.5)
        student_out = MagicMock()
        student_out.logits = torch.randn(1, 5, 32000)
        teacher_out = MagicMock()
        teacher_out.logits = torch.randn(1, 5, 32000)
        labels = torch.randint(0, 100, (1, 5))
        loss = trainer.create_distillation_loss(student_out, teacher_out, labels)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Settings Expansion
# ---------------------------------------------------------------------------


class TestDistillationSettings:
    """Tests for new distillation expansion settings."""

    def test_default_lora_rank(self):
        settings = MangoMASSettings()
        assert settings.lora_rank == 16

    def test_default_lora_alpha(self):
        settings = MangoMASSettings()
        assert settings.lora_alpha == 32

    def test_default_alpha_ratio(self):
        """Industry standard: alpha = 2 * rank."""
        settings = MangoMASSettings()
        assert settings.lora_alpha == 2 * settings.lora_rank

    def test_default_use_dora_false(self):
        settings = MangoMASSettings()
        assert settings.use_dora is False

    def test_default_quantize_4bit_false(self):
        settings = MangoMASSettings()
        assert settings.quantize_4bit is False

    def test_default_distill_temperature(self):
        settings = MangoMASSettings()
        assert settings.distill_temperature == 2.0

    def test_default_dpo_beta(self):
        settings = MangoMASSettings()
        assert settings.dpo_beta == 0.1

    def test_default_lora_target_modules_empty(self):
        """Empty string means auto-detection."""
        settings = MangoMASSettings()
        assert settings.lora_target_modules == ""

    @patch.dict("os.environ", {"MANGOMAS_LORA_RANK": "32"})
    def test_env_override_lora_rank(self):
        """Settings should be overridable via MANGOMAS_ env vars."""
        settings = MangoMASSettings()
        assert settings.lora_rank == 32

    @patch.dict("os.environ", {"MANGOMAS_USE_DORA": "true"})
    def test_env_override_use_dora(self):
        settings = MangoMASSettings()
        assert settings.use_dora is True

    @patch.dict("os.environ", {"MANGOMAS_DPO_BETA": "0.05"})
    def test_env_override_dpo_beta(self):
        settings = MangoMASSettings()
        assert settings.dpo_beta == 0.05


# ---------------------------------------------------------------------------
# SageMaker Job Spec Forwarding
# ---------------------------------------------------------------------------


class TestSageMakerJobSpec:
    """Tests that create_job_spec forwards new distillation args."""

    def test_job_spec_includes_lora_args(self):
        from enhanced_system.ops.sagemaker_launcher import MangoMASSageMakerLauncher

        with patch("enhanced_system.ops.sagemaker_launcher.get_settings") as mock_gs:
            settings = MangoMASSettings()
            mock_gs.return_value = settings
            with patch.object(MangoMASSageMakerLauncher, "__init__", lambda self, *a, **kw: None):
                launcher = MangoMASSageMakerLauncher.__new__(MangoMASSageMakerLauncher)
            launcher.settings = settings
            launcher.instance_type = "ml.g4dn.xlarge"
            launcher._get_execution_role = MagicMock(return_value="arn:aws:iam::role/test")
            launcher._get_s3_bucket = MagicMock(return_value="test-bucket")

            config = MagicMock()
            config.agent_name = "test-agent"
            config.training_file = "train.jsonl"
            config.instance_type = None
            config.model_name = None
            config.student_model = None
            config.epochs = 3
            config.batch_size = 16
            config.learning_rate = 1e-4
            config.trajectory_mode = True

            spec = launcher.create_job_spec(config)

        hp = spec["hyperparameters"]
        assert "lora_r" in hp
        assert hp["lora_r"] == "16"
        assert "lora_alpha" in hp
        assert hp["lora_alpha"] == "32"
        assert "use_dora" in hp
        assert hp["use_dora"] == "false"
        assert "trajectory_mode" in hp
        assert hp["trajectory_mode"] == "True"

    def test_job_spec_omits_empty_target_modules(self):
        """When lora_target_modules is empty, it should not appear in hyperparams."""
        from enhanced_system.ops.sagemaker_launcher import MangoMASSageMakerLauncher

        with patch("enhanced_system.ops.sagemaker_launcher.get_settings") as mock_gs:
            settings = MangoMASSettings()
            mock_gs.return_value = settings
            with patch.object(MangoMASSageMakerLauncher, "__init__", lambda self, *a, **kw: None):
                launcher = MangoMASSageMakerLauncher.__new__(MangoMASSageMakerLauncher)
            launcher.settings = settings
            launcher.instance_type = "ml.g4dn.xlarge"
            launcher._get_execution_role = MagicMock(return_value="arn:aws:iam::role/test")
            launcher._get_s3_bucket = MagicMock(return_value="test-bucket")

            config = MagicMock()
            config.agent_name = "test-agent"
            config.training_file = "train.jsonl"
            config.instance_type = None
            config.model_name = None
            config.student_model = None
            config.epochs = 3
            config.batch_size = 16
            config.learning_rate = 1e-4

            spec = launcher.create_job_spec(config)

        assert "lora_target_modules" not in spec["hyperparameters"]


# ---------------------------------------------------------------------------
# LoRA Module Map Completeness
# ---------------------------------------------------------------------------


class TestLoraModuleMap:
    """Verify the module map covers all expected architectures."""

    def test_module_map_has_gpt2(self):
        from scripts.training.distill.trainer import _LORA_MODULE_MAP

        assert "gpt2" in _LORA_MODULE_MAP

    def test_module_map_has_llama(self):
        from scripts.training.distill.trainer import _LORA_MODULE_MAP

        assert "llama" in _LORA_MODULE_MAP

    def test_module_map_has_mistral(self):
        from scripts.training.distill.trainer import _LORA_MODULE_MAP

        assert "mistral" in _LORA_MODULE_MAP

    def test_module_map_has_qwen2(self):
        from scripts.training.distill.trainer import _LORA_MODULE_MAP

        assert "qwen2" in _LORA_MODULE_MAP

    def test_module_map_has_gemma(self):
        from scripts.training.distill.trainer import _LORA_MODULE_MAP

        assert "gemma" in _LORA_MODULE_MAP

    def test_all_values_are_nonempty_lists(self):
        from scripts.training.distill.trainer import _LORA_MODULE_MAP

        for arch, modules in _LORA_MODULE_MAP.items():
            assert isinstance(modules, list), f"{arch} modules is not a list"
            assert len(modules) > 0, f"{arch} has empty module list"

    def test_default_modules_is_llama_style(self):
        from scripts.training.distill.trainer import _DEFAULT_MODULES

        assert "q_proj" in _DEFAULT_MODULES
        assert len(_DEFAULT_MODULES) == 7
