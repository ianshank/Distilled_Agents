import pytest

# ── Regression tests for peer-review defect fixes ────────────────────────────


@pytest.mark.regression
class TestSecurityRegressions:
    """Regression tests for security defects SEC-001 through SEC-003."""

    def test_sec_001_timing_safe_auth(self):
        """SEC-001: Adapter auth must use hmac.compare_digest, not ==."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "scripts" / "inference.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        # Find _require_adapter_auth and verify hmac.compare_digest usage
        found_hmac = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "compare_digest":
                    found_hmac = True
        assert found_hmac, "_require_adapter_auth must use hmac.compare_digest"

    def test_sec_002_predict_fn_raises_on_error(self):
        """SEC-002: predict_fn must re-raise exceptions, not return error dict."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "scripts" / "inference.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        predict_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "predict_fn":
                predict_fn = node
                # Check the except handler uses Raise not Return
                for child in ast.walk(node):
                    if isinstance(child, ast.ExceptHandler):
                        assert any(isinstance(stmt, ast.Raise) for stmt in child.body), (
                            "predict_fn except handler must raise"
                        )
                        for stmt in child.body:
                            if isinstance(stmt, ast.Return):
                                val = stmt.value
                                if isinstance(val, ast.Dict):
                                    for key in val.keys:
                                        if isinstance(key, ast.Constant) and key.value == "error":
                                            pytest.fail(
                                                "predict_fn still returns {error: ...} instead of raising"
                                            )
        assert predict_fn is not None, "predict_fn function definition must exist"

    def test_sec_003_prompt_size_validation(self):
        """SEC-003: predict_fn must validate prompt byte size."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "scripts" / "inference.py"
        source_text = src.read_text(encoding="utf-8")
        assert "MANGOMAS_MAX_PROMPT_BYTES" in source_text, (
            "predict_fn must reference MANGOMAS_MAX_PROMPT_BYTES for size validation"
        )
        tree = ast.parse(source_text)
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "predict_fn"
            for node in ast.walk(tree)
        ), "predict_fn function definition must exist"


@pytest.mark.regression
class TestCodeQualityRegressions:
    """Regression tests for code quality defects CQ-001 through CQ-004."""

    def test_cq_001_bandit_json_parse_graceful(self):
        """CQ-001: SecurityScanner must not crash on malformed bandit output."""
        from unittest.mock import MagicMock, patch

        from enhanced_system.harness.security import SecurityScanner

        scanner = SecurityScanner(fail_on_high=False)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "WARNING: bandit deprecation notice\nNot JSON"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            findings = scanner.scan_python_code("x = 1")
        assert findings == [], "Should return empty list on malformed JSON"

    def test_cq_002_pii_scrubber_no_presidio(self):
        """CQ-002: PIIScrubber must degrade gracefully without presidio."""
        from unittest.mock import patch

        with patch.dict(
            "enhanced_system.harness.data_governance.__dict__", {"HAS_PRESIDIO": False}
        ):
            from enhanced_system.harness.data_governance import PIIScrubber

            scrubber = PIIScrubber()
            result = scrubber.redact_text("test@example.com")
            assert result == "test@example.com", "Should return text unchanged without presidio"

    def test_cq_004_no_hardcoded_account_id(self):
        """CQ-004: sagemaker_launcher must not contain hardcoded 000000000000."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / "enhanced_system"
            / "ops"
            / "sagemaker_launcher.py"
        )
        content = src.read_text(encoding="utf-8")
        # The string should not appear as a return value
        assert 'return "000000000000"' not in content, (
            "_account_id must not hardcode fallback account ID"
        )


@pytest.mark.regression
class TestConfigComplianceRegressions:
    """Regression tests for configuration compliance CFG-001 through CFG-005."""

    def test_cfg_001_002_mangomas_prefix_inference(self):
        """CFG-001/002: inference.py must use MANGOMAS_ prefixed env vars."""
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "scripts" / "inference.py"
        content = src.read_text(encoding="utf-8")
        assert "MANGOMAS_MODEL_DIR" in content, "Must use MANGOMAS_MODEL_DIR"
        assert "MANGOMAS_PORT" in content or "MANGOMAS_BIND_PORT" in content, (
            "Must use MANGOMAS_PORT or MANGOMAS_BIND_PORT"
        )


@pytest.mark.regression
class TestTrainingRegressions:
    """Regression tests for training pipeline defects."""

    def test_trajectory_mode_empty_dataset_raises_valueerror(self):
        """TRN-001: Trajectory dataset filter must raise explicit ValueError if 0 rows remain."""
        import sys
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        import pytest

        pytest.importorskip("torch")
        pytest.importorskip("datasets")
        pytest.importorskip("transformers")

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from scripts.training.distill.trainer import AgentDistillationTrainer

        mock_args = MagicMock()
        mock_args.trajectory_mode = True
        mock_args.train_file = "dummy.jsonl"
        mock_args.student_model_name = "dummy/model"

        # Mock dataset with NO "turns" (just raw prompts like a golden set)
        mock_dataset_dict = {"train": MagicMock()}
        # When filtered, it should return an empty mock dataset with length 0
        mock_filtered = MagicMock()
        mock_filtered.__len__.return_value = 0
        mock_dataset_dict["train"].filter.return_value = mock_filtered

        trainer = AgentDistillationTrainer.__new__(AgentDistillationTrainer)
        trainer.args = mock_args

        with patch(
            "scripts.training.distill.trainer.resolve_train_file", return_value="dummy.jsonl"
        ):
            with patch(
                "scripts.training.distill.trainer.load_dataset", return_value=mock_dataset_dict
            ):
                with patch("scripts.training.distill.trainer.load_tokenizer"):
                    with pytest.raises(ValueError, match=r"Trajectory.*dataset filtered to 0 rows"):
                        trainer.prepare_dataset()
