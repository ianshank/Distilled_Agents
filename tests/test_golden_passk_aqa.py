"""Unit and integration tests for multi-trial pass@k, golden sets, rule matrix, and AQA gate."""

from __future__ import annotations

import json

import pytest
import yaml
from enhanced_system.harness.backends.echo import EchoBackend
from enhanced_system.harness.types import GoldenRow
from scripts.harness.check_rule_matrix import verify_rule_matrix
from scripts.harness.eval_harness import _evaluate_file
from scripts.harness.run_aqa_gate import run_gate


@pytest.mark.unit
@pytest.mark.harness
def test_golden_row_schema_backwards_compatibility():
    """Existing golden rows with only prompt and expected load with correct defaults."""
    raw = {"prompt": "What is Python?", "expected": "A programming language"}
    row = GoldenRow.model_validate(raw)
    assert row.prompt == "What is Python?"
    assert row.expected == "A programming language"
    assert row.slice == "core"
    assert row.allow_semantic is False
    assert row.id is None
    assert row.harness_id is None
    assert row.expected_tools is None


@pytest.mark.unit
@pytest.mark.harness
def test_golden_row_empty_prompt_and_slice_normalize():
    """Empty prompts are rejected and slice casing is normalized."""
    with pytest.raises(ValueError, match="must not be empty"):
        GoldenRow.model_validate({"prompt": ""})

    with pytest.raises(ValueError, match="must not be empty"):
        GoldenRow.model_validate({"prompt": "   "})

    row_upper = GoldenRow.model_validate({"prompt": "Valid", "slice": "HARD"})
    assert row_upper.slice == "hard"

    row_core_upper = GoldenRow.model_validate({"prompt": "Valid", "slice": "CORE"})
    assert row_core_upper.slice == "core"

    row_none = GoldenRow.model_validate({"prompt": "Valid", "slice": None})
    assert row_none.slice == "core"


@pytest.mark.unit
@pytest.mark.harness
def test_golden_row_schema_hard_slice_validation():
    """Hard slice rows require id, non-empty expected, and slice=='hard'."""
    valid_hard = {
        "id": "hard-001",
        "prompt": "Fix deadlock",
        "expected": "transaction retried",
        "slice": "hard",
        "allow_semantic": False,
    }
    row = GoldenRow.model_validate(valid_hard)
    row.validate_for_hard_slice()

    # Missing id
    missing_id = {"prompt": "Fix deadlock", "expected": "ok", "slice": "hard"}
    row_no_id = GoldenRow.model_validate(missing_id)
    with pytest.raises(ValueError, match="missing stable id"):
        row_no_id.validate_for_hard_slice()

    # Missing expected
    missing_expected = {"id": "hard-002", "prompt": "Fix deadlock", "slice": "hard"}
    row_no_exp = GoldenRow.model_validate(missing_expected)
    with pytest.raises(ValueError, match="must have non-empty expected"):
        row_no_exp.validate_for_hard_slice()

    # Core row cannot validate as hard
    core_row = GoldenRow.model_validate({"prompt": "Hello", "expected": "Hi", "slice": "core"})
    with pytest.raises(ValueError, match="must have slice='hard'"):
        core_row.validate_for_hard_slice()


@pytest.mark.unit
@pytest.mark.harness
def test_echo_backend_dict_mapping_and_reset():
    """EchoBackend supports dictionary mapping, lists per prompt, default fallback, and reset."""
    mapping = {
        "task1": '{"tool": "final_answer", "args": {"text": "resp1"}}',
        "task2": [
            '{"tool": "final_answer", "args": {"text": "trial1"}}',
            '{"tool": "final_answer", "args": {"text": "trial2"}}',
        ],
        "default": '{"tool": "final_answer", "args": {"text": "fallback"}}',
    }
    backend = EchoBackend(mapping)

    # Matches task1
    res1 = backend.generate([{"role": "user", "content": "Execute task1"}])
    assert "resp1" in res1[0]

    # Matches task2 first trial then second trial
    res2a = backend.generate([{"role": "user", "content": "Execute task2"}])
    assert "trial1" in res2a[0]
    res2b = backend.generate([{"role": "user", "content": "Execute task2"}])
    assert "trial2" in res2b[0]

    # Unmatched uses default
    res_def = backend.generate([{"role": "user", "content": "Unknown task"}])
    assert "fallback" in res_def[0]

    # After reset, task2 list is restored
    backend.reset()
    res2_reset = backend.generate([{"role": "user", "content": "Execute task2"}])
    assert "trial1" in res2_reset[0]


class MultiTrialScriptedBackend(EchoBackend):
    """Scripted backend returning varying responses across trials for testing pass@k."""

    def __init__(self, responses_by_prompt: dict[str, list[str]]) -> None:
        super().__init__()
        self.responses = {k: list(v) for k, v in responses_by_prompt.items()}

    def generate(self, messages, *, prefix=None, n=1, temperature=None):
        user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        full_user = " ".join(user_texts)
        matched_queue = None
        for key, q in self.responses.items():
            if key in full_user:
                matched_queue = q
                break
        if matched_queue and len(matched_queue) > 0:
            item = matched_queue.pop(0)
        else:
            item = '{"tool": "final_answer", "args": {"text": "default"}}'
        return [item] * (n if n and n > 0 else 1)


@pytest.mark.unit
@pytest.mark.harness
def test_pass_at_1_vs_pass_at_k_multi_trial(tmp_path):
    """Verify that a prompt failing trial 1 but passing trial 2 gives pass@1=0% and pass@k=100%."""
    golden_file = tmp_path / "test_golden.jsonl"
    golden_file.write_text(
        json.dumps(
            {
                "id": "trial-test-01",
                "prompt": "Test multi-trial",
                "expected": "Success",
                "slice": "core",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Trial 1: wrong answer, Trial 2: exact match
    backend = MultiTrialScriptedBackend(
        {
            "Test multi-trial": [
                '{"tool": "final_answer", "args": {"text": "Wrong"}}',
                '{"tool": "final_answer", "args": {"text": "Success"}}',
                '{"tool": "final_answer", "args": {"text": "Success"}}',
            ]
        }
    )

    from enhanced_system.harness.registry import load_spec
    from enhanced_system.harness.runtime import AgentRuntime

    spec = load_spec("base_react")
    runtime = AgentRuntime(backend, spec=spec, teacher=False)

    summary = _evaluate_file(
        runtime,
        golden_file,
        "base_react",
        strict=True,
        pass_k=2,
    )

    assert summary["total"] == 1
    assert summary["k"] == 2
    assert summary["pass_at_1"] == 0.0
    assert summary["pass_at_k"] == 100.0


@pytest.mark.unit
@pytest.mark.harness
def test_hard_slice_rejects_semantic_only_match(tmp_path):
    """Hard slice requires exact answers_match; semantic near-miss counts as failure."""
    golden_file = tmp_path / "hard_test.jsonl"
    golden_file.write_text(
        json.dumps(
            {
                "id": "hard-001",
                "prompt": "Database connection pool",
                "expected": "ConnectionRefusedError handled properly",
                "slice": "hard",
                "allow_semantic": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Returns "connectionrefusederror handled properly!" - semantic match but not exact
    backend = MultiTrialScriptedBackend(
        {
            "Database connection pool": [
                '{"tool": "final_answer", "args": {"text": "connectionrefusederror handled properly!"}}',
            ]
        }
    )

    from enhanced_system.harness.registry import load_spec
    from enhanced_system.harness.runtime import AgentRuntime

    spec = load_spec("base_react")
    runtime = AgentRuntime(backend, spec=spec, teacher=False)

    summary = _evaluate_file(
        runtime,
        golden_file,
        "base_react",
        strict=True,
        pass_k=1,
    )

    assert summary["total"] == 1
    assert summary["hard_total"] == 1
    assert summary["semantic_match"] == 1
    assert summary["exact_match"] == 0
    assert summary["hard_pass_at_k"] == 0.0
    assert summary["pass_at_k"] == 0.0


@pytest.mark.unit
@pytest.mark.harness
def test_core_slice_allow_semantic_true_vs_false(tmp_path):
    """Core slice allows semantic match ONLY if allow_semantic is true."""
    golden_file = tmp_path / "core_test.jsonl"
    rows = [
        {
            "id": "core-sem-false",
            "prompt": "Prompt 1",
            "expected": "Hello World",
            "slice": "core",
            "allow_semantic": False,
        },
        {
            "id": "core-sem-true",
            "prompt": "Prompt 2",
            "expected": "Hello World",
            "slice": "core",
            "allow_semantic": True,
        },
    ]
    golden_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Both return semantic near-match: "hello world!"
    backend = MultiTrialScriptedBackend(
        {
            "Prompt 1": ['{"tool": "final_answer", "args": {"text": "hello world!"}}'],
            "Prompt 2": ['{"tool": "final_answer", "args": {"text": "hello world!"}}'],
        }
    )

    from enhanced_system.harness.registry import load_spec
    from enhanced_system.harness.runtime import AgentRuntime

    spec = load_spec("base_react")
    runtime = AgentRuntime(backend, spec=spec, teacher=False)

    summary = _evaluate_file(
        runtime,
        golden_file,
        "base_react",
        strict=True,
        pass_k=1,
    )

    assert summary["total"] == 2
    # Prompt 1 failed (allow_semantic=False), Prompt 2 succeeded (allow_semantic=True)
    assert summary["pass_at_k"] == 50.0
    assert summary["core_pass_at_k"] == 50.0


@pytest.mark.unit
@pytest.mark.harness
def test_rule_matrix_verification(tmp_path):
    """Rule traceability check passes when all hard IDs are present and fails when missing."""
    matrix_file = tmp_path / "matrix.yaml"
    golden_file = tmp_path / "hard.jsonl"

    matrix_file.write_text(
        yaml.dump(
            {
                "schema_version": "1",
                "rules": [
                    {
                        "rule_id": "R1",
                        "source_trace": "t1",
                        "harness_id": "h1",
                        "golden_id": "hard-1",
                        "solver_status": "pending",
                    },
                    {
                        "rule_id": "R2",
                        "source_trace": "t2",
                        "harness_id": "h2",
                        "golden_id": "hard-2",
                        "solver_status": "pending",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    golden_file.write_text(
        json.dumps({"id": "hard-1", "prompt": "p1", "expected": "e1", "slice": "hard"})
        + "\n"
        + json.dumps({"id": "hard-2", "prompt": "p2", "expected": "e2", "slice": "hard"})
        + "\n",
        encoding="utf-8",
    )

    # All covered
    assert verify_rule_matrix(matrix_file, golden_file) == []

    # Add uncovered row hard-3
    golden_file.write_text(
        golden_file.read_text(encoding="utf-8")
        + json.dumps({"id": "hard-3", "prompt": "p3", "expected": "e3", "slice": "hard"})
        + "\n",
        encoding="utf-8",
    )
    assert verify_rule_matrix(matrix_file, golden_file) == ["hard-3"]


@pytest.mark.integration
@pytest.mark.harness
def test_run_aqa_gate_cli_integration(tmp_path):
    """Test run_aqa_gate CLI with scripted responses and hard threshold."""
    golden_file = tmp_path / "golden.jsonl"
    golden_file.write_text(
        json.dumps(
            {
                "id": "gate-hard-001",
                "prompt": "Do task",
                "expected": "Task Done",
                "slice": "hard",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    scripted_file = tmp_path / "mock.json"
    scripted_file.write_text(
        json.dumps({"Do task": '{"tool": "final_answer", "args": {"text": "Task Done"}}'}),
        encoding="utf-8",
    )

    ret = run_gate(
        [
            "--golden-set",
            str(golden_file),
            "--threshold",
            "100.0",
            "--hard-threshold",
            "100.0",
            "--pass-k",
            "1",
            "--require-hard",
            "--scripted",
            str(scripted_file),
        ]
    )
    assert ret == 0


@pytest.mark.integration
@pytest.mark.harness
def test_run_aqa_gate_fails_below_threshold(tmp_path):
    """Test run_aqa_gate CLI fails when pass@k is below threshold."""
    golden_file = tmp_path / "golden.jsonl"
    golden_file.write_text(
        json.dumps(
            {
                "id": "gate-hard-002",
                "prompt": "Do task",
                "expected": "Task Done",
                "slice": "hard",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    scripted_file = tmp_path / "mock.json"
    scripted_file.write_text(
        json.dumps({"Do task": '{"tool": "final_answer", "args": {"text": "Wrong"}}'}),
        encoding="utf-8",
    )

    ret = run_gate(
        [
            "--golden-set",
            str(golden_file),
            "--threshold",
            "80.0",
            "--scripted",
            str(scripted_file),
        ]
    )
    assert ret == 1


@pytest.mark.unit
@pytest.mark.harness
def test_security_scanner_empty_and_safe_and_unsafe_code():
    """SecurityScanner scans python code and trajectories with Bandit."""
    from enhanced_system.harness.security import SecurityScanner
    from enhanced_system.harness.types import Step, Trajectory

    scanner = SecurityScanner(fail_on_high=True)

    # Empty code
    assert scanner.scan_python_code("") == []
    assert scanner.scan_python_code("   ") == []

    # Safe code
    safe_findings = scanner.scan_python_code("result = [x * 2 for x in range(10)]\n")
    assert safe_findings == []

    # Unsafe code (B307 eval)
    unsafe_findings = scanner.scan_python_code("eval('1 + 1')\n")
    assert len(unsafe_findings) > 0
    assert any("eval" in f.get("issue_text", "") for f in unsafe_findings)

    # Trajectory scanning
    traj = Trajectory(
        harness_id="test",
        steps=[
            Step(
                action=json.dumps({"tool": "code_interpreter", "args": {"code": "x = 42"}}),
                tool_id="code_interpreter",
            ),
            Step(
                action=json.dumps({"tool": "code_interpreter", "code": "eval('2+2')"}),
                tool_id="code_interpreter",
            ),
        ],
    )
    traj_findings = scanner.scan_trajectory(traj)
    assert len(traj_findings) > 0

    # Non-trajectory object
    assert scanner.scan_trajectory(object()) == []


@pytest.mark.unit
@pytest.mark.harness
def test_pii_scrubber_behavior(monkeypatch):
    """PIIScrubber handles missing Presidio dependency gracefully."""
    import enhanced_system.harness.data_governance as dg

    if not dg.HAS_PRESIDIO:
        with pytest.raises(ImportError, match="Presidio libraries not found"):
            dg.PIIScrubber()
    else:
        # Mock presidio behavior if installed
        scrubber = dg.PIIScrubber()
        assert scrubber.entities is not None
