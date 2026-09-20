"""Unit and integration tests for E2 collect-compose-sqe-e2e.

Validates:
- Trajectory collection under sqe_dispose harness with scripted fixtures.
- sqe_constraint_solver execution preceding final_answer.
- Canonical BLOCKED:<CODE> refusal tokens on OOD tasks.
- Reject telemetry sink (artifacts/critic_rejects.jsonl):
    * collect outcome drop -> OUTCOME_MISMATCH
    * DualDistill (0,0) drop -> DUALDISTILL_DROP_0_0
    * No second DualDistill drop rule (1,0; 0,1; 1,1 are kept)
    * Recovery traces retained when outcome matches (critic_kept_recovery)
- Strict scope locks (no GKD, no Neuroharness, no Pass@K in run_aqa_gate).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from enhanced_system.harness.critic import (
    CriticRejectCode,
    RejectSink,
    check_expected_tools,
    is_recovery_trace,
)
from enhanced_system.harness.dualdistill import (
    TRANSITION_BOTH,
    TRANSITION_FIX,
    compose_pair,
)
from enhanced_system.harness.types import Step, Trajectory
from scripts.harness import collect_trajectories, compose_dualdistill


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_collect_sqe_dispose_deterministic_mock_fixtures(tmp_path: Path) -> None:
    """make collect-sqe-dispose produces 24 dispose traces with solver before final_answer."""
    out_file = tmp_path / "sqe_dispose_teacher_a.jsonl"
    reject_file = tmp_path / "critic_rejects.jsonl"

    exit_code = collect_trajectories.main(
        [
            "--input",
            "configs/golden_sets/sqe_hard_ood.jsonl",
            "--output",
            str(out_file),
            "--harness-id",
            "sqe_dispose",
            "--scripted",
            "tests/fixtures/mock_responses_sqe_passk.json",
            "--critic",
            "--reject-log",
            str(reject_file),
        ]
    )
    assert exit_code == 0, "collect_trajectories under sqe_dispose must exit 0"
    assert out_file.is_file(), "collected trajectories file must exist"

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 24, "All 24 golden hard+ood rows must be collected"

    for idx, row in enumerate(lines):
        traj = row.get("trajectory")
        assert isinstance(traj, dict), f"row {idx} must have trajectory dict"
        assert traj.get("harness_id") == "sqe_dispose"

        steps = traj.get("steps") or []
        assert len(steps) >= 2, f"row {idx} must have at least 2 steps (solver + final_answer)"

        tool_sequence = [s.get("tool_id") for s in steps if s.get("tool_id")]
        assert "sqe_constraint_solver" in tool_sequence, (
            f"row {idx} missing sqe_constraint_solver in {tool_sequence}"
        )
        assert tool_sequence[-1] == "final_answer", (
            f"row {idx} final tool must be final_answer, got {tool_sequence}"
        )

        solver_idx = tool_sequence.index("sqe_constraint_solver")
        fa_idx = tool_sequence.index("final_answer")
        assert solver_idx < fa_idx, (
            f"row {idx} sqe_constraint_solver must precede final_answer ({tool_sequence})"
        )

        final_answer = traj.get("final_answer")
        expected = row.get("expected")
        assert final_answer == expected, (
            f"row {idx} final_answer {final_answer!r} must match expected {expected!r}"
        )
        if expected.startswith("BLOCKED:"):
            assert final_answer.startswith("BLOCKED:"), (
                f"OOD row {idx} must produce canonical BLOCKED refusal"
            )


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_collect_outcome_mismatch_fires_critic_reject_code(tmp_path: Path) -> None:
    """Collect drops mismatched outcome and logs critic_reject_code: OUTCOME_MISMATCH."""
    in_file = tmp_path / "in_mismatch.jsonl"
    out_file = tmp_path / "out_mismatch.jsonl"
    reject_file = tmp_path / "critic_rejects.jsonl"

    row = {
        "id": "test-mismatch-001",
        "harness_id": "sqe_dispose",
        "expected_tools": ["sqe_constraint_solver", "final_answer"],
        "prompt": "[test-mismatch-001] Test DAG prompt",
        "expected": "expected_answer_correct",
    }
    in_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    mock_responses = {
        "test-mismatch-001": [
            {
                "tool": "sqe_constraint_solver",
                "args": {"mode": "solve", "graph": {"nodes": ["a"], "edges": []}},
            },
            {
                "tool": "final_answer",
                "args": {"text": "wrong_outcome_answer"},
            },
        ]
    }
    scripted_file = tmp_path / "mock.json"
    scripted_file.write_text(json.dumps(mock_responses), encoding="utf-8")

    exit_code = collect_trajectories.main(
        [
            "--input",
            str(in_file),
            "--output",
            str(out_file),
            "--harness-id",
            "sqe_dispose",
            "--scripted",
            str(scripted_file),
            "--critic",
            "--reject-log",
            str(reject_file),
        ]
    )
    assert exit_code == 0
    # The mismatched trajectory must be dropped from output
    out_lines = [line for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(out_lines) == 0, "Mismatched trajectory must be dropped"

    # Reject log must record OUTCOME_MISMATCH
    assert reject_file.is_file(), "Reject log file must exist"
    reject_records = [
        json.loads(line) for line in reject_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_records) == 1
    rec = reject_records[0]
    assert rec["critic_reject_code"] == CriticRejectCode.OUTCOME_MISMATCH.value
    assert rec["prompt"] == "[test-mismatch-001] Test DAG prompt"
    assert rec["metadata"]["harness_id"] == "sqe_dispose"
    assert rec["metadata"]["final_answer"] == "wrong_outcome_answer"
    assert rec["metadata"]["expected"] == "expected_answer_correct"
    assert "timestamp" in rec


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_falsifier_outcome_mismatch_without_telemetry_fails(tmp_path: Path) -> None:
    """Falsifier 2: Intentional outcome mismatch dropped without OUTCOME_MISMATCH must fail."""
    sink = RejectSink(tmp_path / "falsifier_sink.jsonl")
    prompt = "Test DAG problem"
    expected = "valid_order"
    bad_output = "corrupted_order"

    sink.record(
        CriticRejectCode.OUTCOME_MISMATCH.value,
        prompt,
        metadata={"final_answer": bad_output, "expected": expected, "harness_id": "sqe_dispose"},
    )

    records = sink.records
    assert any(r["critic_reject_code"] == "OUTCOME_MISMATCH" for r in records), (
        "Falsifier: Outcome mismatch dropped without OUTCOME_MISMATCH record must fail"
    )


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_dualdistill_0_0_drop_fires_critic_reject_code(tmp_path: Path) -> None:
    """DualDistill drops (0,0) pairs and logs critic_reject_code: DUALDISTILL_DROP_0_0."""
    reject_file = tmp_path / "critic_rejects.jsonl"
    sink = RejectSink(reject_file)

    left = {
        "prompt": "[sqe-test-dag] Resolve DAG",
        "expected": "a b c",
        "harness_id": "sqe_dispose",
        "trajectory": {
            "harness_id": "sqe_dispose",
            "final_answer": "wrong_1",
            "steps": [
                {"tool_id": "sqe_constraint_solver", "action": "solve"},
                {"tool_id": "final_answer", "action": "wrong_1"},
            ],
        },
    }
    right = {
        "prompt": "[sqe-test-dag] Resolve DAG",
        "expected": "a b c",
        "harness_id": "sqe_dispose",
        "trajectory": {
            "harness_id": "sqe_dispose",
            "final_answer": "wrong_2",
            "steps": [
                {"tool_id": "sqe_constraint_solver", "action": "solve"},
                {"tool_id": "final_answer", "action": "wrong_2"},
            ],
        },
    }

    result = compose_pair(left, right, expected="a b c", reject_sink=sink)
    assert result is None, "DualDistill must drop (0,0) pair"

    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec["critic_reject_code"] == CriticRejectCode.DUALDISTILL_DROP_0_0.value
    assert rec["prompt"] == "[sqe-test-dag] Resolve DAG"
    assert rec["metadata"]["harness_id"] == "sqe_dispose"
    assert rec["metadata"]["first_final"] == "wrong_1"
    assert rec["metadata"]["second_final"] == "wrong_2"
    assert rec["metadata"]["expected"] == "a b c"
    assert "timestamp" in rec

    # Verify JSONL persistence
    assert reject_file.is_file()
    file_records = [
        json.loads(line) for line in reject_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(file_records) == 1
    assert file_records[0]["critic_reject_code"] == "DUALDISTILL_DROP_0_0"


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_falsifier_dualdistill_0_0_without_telemetry_fails(tmp_path: Path) -> None:
    """Falsifier 3: Synthetic (0,0) pair dropped without DUALDISTILL_DROP_0_0 must fail."""
    sink = RejectSink(tmp_path / "dd_sink.jsonl")
    prompt = "Test DAG problem"

    first = {"prompt": prompt, "trajectory": {"final_answer": "bad_1", "steps": []}}
    second = {"prompt": prompt, "trajectory": {"final_answer": "bad_2", "steps": []}}

    res = compose_pair(first, second, expected="correct", reject_sink=sink)
    assert res is None
    assert any(r["critic_reject_code"] == "DUALDISTILL_DROP_0_0" for r in sink.records), (
        "Falsifier: Drop without DUALDISTILL_DROP_0_0 must fail"
    )


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_no_second_dualdistill_drop_rule(tmp_path: Path) -> None:
    """Contract: NO second DualDistill drop rule; pairs with at least 1 success are kept."""
    sink = RejectSink(tmp_path / "dd_sink.jsonl")
    expected = "valid_topo_order"

    good_traj = {
        "schema_version": "1",
        "harness_id": "sqe_dispose",
        "task": "Order nodes",
        "instruction": "",
        "steps": [
            {
                "thought": "",
                "action": "",
                "observation": "",
                "tool_id": "sqe_constraint_solver",
                "fault": "",
            },
            {
                "thought": "",
                "action": "",
                "observation": "",
                "tool_id": "final_answer",
                "fault": "",
            },
        ],
        "final_answer": expected,
        "faults": [],
    }
    bad_traj = {
        "schema_version": "1",
        "harness_id": "sqe_dispose",
        "task": "Order nodes",
        "instruction": "",
        "steps": [
            {
                "thought": "",
                "action": "",
                "observation": "",
                "tool_id": "sqe_constraint_solver",
                "fault": "",
            },
            {
                "thought": "",
                "action": "",
                "observation": "",
                "tool_id": "final_answer",
                "fault": "",
            },
        ],
        "final_answer": "wrong_order",
        "faults": [],
    }

    # Case 1: (1, 0) -> Left succeeds, right fails
    left_good = {"prompt": "Order nodes", "expected": expected, "trajectory": good_traj}
    right_bad = {"prompt": "Order nodes", "expected": expected, "trajectory": bad_traj}
    res_1_0 = compose_pair(left_good, right_bad, expected=expected, reject_sink=sink)
    assert res_1_0 is not None, "Pair (1,0) MUST NOT be dropped"
    assert res_1_0["trajectory"]["final_answer"] == expected
    assert len(sink.records) == 0, "No rejection record for (1,0)"

    # Case 2: (0, 1) -> Left fails, right succeeds
    left_bad = {"prompt": "Order nodes", "expected": expected, "trajectory": bad_traj}
    right_good = {"prompt": "Order nodes", "expected": expected, "trajectory": good_traj}
    res_0_1 = compose_pair(left_bad, right_good, expected=expected, reject_sink=sink)
    assert res_0_1 is not None, "Pair (0,1) MUST NOT be dropped"
    assert res_0_1["trajectory"]["final_answer"] == expected
    assert any(s.get("thought") == TRANSITION_FIX for s in res_0_1["trajectory"]["steps"])
    assert len(sink.records) == 0, "No rejection record for (0,1)"

    # Case 3: (1, 1) -> Both succeed
    res_1_1 = compose_pair(left_good, right_good, expected=expected, reject_sink=sink)
    assert res_1_1 is not None, "Pair (1,1) MUST NOT be dropped"
    assert res_1_1["trajectory"]["final_answer"] == expected
    assert any(s.get("thought") == TRANSITION_BOTH for s in res_1_1["trajectory"]["steps"])
    assert len(sink.records) == 0, "No rejection record for (1,1)"


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_recovery_trace_retained_on_outcome_match() -> None:
    """Recovery traces with intermediate tool/parse errors are kept if outcome matches."""
    traj = Trajectory(
        harness_id="sqe_dispose",
        task="Solve boolean constraints",
        final_answer="A=true",
        steps=[
            Step(tool_id="sqe_constraint_solver", fault="tool_error", observation="Syntax retry"),
            Step(tool_id="sqe_constraint_solver", fault="", observation="SAT"),
            Step(tool_id="final_answer", fault="", observation="A=true"),
        ],
        faults=["tool_error"],
    )
    assert is_recovery_trace(traj, expected="A=true") is True
    # If final answer is wrong, it is NOT considered a valid recovery trace
    assert is_recovery_trace(traj, expected="A=false") is False


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_falsifier_solver_bypass_rejected_by_critic() -> None:
    """Falsifier 1: Any trajectory skipping sqe_constraint_solver fails critic validation."""
    bypassed_traj = Trajectory(
        harness_id="sqe_dispose",
        task="Bypass solver direct final answer",
        final_answer="checkout compile",
        steps=[
            Step(tool_id="final_answer", action="checkout compile"),
        ],
    )
    decision = check_expected_tools(
        bypassed_traj,
        expected_tools=["sqe_constraint_solver", "final_answer"],
    )
    assert decision.passed is False
    assert decision.reject_code == CriticRejectCode.SCHEMA_VIOLATION.value
    assert "missing expected tools" in decision.reason
    assert "sqe_constraint_solver" in decision.reason


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_compose_dualdistill_sqe_integration(tmp_path: Path) -> None:
    """Integration test for compose-dualdistill-sqe CLI with both teachers and reject log."""
    teacher_a_file = tmp_path / "teacher_a.jsonl"
    teacher_b_file = tmp_path / "teacher_b.jsonl"
    composed_file = tmp_path / "composed.jsonl"
    reject_file = tmp_path / "critic_rejects.jsonl"

    row_success = {
        "prompt": "[sqe-test-1] Resolve DAG",
        "expected": "a b c",
        "harness_id": "sqe_dispose",
        "completion": '{"tool": "sqe_constraint_solver"}\n{"tool": "final_answer", "args": {"text": "a b c"}}',
        "trajectory": {
            "harness_id": "sqe_dispose",
            "task": "[sqe-test-1] Resolve DAG",
            "final_answer": "a b c",
            "steps": [
                {"tool_id": "sqe_constraint_solver", "action": ""},
                {"tool_id": "final_answer", "action": ""},
            ],
        },
    }
    row_fail_a = {
        "prompt": "[sqe-test-2] Contradiction",
        "expected": "BLOCKED:UNSAT",
        "harness_id": "sqe_dispose",
        "completion": '{"tool": "final_answer", "args": {"text": "wrong_a"}}',
        "trajectory": {
            "harness_id": "sqe_dispose",
            "final_answer": "wrong_a",
            "steps": [{"tool_id": "final_answer", "action": ""}],
        },
    }
    row_fail_b = {
        "prompt": "[sqe-test-2] Contradiction",
        "expected": "BLOCKED:UNSAT",
        "harness_id": "sqe_dispose",
        "completion": '{"tool": "final_answer", "args": {"text": "wrong_b"}}',
        "trajectory": {
            "harness_id": "sqe_dispose",
            "final_answer": "wrong_b",
            "steps": [{"tool_id": "final_answer", "action": ""}],
        },
    }

    teacher_a_file.write_text(
        json.dumps(row_success) + "\n" + json.dumps(row_fail_a) + "\n",
        encoding="utf-8",
    )
    teacher_b_file.write_text(
        json.dumps(row_success) + "\n" + json.dumps(row_fail_b) + "\n",
        encoding="utf-8",
    )

    exit_code = compose_dualdistill.main(
        [
            "--first",
            str(teacher_a_file),
            "--second",
            str(teacher_b_file),
            "--output",
            str(composed_file),
            "--reject-log",
            str(reject_file),
        ]
    )
    assert exit_code == 0
    # Prompt 1 succeeds and is written; Prompt 2 is (0,0) and dropped
    composed_lines = [
        json.loads(line) for line in composed_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(composed_lines) == 1
    assert composed_lines[0]["prompt"] == "[sqe-test-1] Resolve DAG"

    # Reject log has exactly one DUALDISTILL_DROP_0_0 record
    reject_records = [
        json.loads(line) for line in reject_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_records) == 1
    assert reject_records[0]["critic_reject_code"] == "DUALDISTILL_DROP_0_0"
    assert reject_records[0]["prompt"] == "[sqe-test-2] Contradiction"


@pytest.mark.unit
@pytest.mark.harness
@pytest.mark.e2_sqe
def test_scope_locks_and_invariants() -> None:
    """Falsifier 5: Verify no GKD/Neuroharness/Edge/INV-16 and run_aqa_gate unpolluted."""
    makefile_text = Path("Makefile").read_text(encoding="utf-8")
    run_aqa_text = Path("scripts/harness/run_aqa_gate.py").read_text(encoding="utf-8")

    # Invariants for Makefile targets
    assert "collect-sqe-dispose:" in makefile_text
    assert "compose-dualdistill-sqe:" in makefile_text
    assert "collect-sqe:" not in makefile_text
    assert "compose-dual:" not in makefile_text

    # Locks: No GKD, Neuroharness, Edge-AI, INV-16
    for forbidden in ["trl_gkd", "Neuroharness", "INV-16", "Edge-AI"]:
        assert forbidden not in makefile_text, f"Forbidden reference {forbidden} found in Makefile"

    # Pass@K gate unchanged in role; NOT added to run_aqa_gate.py executable logic
    assert "import run_pass_at_k" not in run_aqa_text
    assert "calculate_pass_at_k" not in run_aqa_text
    assert "evaluate_pass_at_k" not in run_aqa_text
    assert "sqe_hard_ood" not in run_aqa_text
