"""Unit tests for Phase P2 critic-cascade: pure critic functions, reject codes, telemetry sink, and recovery traces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from enhanced_system.harness.critic import (
    VALID_REJECT_CODES,
    CriticDecision,
    CriticMetrics,
    CriticRejectCode,
    RejectSink,
    check_expected_tools,
    check_outcome,
    check_tool_allowlist,
    evaluate_trace,
    is_recovery_trace,
)
from enhanced_system.harness.dualdistill import compose_pair
from enhanced_system.harness.types import Step, Trajectory


@pytest.mark.unit
@pytest.mark.harness
def test_critic_reject_codes_conform_to_shared_standard():
    """Canonical reject codes match openspec/changes/_shared/blocked-reject-codes.md."""
    expected_codes = {
        # Deterministic solver / teacher-rule codes
        "CYCLE_DETECTED",
        "UNSAT",
        "SCHEMA_VIOLATION",
        "SYNTAX_INVALID",
        "UNSUPPORTED_THEORY",
        "RESOURCE_LIMIT",
        # Upstream pipeline codes
        "OUTCOME_MISMATCH",
        "DUALDISTILL_DROP_0_0",
    }
    assert VALID_REJECT_CODES == expected_codes
    for code in expected_codes:
        assert CriticRejectCode(code).value == code


@pytest.mark.unit
@pytest.mark.harness
def test_critic_decision_bool_protocol():
    """CriticDecision is truthy when passed=True, falsy when passed=False."""
    passed = CriticDecision(passed=True)
    assert bool(passed) is True
    assert passed.reject_code is None

    failed = CriticDecision(
        passed=False,
        reject_code=CriticRejectCode.OUTCOME_MISMATCH.value,
        reason="mismatch",
    )
    assert bool(failed) is False
    assert failed.reject_code == "OUTCOME_MISMATCH"
    assert "mismatch" in failed.reason


@pytest.mark.unit
@pytest.mark.harness
def test_check_tool_allowlist_valid():
    """Valid tools in trajectory pass allowlist check."""
    traj = Trajectory(
        task="Test task",
        steps=[
            Step(thought="Thinking", action="sqe_checklist()", tool_id="sqe_checklist"),
            Step(thought="Done", action="final_answer(text='ok')", tool_id="final_answer"),
        ],
        final_answer="ok",
    )
    decision = check_tool_allowlist(traj, {"sqe_checklist", "final_answer"})
    assert decision.passed is True
    assert decision.reject_code is None


@pytest.mark.unit
@pytest.mark.harness
def test_check_tool_allowlist_unknown_tool_id():
    """Unknown tool_id fails allowlist check with SCHEMA_VIOLATION."""
    traj = Trajectory(
        task="Test task",
        steps=[
            Step(thought="Hack", action="bash(cmd='rm -rf')", tool_id="bash"),
        ],
    )
    decision = check_tool_allowlist(traj, {"sqe_checklist", "final_answer"})
    assert decision.passed is False
    assert decision.reject_code == CriticRejectCode.SCHEMA_VIOLATION.value
    assert "bash" in decision.reason


@pytest.mark.unit
@pytest.mark.harness
def test_check_tool_allowlist_observation_error():
    """Observation containing dispatch/allowlist error flags allowlist violation."""
    traj = Trajectory(
        task="Test task",
        steps=[
            Step(
                thought="Try tool",
                action="",
                observation="unknown tool id: malicious_tool",
                fault="parse_error",
            ),
        ],
    )
    decision = check_tool_allowlist(traj, {"final_answer"})
    assert decision.passed is False
    assert decision.reject_code == CriticRejectCode.SCHEMA_VIOLATION.value
    assert "malicious_tool" in decision.reason


@pytest.mark.unit
@pytest.mark.harness
def test_check_tool_allowlist_from_dict_and_legacy_row():
    """Allowlist check works across dictionary and legacy row structures."""
    dict_row = {
        "trajectory": {
            "steps": [
                {"tool_id": "unauthorized_tool", "action": "unauthorized_tool()"},
            ],
            "final_answer": "ans",
        }
    }
    decision = check_tool_allowlist(dict_row, {"final_answer"})
    assert decision.passed is False
    assert decision.reject_code == CriticRejectCode.SCHEMA_VIOLATION.value


@pytest.mark.unit
@pytest.mark.harness
def test_check_outcome_exact_match():
    """Outcome check passes on exact match and unlabeled rows."""
    assert check_outcome("expected_answer", "expected_answer").passed is True
    assert check_outcome("any_answer", None).passed is True
    assert check_outcome("any_answer", "").passed is True
    assert check_outcome("any_answer", "   ").passed is True


@pytest.mark.unit
@pytest.mark.harness
def test_check_outcome_mismatch():
    """Outcome check fails on mismatch with OUTCOME_MISMATCH."""
    decision = check_outcome("actual_answer", "expected_answer")
    assert decision.passed is False
    assert decision.reject_code == CriticRejectCode.OUTCOME_MISMATCH.value
    assert "outcome mismatch" in decision.reason


@pytest.mark.unit
@pytest.mark.harness
def test_check_outcome_semantic_match():
    """Semantic match passes when allow_semantic=True."""
    decision_strict = check_outcome("hello world!", "Hello, World", allow_semantic=False)
    assert decision_strict.passed is False

    decision_sem = check_outcome("hello world!", "Hello, World", allow_semantic=True)
    assert decision_sem.passed is True


@pytest.mark.unit
@pytest.mark.harness
def test_check_expected_tools():
    """check_expected_tools checks tool presence and ordering."""
    traj = Trajectory(
        steps=[
            Step(tool_id="sqe_checklist", action="a"),
            Step(tool_id="pytest_runner", action="b"),
            Step(tool_id="final_answer", action="c"),
        ]
    )
    # Unordered check
    assert check_expected_tools(traj, ["pytest_runner", "sqe_checklist"]).passed is True
    # Missing tool
    missing = check_expected_tools(traj, ["architect_structure"])
    assert missing.passed is False
    assert missing.reject_code == CriticRejectCode.SCHEMA_VIOLATION.value
    assert "architect_structure" in missing.reason

    # Ordered check
    assert (
        check_expected_tools(
            traj, ["sqe_checklist", "pytest_runner", "final_answer"], ordered=True
        ).passed
        is True
    )
    assert (
        check_expected_tools(
            traj, ["pytest_runner", "sqe_checklist", "final_answer"], ordered=True
        ).passed
        is False
    )


@pytest.mark.unit
@pytest.mark.harness
def test_is_recovery_trace():
    """Recovery trace is detected when intermediate fault occurs but final outcome matches."""
    # Case 1: Early parse error, recovered with matching final answer
    recovered_parse = Trajectory(
        steps=[
            Step(thought="", action="", observation="not json", fault="parse_error"),
            Step(thought="fix", action="final_answer(text='42')", tool_id="final_answer"),
        ],
        final_answer="42",
        faults=["parse_error"],
    )
    assert is_recovery_trace(recovered_parse, expected="42") is True
    assert is_recovery_trace(recovered_parse, expected="wrong") is False
    assert is_recovery_trace(recovered_parse, expected=None) is False

    # Case 2: Early tool error, recovered with matching final answer
    recovered_tool = Trajectory(
        steps=[
            Step(
                thought="check",
                action="sqe_checklist()",
                observation="ValueError",
                fault="tool_error",
                tool_id="sqe_checklist",
            ),
            Step(thought="retry", action="final_answer(text='ok')", tool_id="final_answer"),
        ],
        final_answer="ok",
        faults=["tool_error"],
    )
    assert is_recovery_trace(recovered_tool, expected="ok") is True

    # Case 3: Clean trace without faults is not a recovery trace
    clean = Trajectory(
        steps=[
            Step(thought="ok", action="final_answer(text='ok')", tool_id="final_answer"),
        ],
        final_answer="ok",
        faults=[],
    )
    assert is_recovery_trace(clean, expected="ok") is False


@pytest.mark.unit
@pytest.mark.harness
def test_evaluate_trace_cascade():
    """evaluate_trace runs allowlist -> expected_tools -> outcome cascade."""
    traj = Trajectory(
        steps=[
            Step(tool_id="disallowed", action="x"),
        ],
        final_answer="42",
    )
    dec = evaluate_trace(
        traj,
        allowed_tools=["final_answer"],
        expected="42",
    )
    assert dec.passed is False
    assert dec.reject_code == CriticRejectCode.SCHEMA_VIOLATION.value

    # Pass allowlist, fail expected tools
    traj2 = Trajectory(
        steps=[
            Step(tool_id="final_answer", action="x"),
        ],
        final_answer="42",
    )
    dec2 = evaluate_trace(
        traj2,
        allowed_tools=["final_answer"],
        expected_tools=["sqe_checklist"],
        expected="42",
    )
    assert dec2.passed is False

    # Pass allowlist and expected tools, fail outcome
    dec3 = evaluate_trace(
        traj2,
        allowed_tools=["final_answer"],
        expected_tools=["final_answer"],
        expected="99",
    )
    assert dec3.passed is False
    assert dec3.reject_code == CriticRejectCode.OUTCOME_MISMATCH.value

    # All pass
    dec4 = evaluate_trace(
        traj2,
        allowed_tools=["final_answer"],
        expected_tools=["final_answer"],
        expected="42",
    )
    assert dec4.passed is True


@pytest.mark.unit
@pytest.mark.harness
def test_reject_sink_records_jsonl(tmp_path):
    """RejectSink writes structured records to JSONL sink file."""
    sink_path = tmp_path / "artifacts" / "critic_rejects.jsonl"
    sink = RejectSink(sink_path)

    entry1 = sink.record(
        CriticRejectCode.OUTCOME_MISMATCH.value,
        "Calculate 2+2",
        metadata={"expected": "4", "final_answer": "5", "line_no": 1},
    )
    assert entry1["critic_reject_code"] == "OUTCOME_MISMATCH"
    assert entry1["prompt"] == "Calculate 2+2"
    assert len(sink.records) == 1

    entry2 = sink.record(
        CriticRejectCode.DUALDISTILL_DROP_0_0.value,
        "Explain quantum gravity",
        metadata={"expected": "theory"},
    )
    assert entry2["critic_reject_code"] == "DUALDISTILL_DROP_0_0"
    assert len(sink.records) == 2

    # Verify file contents
    assert sink_path.is_file()
    lines = [json.loads(line) for line in sink_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["critic_reject_code"] == "OUTCOME_MISMATCH"
    assert lines[0]["metadata"]["line_no"] == 1
    assert lines[1]["critic_reject_code"] == "DUALDISTILL_DROP_0_0"


@pytest.mark.unit
@pytest.mark.harness
def test_critic_metrics_counters():
    """CriticMetrics tracks reject codes and recovery counts."""
    metrics = CriticMetrics()
    assert metrics.critic_kept_recovery == 0
    assert metrics.critic_rejected_outcome_mismatch == 0
    assert metrics.critic_rejected_allowlist == 0

    metrics.record_reject(CriticRejectCode.OUTCOME_MISMATCH.value)
    metrics.record_reject(CriticRejectCode.SCHEMA_VIOLATION.value)
    metrics.record_reject(CriticRejectCode.DUALDISTILL_DROP_0_0.value)
    metrics.record_recovery()
    metrics.record_recovery()

    assert metrics.critic_rejected_outcome_mismatch == 1
    assert metrics.critic_rejected_schema_violation == 1
    assert metrics.critic_rejected_dualdistill_drop_0_0 == 1
    assert metrics.critic_kept_recovery == 2
    d = metrics.as_dict()
    assert d["critic_kept_recovery"] == 2
    assert d["critic_rejected_outcome_mismatch"] == 1


# --- SPEC FALSIFIERS ---


@pytest.mark.unit
@pytest.mark.harness
def test_falsifier_unknown_tool_rejected_when_critic_enabled(monkeypatch, tmp_path):
    """Spec falsifier: A unit test that injects an unknown tool_id MUST fail if collect writes the row with critic enabled."""
    import enhanced_system.harness.runtime as rt
    from scripts.harness.collect_trajectories import main

    orig_run = rt.AgentRuntime.run

    def injected_run(self, *args, **kwargs):
        res = orig_run(self, *args, **kwargs)
        res.trajectory.steps.insert(
            0,
            Step(
                thought="Injected bad step",
                action="unknown_tool()",
                tool_id="unknown_tool",
            ),
        )
        return res

    monkeypatch.setattr(rt.AgentRuntime, "run", injected_run)

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Run security check", "expected": "safe"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    reject_log = tmp_path / "rejects.jsonl"

    code = main(
        [
            "--input",
            str(source),
            "--output",
            str(dest),
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"safe\\"}}"]',
            "--critic",
            "--reject-log",
            str(reject_log),
        ]
    )
    assert code == 0
    # Collect MUST NOT write the row with critic enabled
    written_lines = [line for line in dest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(written_lines) == 0, "Collect must not write row with injected unknown tool_id"

    # Reject log MUST contain the reject entry with SCHEMA_VIOLATION
    assert reject_log.is_file()
    reject_entries = [
        json.loads(line) for line in reject_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_entries) == 1
    assert reject_entries[0]["critic_reject_code"] == CriticRejectCode.SCHEMA_VIOLATION.value
    assert "unknown_tool" in str(reject_entries[0]["metadata"])


@pytest.mark.unit
@pytest.mark.harness
def test_falsifier_dualdistill_0_0_emits_telemetry(caplog):
    """Spec falsifier: A unit test dropping a (0,0) pair without emitting DUALDISTILL_DROP_0_0 MUST fail."""
    import logging

    first = {
        "prompt": "Hard math problem",
        "expected": "42",
        "trajectory": {"final_answer": "0", "steps": []},
    }
    second = {
        "prompt": "Hard math problem",
        "expected": "42",
        "trajectory": {"final_answer": "100", "steps": []},
    }

    sink = RejectSink(path=None)
    with caplog.at_level(logging.WARNING):
        result = compose_pair(first, second, expected="42", reject_sink=sink)

    # Must drop (0,0) pair
    assert result is None

    # Telemetry requirement: MUST log critic_reject_code: DUALDISTILL_DROP_0_0
    assert any("DUALDISTILL_DROP_0_0" in record.message for record in caplog.records)
    # Reject sink MUST record DUALDISTILL_DROP_0_0
    assert len(sink.records) == 1
    assert sink.records[0]["critic_reject_code"] == "DUALDISTILL_DROP_0_0"
    assert sink.records[0]["prompt"] == "Hard math problem"


@pytest.mark.unit
@pytest.mark.harness
def test_collect_recovery_trace_kept_and_counter_incremented(tmp_path, caplog):
    """Intermediate parse_error is kept when final outcome matches expected, incrementing critic_kept_recovery."""
    import logging

    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Greet me", "expected": "hi"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"

    with caplog.at_level(logging.INFO):
        code = main(
            [
                "--input",
                str(source),
                "--output",
                str(dest),
                "--harness-id",
                "base_react",
                "--scripted",
                '["not-json", "{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
                "--critic",
            ]
        )
    assert code == 0

    # The recovered row is kept
    rows = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trajectory"]["final_answer"] == "hi"
    assert "parse_error" in rows[0]["trajectory"]["faults"]

    # critic_kept_recovery counter is logged
    assert any("critic_kept_recovery" in record.message for record in caplog.records)


@pytest.mark.unit
@pytest.mark.harness
def test_collect_outcome_mismatch_logged_and_recorded(tmp_path, caplog):
    """Outcome mismatch logs OUTCOME_MISMATCH and appends to reject sink."""
    import logging

    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Greet me", "expected": "hello"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    reject_log = tmp_path / "rejects.jsonl"

    with caplog.at_level(logging.WARNING):
        code = main(
            [
                "--input",
                str(source),
                "--output",
                str(dest),
                "--harness-id",
                "base_react",
                "--scripted",
                '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"bye\\"}}"]',
                "--critic",
                "--reject-log",
                str(reject_log),
            ]
        )
    assert code == 0
    assert dest.read_text(encoding="utf-8").strip() == ""

    assert any("OUTCOME_MISMATCH" in record.message for record in caplog.records)
    reject_entries = [
        json.loads(line) for line in reject_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_entries) == 1
    assert reject_entries[0]["critic_reject_code"] == "OUTCOME_MISMATCH"
    assert reject_entries[0]["metadata"]["final_answer"] == "bye"
    assert reject_entries[0]["metadata"]["expected"] == "hello"


@pytest.mark.unit
@pytest.mark.harness
def test_compose_dualdistill_cli_with_reject_log(tmp_path):
    """compose_dualdistill CLI logs DUALDISTILL_DROP_0_0 and records to reject sink."""
    from scripts.harness.compose_dualdistill import main

    first = tmp_path / "teacher_a.jsonl"
    second = tmp_path / "teacher_b.jsonl"
    out = tmp_path / "composed.jsonl"
    reject_log = tmp_path / "rejects.jsonl"

    row_bad_a = {
        "prompt": "Task 1",
        "expected": "answer",
        "trajectory": {"final_answer": "wrong_a", "steps": []},
    }
    row_bad_b = {
        "prompt": "Task 1",
        "expected": "answer",
        "trajectory": {"final_answer": "wrong_b", "steps": []},
    }

    first.write_text(json.dumps(row_bad_a) + "\n", encoding="utf-8")
    second.write_text(json.dumps(row_bad_b) + "\n", encoding="utf-8")

    code = main(
        [
            "--first",
            str(first),
            "--second",
            str(second),
            "--output",
            str(out),
            "--reject-log",
            str(reject_log),
        ]
    )
    assert code == 0
    assert out.read_text(encoding="utf-8").strip() == ""
    assert reject_log.is_file()

    entries = [json.loads(line) for line in reject_log.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["critic_reject_code"] == "DUALDISTILL_DROP_0_0"
    assert entries[0]["prompt"] == "Task 1"
    assert entries[0]["metadata"]["first_final"] == "wrong_a"
    assert entries[0]["metadata"]["second_final"] == "wrong_b"


@pytest.mark.unit
@pytest.mark.harness
def test_collect_reads_mangomas_critic_enabled_env(monkeypatch, tmp_path):
    """MANGOMAS_CRITIC_ENABLED=true environment variable enables critic cascade and reject logging."""
    import enhanced_system.ops.settings as st
    from scripts.harness.collect_trajectories import main

    st.get_settings.cache_clear()
    monkeypatch.setenv("MANGOMAS_CRITIC_ENABLED", "true")
    monkeypatch.setenv("MANGOMAS_CRITIC_REJECT_LOG", str(tmp_path / "env_rejects.jsonl"))

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Task", "expected": "right"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"

    code = main(
        [
            "--input",
            str(source),
            "--output",
            str(dest),
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"wrong\\"}}"]',
        ]
    )
    assert code == 0
    assert dest.read_text(encoding="utf-8").strip() == ""
    reject_entries = [
        json.loads(line)
        for line in (tmp_path / "env_rejects.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_entries) == 1
    assert reject_entries[0]["critic_reject_code"] == "OUTCOME_MISMATCH"
    st.get_settings.cache_clear()


@pytest.mark.unit
@pytest.mark.harness
def test_collect_backwards_compatible_when_critic_disabled(tmp_path):
    """When critic is disabled (--no-critic), collection behaves as before."""
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Greet", "expected": "hi"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"

    code = main(
        [
            "--input",
            str(source),
            "--output",
            str(dest),
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
            "--no-critic",
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trajectory"]["final_answer"] == "hi"


@pytest.mark.unit
@pytest.mark.harness
def test_produce_kill_artifact_critic_rejects(monkeypatch):
    """CI/test producing uploadable artifact artifacts/critic_rejects.jsonl with both codes."""
    from scripts.harness.collect_trajectories import main as collect_main
    from scripts.harness.compose_dualdistill import main as compose_main

    artifact_path = Path("artifacts/critic_rejects.jsonl")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        artifact_path.unlink()

    monkeypatch.setenv("MANGOMAS_CRITIC_ENABLED", "true")
    monkeypatch.setenv("MANGOMAS_CRITIC_REJECT_LOG", str(artifact_path))

    temp_dir = Path("artifacts/_tmp_test")
    temp_dir.mkdir(parents=True, exist_ok=True)
    collect_in = temp_dir / "collect_in.jsonl"
    collect_out = temp_dir / "collect_out.jsonl"
    collect_in.write_text(
        json.dumps({"prompt": "Task outcome mismatch", "expected": "expected_val"}) + "\n",
        encoding="utf-8",
    )

    code = collect_main(
        [
            "--input",
            str(collect_in),
            "--output",
            str(collect_out),
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"actual_val\\"}}"]',
            "--reject-log",
            str(artifact_path),
            "--critic",
        ]
    )
    assert code == 0

    t1_file = temp_dir / "t1.jsonl"
    t2_file = temp_dir / "t2.jsonl"
    comp_out = temp_dir / "comp_out.jsonl"
    t1_file.write_text(
        json.dumps({"prompt": "Task dualdistill drop", "expected": "gold", "completion": "ans1"})
        + "\n",
        encoding="utf-8",
    )
    t2_file.write_text(
        json.dumps({"prompt": "Task dualdistill drop", "expected": "gold", "completion": "ans2"})
        + "\n",
        encoding="utf-8",
    )

    code = compose_main(
        [
            "--first",
            str(t1_file),
            "--second",
            str(t2_file),
            "--output",
            str(comp_out),
            "--reject-log",
            str(artifact_path),
        ]
    )
    assert code == 0

    assert artifact_path.is_file(), f"{artifact_path} was not created"
    records = [json.loads(line) for line in artifact_path.read_text(encoding="utf-8").splitlines()]
    codes = {r.get("critic_reject_code") for r in records}
    assert "OUTCOME_MISMATCH" in codes, f"Missing OUTCOME_MISMATCH in {codes}"
    assert "DUALDISTILL_DROP_0_0" in codes, f"Missing DUALDISTILL_DROP_0_0 in {codes}"
    for r in records:
        assert "critic_reject_code" in r
        assert "prompt" in r

    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)
