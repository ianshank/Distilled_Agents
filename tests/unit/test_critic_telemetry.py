"""Unit and integration tests for Phase 0 I2 critic cascade reject telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from enhanced_system.harness.critic import (
    CriticRejectCode,
    CriticTelemetry,
    check_expected_tools,
    check_outcome,
    check_tool_allowlist,
    is_recovery_trace,
)
from enhanced_system.harness.dualdistill import compose_pair
from enhanced_system.harness.types import HarnessRunResult, Step, Trajectory
from enhanced_system.ops.settings import get_settings


@pytest.mark.unit
@pytest.mark.harness
def test_critic_reject_codes_enum_values():
    """Verify all standard codes from blocked-reject-codes.md are defined and match string values."""
    assert CriticRejectCode.OUTCOME_MISMATCH.value == "OUTCOME_MISMATCH"
    assert CriticRejectCode.DUALDISTILL_DROP_0_0.value == "DUALDISTILL_DROP_0_0"
    assert CriticRejectCode.CYCLE_DETECTED.value == "CYCLE_DETECTED"
    assert CriticRejectCode.UNSAT.value == "UNSAT"
    assert CriticRejectCode.SCHEMA_VIOLATION.value == "SCHEMA_VIOLATION"
    assert CriticRejectCode.SYNTAX_INVALID.value == "SYNTAX_INVALID"
    assert CriticRejectCode.UNSUPPORTED_THEORY.value == "UNSUPPORTED_THEORY"
    assert CriticRejectCode.RESOURCE_LIMIT.value == "RESOURCE_LIMIT"
    assert CriticRejectCode.TOOL_NOT_ALLOWED.value == "TOOL_NOT_ALLOWED"


@pytest.mark.unit
@pytest.mark.harness
def test_check_outcome():
    """Verify check_outcome exact matching and reject emission."""
    ok, code, reason = check_outcome("foo", "foo")
    assert ok is True
    assert code is None
    assert reason is None

    # Missing or empty expected passes
    ok, code, _ = check_outcome("foo", None)
    assert ok is True
    ok, code, _ = check_outcome("foo", "   ")
    assert ok is True

    # Mismatch emits OUTCOME_MISMATCH
    ok, code, reason = check_outcome("actual", "expected")
    assert ok is False
    assert code == CriticRejectCode.OUTCOME_MISMATCH
    assert "Outcome mismatch" in str(reason)


@pytest.mark.unit
@pytest.mark.harness
def test_check_tool_allowlist():
    """Verify check_tool_allowlist enforces allowed tools on steps."""
    traj = Trajectory(
        task="demo",
        steps=[
            Step(tool_id="final_answer", action="final_answer(text='ok')"),
        ],
        final_answer="ok",
    )
    ok, code, _ = check_tool_allowlist(traj, ["final_answer", "sqe_checklist"])
    assert ok is True
    assert code is None

    # Step invoking disallowed tool
    bad_traj = Trajectory(
        task="demo",
        steps=[
            Step(tool_id="unauthorized_exec", action="unauthorized_exec()"),
            Step(tool_id="final_answer", action="final_answer(text='ok')"),
        ],
        final_answer="ok",
    )
    ok, code, reason = check_tool_allowlist(bad_traj, ["final_answer"])
    assert ok is False
    assert code == CriticRejectCode.SCHEMA_VIOLATION
    assert "unauthorized_exec" in str(reason)

    # Check dict payload format
    dict_payload = {
        "steps": [{"tool_id": "bad_tool", "action": "bad_tool()"}],
    }
    ok, code, reason = check_tool_allowlist(dict_payload, ["final_answer"])
    assert ok is False
    assert code == CriticRejectCode.SCHEMA_VIOLATION


@pytest.mark.unit
@pytest.mark.harness
def test_check_expected_tools():
    """Verify check_expected_tools validates invocation of expected tools."""
    traj = Trajectory(
        task="demo",
        steps=[
            Step(tool_id="sqe_checklist"),
            Step(tool_id="final_answer"),
        ],
        final_answer="done",
    )
    ok, code, _ = check_expected_tools(traj, ["sqe_checklist"])
    assert ok is True

    ok, code, _ = check_expected_tools(traj, None)
    assert ok is True

    ok, code, _ = check_expected_tools(traj, [])
    assert ok is True

    ok, code, reason = check_expected_tools(traj, ["sqe_checklist", "other_tool"])
    assert ok is False
    assert code == CriticRejectCode.SCHEMA_VIOLATION
    assert "other_tool" in str(reason)


@pytest.mark.unit
@pytest.mark.harness
def test_is_recovery_trace():
    """Verify is_recovery_trace detects intermediate faults when outcome matches."""
    # Recovered parse error via trajectory.faults
    recovered_traj = Trajectory(
        task="demo",
        steps=[
            Step(thought="bad", fault="parse_error"),
            Step(thought="good", tool_id="final_answer"),
        ],
        faults=["parse_error"],
        final_answer="expected_ans",
    )
    assert is_recovery_trace(recovered_traj, expected="expected_ans") is True

    # Recovered step fault without trajectory.faults
    step_fault_traj = Trajectory(
        task="demo",
        steps=[
            Step(thought="bad", fault="tool_error"),
            Step(thought="good", tool_id="final_answer"),
        ],
        faults=[],
        final_answer="expected_ans",
    )
    assert is_recovery_trace(step_fault_traj, expected="expected_ans") is True

    # Fault present but outcome mismatch -> False (not a successful recovery)
    assert is_recovery_trace(recovered_traj, expected="wrong_ans") is False

    # Clean run without faults -> False (not a recovery trace)
    clean_traj = Trajectory(
        task="demo",
        steps=[Step(tool_id="final_answer")],
        faults=[],
        final_answer="expected_ans",
    )
    assert is_recovery_trace(clean_traj, expected="expected_ans") is False


@pytest.mark.unit
@pytest.mark.harness
def test_extract_trajectory_components_fallback_parsing():
    """Verify fallback parsing of action and observation when tool_id is not set."""
    dict_payload = {
        "steps": [
            {
                "tool_id": "",
                "observation": "tool 'forbidden' not allowed (must be one of ['final_answer'])",
            },
            {"tool_id": "", "action": "custom_call(arg=1)"},
        ]
    }
    ok, code, reason = check_tool_allowlist(dict_payload, ["final_answer"])
    assert ok is False
    assert code == CriticRejectCode.SCHEMA_VIOLATION
    assert "forbidden" in str(reason)

    telemetry_no_sink = CriticTelemetry(sink_path=None, enabled=True)
    entry = telemetry_no_sink.record_reject(CriticRejectCode.OUTCOME_MISMATCH, "no sink prompt")
    assert entry["critic_reject_code"] == "OUTCOME_MISMATCH"


@pytest.mark.unit
@pytest.mark.harness
def test_critic_telemetry_sink_and_counters(tmp_path):
    """Verify CriticTelemetry writes JSONL entries and increments counters."""
    sink_file = tmp_path / "rejects.jsonl"
    telemetry = CriticTelemetry(sink_path=sink_file, enabled=True)

    telemetry.record_reject(
        code=CriticRejectCode.OUTCOME_MISMATCH,
        prompt="test prompt 1",
        expected="gold",
        final_answer="pred",
        metadata={"step": 1},
    )
    telemetry.record_reject(
        code=CriticRejectCode.DUALDISTILL_DROP_0_0,
        prompt="test prompt 2",
        expected="gold",
        metadata={"reason": "both 0"},
    )
    telemetry.record_recovery("recovered prompt")
    telemetry.record_kept("clean prompt", is_recovery=False)

    summary = telemetry.emit_summary()
    assert summary["critic_rejected_outcome_mismatch"] == 1
    assert summary["critic_rejected_dualdistill_drop_0_0"] == 1
    assert summary["critic_rejected_total"] == 2
    assert summary["critic_kept_recovery"] == 1
    assert summary["critic_kept_total"] == 1

    lines = sink_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["critic_reject_code"] == "OUTCOME_MISMATCH"
    assert rec1["prompt"] == "test prompt 1"
    assert rec1["expected"] == "gold"
    assert rec1["final_answer"] == "pred"
    assert rec1["metadata"] == {"step": 1}

    rec2 = json.loads(lines[1])
    assert rec2["critic_reject_code"] == "DUALDISTILL_DROP_0_0"
    assert rec2["prompt"] == "test prompt 2"


@pytest.mark.unit
@pytest.mark.harness
def test_critic_telemetry_disabled_does_not_write(tmp_path):
    """Verify disabled CriticTelemetry does not write to sink file."""
    sink_file = tmp_path / "no_rejects.jsonl"
    telemetry = CriticTelemetry(sink_path=sink_file, enabled=False)
    telemetry.record_reject(
        code=CriticRejectCode.OUTCOME_MISMATCH,
        prompt="p",
    )
    assert not sink_file.exists()
    summary = telemetry.emit_summary()
    assert summary["critic_rejected_outcome_mismatch"] == 1


@pytest.mark.unit
@pytest.mark.harness
def test_dualdistill_0_0_drop_emits_telemetry(tmp_path, caplog):
    """Verify compose_pair drops (0,0) and emits DUALDISTILL_DROP_0_0."""
    sink_file = tmp_path / "dualdistill_rejects.jsonl"
    telemetry = CriticTelemetry(sink_path=sink_file, enabled=True)

    t1 = {"prompt": "p1", "completion": "bad1"}
    t2 = {"prompt": "p1", "completion": "bad2"}

    with caplog.at_level("INFO"):
        composed = compose_pair(t1, t2, expected="gold", telemetry=telemetry)

    assert composed is None
    assert "critic_reject_code: DUALDISTILL_DROP_0_0" in caplog.text

    lines = sink_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["critic_reject_code"] == "DUALDISTILL_DROP_0_0"
    assert record["prompt"] == "p1"
    assert record["expected"] == "gold"
    assert telemetry.counters["critic_rejected_dualdistill_drop_0_0"] == 1


@pytest.mark.unit
@pytest.mark.harness
def test_dualdistill_no_alternate_drop_path(tmp_path):
    """Prove that only (0,0) drops and (1,0), (0,1), (1,1) all succeed without dropping."""
    sink_file = tmp_path / "no_drop_rejects.jsonl"
    telemetry = CriticTelemetry(sink_path=sink_file, enabled=True)

    row_good_1 = {
        "prompt": "p",
        "expected": "gold",
        "trajectory": {"task": "p", "final_answer": "gold", "steps": []},
    }
    row_good_2 = {
        "prompt": "p",
        "expected": "gold",
        "trajectory": {"task": "p", "final_answer": "gold", "steps": []},
    }
    row_bad = {
        "prompt": "p",
        "expected": "gold",
        "trajectory": {"task": "p", "final_answer": "wrong", "steps": []},
    }

    # (1, 0) -> keep first
    res_1_0 = compose_pair(row_good_1, row_bad, expected="gold", telemetry=telemetry)
    assert res_1_0 is not None
    assert res_1_0["expected"] == "gold"

    # (0, 1) -> stitch fix
    res_0_1 = compose_pair(row_bad, row_good_2, expected="gold", telemetry=telemetry)
    assert res_0_1 is not None
    assert res_0_1["expected"] == "gold"

    # (1, 1) -> stitch both
    res_1_1 = compose_pair(row_good_1, row_good_2, expected="gold", telemetry=telemetry)
    assert res_1_1 is not None
    assert res_1_1["expected"] == "gold"

    # Prove that no reject entries were written for (1,0), (0,1), or (1,1)
    assert not sink_file.exists()
    assert telemetry.counters.get("critic_rejected_dualdistill_drop_0_0", 0) == 0
    assert telemetry.counters.get("critic_kept_total", 0) == 3


@pytest.mark.unit
@pytest.mark.harness
def test_compose_dualdistill_cli_with_critic(tmp_path):
    """Verify compose_dualdistill.py CLI logs and records DUALDISTILL_DROP_0_0 when critic is enabled."""
    from scripts.harness.compose_dualdistill import main

    first_file = tmp_path / "t1.jsonl"
    second_file = tmp_path / "t2.jsonl"
    out_file = tmp_path / "out.jsonl"
    reject_file = tmp_path / "rejects.jsonl"

    row_bad_1 = {"prompt": "task_drop", "expected": "gold", "completion": "wrong1"}
    row_bad_2 = {"prompt": "task_drop", "expected": "gold", "completion": "wrong2"}
    row_good_1 = {"prompt": "task_keep", "expected": "gold", "completion": "gold"}
    row_good_2 = {"prompt": "task_keep", "expected": "gold", "completion": "gold"}

    first_file.write_text(
        json.dumps(row_bad_1) + "\n" + json.dumps(row_good_1) + "\n", encoding="utf-8"
    )
    second_file.write_text(
        json.dumps(row_bad_2) + "\n" + json.dumps(row_good_2) + "\n", encoding="utf-8"
    )

    exit_code = main(
        [
            "--first",
            str(first_file),
            "--second",
            str(second_file),
            "--output",
            str(out_file),
            "--reject-log",
            str(reject_file),
            "--critic",
        ]
    )
    assert exit_code == 0
    out_lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(out_lines) == 1
    assert out_lines[0]["prompt"] == "task_keep"

    reject_lines = [
        json.loads(line) for line in reject_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_lines) == 1
    assert reject_lines[0]["critic_reject_code"] == "DUALDISTILL_DROP_0_0"
    assert reject_lines[0]["prompt"] == "task_drop"


@pytest.mark.unit
@pytest.mark.harness
def test_collect_trajectories_cli_critic_outcome_mismatch(tmp_path, caplog):
    """Verify collect_trajectories CLI logs OUTCOME_MISMATCH and appends to sink."""
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Greet politely", "expected": "hello"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    reject_file = tmp_path / "critic_rejects.jsonl"

    with caplog.at_level("WARNING"):
        code = main(
            [
                "--input",
                str(source),
                "--output",
                str(dest),
                "--harness-id",
                "base_react",
                "--scripted",
                '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"wrong_greeting\\"}}"]',
                "--reject-log",
                str(reject_file),
                "--critic",
            ]
        )
    assert code == 0
    assert dest.read_text(encoding="utf-8").strip() == ""
    assert "OUTCOME_MISMATCH" in caplog.text

    reject_lines = [
        json.loads(line) for line in reject_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_lines) == 1
    assert reject_lines[0]["critic_reject_code"] == "OUTCOME_MISMATCH"
    assert reject_lines[0]["prompt"] == "Greet politely"
    assert reject_lines[0]["expected"] == "hello"
    assert reject_lines[0]["final_answer"] == "wrong_greeting"


@pytest.mark.unit
@pytest.mark.harness
def test_collect_trajectories_critic_unknown_tool_id_injected_drop(tmp_path):
    """Falsifier: injecting an unknown tool_id MUST fail if collect writes the row with critic enabled."""
    from scripts.harness.collect_trajectories import _collect_rows

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Greet politely", "expected": "hello"}) + "\n",
        encoding="utf-8",
    )
    reject_file = tmp_path / "critic_rejects.jsonl"
    telemetry = CriticTelemetry(sink_path=reject_file, enabled=True)

    class InjectedToolRuntime:
        def run(self, task: str, harness_id: str | None = None) -> HarnessRunResult:
            return HarnessRunResult(
                final_answer="hello",
                trajectory=Trajectory(
                    harness_id=harness_id or "base_react",
                    task=task,
                    steps=[
                        Step(tool_id="unknown_injected_tool", action="unknown_injected_tool()"),
                        Step(tool_id="final_answer", action="final_answer(text='hello')"),
                    ],
                    final_answer="hello",
                ),
            )

    rows = _collect_rows(
        InjectedToolRuntime(),
        source,
        "base_react",
        strict=False,
        telemetry=telemetry,
    )
    # Critic must drop the row containing unknown tool_id
    assert rows == []

    reject_lines = [
        json.loads(line) for line in reject_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_lines) == 1
    assert reject_lines[0]["critic_reject_code"] == "SCHEMA_VIOLATION"
    assert "unknown_injected_tool" in reject_lines[0]["metadata"]["reason"]


@pytest.mark.unit
@pytest.mark.harness
def test_collect_keeps_recovery_with_critic(tmp_path):
    """Verify recovery trace is kept and critic_kept_recovery counter increments."""
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Greet politely", "expected": "hello"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    reject_file = tmp_path / "critic_rejects.jsonl"

    code = main(
        [
            "--input",
            str(source),
            "--output",
            str(dest),
            "--harness-id",
            "base_react",
            "--scripted",
            '["not-valid-json", "{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hello\\"}}"]',
            "--reject-log",
            str(reject_file),
            "--critic",
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trajectory"]["final_answer"] == "hello"
    assert "parse_error" in rows[0]["trajectory"]["faults"]
    # No rejects recorded
    assert not reject_file.exists()


@pytest.mark.unit
@pytest.mark.harness
def test_collect_integration_with_echo_backend(tmp_path):
    """Integration test with EchoBackend verifying outcome mismatch emission and telemetry recording."""
    from enhanced_system.harness.backends.echo import EchoBackend
    from enhanced_system.harness.registry import load_spec
    from enhanced_system.harness.runtime import AgentRuntime
    from scripts.harness.collect_trajectories import _collect_rows

    mapping = {
        "mismatch_task": '{"tool": "final_answer", "args": {"text": "echo_actual"}}',
        "matching_task": '{"tool": "final_answer", "args": {"text": "gold_answer"}}',
    }
    backend = EchoBackend(mapping)
    spec = load_spec("base_react")
    runtime = AgentRuntime(backend=backend, spec=spec)

    source = tmp_path / "echo_in.jsonl"
    source.write_text(
        json.dumps({"prompt": "mismatch_task", "expected": "gold_answer"})
        + "\n"
        + json.dumps({"prompt": "matching_task", "expected": "gold_answer"})
        + "\n",
        encoding="utf-8",
    )
    reject_file = tmp_path / "echo_rejects.jsonl"
    telemetry = CriticTelemetry(sink_path=reject_file, enabled=True)

    rows = _collect_rows(
        runtime,
        source,
        "base_react",
        strict=False,
        telemetry=telemetry,
    )
    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["prompt"] == "matching_task"

    reject_lines = [
        json.loads(line) for line in reject_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reject_lines) == 1
    assert reject_lines[0]["critic_reject_code"] == "OUTCOME_MISMATCH"
    assert reject_lines[0]["prompt"] == "mismatch_task"
    assert reject_lines[0]["expected"] == "gold_answer"
    assert reject_lines[0]["final_answer"] == "echo_actual"
    assert telemetry.counters["critic_rejected_outcome_mismatch"] == 1
    assert telemetry.counters["critic_kept_total"] == 1


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
    get_settings.cache_clear()

    # Step 1: Trigger OUTCOME_MISMATCH via collect
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

    # Step 2: Trigger DUALDISTILL_DROP_0_0 via compose
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
            "--critic",
        ]
    )
    assert code == 0

    # Step 3: Validate artifact contents
    assert artifact_path.is_file(), f"{artifact_path} was not created"
    records = [json.loads(line) for line in artifact_path.read_text(encoding="utf-8").splitlines()]
    codes = {r.get("critic_reject_code") for r in records}
    assert "OUTCOME_MISMATCH" in codes, f"Missing OUTCOME_MISMATCH in {codes}"
    assert "DUALDISTILL_DROP_0_0" in codes, f"Missing DUALDISTILL_DROP_0_0 in {codes}"
    for r in records:
        assert "critic_reject_code" in r
        assert "prompt" in r

    # Cleanup temp test inputs
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)
