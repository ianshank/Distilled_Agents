"""Unit tests for format unification, eval, AMD, DualDistill, and SCoRe-SFT."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from enhanced_system.harness.backends.echo import EchoBackend
from enhanced_system.harness.backends.transformers import TransformersBackend
from enhanced_system.harness.dispatch import DispatchError, parse_action, split_thought_action
from enhanced_system.harness.dualdistill import TRANSITION_BOTH, TRANSITION_FIX, compose_pair
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.memory_bank import MemoryBank, build_bank, function_hint, workflow_hint
from enhanced_system.harness.prompt_render import (
    messages_from_steps,
    render_messages,
    render_prompt,
    supervised_spans,
)
from enhanced_system.harness.registry import load_spec
from enhanced_system.harness.runtime import AgentRuntime
from enhanced_system.harness.score import earliest_error_index, preference_pair, review_prompt
from enhanced_system.harness.types import Step, Trajectory
from scripts.training.distill import prompt_render as distill_render
from scripts.training.distill.trajectory_collator import encode_row


@pytest.mark.unit
@pytest.mark.harness
def test_prompt_render_copies_match():
    repo = Path(__file__).resolve().parents[3]
    left = (repo / "enhanced_system" / "harness" / "prompt_render.py").read_text(encoding="utf-8")
    right = (repo / "scripts" / "training" / "distill" / "prompt_render.py").read_text(
        encoding="utf-8"
    )
    assert left.split('"""', 2)[-1] == right.split('"""', 2)[-1]
    messages = [{"role": "user", "content": "task"}]
    assert render_prompt(messages, prefix="pre") == distill_render.render_prompt(
        messages, prefix="pre"
    )
    steps = [{"thought": "T", "action": "A", "observation": "O", "tool_id": "json_schema"}]
    left_msg = messages_from_steps("task", steps, instruction="inst")
    right_msg = distill_render.messages_from_steps("task", steps, instruction="inst")
    assert left_msg == right_msg
    assert supervised_spans("task", steps, instruction="inst") == distill_render.supervised_spans(
        "task", steps, instruction="inst"
    )
    fault_steps = [{"fault": "parse_error", "observation": "bad"}]
    assert "parse_error: bad" in "".join(text for text, _s in supervised_spans("task", fault_steps))
    assert render_prompt([]) == "assistant:"
    assert render_prompt([], prefix="p") == "assistant: p"


@pytest.mark.unit
@pytest.mark.harness
def test_split_thought_action_json_and_call():
    allowed = {"final_answer", "json_schema"}
    thought, action = split_thought_action(
        'Need an answer. {"tool": "final_answer", "args": {"text": "ok"}}',
        allowed,
    )
    assert thought.startswith("Need")
    assert parse_action(action, allowed)[0] == "final_answer"
    thought, action = split_thought_action(
        'plan first\nfinal_answer(text="hi")',
        allowed,
    )
    assert "plan" in thought
    assert parse_action(action, allowed)[0] == "final_answer"
    with pytest.raises(DispatchError):
        split_thought_action("", allowed)
    with pytest.raises(DispatchError, match="no parseable"):
        split_thought_action("not an action", allowed)
    thought, action = split_thought_action(
        'example {"tool": "json_schema", "args": {}}\n'
        '{"tool": "final_answer", "args": {"text": "ok"}}',
        {"final_answer"},
    )
    assert parse_action(action, {"final_answer"})[0] == "final_answer"
    assert "example" in thought
    with pytest.raises(DispatchError, match="no parseable"):
        split_thought_action('not_final_answer(text="x")', {"final_answer"})


@pytest.mark.unit
@pytest.mark.harness
def test_collator_matches_runtime_messages():
    class Tok:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 20 + 1 for ch in text]

    steps = [
        {
            "thought": "T",
            "action": '{"tool": "final_answer", "args": {"text": "a"}}',
            "observation": "a",
            "tool_id": "final_answer",
        }
    ]
    row = {
        "prompt": "task",
        "completion": "x",
        "trajectory": {"instruction": "inst", "steps": steps},
    }
    encoded = encode_row(Tok(), row, max_length=256)
    messages = messages_from_steps("task", steps, instruction="inst")
    rendered = render_messages(messages)
    assert encoded["input_ids"] == Tok().encode(rendered)[:256]
    prompt = TransformersBackend(generate_fn=lambda *_a, **_k: ["x"])._render_prompt(messages, None)
    assert prompt == f"{rendered}\nassistant:"

    class OffsetTok:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 20 + 1 for ch in text]

        def __call__(
            self,
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=None,
            **_kwargs,
        ):
            ids = self.encode(text)
            if max_length:
                ids = ids[:max_length]
            offsets = [(index, index + 1) for index in range(len(ids))]
            return {"input_ids": ids, "offset_mapping": offsets}

    offset_encoded = encode_row(OffsetTok(), row, max_length=256)
    assert offset_encoded["input_ids"] == Tok().encode(rendered)[:256]

    class SubwordTok:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3] if text else []

    with pytest.raises(ValueError, match="offset_mapping"):
        encode_row(SubwordTok(), row, max_length=256)

    class SlowTok:
        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 20 + 1 for ch in text]

        def __call__(self, *_args, **_kwargs):
            raise NotImplementedError("offset mapping")

    slow_encoded = encode_row(SlowTok(), row, max_length=256)
    assert slow_encoded["input_ids"] == Tok().encode(rendered)[:256]


@pytest.mark.unit
@pytest.mark.harness
def test_runtime_splits_thought_and_sends_instruction():
    spec = load_spec("base_react")
    spec.policy.teacher = False
    spec.planning.first_thought_prefix = False
    backend = EchoBackend(['Need a greeting. {"tool": "final_answer", "args": {"text": "hi"}}'])
    result = AgentRuntime(backend, spec=spec, teacher=False).run(
        "Write a short greeting", harness_id="base_react"
    )
    assert result.final_answer == "hi"
    assert "Need" in result.trajectory.steps[0].thought
    assert backend.last_messages[0]["role"] == "system"
    assert result.trajectory.instruction


@pytest.mark.unit
@pytest.mark.harness
def test_resume_and_inject_action():
    spec = load_spec("base_react")
    spec.policy.teacher = False
    spec.planning.first_thought_prefix = False
    runtime = AgentRuntime(
        EchoBackend([]),
        spec=spec,
        teacher=False,
    )
    prefix = [Step(thought="oops", action="", observation="bad", fault="parse_error")]
    result = runtime.run(
        "Write a short greeting",
        harness_id="base_react",
        resume_steps=prefix,
        inject_action='{"tool": "final_answer", "args": {"text": "patched"}}',
    )
    assert result.final_answer == "patched"
    assert result.trajectory.steps[0].fault == "parse_error"
    assert "parse_error" in result.trajectory.faults


@pytest.mark.unit
@pytest.mark.harness
def test_memory_bank_workflow_and_function_hints(tmp_path):
    traj = Trajectory(
        task="Write a short greeting",
        harness_id="base_react",
        final_answer="hi",
        steps=[
            Step(thought="greet briefly", action='final_answer(text="hi")', tool_id="final_answer"),
            Step(tool_id="json_schema", fault="tool_error", observation="missing keys"),
        ],
    )
    bank = build_bank([traj])
    path = tmp_path / "bank.json"
    bank.save(path)
    loaded = MemoryBank.load(path)
    assert workflow_hint(loaded, "Write a short greeting")
    assert workflow_hint(loaded, "Completely unrelated calculus homework") == ""
    assert "avoid" in function_hint(loaded, "json_schema")
    assert workflow_hint(None, "x") == ""
    assert function_hint(loaded, "missing") == ""
    assert function_hint(loaded, "") == ""
    assert function_hint(MemoryBank(functions={"json_schema": []}), "json_schema") == ""
    spec = load_spec("base_react")
    spec.policy.teacher = False
    spec.planning.first_thought_prefix = False
    runtime = AgentRuntime(
        EchoBackend(['{"tool": "final_answer", "args": {"text": "hi"}}']),
        spec=spec,
        teacher=False,
        memory_bank=loaded,
    )
    result = runtime.run("Write a short greeting", harness_id="base_react")
    assert "Workflow:" in result.trajectory.instruction
    skipped = build_bank([Trajectory(task="t", final_answer="", faults=["loop"])])
    assert skipped.workflows == []
    action_only = build_bank(
        [
            Trajectory(
                task="other task",
                final_answer="x",
                steps=[Step(action="final_answer(text='x')", tool_id="final_answer")],
            )
        ]
    )
    assert action_only.workflows
    empty_steps = build_bank(
        [Trajectory(task="t", final_answer="x", steps=[Step(fault="parse_error")])]
    )
    assert empty_steps.workflows == []
    spec_swe = load_spec("swe_codeact")
    spec_swe.policy.teacher = False
    spec_swe.planning.first_thought_prefix = False
    hinted = AgentRuntime(
        EchoBackend(
            [
                '{"tool": "json_schema", "args": {"required": ["a"], "document": {}}}',
                '{"tool": "final_answer", "args": {"text": "ok"}}',
            ]
        ),
        spec=spec_swe,
        teacher=False,
        memory_bank=loaded,
    ).run("Write a short greeting")
    assert "Function memory" in hinted.trajectory.steps[0].observation
    assert hinted.trajectory.steps[0].tool_id == "json_schema"
    assert hinted.trajectory.steps[0].fault == "tool_error"


@pytest.mark.unit
@pytest.mark.harness
def test_factory_skips_missing_memory_bank(tmp_path):
    runtime = HarnessFactory.create(
        {
            "harness_id": "base_react",
            "scripted": ['{"tool": "final_answer", "args": {"text": "hi"}}'],
            "bank_path": str(tmp_path / "missing.json"),
        }
    )
    result = runtime.run("Write a short greeting", harness_id="base_react")
    assert result.final_answer == "hi"
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    runtime = HarnessFactory.create(
        {
            "harness_id": "base_react",
            "scripted": ['{"tool": "final_answer", "args": {"text": "hi"}}'],
            "bank_path": str(bad),
        }
    )
    assert runtime.run("Write a short greeting", harness_id="base_react").final_answer == "hi"
    good = tmp_path / "good.json"
    MemoryBank(workflows=[{"task": "Write a short greeting", "strategy": "be brief"}]).save(good)
    loaded = HarnessFactory.create(
        {
            "harness_id": "base_react",
            "scripted": ['{"tool": "final_answer", "args": {"text": "hi"}}'],
            "bank_path": str(good),
            "teacher": False,
        }
    )
    assert (
        "Workflow:"
        in loaded.run("Write a short greeting", harness_id="base_react").trajectory.instruction
    )


@pytest.mark.unit
@pytest.mark.harness
def test_runtime_loads_bank_from_spec_path(tmp_path):
    bank = MemoryBank(workflows=[{"task": "Write a short greeting", "strategy": "be brief"}])
    path = tmp_path / "bank.json"
    bank.save(path)
    spec = load_spec("base_react")
    spec.policy.teacher = False
    spec.planning.first_thought_prefix = False
    spec.memory.bank_path = str(path)
    result = AgentRuntime(
        EchoBackend(['{"tool": "final_answer", "args": {"text": "hi"}}']),
        spec=spec,
        teacher=False,
    ).run("Write a short greeting", harness_id="base_react")
    assert "Workflow:" in result.trajectory.instruction
    spec.memory.bank_path = str(tmp_path / "missing.json")
    skipped_bank = AgentRuntime(
        EchoBackend(['{"tool": "final_answer", "args": {"text": "hi"}}']),
        spec=spec,
        teacher=False,
    ).run("Write a short greeting", harness_id="base_react")
    assert skipped_bank.final_answer == "hi"


@pytest.mark.unit
@pytest.mark.harness
def test_dualdistill_compose_table():
    expected = "ok"
    bad = {
        "prompt": "p",
        "expected": expected,
        "trajectory": {
            "task": "p",
            "final_answer": "no",
            "steps": [{"thought": "a", "action": "x", "tool_id": "final_answer"}],
        },
    }
    good = {
        "prompt": "p",
        "expected": expected,
        "trajectory": {
            "task": "p",
            "final_answer": "ok",
            "steps": [{"thought": "b", "action": "y", "tool_id": "final_answer"}],
        },
    }
    composed = compose_pair(bad, good, expected=expected)
    assert composed is not None
    assert TRANSITION_FIX in composed["completion"]
    assert "x" in composed["completion"]
    assert "y" in composed["completion"]
    assert compose_pair(bad, bad, expected=expected) is None
    kept = compose_pair(good, bad, expected=expected)
    assert kept is not None
    assert kept["trajectory"]["final_answer"] == "ok"
    unlabeled_good = dict(good)
    unlabeled_good["expected"] = ""
    relabeled = compose_pair(unlabeled_good, bad, expected=expected)
    assert relabeled is not None
    assert relabeled["expected"] == expected
    faulty = dict(bad)
    faulty["trajectory"] = dict(bad["trajectory"])
    faulty["trajectory"]["faults"] = ["parse_error"]
    stitched = compose_pair(faulty, good, expected=expected)
    assert stitched is not None
    assert "parse_error" in stitched["trajectory"]["faults"]
    both = compose_pair(good, good, expected=expected)
    assert both is not None
    assert TRANSITION_BOTH in both["completion"]
    legacy = compose_pair(
        {"prompt": "p", "completion": "ok", "expected": expected},
        {"prompt": "p", "completion": "ok", "expected": expected},
        expected=expected,
    )
    assert legacy is not None


@pytest.mark.unit
@pytest.mark.harness
def test_score_earliest_error_and_prefs():
    recovered = Trajectory(
        task="t",
        final_answer="ok",
        steps=[
            Step(fault="parse_error", observation="bad"),
            Step(action="later", tool_id="final_answer"),
        ],
    )
    assert earliest_error_index(recovered, expected="ok") is None
    student = Trajectory(
        task="t",
        final_answer="",
        steps=[
            Step(fault="parse_error", observation="bad"),
            Step(action="later"),
        ],
    )
    assert earliest_error_index(student, expected="ok") == 1
    corrected = Trajectory(
        task="t",
        final_answer="ok",
        steps=[
            Step(fault="parse_error", observation="bad"),
            Step(
                action='{"tool": "final_answer", "args": {"text": "ok"}}',
                tool_id="final_answer",
            ),
        ],
    )
    pair = preference_pair(student, corrected, 1)
    assert pair is not None
    assert pair["rejected"]["action"] == "later"
    miss = Trajectory(task="t", final_answer="no", steps=[Step(action="x")])
    assert earliest_error_index(miss, expected="yes") == 0
    prompt = review_prompt(miss, expected="yes")
    assert prompt[0]["role"] == "system"
    assert "yes" in prompt[1]["content"]
    assert preference_pair(student, corrected, 99) is None
    looped = Trajectory(task="t", final_answer="", faults=["loop"], steps=[])
    assert earliest_error_index(looped) is None
    assert (
        earliest_error_index(Trajectory(task="t", final_answer="ok", steps=[Step(action="x")]))
        is None
    )
    only_fault = Trajectory(task="t", final_answer="", steps=[Step(fault="parse_error")])
    assert earliest_error_index(only_fault, expected="ok") == 0


@pytest.mark.unit
@pytest.mark.harness
def test_iter_jsonl_dicts_skips_and_strict(tmp_path, caplog):
    from enhanced_system.harness.jsonl import JsonlRowError, iter_jsonl_dicts

    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            [
                "",
                "[1, 2]",
                "{not json",
                '{"prompt": ""}',
                '{"prompt": "ok"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        rows = list(iter_jsonl_dicts(path, require_prompt=True, strict=False))
    assert len(rows) == 1
    assert rows[0][0] == 5
    assert rows[0][1]["prompt"] == "ok"
    assert "skipping line" in caplog.text
    yielded = list(iter_jsonl_dicts(path, require_prompt=False, strict=False))
    prompts = [row["prompt"] for _line, row in yielded]
    assert prompts == ["", "ok"]
    with pytest.raises(JsonlRowError, match="skipping line"):
        list(iter_jsonl_dicts(path, require_prompt=True, strict=True))


@pytest.mark.unit
@pytest.mark.harness
def test_collect_copies_expected_and_strict_mismatch(tmp_path):
    from enhanced_system.harness.convert import trajectory_to_legacy
    from scripts.harness.collect_trajectories import main as collect_main

    traj = Trajectory(task="t", final_answer="hi", steps=[Step(action="a")])
    assert trajectory_to_legacy(traj, expected="hi")["expected"] == "hi"
    assert "expected" not in trajectory_to_legacy(traj)

    source = tmp_path / "in.jsonl"
    dest = tmp_path / "out.jsonl"
    source.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "hi"}) + "\n",
        encoding="utf-8",
    )
    code = collect_main(
        [
            "--input",
            str(source),
            "--output",
            str(dest),
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
        ]
    )
    assert code == 0
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert row["expected"] == "hi"

    source.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "nope"}) + "\n",
        encoding="utf-8",
    )
    code = collect_main(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "strict.jsonl"),
            "--harness-id",
            "base_react",
            "--strict",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
        ]
    )
    assert code == 1


@pytest.mark.unit
@pytest.mark.harness
def test_eval_skips_unlabeled_threshold_and_strict(tmp_path, caplog, capsys, monkeypatch):
    from enhanced_system.ops.settings import get_settings
    from scripts.harness.eval_harness import main as eval_main

    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        "\n".join(
            [
                '{"prompt": ""}',
                "not-json",
                json.dumps({"prompt": "Write a short greeting", "expected": "hi"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        code = eval_main(
            [
                "--input",
                str(mixed),
                "--harness-id",
                "base_react",
                "--scripted",
                '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
                "--threshold",
                "50",
            ]
        )
    assert code == 0
    assert "skipping line" in caplog.text
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 1

    unlabeled = tmp_path / "unlabeled.jsonl"
    unlabeled.write_text(json.dumps({"prompt": "Write a short greeting"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("MANGOMAS_HARNESS_MAX_STEPS", "1")
    get_settings.cache_clear()
    try:
        code = eval_main(
            [
                "--input",
                str(unlabeled),
                "--harness-id",
                "base_react",
                "--scripted",
                '["not-json"]',
                "--threshold",
                "85",
            ]
        )
        assert code == 1
        summary = json.loads(capsys.readouterr().out)
        assert summary["total"] == 1
        assert summary["with_expected"] == 0
        assert summary["truncated"] == 1

        mixed_gate = tmp_path / "mix.jsonl"
        mixed_gate.write_text(
            json.dumps({"prompt": "Write a short greeting", "expected": "hi"})
            + "\n"
            + json.dumps({"prompt": "Write a short greeting"})
            + "\n",
            encoding="utf-8",
        )
        mix_code = eval_main(
            [
                "--input",
                str(mixed_gate),
                "--harness-id",
                "base_react",
                "--scripted",
                json.dumps(['{"tool": "final_answer", "args": {"text": "hi"}}', "not-json"]),
                "--threshold",
                "85",
            ]
        )
        assert mix_code == 1
        mix_summary = json.loads(capsys.readouterr().out)
        assert mix_summary["total"] == 2
        assert mix_summary["with_expected"] == 1
        assert mix_summary["pass_rate"] == 50.0
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()

    bad = tmp_path / "bad.jsonl"
    bad.write_text("not-json\n", encoding="utf-8")
    assert eval_main(["--input", str(bad), "--harness-id", "base_react", "--strict"]) == 1

    def _boom(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("scripts.harness.eval_harness.iter_jsonl_dicts", _boom)
    readable = tmp_path / "readable.jsonl"
    readable.write_text(json.dumps({"prompt": "Write a short greeting"}) + "\n", encoding="utf-8")
    assert eval_main(["--input", str(readable), "--harness-id", "base_react"]) == 1


@pytest.mark.unit
@pytest.mark.harness
def test_score_empty_generate_strict_and_non_dict(tmp_path, caplog):
    from scripts.harness.collect_score import main as score_main

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "ok"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "score.jsonl"
    with caplog.at_level(logging.WARNING):
        code = score_main(
            [
                "--input",
                str(prompts),
                "--output",
                str(out),
                "--harness-id",
                "base_react",
                "--student-scripted",
                '["not-json"]',
                "--teacher-scripted",
                '[""]',
            ]
        )
    assert code == 0
    assert out.read_text(encoding="utf-8").strip() == ""
    assert "teacher produced no correction" in caplog.text

    success = tmp_path / "ok.jsonl"
    success.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "hi"}) + "\n",
        encoding="utf-8",
    )
    kept = tmp_path / "kept.jsonl"
    assert (
        score_main(
            [
                "--input",
                str(success),
                "--output",
                str(kept),
                "--harness-id",
                "base_react",
                "--student-scripted",
                '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
                "--teacher-scripted",
                '[""]',
            ]
        )
        == 0
    )
    assert json.loads(kept.read_text(encoding="utf-8"))["expected"] == "hi"

    blank = tmp_path / "blank.jsonl"
    blank.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "   "}) + "\n",
        encoding="utf-8",
    )
    blank_out = tmp_path / "blank-out.jsonl"
    assert (
        score_main(
            [
                "--input",
                str(blank),
                "--output",
                str(blank_out),
                "--harness-id",
                "base_react",
                "--student-scripted",
                '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
                "--teacher-scripted",
                '[""]',
            ]
        )
        == 0
    )
    assert json.loads(blank_out.read_text(encoding="utf-8"))["trajectory"]["final_answer"] == "hi"

    prompts.write_text("not-json\n", encoding="utf-8")
    assert (
        score_main(
            [
                "--input",
                str(prompts),
                "--output",
                str(tmp_path / "strict.jsonl"),
                "--harness-id",
                "base_react",
                "--strict",
            ]
        )
        == 1
    )

    prompts.write_text("[1, 2]\n", encoding="utf-8")
    skipped = tmp_path / "skip.jsonl"
    assert (
        score_main(
            [
                "--input",
                str(prompts),
                "--output",
                str(skipped),
                "--harness-id",
                "base_react",
            ]
        )
        == 0
    )
    assert skipped.read_text(encoding="utf-8").strip() == ""


@pytest.mark.unit
@pytest.mark.harness
def test_compose_mismatch_and_duplicate_prompt(tmp_path, caplog):
    from scripts.harness.compose_dualdistill import main as compose_main

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    out = tmp_path / "c.jsonl"
    first.write_text(json.dumps({"prompt": "a", "expected": "x", "trajectory": {}}) + "\n")
    second.write_text(json.dumps({"prompt": "b", "expected": "x", "trajectory": {}}) + "\n")
    with caplog.at_level(logging.WARNING):
        assert (
            compose_main(["--first", str(first), "--second", str(second), "--output", str(out)])
            == 0
        )
    assert out.read_text(encoding="utf-8").strip() == ""
    assert "unmatched" in caplog.text

    row_bad = {
        "prompt": "p",
        "expected": "ok",
        "trajectory": {
            "task": "p",
            "final_answer": "no",
            "steps": [{"thought": "a", "action": "x"}],
        },
    }
    row_good = {
        "prompt": "p",
        "expected": "ok",
        "trajectory": {
            "task": "p",
            "final_answer": "ok",
            "steps": [{"thought": "b", "action": "y"}],
        },
    }
    first.write_text(json.dumps(row_bad) + "\n" + json.dumps(row_good) + "\n", encoding="utf-8")
    second.write_text(json.dumps(row_good) + "\n", encoding="utf-8")
    composed = tmp_path / "d.jsonl"
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert (
            compose_main(
                ["--first", str(first), "--second", str(second), "--output", str(composed)]
            )
            == 0
        )
    assert "duplicate prompt" in caplog.text
    assert composed.read_text(encoding="utf-8").strip()


@pytest.mark.unit
@pytest.mark.harness
def test_empty_instruction_ftp_resume_inject_and_parse_error():
    spec = load_spec("base_react")
    spec.policy.teacher = False
    spec.planning.first_thought_prefix = False
    spec.planning.instruction = ""
    backend = EchoBackend(['{"tool": "final_answer", "args": {"text": "hi"}}'])
    AgentRuntime(backend, spec=spec, teacher=False).run(
        "Write a short greeting", harness_id="base_react"
    )
    assert backend.last_messages[0]["role"] == "user"

    class CaptureEcho(EchoBackend):
        def __init__(self) -> None:
            super().__init__(
                ["first thought", '{"tool": "final_answer", "args": {"text": "done"}}']
            )
            self.calls: list[list[dict[str, str]]] = []

        def generate(self, messages, *, prefix=None, n=1, temperature=None):
            self.calls.append([dict(item) for item in messages])
            return super().generate(messages, prefix=prefix, n=n, temperature=temperature)

    swe = load_spec("swe_codeact")
    capture = CaptureEcho()
    AgentRuntime(capture, spec=swe, teacher=True).run(
        "Explain a ping endpoint", harness_id="swe_codeact"
    )
    assert capture.calls[0][0]["role"] == "system"
    assert swe.planning.instruction.strip() in capture.calls[0][0]["content"]

    ftp_spec = load_spec("swe_codeact")
    ftp_spec.planning.first_thought_prefix = True
    unused = EchoBackend(["SHOULD_NOT_BE_USED"])
    injected = AgentRuntime(unused, spec=ftp_spec, teacher=True).run(
        "Explain a ping endpoint",
        harness_id="swe_codeact",
        resume_steps=[],
        inject_action='{"tool": "final_answer", "args": {"text": "from-inject"}}',
    )
    assert unused.call_count == 0
    assert injected.final_answer == "from-inject"
    assert injected.trajectory.steps[0].action == (
        '{"tool": "final_answer", "args": {"text": "from-inject"}}'
    )

    recover = EchoBackend(['{"tool": "final_answer", "args": {"text": "ok"}}'])
    parsed = AgentRuntime(recover, spec=spec, teacher=False).run(
        "Write a short greeting",
        harness_id="base_react",
        inject_action="not-json",
    )
    assert parsed.trajectory.steps[0].fault == "parse_error"
    assert parsed.final_answer == "ok"


@pytest.mark.unit
@pytest.mark.harness
def test_factory_logs_skipped_memory_bank(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        HarnessFactory.create(
            {
                "harness_id": "base_react",
                "scripted": ['{"tool": "final_answer", "args": {"text": "hi"}}'],
                "bank_path": str(tmp_path / "missing.json"),
            }
        )
    assert "memory bank skipped" in caplog.text


@pytest.mark.unit
@pytest.mark.harness
def test_score_skips_failed_inject_and_truncates_prefs(tmp_path, monkeypatch):
    from enhanced_system.ops.settings import get_settings
    from scripts.harness.collect_score import main as score_main

    monkeypatch.setenv("MANGOMAS_HARNESS_MAX_STEPS", "1")
    get_settings.cache_clear()
    prompts = tmp_path / "in.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "ok"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text('{"stale": true}\n', encoding="utf-8")
    try:
        code = score_main(
            [
                "--input",
                str(prompts),
                "--output",
                str(out),
                "--harness-id",
                "base_react",
                "--student-scripted",
                '["not-json"]',
                "--teacher-scripted",
                '["not-json"]',
                "--prefs",
                str(prefs),
            ]
        )
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
    assert code == 0
    assert out.read_text(encoding="utf-8").strip() == ""
    assert prefs.read_text(encoding="utf-8") == ""


@pytest.mark.unit
@pytest.mark.harness
def test_eval_skips_runtime_value_error(tmp_path, monkeypatch, capsys):
    from scripts.harness.eval_harness import main as eval_main

    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps({"prompt": "Write a short greeting"}) + "\n", encoding="utf-8")

    def _boom(self, *_args, **_kwargs):
        raise ValueError("too long")

    monkeypatch.setattr("enhanced_system.harness.runtime.AgentRuntime.run", _boom)
    assert (
        eval_main(["--input", str(source), "--harness-id", "base_react", "--threshold", "0"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 0
    assert eval_main(["--input", str(source), "--harness-id", "base_react", "--strict"]) == 1
