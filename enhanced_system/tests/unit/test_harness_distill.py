"""Unit tests for format unification, eval, AMD, DualDistill, and SCoRe-SFT."""

from __future__ import annotations

import json
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
    assert TRANSITION_FIX in json.dumps(composed)
    assert compose_pair(bad, bad, expected=expected) is None
    kept = compose_pair(good, bad, expected=expected)
    assert kept is not None
    assert kept["trajectory"]["final_answer"] == "ok"
    both = compose_pair(good, good, expected=expected)
    assert both is not None
    assert TRANSITION_BOTH in json.dumps(both)
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
