"""Unit tests for the MangoMAS agent harness."""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from enhanced_system.config import load_config
from enhanced_system.harness.adapters import RouterAdapter, SkillEvalAdapter, harness_id_for_agent
from enhanced_system.harness.backends.echo import EchoBackend
from enhanced_system.harness.backends.transformers import TransformersBackend
from enhanced_system.harness.convert import legacy_to_stub_trajectory, trajectory_to_legacy
from enhanced_system.harness.dispatch import DispatchError, parse_action
from enhanced_system.harness.factory import HarnessFactory
from enhanced_system.harness.registry import (
    apply_settings,
    load_schema,
    load_spec,
    resolve_spec_path,
    validate_payload,
)
from enhanced_system.harness.runtime import AgentRuntime
from enhanced_system.harness.tailor import HarnessTailor
from enhanced_system.harness.tools.registry import get_tool
from enhanced_system.harness.traces import JsonlTraceStore
from enhanced_system.harness.types import HarnessSpec, Step, Trajectory
from enhanced_system.ops.sagemaker_launcher import MangoMASSageMakerLauncher
from enhanced_system.ops.settings import get_settings
from scripts.training.distill.sagemaker_io import texts_from_examples
from scripts.training.distill.trajectory_collator import TrajectoryDataCollator, encode_row

REPO = Path(__file__).resolve().parents[3]
HARNESS_ROOT = REPO / "enhanced_system" / "harness"
SCRIPTS_HARNESS = REPO / "scripts" / "harness"
FORBIDDEN = (
    r"us-east-1",
    r"(?<![A-Za-z])ml\.[a-z0-9]",
    r"mistralai/",
    r"DialoGPT",
    r"s3://",
)


@pytest.fixture(autouse=True)
def _clear_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.unit
@pytest.mark.harness
def test_config_harness_pointers_default():
    config = load_config("default")
    assert config.harness.enabled is True
    assert config.harness.default_id == ""


@pytest.mark.unit
@pytest.mark.harness
def test_load_spec_inherits_settings(monkeypatch):
    monkeypatch.setenv("MANGOMAS_HARNESS_MAX_STEPS", "5")
    get_settings.cache_clear()
    spec = load_spec("base_react")
    assert spec.id == "base_react"
    assert spec.planning.max_steps == 5
    assert spec.memory.window_turns == get_settings().harness_memory_window


@pytest.mark.unit
@pytest.mark.harness
def test_yaml_caps_max_steps_to_settings(monkeypatch):
    monkeypatch.setenv("MANGOMAS_HARNESS_MAX_STEPS", "3")
    get_settings.cache_clear()
    spec = HarnessSpec(id="tmp", planning={"max_steps": 99}, action={"tool_ids": ["final_answer"]})
    filled = apply_settings(spec)
    assert filled.planning.max_steps == 3


@pytest.mark.unit
@pytest.mark.harness
def test_schema_lists_required_keys():
    schema = load_schema()
    assert "id" in schema["required"]
    assert "schema_version" in schema["required"]


@pytest.mark.unit
@pytest.mark.harness
def test_unknown_tool_id_in_spec_rejected():
    spec = HarnessSpec(id="tmp", action={"tool_ids": ["not_a_tool"]})
    with pytest.raises(DispatchError):
        apply_settings(spec)


@pytest.mark.unit
@pytest.mark.harness
def test_json_and_ast_dispatch():
    allowed = {"json_schema", "final_answer"}
    tool_id, args = parse_action(
        '{"tool": "json_schema", "args": {"required": ["a"], "document": {"a": 1}}}',
        allowed,
    )
    assert tool_id == "json_schema"
    tool_id, args = parse_action('final_answer(text="ok")', allowed)
    assert tool_id == "final_answer"
    assert args["text"] == "ok"


@pytest.mark.unit
@pytest.mark.harness
def test_dispatch_rejects_exec_trees():
    allowed = {"final_answer"}
    with pytest.raises(DispatchError):
        parse_action("import os", allowed)
    with pytest.raises(DispatchError):
        parse_action("__import__('os').system('x')", allowed)
    with pytest.raises(DispatchError):
        parse_action("final_answer(text=open('x'))", allowed)


@pytest.mark.unit
@pytest.mark.harness
def test_tools_valid_and_invalid():
    assert json.loads(get_tool("json_schema").run({"required": ["a"], "document": {"a": 1}}))["ok"]
    with pytest.raises(ValueError):
        get_tool("json_schema").run({"required": ["a"], "document": {}})
    assert json.loads(get_tool("pytest_runner").run({"suite": ["test_a"]}))["count"] == 1
    with pytest.raises(ValueError):
        get_tool("pytest_runner").run({"suite": []})
    assert json.loads(get_tool("sqe_checklist").run({"assertions": ["assert status == 200"]}))["ok"]
    with pytest.raises(ValueError):
        get_tool("sqe_checklist").run({"assertions": ["nope"]})
    assert (
        json.loads(
            get_tool("architect_structure").run({"nodes": ["api"], "edges": [["api", "db"]]})
        )["nodes"]
        == 1
    )
    assert json.loads(get_tool("pm_acceptance").run({"criteria": ["REQ-1: login"]}))["ids"] == [
        "REQ-1"
    ]
    with pytest.raises(ValueError):
        get_tool("final_answer").run({"text": "  "})


@pytest.mark.unit
@pytest.mark.harness
def test_runtime_reaches_final_answer():
    backend = EchoBackend(['{"tool": "final_answer", "args": {"text": "hello"}}'])
    runtime = HarnessFactory.create({"harness_id": "base_react", "backend": backend})
    result = runtime.run("Write a short greeting", harness_id="base_react")
    assert result.final_answer == "hello"
    assert result.truncated is False


@pytest.mark.unit
@pytest.mark.harness
def test_ftp_prefix_only_for_teacher():
    backend = EchoBackend(
        [
            "first thought",
            '{"tool": "final_answer", "args": {"text": "done"}}',
        ]
    )
    runtime = AgentRuntime(backend, spec=load_spec("swe_codeact"), teacher=True)
    result = runtime.run("Explain a ping endpoint", harness_id="swe_codeact")
    assert backend.call_count >= 2
    assert result.final_answer == "done"
    assert backend.last_prefix == "first thought"


@pytest.mark.unit
@pytest.mark.harness
def test_sag_picks_parse_valid_sample():
    class Flaky(EchoBackend):
        def generate(self, messages, *, prefix=None, n=1, temperature=None):
            self.call_count += 1
            self.last_prefix = prefix
            return ["not-json", '{"tool": "final_answer", "args": {"text": "ok"}}'][:n]

    runtime = AgentRuntime(Flaky(), spec=load_spec("base_react"), teacher=False)
    # Force two samples via spec copy
    spec = load_spec("base_react")
    spec.policy.sag_samples = 2
    runtime.spec = spec
    result = runtime.run("Say ok", harness_id="base_react")
    assert result.final_answer == "ok"


@pytest.mark.unit
@pytest.mark.harness
def test_loop_fault_on_max_steps(monkeypatch):
    monkeypatch.setenv("MANGOMAS_HARNESS_MAX_STEPS", "1")
    get_settings.cache_clear()
    spec = load_spec("swe_codeact")
    spec.policy.teacher = False
    spec.planning.first_thought_prefix = False
    backend = EchoBackend(
        ['{"tool": "json_schema", "args": {"required": ["a"], "document": {"a": 1}}}']
    )
    runtime = AgentRuntime(backend, spec=spec)
    result = runtime.run("Describe a schema", harness_id="swe_codeact")
    assert result.truncated is True
    assert "loop" in result.trajectory.faults


@pytest.mark.unit
@pytest.mark.harness
def test_convert_keeps_prompt_completion():
    traj = Trajectory(
        task="t",
        final_answer="a",
        steps=[Step(thought="think", action='final_answer(text="a")', observation="hello")],
    )
    row = trajectory_to_legacy(traj)
    assert set(row) >= {"prompt", "completion", "trajectory"}
    assert "hello" not in row["completion"]
    assert texts_from_examples(row) == f"{row['prompt']}\n{row['completion']}"
    extra = dict(row)
    extra["noise"] = "x"
    assert (
        texts_from_examples(
            {
                "prompt": extra["prompt"],
                "completion": extra["completion"],
                "trajectory": extra["trajectory"],
            }
        )
        == f"{row['prompt']}\n{row['completion']}"
    )
    stub = legacy_to_stub_trajectory({"prompt": "p", "completion": "c"})
    assert stub.task == "p"


@pytest.mark.unit
@pytest.mark.harness
def test_trace_store_appends(tmp_path):
    store = JsonlTraceStore(tmp_path / "t.jsonl")
    store.append(Trajectory(task="x", final_answer="y"))
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["task"] == "x"


@pytest.mark.unit
@pytest.mark.harness
def test_skill_eval_adapter_returns_final_answer():
    backend = EchoBackend(['{"tool": "final_answer", "args": {"text": "passed"}}'])
    runtime = HarnessFactory.create({"harness_id": "base_react", "backend": backend})
    adapter = SkillEvalAdapter(runtime)
    assert adapter("Write a short greeting") == "passed"


@pytest.mark.unit
@pytest.mark.harness
def test_router_adapter_reads_harness_id():
    assert harness_id_for_agent("swe_agent").endswith("swe_codeact")
    adapter = RouterAdapter()
    harness_id = adapter.resolve("Write a Python function to add numbers")
    assert harness_id


@pytest.mark.unit
@pytest.mark.harness
def test_tailor_clamps_and_archives_without_apply(tmp_path):
    spec = load_spec("base_react")
    spec.planning.max_steps = 2
    traj = Trajectory(task="t", faults=["loop"], steps=[])
    proposed = HarnessTailor().propose(spec, [traj])
    assert proposed.planning.max_steps == get_settings().harness_max_steps
    path = HarnessTailor().persist(proposed, archive_dir=tmp_path / "archive", apply_patches=False)
    assert path.is_file()
    live = tmp_path / "live.yaml"
    HarnessTailor().persist(
        proposed, archive_dir=tmp_path / "archive2", apply_patches=False, live_path=live
    )
    assert not live.exists()


@pytest.mark.unit
@pytest.mark.harness
def test_tailor_drops_repeated_tool_errors():
    spec = load_spec("swe_codeact")
    steps = [
        Step(tool_id="json_schema", fault="tool_error"),
        Step(tool_id="json_schema", fault="tool_error"),
        Step(tool_id="pytest_runner"),
    ]
    proposed = HarnessTailor().propose(spec, [Trajectory(task="t", steps=steps)])
    assert "json_schema" not in proposed.action.tool_ids
    assert "pytest_runner" in proposed.action.tool_ids


@pytest.mark.unit
@pytest.mark.harness
def test_collator_masks_observations():
    class Tok:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 20 + 1 for ch in text]

    row = {
        "prompt": "P",
        "completion": "ignored",
        "trajectory": {
            "steps": [
                {"thought": "T", "action": "A", "observation": "OBS"},
            ]
        },
    }
    encoded = encode_row(Tok(), row, max_length=64)
    obs_ids = Tok().encode("OBS")
    labeled = Tok().encode("T A")
    # observation ids appear after prompt + labeled
    assert encoded["labels"][-len(obs_ids) :] == [-100] * len(obs_ids)
    assert (
        encoded["labels"][len(Tok().encode("P")) : len(Tok().encode("P")) + len(labeled)] == labeled
    )
    collator = TrajectoryDataCollator(Tok(), max_length=64)
    batch = collator([row])
    assert "labels" in batch


@pytest.mark.unit
@pytest.mark.harness
def test_collator_without_trajectory_uses_completion():
    class Tok:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [1, 2] if text else []

    encoded = encode_row(Tok(), {"prompt": "p", "completion": "c"}, max_length=16)
    assert -100 in encoded["labels"]
    assert 1 in encoded["labels"] or 2 in encoded["labels"]


@pytest.mark.unit
@pytest.mark.harness
def test_launcher_job_spec_keys_unchanged(tmp_path):
    launcher = MangoMASSageMakerLauncher(data_dir=tmp_path)
    spec = launcher.create_job_spec(launcher.agent_configs[0])
    keys = spec["hyperparameters"]
    assert "teacher_model_name" in keys
    assert "student_model_name" in keys
    assert "model_name_or_path" not in keys
    assert "trajectory_mode" not in keys


@pytest.mark.unit
@pytest.mark.harness
def test_transformers_backend_requires_inject():
    backend = TransformersBackend()
    with pytest.raises(ImportError):
        backend.generate([{"role": "user", "content": "x"}])
    backend = TransformersBackend(generate_fn=lambda messages, **kwargs: ["ok"])
    assert backend.generate([{"role": "user", "content": "x"}]) == ["ok"]


@pytest.mark.unit
@pytest.mark.harness
def test_no_hardcoded_cloud_literals():
    pattern = re.compile("|".join(FORBIDDEN))
    roots = [HARNESS_ROOT, SCRIPTS_HARNESS]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, offenders


@pytest.mark.unit
@pytest.mark.harness
def test_run_agent_cli(tmp_path, capsys):
    from scripts.harness.run_agent import main

    code = main(
        [
            "--task",
            "Write a short greeting",
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["final_answer"] == "hi"


@pytest.mark.unit
@pytest.mark.harness
def test_collect_trajectories_cli(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text('{"prompt": "Write a short greeting"}\n', encoding="utf-8")
    dest = tmp_path / "out.jsonl"
    code = main(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
        ]
    )
    assert code == 0
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert "prompt" in row and "completion" in row


@pytest.mark.unit
@pytest.mark.harness
def test_validator_rejects_empty_task():
    runtime = HarnessFactory.create({"harness_id": "base_react"})
    with pytest.raises(ValueError):
        runtime.run("   ", harness_id="base_react")


@pytest.mark.unit
@pytest.mark.harness
def test_harness_python_is_parseable():
    for path in HARNESS_ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.unit
@pytest.mark.harness
def test_package_exports():
    from enhanced_system import harness
    from enhanced_system.harness.backends import EchoBackend as ExportedEcho
    from enhanced_system.harness.policies import generate_action, maybe_prefix
    from enhanced_system.harness.tools import TOOL_REGISTRY as ExportedTools

    assert harness.HarnessFactory is HarnessFactory
    assert ExportedEcho is EchoBackend
    assert callable(maybe_prefix) and callable(generate_action)
    assert "final_answer" in ExportedTools


@pytest.mark.unit
@pytest.mark.harness
def test_dispatch_json_and_ast_error_paths():
    allowed = {"json_schema", "final_answer"}
    with pytest.raises(DispatchError):
        parse_action("   ", allowed)
    with pytest.raises(DispatchError):
        parse_action("{", allowed)
    with pytest.raises(DispatchError):
        parse_action("[]", allowed)
    with pytest.raises(DispatchError):
        parse_action('{"args": {}}', allowed)
    with pytest.raises(DispatchError):
        parse_action('{"tool": "final_answer", "args": []}', allowed)
    with pytest.raises(DispatchError):
        parse_action('{"tool": "not_a_tool", "args": {}}', allowed)
    with pytest.raises(DispatchError):
        parse_action('pytest_runner(suite=["a"])', allowed)
    tool_id, args = parse_action(
        'json_schema(required=["a"], document={"a": 1, "nested": [1, 2]})',
        allowed,
    )
    assert tool_id == "json_schema"
    assert args["document"]["nested"] == [1, 2]
    with pytest.raises(DispatchError):
        parse_action("final_answer(*args)", allowed)
    with pytest.raises(DispatchError):
        parse_action("final_answer(text=unknown)", allowed)


@pytest.mark.unit
@pytest.mark.harness
def test_schema_rejects_extra_and_empty_id():
    schema = load_schema()
    with pytest.raises(ValueError, match="unknown keys"):
        validate_payload({"id": "x", "schema_version": "1", "nope": 1}, schema)
    with pytest.raises(ValueError, match="minLength"):
        validate_payload({"id": "", "schema_version": "1"}, schema)
    with pytest.raises(ValueError, match="missing required"):
        validate_payload({"schema_version": "1"}, schema)
    with pytest.raises(ValueError, match="one of"):
        validate_payload(
            {"id": "x", "schema_version": "1", "planning": {"style": "nope"}},
            schema,
        )


@pytest.mark.unit
@pytest.mark.harness
def test_load_spec_rejects_non_mapping_and_prefers_explicit_dir(tmp_path, monkeypatch):
    (tmp_path / "bad.yaml").write_text("- not a mapping\n", encoding="utf-8")
    monkeypatch.setenv("MANGOMAS_HARNESS_DIR", str(tmp_path))
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="mapping"):
        load_spec("bad")
    (tmp_path / "base_react.yaml").write_text(
        'id: base_react\nschema_version: "1"\naction:\n  tool_ids: [final_answer]\n',
        encoding="utf-8",
    )
    spec = load_spec("base_react")
    assert spec.action.tool_ids == ["final_answer"]
    assert resolve_spec_path("base_react").parent == tmp_path


@pytest.mark.unit
@pytest.mark.harness
def test_load_schema_missing(monkeypatch):
    monkeypatch.setattr(
        "enhanced_system.harness.registry.harness_search_dirs",
        lambda settings=None: [],
    )
    with pytest.raises(FileNotFoundError, match="schema.json"):
        load_schema()
    with pytest.raises(FileNotFoundError, match="harness id is required"):
        resolve_spec_path("")


@pytest.mark.unit
@pytest.mark.harness
def test_runtime_store_tool_error_memory_and_reload(tmp_path):
    store = JsonlTraceStore(tmp_path / "trace.jsonl")
    calls = [
        "first thought",
        '{"tool": "json_schema", "args": {"required": ["a"], "document": {}}}',
        '{"tool": "final_answer", "args": {"text": "recovered"}}',
    ]
    backend = EchoBackend(calls)
    runtime = AgentRuntime(backend, spec=load_spec("base_react"), store=store)
    result = runtime.run("Write a short greeting", harness_id="swe_codeact")
    assert result.harness_id == "swe_codeact"
    assert result.final_answer == "recovered"
    assert "tool_error" in result.trajectory.faults
    assert (tmp_path / "trace.jsonl").is_file()

    spec = load_spec("swe_codeact")
    spec.memory.write_observations = False
    spec.memory.window_turns = 1
    backend = EchoBackend(
        [
            '{"tool": "json_schema", "args": {"required": ["a"], "document": {"a": 1}}}',
            '{"tool": "final_answer", "args": {"text": "ok"}}',
        ]
    )
    result = AgentRuntime(backend, spec=spec).run("Write a short greeting")
    assert result.final_answer == "ok"

    spec.memory.write_observations = True
    spec.memory.window_turns = 0
    backend = EchoBackend(
        [
            '{"tool": "json_schema", "args": {"required": ["a"], "document": {"a": 1}}}',
            '{"tool": "final_answer", "args": {"text": "ok"}}',
        ]
    )
    result = AgentRuntime(backend, spec=spec).run("Write a short greeting")
    assert result.final_answer == "ok"


@pytest.mark.unit
@pytest.mark.harness
def test_ftp_empty_prefix_and_sag_no_valid():
    class EmptyThenAnswer(EchoBackend):
        def generate(self, messages, *, prefix=None, n=1, temperature=None):
            self.call_count += 1
            self.last_prefix = prefix
            if self.call_count == 1:
                return []
            return ['{"tool": "final_answer", "args": {"text": "seedless"}}']

    spec = load_spec("swe_codeact")
    runtime = AgentRuntime(EmptyThenAnswer(), spec=spec, teacher=True)
    result = runtime.run("Write a short greeting", harness_id="swe_codeact")
    assert result.final_answer == "seedless"

    class Invalid(EchoBackend):
        def generate(self, messages, *, prefix=None, n=1, temperature=None):
            self.call_count += 1
            return ["not-json"] * max(n, 1)

    spec = load_spec("base_react")
    spec.policy.sag_samples = 0
    spec.policy.sag_temperature = 0.5
    runtime = AgentRuntime(Invalid(), spec=spec, teacher=False)
    result = runtime.run("Write a short greeting", harness_id="base_react")
    assert result.truncated is True
    assert "parse_error" in result.trajectory.faults


@pytest.mark.unit
@pytest.mark.harness
def test_tools_string_document_and_invalid_payloads():
    assert json.loads(get_tool("json_schema").run({"required": ["a"], "document": '{"a": 1}'}))[
        "ok"
    ]
    with pytest.raises(ValueError):
        get_tool("json_schema").run({"required": [], "document": "{"})
    with pytest.raises(ValueError):
        get_tool("json_schema").run({"required": [], "document": [1]})
    with pytest.raises(ValueError):
        get_tool("architect_structure").run({"nodes": [], "edges": []})
    with pytest.raises(ValueError):
        get_tool("architect_structure").run({"nodes": ["api"], "edges": "nope"})
    with pytest.raises(ValueError):
        get_tool("pm_acceptance").run({"criteria": []})
    with pytest.raises(ValueError):
        get_tool("pm_acceptance").run({"criteria": [""]})
    with pytest.raises(KeyError):
        get_tool("missing")
    echo = EchoBackend()
    assert echo.generate([{"role": "user", "content": "x"}], n=0)


@pytest.mark.unit
@pytest.mark.harness
def test_convert_stub_passthrough_and_length_warning(caplog):
    existing = Trajectory(task="p", final_answer="c", steps=[Step(thought="keep")])
    stub = legacy_to_stub_trajectory(
        {"prompt": "p", "completion": "c", "trajectory": existing.model_dump()}
    )
    assert stub.steps[0].thought == "keep"
    stub = legacy_to_stub_trajectory({"prompt": "p", "completion": "c"})
    row = trajectory_to_legacy(stub)
    assert row["completion"] == "c"
    long_traj = Trajectory(task="t" * 400, final_answer="a" * 200)
    with caplog.at_level(logging.WARNING):
        trajectory_to_legacy(long_traj)
    assert "harness_max_length" in caplog.text


@pytest.mark.unit
@pytest.mark.harness
def test_tailor_apply_live_and_keep_last_tool(tmp_path):
    spec = load_spec("swe_codeact")
    proposed = HarnessTailor().propose(spec, [Trajectory(task="ok", steps=[])])
    assert proposed.action.tool_ids == spec.action.tool_ids
    live = tmp_path / "live.yaml"
    archive = HarnessTailor().persist(
        proposed,
        archive_dir=tmp_path / "archive",
        apply_patches=True,
        live_path=live,
    )
    assert archive.is_file()
    assert live.is_file()
    only = HarnessSpec(id="tmp", action={"tool_ids": ["json_schema"]})
    steps = [Step(tool_id="json_schema", fault="tool_error")] * 2
    kept = HarnessTailor().propose(only, [Trajectory(task="t", steps=steps)])
    assert "json_schema" in kept.action.tool_ids


@pytest.mark.unit
@pytest.mark.harness
def test_factory_settings_store_and_adapters(tmp_path, monkeypatch):
    monkeypatch.setenv("MANGOMAS_HARNESS_ID", "base_react")
    get_settings.cache_clear()
    store = JsonlTraceStore(tmp_path / "s.jsonl")
    runtime = HarnessFactory.create({"store": store})
    assert runtime.spec.id == "base_react"
    runtime.run("Write a short greeting")
    assert (tmp_path / "s.jsonl").is_file()
    assert harness_id_for_agent("missing_agent") == ""
    assert harness_id_for_agent("swe_agent", str(tmp_path / "missing.json")) == ""
    adapter = RouterAdapter()
    monkeypatch.setattr(
        adapter.router,
        "route_task",
        lambda _task: SimpleNamespace(selected_agents=[]),
    )
    assert adapter.resolve("Write a short greeting") == get_settings().harness_id


@pytest.mark.unit
@pytest.mark.harness
def test_collator_empty_and_non_trajectory():
    class Tok:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [] if not text else [3]

    collator = TrajectoryDataCollator(Tok(), max_length=8)
    empty = collator([])
    assert "labels" in empty
    encoded = encode_row(Tok(), {"prompt": "p", "completion": "c", "trajectory": []}, max_length=8)
    assert encoded["labels"][-1] != -100 or encoded["input_ids"]


@pytest.mark.unit
@pytest.mark.harness
def test_tailor_and_collect_cli_blank_lines(tmp_path):
    from scripts.harness.tailor_harness import main as tailor_main

    traces = tmp_path / "t.jsonl"
    traces.write_text(
        "\n" + json.dumps({"task": "t", "faults": ["loop"], "steps": []}) + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive"
    live = tmp_path / "live.yaml"
    code = tailor_main(
        [
            "--harness-id",
            "base_react",
            "--traces",
            str(traces),
            "--archive-dir",
            str(archive),
            "--apply",
            "--live-path",
            str(live),
        ]
    )
    assert code == 0
    assert live.is_file()
    from scripts.harness.collect_trajectories import main as collect_main

    source = tmp_path / "in.jsonl"
    source.write_text('\n{"prompt": "Write a short greeting"}\n', encoding="utf-8")
    dest = tmp_path / "out.jsonl"
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


@pytest.mark.unit
@pytest.mark.harness
def test_trajectory_mode_zeros_distillation_alpha():
    text = (REPO / "scripts" / "training" / "train_distilled_adapter.py").read_text(
        encoding="utf-8"
    )
    assert "--trajectory_mode" in text
    assert 'type=str, default="False"' in text
    assert "args.distillation_alpha = 0.0" in text
    trainer = (REPO / "scripts" / "training" / "distill" / "trainer.py").read_text(encoding="utf-8")
    assert "TrajectoryDataCollator" in trainer
    assert 'getattr(self.args, "trajectory_mode", False)' in trainer


@pytest.mark.unit
@pytest.mark.harness
def test_schema_type_branches():
    schema = {
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "count": {"type": "integer", "minimum": 1},
            "temp": {"type": "number", "minimum": 0},
            "items": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
    }
    validate_payload({"flag": True, "count": 1, "temp": 0.5, "items": ["a"]}, schema)
    with pytest.raises(ValueError):
        validate_payload("nope", {"type": "object"})
    with pytest.raises(ValueError):
        validate_payload(
            {"flag": "yes"}, {"type": "object", "properties": {"flag": {"type": "boolean"}}}
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"count": True}, {"type": "object", "properties": {"count": {"type": "integer"}}}
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"count": 0},
            {"type": "object", "properties": {"count": {"type": "integer", "minimum": 1}}},
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"temp": True}, {"type": "object", "properties": {"temp": {"type": "number"}}}
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"temp": -1},
            {"type": "object", "properties": {"temp": {"type": "number", "minimum": 0}}},
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"items": "a"}, {"type": "object", "properties": {"items": {"type": "array"}}}
        )
    with pytest.raises(ValueError):
        validate_payload(
            {"items": []},
            {"type": "object", "properties": {"items": {"type": "array", "minItems": 1}}},
        )
    with pytest.raises(ValueError):
        validate_payload({"id": 1}, {"type": "object", "properties": {"id": {"type": "string"}}})
