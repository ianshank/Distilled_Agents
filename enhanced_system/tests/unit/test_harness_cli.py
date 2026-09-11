"""Harness CLI unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.unit
@pytest.mark.harness
def test_run_agent_cli(capsys):
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
def test_run_agent_cli_bad_scripted_json():
    from scripts.harness.run_agent import main

    code = main(
        ["--task", "Write a short greeting", "--harness-id", "base_react", "--scripted", "not-json"]
    )
    assert code == 1


@pytest.mark.unit
@pytest.mark.harness
def test_run_agent_rejects_sql_task():
    from scripts.harness.run_agent import main

    code = main(["--task", "SELECT * FROM users WHERE id = 1", "--harness-id", "base_react"])
    assert code == 1


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
            str(dest),
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
def test_collect_allows_coding_hashes_and_skips_empty(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"prompt": ""}),
                json.dumps({"prompt": 'Fix the # TODO and add "quotes"'}),
                json.dumps({"prompt": "Write a short greeting"}),
            ]
        )
        + "\n",
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
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"ok\\"}}"]',
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert "#" in rows[0]["prompt"]


@pytest.mark.unit
@pytest.mark.harness
def test_collect_skips_non_mapping_row(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        '["not", "an", "object"]\n{"prompt": "Write a short greeting"}\n',
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
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"ok\\"}}"]',
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1


@pytest.mark.unit
@pytest.mark.harness
def test_collect_strict_fails_non_mapping_row(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text("null\n", encoding="utf-8")
    code = main(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--harness-id",
            "base_react",
            "--strict",
        ]
    )
    assert code == 1


@pytest.mark.unit
@pytest.mark.harness
def test_collect_raw_path_avoids_output_collision(tmp_path):
    from scripts.harness.collect_trajectories import main, raw_store_path

    dest = tmp_path / "rollout.raw.jsonl"
    assert raw_store_path(dest).name == "rollout.raw.traces.jsonl"
    source = tmp_path / "in.jsonl"
    source.write_text('{"prompt": "Write a short greeting"}\n', encoding="utf-8")
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
        ]
    )
    assert code == 0
    assert dest.is_file()
    assert (tmp_path / "rollout.raw.traces.jsonl").is_file()
    assert dest.read_text(encoding="utf-8").strip()
    assert (tmp_path / "rollout.raw.traces.jsonl").read_text(encoding="utf-8").strip()


@pytest.mark.unit
@pytest.mark.harness
def test_collect_strict_fails_empty_prompt(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text('{"prompt": ""}\n', encoding="utf-8")
    code = main(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--harness-id",
            "base_react",
            "--strict",
        ]
    )
    assert code == 1


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


@pytest.mark.unit
@pytest.mark.harness
def test_tailor_cli_bad_json(tmp_path):
    from scripts.harness.tailor_harness import main as tailor_main

    traces = tmp_path / "t.jsonl"
    traces.write_text("{not json\n", encoding="utf-8")
    code = tailor_main(
        [
            "--harness-id",
            "base_react",
            "--traces",
            str(traces),
            "--archive-dir",
            str(tmp_path / "archive"),
        ]
    )
    assert code == 1


@pytest.mark.unit
@pytest.mark.harness
def test_trajectory_mode_reads_distill_alpha_env():
    text = (REPO / "scripts" / "training" / "train_distilled_adapter.py").read_text(
        encoding="utf-8"
    )
    assert "--trajectory_mode" in text
    assert 'type=str, default="False"' in text
    assert "MANGOMAS_TRAJECTORY_DISTILL_ALPHA" in text
    assert "parse_trajectory_distill_alpha" in text
    from scripts.training.distill.alpha import parse_trajectory_distill_alpha

    assert parse_trajectory_distill_alpha("0.0") == 0.0
    with pytest.raises(ValueError):
        parse_trajectory_distill_alpha("1.5")
    with pytest.raises(ValueError):
        parse_trajectory_distill_alpha("nan")
    trainer = (REPO / "scripts" / "training" / "distill" / "trainer.py").read_text(encoding="utf-8")
    assert "TrajectoryDataCollator" in trainer
    assert 'getattr(self.args, "trajectory_mode", False)' in trainer
