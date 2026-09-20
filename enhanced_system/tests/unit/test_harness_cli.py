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
def test_collect_outcome_filter_keeps_faults_when_expected_matches(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "hi"}) + "\n",
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
        ]
    )
    assert code == 0
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert row["trajectory"]["final_answer"] == "hi"


@pytest.mark.unit
@pytest.mark.harness
def test_collect_outcome_filter_skips_mismatch(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "nope"}) + "\n",
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
        ]
    )
    assert code == 0
    assert dest.read_text(encoding="utf-8").strip() == ""


@pytest.mark.unit
@pytest.mark.harness
def test_collect_outcome_filter_skips_mismatch_with_reject_log(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "nope"}) + "\n",
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
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
            "--reject-log",
            str(reject_log),
        ]
    )
    assert code == 0
    assert dest.read_text(encoding="utf-8").strip() == ""
    assert reject_log.is_file()
    records = [json.loads(line) for line in reject_log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["critic_reject_code"] == "OUTCOME_MISMATCH"


@pytest.mark.unit
@pytest.mark.harness
def test_collect_student_flag(tmp_path):
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
            "--student",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
        ]
    )
    assert code == 0
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert row["trajectory"]["final_answer"] == "hi"


@pytest.mark.unit
@pytest.mark.harness
def test_collect_keeps_recovery_when_expected_matches(tmp_path):
    from scripts.harness.collect_trajectories import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "hi"}) + "\n",
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
            '["not-json", "{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
        ]
    )
    assert code == 0
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert row["trajectory"]["final_answer"] == "hi"
    assert "parse_error" in row["trajectory"]["faults"]


@pytest.mark.unit
@pytest.mark.harness
def test_eval_harness_cli(tmp_path, capsys):
    from scripts.harness.eval_harness import main

    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "hi"}) + "\n",
        encoding="utf-8",
    )
    code = main(
        [
            "--input",
            str(source),
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
            "--threshold",
            "50",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exact_match"] == 1
    assert payload["pass_rate"] == 100.0


@pytest.mark.unit
@pytest.mark.harness
def test_eval_harness_missing_input_and_threshold(tmp_path, capsys):
    from scripts.harness.eval_harness import main

    assert main(["--input", str(tmp_path / "missing.jsonl"), "--harness-id", "base_react"]) == 1
    source = tmp_path / "in.jsonl"
    source.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "nope"}) + "\n",
        encoding="utf-8",
    )
    code = main(
        [
            "--input",
            str(source),
            "--harness-id",
            "base_react",
            "--scripted",
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"hi\\"}}"]',
            "--threshold",
            "100",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["exact_match"] == 0


@pytest.mark.unit
@pytest.mark.harness
def test_build_memory_and_dualdistill_and_score_cli(tmp_path):
    from scripts.harness.build_memory import main as build_main
    from scripts.harness.collect_score import main as score_main
    from scripts.harness.compose_dualdistill import main as compose_main

    traces = tmp_path / "t.jsonl"
    traces.write_text(
        json.dumps(
            {
                "task": "Write a short greeting",
                "final_answer": "hi",
                "steps": [
                    {
                        "thought": "greet",
                        "action": "final_answer(text='hi')",
                        "tool_id": "final_answer",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bank = tmp_path / "bank.json"
    assert build_main(["--traces", str(traces), "--output", str(bank)]) == 0
    assert bank.is_file()

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
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
    first.write_text(json.dumps(row_bad) + "\n", encoding="utf-8")
    second.write_text(json.dumps(row_good) + "\n", encoding="utf-8")
    composed = tmp_path / "c.jsonl"
    assert (
        compose_main(["--first", str(first), "--second", str(second), "--output", str(composed)])
        == 0
    )
    assert composed.read_text(encoding="utf-8").strip()
    unlabeled = tmp_path / "u.jsonl"
    unlabeled.write_text(json.dumps({"prompt": "z", "trajectory": {"final_answer": "ok"}}) + "\n")
    empty = tmp_path / "c2.jsonl"
    assert (
        compose_main(
            ["--first", str(unlabeled), "--second", str(unlabeled), "--output", str(empty)]
        )
        == 0
    )
    assert empty.read_text(encoding="utf-8").strip() == ""

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "Write a short greeting", "expected": "ok"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "score.jsonl"
    prefs = tmp_path / "prefs.jsonl"
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
            '["{\\"tool\\": \\"final_answer\\", \\"args\\": {\\"text\\": \\"ok\\"}}"]',
            "--prefs",
            str(prefs),
        ]
    )
    assert code == 0
    assert out.is_file()
    assert prefs.is_file()


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
