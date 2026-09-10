"""Root training-data checks using pytest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TRAINING_FILE = Path(__file__).resolve().parents[1] / "data" / "training" / "sample_training_data.jsonl"


@pytest.mark.unit
def test_sample_training_data_exists():
    assert TRAINING_FILE.exists(), f"Missing {TRAINING_FILE}"


@pytest.mark.unit
def test_sample_training_data_is_valid_jsonl():
    lines = TRAINING_FILE.read_text(encoding="utf-8").splitlines()
    assert lines, "training file is empty"
    valid = 0
    for line in lines:
        payload = json.loads(line)
        assert "prompt" in payload and "completion" in payload
        valid += 1
    assert valid == len(lines)
