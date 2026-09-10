"""Unit tests for data curation and training orchestration."""

from __future__ import annotations

import pytest
from enhanced_system.core.enums import TrainingStatus
from enhanced_system.training.data_curator import DataCurator
from enhanced_system.training.training_orchestrator import TrainingOrchestrator


@pytest.mark.unit
def test_data_curator_filters_and_returns_dicts():
    curator = DataCurator(
        {
            "quality_threshold": 0.1,
            "enable_augmentation": False,
            "enable_balancing": False,
        }
    )
    data = [
        {
            "id": "1",
            "prompt": "Write a detailed Python function with error handling",
            "completion": "def add(a, b):\n    return a + b\n",
            "category": "coding",
        },
        {"id": "2", "prompt": "x", "completion": "y", "category": "coding"},
    ]
    curated = curator.curate_dataset(data)
    assert isinstance(curated, list)
    assert curated
    assert "prompt" in curated[0]


@pytest.mark.unit
def test_orchestrator_adds_jobs():
    orchestrator = TrainingOrchestrator({"parallel_training": False, "max_parallel_jobs": 1})
    job_id = orchestrator.add_training_job(
        job_id="job-1",
        agent_name="swe",
        dataset_path="data.jsonl",
        model_config={"epochs": 1},
    )
    assert job_id == "job-1"
    assert orchestrator.jobs["job-1"].status == TrainingStatus.PENDING
    assert orchestrator.get_job_status("job-1") is not None
    progress = orchestrator.get_progress()
    assert progress["total_jobs"] == 1
    assert progress["pending"] == 1
    assert orchestrator.cancel_job("job-1") is True
    assert orchestrator.get_job_status("job-1").status == TrainingStatus.CANCELLED
