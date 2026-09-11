"""Rule-based harness tailor unit tests."""

from __future__ import annotations

import pytest
from enhanced_system.harness.registry import load_spec
from enhanced_system.harness.tailor import HarnessTailor
from enhanced_system.harness.types import HarnessSpec, Step, Trajectory
from enhanced_system.ops.settings import get_settings


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
def test_tailor_empty_traces_noop():
    spec = load_spec("base_react")
    proposed = HarnessTailor().propose(spec, [])
    assert proposed.action.tool_ids == spec.action.tool_ids
    assert proposed.planning.max_steps == spec.planning.max_steps
