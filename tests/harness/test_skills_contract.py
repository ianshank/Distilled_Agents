"""Deterministic skill/CLI contract tests (no live AWS)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from enhanced_system.ops.settings import get_settings

REPO = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO / ".cursor" / "skills"
REQUIRED_FRONTMATTER = ("name", "description", "cli", "inputs")

CLI_REQUIRED_FLAGS = {
    "scripts/training/train_agent_skill.py": {
        "--role",
        "--dataset",
        "--model",
        "--output_dir",
        "--region",
        "--bucket",
        "--table",
    },
    "scripts/evaluation/evaluate_agent_skill.py": {
        "--role",
        "--test_suite",
        "--threshold",
        "--region",
    },
    "scripts/deployment/simple_launch_sagemaker.py": {"--cpu"},
    "scripts/infrastructure/security_scan.py": {"--output"},
    "scripts/deployment/run_sagemaker_training.py": {
        "--region",
        "--dry-run",
        "--max-concurrent",
        "--wait",
    },
}


@pytest.fixture(autouse=True)
def _no_boto3_network(monkeypatch):
    """Fail closed if a test would construct a real AWS client."""
    import sys
    from types import ModuleType

    def _blocked(*_args, **_kwargs):
        raise RuntimeError("boto3 disabled in harness tests")

    fake = ModuleType("boto3")
    fake.client = _blocked  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)


def _parse_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise AssertionError(f"{path} missing YAML frontmatter")
    payload = text.split("---", 2)
    data = yaml.safe_load(payload[1])
    if not isinstance(data, dict):
        raise AssertionError(f"{path} frontmatter is not a mapping")
    return data


def _help_flags(relpath: str) -> set[str]:
    script = REPO / relpath
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    return set(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", completed.stdout))


@pytest.mark.regression
def test_skills_frontmatter_and_cli_paths():
    skill_files = sorted(SKILLS_ROOT.glob("*/SKILL.md"))
    assert skill_files, "expected Cursor skills under .cursor/skills"
    names = set()
    for path in skill_files:
        data = _parse_frontmatter(path)
        missing = [key for key in REQUIRED_FRONTMATTER if key not in data]
        assert not missing, f"{path} missing keys {missing}"
        cli = Path(data["cli"])
        assert not cli.is_absolute()
        assert (REPO / cli).is_file(), f"cli path missing: {cli}"
        assert isinstance(data["inputs"], dict) and data["inputs"]
        names.add(data["name"])
    assert names == {
        "mangomas-train",
        "mangomas-evaluate",
        "mangomas-launch",
        "mangomas-scan",
    }


@pytest.mark.regression
def test_public_cli_help_flags_are_stable():
    for relpath, required in CLI_REQUIRED_FLAGS.items():
        flags = _help_flags(relpath)
        missing = required - flags
        assert not missing, f"{relpath} missing flags {missing}; got {sorted(flags)}"


@pytest.mark.regression
def test_settings_secure_defaults(monkeypatch):
    monkeypatch.delenv("MANGOMAS_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.delenv("MANGOMAS_BIND_HOST", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.trust_remote_code is False
    assert settings.bind_host == "127.0.0.1"


@pytest.mark.regression
def test_composite_action_wraps_make_validate():
    action = (REPO / ".github" / "actions" / "mangomas-validate" / "action.yml").read_text(
        encoding="utf-8"
    )
    assert "make validate" in action
    assert "gitleaks" in action.lower()
