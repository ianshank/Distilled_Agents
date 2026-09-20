"""Test that enhanced_system/harness/prompt_render.py and scripts/training/distill/prompt_render.py stay in sync."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_RENDER = REPO_ROOT / "enhanced_system" / "harness" / "prompt_render.py"
DISTILL_RENDER = REPO_ROOT / "scripts" / "training" / "distill" / "prompt_render.py"


def _normalize_module_code(text: str) -> str:
    """Normalize the top-level docstring which cross-references the other file."""
    stripped = re.sub(r'^"""[\s\S]*?"""\s*', "", text.strip())
    return stripped.strip()


@pytest.mark.unit
def test_prompt_render_modules_in_sync() -> None:
    """Bodies of both prompt_render modules must be identical after docstring normalization."""
    assert HARNESS_RENDER.is_file(), f"Missing {HARNESS_RENDER}"
    assert DISTILL_RENDER.is_file(), f"Missing {DISTILL_RENDER}"

    harness_content = HARNESS_RENDER.read_text(encoding="utf-8")
    distill_content = DISTILL_RENDER.read_text(encoding="utf-8")

    harness_body = _normalize_module_code(harness_content)
    distill_body = _normalize_module_code(distill_content)

    assert harness_body == distill_body, (
        "prompt_render modules are out of sync! "
        "enhanced_system/harness/prompt_render.py and "
        "scripts/training/distill/prompt_render.py must stay identical."
    )
