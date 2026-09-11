"""Shared settings reset for unit tests."""

from __future__ import annotations

import pytest
from enhanced_system.ops.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_mangomas_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
