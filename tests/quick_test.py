#!/usr/bin/env python3
"""Quick training-data smoke test (delegates to pytest)."""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).with_name("test_training_data.py")), "-q"]))
