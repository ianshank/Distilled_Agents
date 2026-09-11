"""JSONL trajectory store (local files; tests use tmp_path)."""

from __future__ import annotations

import json
from pathlib import Path

from enhanced_system.harness.types import Trajectory


class JsonlTraceStore:
    """Append one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trajectory: Trajectory) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory.model_dump(), ensure_ascii=True) + "\n")
