"""JSONL trajectory store (local files; tests use tmp_path)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from enhanced_system.harness.types import Trajectory


class JsonlTraceStore:
    """Append one JSON object per line (in-process lock only)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, trajectory: Trajectory) -> None:
        payload = json.dumps(trajectory.model_dump(), ensure_ascii=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
