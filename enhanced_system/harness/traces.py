"""JSONL trajectory store (local files; tests use tmp_path)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from enhanced_system.harness.types import Trajectory


def raw_store_path(out_path: Path) -> Path:
    """Return a distinct raw-trace path so collect never truncates its own store."""
    if out_path.name.endswith(".raw.jsonl"):
        stem = out_path.name[: -len(".raw.jsonl")]
        return out_path.with_name(f"{stem}.raw.traces.jsonl")
    return out_path.with_suffix(".raw.jsonl")


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
