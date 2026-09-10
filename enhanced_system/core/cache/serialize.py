"""JSON cache serialization (pickle is not used)."""

from __future__ import annotations

import json
from typing import Any


def dumps(value: Any) -> bytes:
    """Serialize a JSON-compatible value to UTF-8 bytes."""
    return json.dumps(value, default=_default, separators=(",", ":")).encode("utf-8")


def loads(data: bytes) -> Any:
    """Deserialize UTF-8 JSON bytes."""
    if data is None:
        return None
    return json.loads(data.decode("utf-8"))


def _default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )
