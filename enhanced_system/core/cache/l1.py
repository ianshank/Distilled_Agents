"""In-memory L1 LRU cache."""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Optional

from enhanced_system.core.base.cache import BaseCache

logger = logging.getLogger(__name__)


class L1Cache(BaseCache):
    """In-memory LRU cache with TTL."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            now = datetime.now().timestamp()
            if (now - timestamp) < self.ttl:
                self._hits += 1
                # Recency for LRU eviction; keep the original timestamp so TTL stays absolute.
                self.cache.move_to_end(key)
                logger.debug("L1 cache hit for key: %s...", key[:20])
                return value
            del self.cache[key]

        self._misses += 1
        logger.debug("L1 cache miss for key: %s...", key[:20])
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if key in self.cache:
            del self.cache[key]
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = (value, datetime.now().timestamp())
        logger.debug("L1 cache set for key: %s...", key[:20])

    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        self.cache.clear()
        logger.info("L1 cache cleared")

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }
