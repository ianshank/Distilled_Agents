"""Redis L2 cache."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from enhanced_system.core.base.cache import BaseCache
from enhanced_system.core.cache.serialize import dumps, loads

logger = logging.getLogger(__name__)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. L2 caching will be disabled.")


class L2Cache(BaseCache):
    """Redis distributed cache using JSON serialization."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True) and REDIS_AVAILABLE
        self.ttl = config.get("ttl", 86400)
        self.redis_client: Any = None
        self._hits = 0
        self._misses = 0

        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=config.get("host", "localhost"),
                    port=config.get("port", 6379),
                    db=config.get("db", 0),
                    socket_timeout=config.get("socket_timeout", 5),
                    socket_connect_timeout=config.get("socket_connect_timeout", 5),
                    decode_responses=False,
                    max_connections=config.get("pool_size", 10),
                )
                self.redis_client.ping()
                logger.info("L2 Redis cache initialized successfully")
            except Exception as exc:
                logger.warning("Failed to initialize Redis: %s. L2 caching disabled.", exc)
                self.enabled = False
                self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled or not self.redis_client:
            return None
        try:
            value = self.redis_client.get(key)
            if value:
                self._hits += 1
                if isinstance(value, str):
                    return loads(value.encode("utf-8"))
                return loads(bytes(value))
            self._misses += 1
            return None
        except Exception as exc:
            logger.error("L2 cache get error: %s", exc)
            return None

    async def get_async(self, key: str) -> Optional[Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self.enabled or not self.redis_client:
            return
        try:
            serialized = dumps(value)
            self.redis_client.setex(key, ttl or self.ttl, serialized)
        except Exception as exc:
            logger.error("L2 cache set error: %s", exc)

    async def set_async(self, key: str, value: Any) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.set, key, value)

    def delete(self, key: str) -> bool:
        if not self.enabled or not self.redis_client:
            return False
        try:
            return bool(self.redis_client.delete(key))
        except Exception as exc:
            logger.error("L2 cache delete error: %s", exc)
            return False

    def clear(self) -> None:
        if self.enabled and self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("L2 cache cleared")
            except Exception as exc:
                logger.error("L2 cache clear error: %s", exc)

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0
        stats = {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "enabled": self.enabled,
        }
        if self.enabled and self.redis_client:
            try:
                info = self.redis_client.info()
                if isinstance(info, dict):
                    stats.update(
                        {
                            "used_memory": info.get("used_memory_human"),
                            "connected_clients": info.get("connected_clients"),
                            "total_commands_processed": info.get("total_commands_processed"),
                        }
                    )
            except Exception as exc:
                logger.error("Failed to get Redis stats: %s", exc)
        return stats
