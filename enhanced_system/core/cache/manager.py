"""Multi-level intelligent cache manager."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Callable, Optional

from enhanced_system.core.cache.l1 import L1Cache
from enhanced_system.core.cache.l2_redis import L2Cache
from enhanced_system.core.cache.l3_s3 import L3Cache
from enhanced_system.core.cache.semantic import SemanticCacheManager

logger = logging.getLogger(__name__)


class IntelligentCacheManager:
    """L1/L2/L3 cache with optional semantic key reuse."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.l1_cache = L1Cache(
            max_size=config.get("l1", {}).get("max_size", 1000),
            ttl=config.get("l1", {}).get("ttl", 3600),
        )
        self.l2_cache = L2Cache(config.get("l2", {}))
        self.l3_cache = L3Cache(config.get("l3", {}))
        self.semantic_cache = SemanticCacheManager(config.get("semantic_similarity", {}))
        logger.info("IntelligentCacheManager initialized")

    def compute_cache_key(self, task: str, agent: str, params: Optional[dict] = None) -> str:
        param_part = str(sorted((params or {}).items()))
        namespace = hashlib.sha256(f"{agent}|{param_part}".encode()).hexdigest()[:16]
        similar_key = self.semantic_cache.find_similar_cached(task, namespace=namespace)
        if similar_key:
            return similar_key

        key_string = "|".join([task, agent, param_part])
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()
        self.semantic_cache.register_embedding(cache_key, task, namespace=namespace)
        return cache_key

    async def get_or_compute(self, key: str, compute_func: Callable, *args, **kwargs) -> Any:
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        value = await self.l2_cache.get_async(key)
        if value is not None:
            self.l1_cache.set(key, value)
            return value

        value = await self.l3_cache.get_async(key)
        if value is not None:
            await self.l2_cache.set_async(key, value)
            self.l1_cache.set(key, value)
            return value

        logger.info("Cache miss on all levels, computing value...")
        if asyncio.iscoroutinefunction(compute_func):
            value = await compute_func(*args, **kwargs)
        else:
            value = compute_func(*args, **kwargs)

        self.l1_cache.set(key, value)
        await self.l2_cache.set_async(key, value)
        await self.l3_cache.set_async(key, value)
        return value

    def get_stats(self) -> dict[str, Any]:
        return {
            "l1": self.l1_cache.get_stats(),
            "l2": self.l2_cache.get_stats(),
            "l3": self.l3_cache.get_stats(),
            "semantic_enabled": self.semantic_cache.enabled,
        }

    def clear_all(self) -> None:
        self.l1_cache.clear()
        self.l2_cache.clear()
        logger.info("All cache levels cleared (except L3)")
