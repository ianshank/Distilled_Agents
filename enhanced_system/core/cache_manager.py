"""Facade for IntelligentCacheManager (backward-compatible import path)."""

from enhanced_system.core.cache.l1 import L1Cache
from enhanced_system.core.cache.l2_redis import L2Cache
from enhanced_system.core.cache.l3_s3 import L3Cache
from enhanced_system.core.cache.manager import IntelligentCacheManager
from enhanced_system.core.cache.semantic import SemanticCacheManager

__all__ = [
    "L1Cache",
    "L2Cache",
    "L3Cache",
    "SemanticCacheManager",
    "IntelligentCacheManager",
]
