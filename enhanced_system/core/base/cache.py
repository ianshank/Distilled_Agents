"""
Abstract Cache Base Class
==========================

Defines the interface for cache implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCache(ABC):
    """
    Abstract base class for cache implementations.
    
    All cache implementations should inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key.
        
        Returns:
            Cached value or None if not found.
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time to live in seconds (optional).
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key.
        
        Returns:
            True if deleted, False if not found.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats (hits, misses, size, etc.).
        """
        pass

