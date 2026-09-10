"""
Abstract Base Classes
=====================

Provides abstract interfaces for pluggable implementations.
"""

from .cache import BaseCache
from .processor import BaseProcessor
from .validator import BaseValidator

__all__ = [
    "BaseValidator",
    "BaseCache",
    "BaseProcessor",
]
