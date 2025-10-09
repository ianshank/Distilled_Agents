"""
Abstract Base Classes
=====================

Provides abstract interfaces for pluggable implementations.
"""

from .validator import BaseValidator
from .cache import BaseCache
from .processor import BaseProcessor

__all__ = [
    "BaseValidator",
    "BaseCache",
    "BaseProcessor",
]

