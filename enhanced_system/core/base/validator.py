"""
Abstract Validator Base Class
==============================

Defines the interface for input validators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    error_message: str | None = None
    warnings: list[str] | None = None
    sanitized_input: str | None = None
    metadata: dict[str, Any] | None = None
    
    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class BaseValidator(ABC):
    """
    Abstract base class for input validators.
    
    All validator implementations should inherit from this class
    and implement the validate method.
    """
    
    @abstractmethod
    def validate(self, input_data: str) -> ValidationResult:
        """
        Validate input data.
        
        Args:
            input_data: Data to validate.
        
        Returns:
            ValidationResult with validation status and details.
        """
        pass
    
    @abstractmethod
    def sanitize(self, input_data: str) -> str:
        """
        Sanitize input data.
        
        Args:
            input_data: Data to sanitize.
        
        Returns:
            Sanitized data.
        """
        pass

