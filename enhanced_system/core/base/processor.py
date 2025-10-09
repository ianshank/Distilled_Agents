"""
Abstract Processor Base Class
==============================

Defines the interface for data processors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar


InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class BaseProcessor(ABC, Generic[InputType, OutputType]):
    """
    Abstract base class for data processors.
    
    All processor implementations should inherit from this class
    and implement the process method.
    
    Type Parameters:
        InputType: Type of input data.
        OutputType: Type of output data.
    """
    
    @abstractmethod
    async def process(self, input_data: InputType) -> OutputType:
        """
        Process input data.
        
        Args:
            input_data: Data to process.
        
        Returns:
            Processed output.
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: InputType) -> bool:
        """
        Validate input data before processing.
        
        Args:
            input_data: Data to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Get processor configuration.
        
        Returns:
            Configuration dictionary.
        """
        pass

