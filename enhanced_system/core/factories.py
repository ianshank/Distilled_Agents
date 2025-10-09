"""
Factory Module
==============

Provides factory methods for creating complex objects with proper dependency injection.

This module follows 2025 Python coding standards:
- Factory pattern for object creation
- Dependency injection
- Type-safe creation methods
"""

from __future__ import annotations

from typing import Any, Optional
import logging

from .constants import *
from .enums import *


class ValidatorFactory:
    """Factory for creating InputValidator instances."""
    
    @staticmethod
    def create(
        config: dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> 'InputValidator':
        """
        Create InputValidator with proper configuration.
        
        Args:
            config: Configuration dictionary.
            logger: Optional logger instance.
        
        Returns:
            Configured InputValidator instance.
        """
        from .input_validator import InputValidator
        
        # Merge with defaults
        full_config = {
            'max_length': DEFAULT_MAX_LENGTH,
            'enable_pii_detection': True,
            'enable_injection_detection': True,
        }
        full_config.update(config)
        
        return InputValidator(full_config)


class CacheManagerFactory:
    """Factory for creating IntelligentCacheManager instances."""
    
    @staticmethod
    def create(
        config: dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> 'IntelligentCacheManager':
        """
        Create IntelligentCacheManager with proper configuration.
        
        Args:
            config: Configuration dictionary.
            logger: Optional logger instance.
        
        Returns:
            Configured IntelligentCacheManager instance.
        """
        from .cache_manager import IntelligentCacheManager
        
        # Merge with defaults
        full_config = {
            'l1': {'max_size': DEFAULT_CACHE_SIZE, 'ttl': DEFAULT_L1_TTL},
            'l2': {'enabled': True, 'ttl': DEFAULT_L2_TTL},
            'l3': {'enabled': True, 'ttl': DEFAULT_L3_TTL},
            'semantic_similarity': {
                'enabled': True,
                'threshold': SEMANTIC_SIMILARITY_THRESHOLD,
                'model': DEFAULT_EMBEDDING_MODEL
            }
        }
        
        # Deep merge configs
        for key in ['l1', 'l2', 'l3', 'semantic_similarity']:
            if key in config:
                full_config[key].update(config[key])
        
        return IntelligentCacheManager(full_config)


class ConfidenceCalibratorFactory:
    """Factory for creating ConfidenceCalibrator instances."""
    
    @staticmethod
    def create(
        config: dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> 'ConfidenceCalibrator':
        """
        Create ConfidenceCalibrator with proper configuration.
        
        Args:
            config: Configuration dictionary.
            logger: Optional logger instance.
        
        Returns:
            Configured ConfidenceCalibrator instance.
        """
        from .confidence_calibrator import ConfidenceCalibrator
        
        # Merge with defaults
        full_config = {
            'enabled': True,
            'calibration_method': CalibrationMethod.ISOTONIC.value,
            'min_samples_for_calibration': DEFAULT_MIN_SAMPLES,
            'reliability_bands': {
                'low': list(RELIABILITY_BAND_LOW),
                'medium': list(RELIABILITY_BAND_MEDIUM),
                'high': list(RELIABILITY_BAND_HIGH),
                'very_high': list(RELIABILITY_BAND_VERY_HIGH),
            }
        }
        full_config.update(config)
        
        return ConfidenceCalibrator(full_config)


class ConsensusInferenceFactory:
    """Factory for creating ConsensusInference instances."""
    
    @staticmethod
    def create(
        config: dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> 'ConsensusInference':
        """
        Create ConsensusInference with proper configuration.
        
        Args:
            config: Configuration dictionary.
            logger: Optional logger instance.
        
        Returns:
            Configured ConsensusInference instance.
        """
        from .consensus_inference import ConsensusInference
        
        # Merge with defaults
        full_config = {
            'enabled': False,
            'min_agents': DEFAULT_MIN_AGENTS,
            'max_agents': DEFAULT_MAX_AGENTS,
            'threshold': DEFAULT_CONSENSUS_THRESHOLD,
            'agreement_method': AgreementMethod.EMBEDDING_SIMILARITY.value,
            'ensemble_method': EnsembleMethod.CONFIDENCE_WEIGHTED.value,
        }
        full_config.update(config)
        
        return ConsensusInference(full_config)


class AdaptiveRouterFactory:
    """Factory for creating AdaptiveRouter instances."""
    
    @staticmethod
    def create(
        config: dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> 'AdaptiveRouter':
        """
        Create AdaptiveRouter with proper configuration.
        
        Args:
            config: Configuration dictionary.
            logger: Optional logger instance.
        
        Returns:
            Configured AdaptiveRouter instance.
        """
        from .adaptive_router import AdaptiveRouter
        
        # Merge with defaults
        full_config = {
            'enabled': True,
            'complexity_model': 'heuristic',
            'routing_strategy': RoutingStrategy.COST_OPTIMIZED.value,
        }
        full_config.update(config)
        
        return AdaptiveRouter(full_config)


class BatchProcessorFactory:
    """Factory for creating BatchProcessor instances."""
    
    @staticmethod
    def create(
        config: dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> 'BatchProcessor':
        """
        Create BatchProcessor with proper configuration.
        
        Args:
            config: Configuration dictionary.
            logger: Optional logger instance.
        
        Returns:
            Configured BatchProcessor instance.
        """
        from .batch_processor import BatchProcessor
        
        # Merge with defaults
        full_config = {
            'enabled': True,
            'max_batch_size': DEFAULT_BATCH_SIZE,
            'batch_timeout_ms': DEFAULT_BATCH_TIMEOUT_MS,
            'num_workers': DEFAULT_NUM_WORKERS,
            'queue_size': DEFAULT_QUEUE_SIZE,
            'priority_levels': 3,
        }
        full_config.update(config)
        
        return BatchProcessor(full_config)


class MonitorFactory:
    """Factory for creating AgentMonitor instances."""
    
    @staticmethod
    def create(
        config: dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> 'AgentMonitor':
        """
        Create AgentMonitor with proper configuration.
        
        Args:
            config: Configuration dictionary.
            logger: Optional logger instance.
        
        Returns:
            Configured AgentMonitor instance.
        """
        from .monitoring import AgentMonitor
        
        # Merge with defaults
        full_config = {
            'enabled': True,
            'prometheus_port': DEFAULT_PROMETHEUS_PORT,
            'metrics_interval': DEFAULT_METRICS_INTERVAL,
            'alerting': {
                'enabled': True,
                'latency_threshold_ms': LATENCY_THRESHOLD_MS,
                'error_rate_threshold': ERROR_RATE_THRESHOLD,
            }
        }
        
        # Deep merge alerting config
        if 'alerting' in config:
            full_config['alerting'].update(config['alerting'])
            config = {k: v for k, v in config.items() if k != 'alerting'}
        
        full_config.update(config)
        
        return AgentMonitor(full_config)

