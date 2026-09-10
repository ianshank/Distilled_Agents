"""
Configuration Builder
====================

Provides builder pattern for constructing configurations programmatically.

This follows 2025 Python coding standards:
- Fluent builder interface
- Type-safe construction
- Validation before building
"""

from __future__ import annotations

from typing import Any, Optional

from enhanced_system.core.constants import *
from enhanced_system.core.enums import *


class ConfigBuilder:
    """
    Builder for creating system configuration.

    Example:
        >>> config = (ConfigBuilder()
        ...     .with_caching(l1_enabled=True, l2_enabled=False)
        ...     .with_validation(max_length=4096)
        ...     .with_monitoring(enabled=True)
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize empty configuration builder."""
        self._config: dict[str, Any] = {}

    def with_caching(
        self,
        l1_enabled: bool = True,
        l1_max_size: int = DEFAULT_CACHE_SIZE,
        l1_ttl: int = DEFAULT_L1_TTL,
        l2_enabled: bool = False,
        l2_host: str = "localhost",
        l2_port: int = 6379,
        l2_ttl: int = DEFAULT_L2_TTL,
        l3_enabled: bool = False,
        l3_bucket: Optional[str] = None,
        semantic_enabled: bool = False,
        semantic_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    ) -> ConfigBuilder:
        """
        Configure caching settings.

        Args:
            l1_enabled: Enable L1 in-memory cache.
            l1_max_size: Maximum L1 cache entries.
            l1_ttl: L1 cache TTL in seconds.
            l2_enabled: Enable L2 Redis cache.
            l2_host: Redis host.
            l2_port: Redis port.
            l2_ttl: L2 cache TTL in seconds.
            l3_enabled: Enable L3 S3 cache.
            l3_bucket: S3 bucket name.
            semantic_enabled: Enable semantic caching.
            semantic_threshold: Similarity threshold for semantic matches.

        Returns:
            Self for chaining.
        """
        l3: dict[str, Any] = {"enabled": l3_enabled}
        if l3_bucket:
            l3["bucket"] = l3_bucket
        self._config["caching"] = {
            "l1": {
                "enabled": l1_enabled,
                "max_size": l1_max_size,
                "ttl": l1_ttl,
            },
            "l2": {
                "enabled": l2_enabled,
                "host": l2_host,
                "port": l2_port,
                "ttl": l2_ttl,
            },
            "l3": l3,
            "semantic_similarity": {
                "enabled": semantic_enabled,
                "threshold": semantic_threshold,
                "model": DEFAULT_EMBEDDING_MODEL,
            },
        }
        return self

    def with_validation(
        self,
        max_length: int = DEFAULT_MAX_LENGTH,
        enable_pii_detection: bool = True,
        enable_injection_detection: bool = True,
        enable_toxicity_check: bool = False,
    ) -> ConfigBuilder:
        """
        Configure input validation settings.

        Args:
            max_length: Maximum input length in characters.
            enable_pii_detection: Enable PII detection.
            enable_injection_detection: Enable SQL/command injection detection.
            enable_toxicity_check: Enable toxicity checking.

        Returns:
            Self for chaining.
        """
        self._config["input_validation"] = {
            "max_length": max_length,
            "enable_pii_detection": enable_pii_detection,
            "enable_injection_detection": enable_injection_detection,
            "enable_toxicity_check": enable_toxicity_check,
        }
        return self

    def with_error_handling(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        enable_fallbacks: bool = True,
        enable_error_learning: bool = False,
    ) -> ConfigBuilder:
        """
        Configure error handling settings.

        Args:
            max_retries: Maximum retry attempts.
            base_delay: Base delay for exponential backoff.
            max_delay: Maximum delay cap.
            enable_fallbacks: Enable fallback strategies.
            enable_error_learning: Enable error learning.

        Returns:
            Self for chaining.
        """
        self._config["error_handling"] = {
            "max_retries": max_retries,
            "base_delay": base_delay,
            "max_delay": max_delay,
            "enable_fallbacks": enable_fallbacks,
            "enable_error_learning": enable_error_learning,
        }
        return self

    def with_confidence_calibration(
        self,
        enabled: bool = True,
        calibration_method: str = CalibrationMethod.ISOTONIC.value,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> ConfigBuilder:
        """
        Configure confidence calibration settings.

        Args:
            enabled: Enable confidence calibration.
            calibration_method: Calibration method to use.
            min_samples: Minimum samples required for calibration.

        Returns:
            Self for chaining.
        """
        self._config["confidence"] = {
            "enabled": enabled,
            "calibration_method": calibration_method,
            "min_samples_for_calibration": min_samples,
            "reliability_bands": {
                "low": list(RELIABILITY_BAND_LOW),
                "medium": list(RELIABILITY_BAND_MEDIUM),
                "high": list(RELIABILITY_BAND_HIGH),
                "very_high": list(RELIABILITY_BAND_VERY_HIGH),
            },
        }
        return self

    def with_consensus(
        self,
        enabled: bool = False,
        min_agents: int = DEFAULT_MIN_AGENTS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
        agreement_method: str = AgreementMethod.EMBEDDING_SIMILARITY.value,
        ensemble_method: str = EnsembleMethod.CONFIDENCE_WEIGHTED.value,
    ) -> ConfigBuilder:
        """
        Configure multi-agent consensus settings.

        Args:
            enabled: Enable consensus inference.
            min_agents: Minimum agents for consensus.
            max_agents: Maximum agents to use.
            threshold: Agreement threshold.
            agreement_method: Method for calculating agreement.
            ensemble_method: Method for combining responses.

        Returns:
            Self for chaining.
        """
        self._config["consensus"] = {
            "enabled": enabled,
            "min_agents": min_agents,
            "max_agents": max_agents,
            "threshold": threshold,
            "agreement_method": agreement_method,
            "ensemble_method": ensemble_method,
        }
        return self

    def with_routing(
        self,
        enabled: bool = True,
        strategy: str = RoutingStrategy.COST_OPTIMIZED.value,
        complexity_model: str = "heuristic",
        track_performance: bool = True,
    ) -> ConfigBuilder:
        """
        Configure adaptive routing settings.

        Args:
            enabled: Enable adaptive routing.
            strategy: Routing optimization strategy.
            complexity_model: Model for complexity estimation.
            track_performance: Track agent performance.

        Returns:
            Self for chaining.
        """
        self._config["routing"] = {
            "enabled": enabled,
            "routing_strategy": strategy,
            "complexity_model": complexity_model,
            "track_performance": track_performance,
        }
        return self

    def with_batch_processing(
        self,
        enabled: bool = True,
        max_batch_size: int = DEFAULT_BATCH_SIZE,
        batch_timeout_ms: int = DEFAULT_BATCH_TIMEOUT_MS,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> ConfigBuilder:
        """
        Configure batch processing settings.

        Args:
            enabled: Enable batch processing.
            max_batch_size: Maximum batch size.
            batch_timeout_ms: Batch timeout in milliseconds.
            num_workers: Number of worker threads.

        Returns:
            Self for chaining.
        """
        self._config["batch_processing"] = {
            "enabled": enabled,
            "max_batch_size": max_batch_size,
            "batch_timeout_ms": batch_timeout_ms,
            "num_workers": num_workers,
            "queue_size": DEFAULT_QUEUE_SIZE,
        }
        return self

    def with_monitoring(
        self,
        enabled: bool = True,
        prometheus_port: int = DEFAULT_PROMETHEUS_PORT,
        metrics_interval: int = DEFAULT_METRICS_INTERVAL,
        enable_alerting: bool = True,
        latency_threshold_ms: int = LATENCY_THRESHOLD_MS,
        error_rate_threshold: float = ERROR_RATE_THRESHOLD,
    ) -> ConfigBuilder:
        """
        Configure monitoring settings.

        Args:
            enabled: Enable monitoring.
            prometheus_port: Prometheus metrics port.
            metrics_interval: Metrics collection interval.
            enable_alerting: Enable alerting.
            latency_threshold_ms: Latency alert threshold.
            error_rate_threshold: Error rate alert threshold.

        Returns:
            Self for chaining.
        """
        self._config["monitoring"] = {
            "enabled": enabled,
            "prometheus_port": prometheus_port,
            "metrics_interval": metrics_interval,
            "alerting": {
                "enabled": enable_alerting,
                "latency_threshold_ms": latency_threshold_ms,
                "error_rate_threshold": error_rate_threshold,
            },
        }
        return self

    def build(self):
        """
        Build and validate configuration.

        Returns:
            Validated Pydantic Config instance.
        """
        from enhanced_system.config import Config as PydanticConfig

        self._validate()
        return PydanticConfig(**self._config)

    def _validate(self) -> None:
        """Validate configuration before building."""
        # Validate caching
        if "caching" in self._config:
            cache_cfg = self._config["caching"]
            if cache_cfg.get("l1", {}).get("max_size", 0) < 0:
                raise ValueError("L1 cache max_size must be positive")

            if cache_cfg.get("semantic_similarity", {}).get("enabled"):
                threshold = cache_cfg["semantic_similarity"].get("threshold", 0)
                if not 0 <= threshold <= 1:
                    raise ValueError("Semantic similarity threshold must be between 0 and 1")

        # Validate error handling
        if "error_handling" in self._config:
            error_cfg = self._config["error_handling"]
            if error_cfg.get("max_retries", 0) < 0:
                raise ValueError("max_retries must be non-negative")

            if error_cfg.get("base_delay", 0) < 0:
                raise ValueError("base_delay must be positive")

        # Validate consensus
        if "consensus" in self._config:
            consensus_cfg = self._config["consensus"]
            min_agents = consensus_cfg.get("min_agents", 2)
            max_agents = consensus_cfg.get("max_agents", 5)
            if min_agents > max_agents:
                raise ValueError("min_agents cannot exceed max_agents")
