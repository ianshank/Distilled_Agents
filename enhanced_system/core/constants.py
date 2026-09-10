"""
Constants Module
================

Centralized constants for the enhanced agent system.
All hardcoded values should be extracted here for maintainability.

This module follows 2025 Python coding standards:
- Type-safe constants using Final
- Clear categorization
- Comprehensive documentation
"""

from typing import Final

# =============================================================================
# Model Constants
# =============================================================================

DEFAULT_EMBEDDING_MODEL: Final[str] = "all-MiniLM-L6-v2"
"""Default sentence transformer model for semantic similarity."""

DEFAULT_NLP_MODEL: Final[str] = "en_core_web_sm"
"""Default spaCy NLP model for PII detection."""

# =============================================================================
# Reliability Band Thresholds
# =============================================================================

RELIABILITY_BAND_LOW: Final[tuple[float, float]] = (0.0, 0.5)
"""Confidence range for LOW reliability band."""

RELIABILITY_BAND_MEDIUM: Final[tuple[float, float]] = (0.5, 0.75)
"""Confidence range for MEDIUM reliability band."""

RELIABILITY_BAND_HIGH: Final[tuple[float, float]] = (0.75, 0.9)
"""Confidence range for HIGH reliability band."""

RELIABILITY_BAND_VERY_HIGH: Final[tuple[float, float]] = (0.9, 1.0)
"""Confidence range for VERY_HIGH reliability band."""

# =============================================================================
# Input Validation Constants
# =============================================================================

DEFAULT_MAX_LENGTH: Final[int] = 4096
"""Default maximum input length in characters."""

DEFAULT_MIN_LENGTH: Final[int] = 1
"""Default minimum input length in characters."""

MAX_WORD_LENGTH: Final[int] = 100
"""Maximum length for a single word (DoS protection)."""

REPETITION_THRESHOLD: Final[float] = 0.1
"""Threshold for detecting excessive repetition (unique/total words)."""

# =============================================================================
# Calibration Constants
# =============================================================================

DEFAULT_MIN_SAMPLES: Final[int] = 100
"""Minimum samples required for calibration."""

CALIBRATION_ADJUSTMENT_FACTOR: Final[float] = 0.1
"""Conservative adjustment factor for uncalibrated models."""

# =============================================================================
# Caching Constants
# =============================================================================

DEFAULT_CACHE_SIZE: Final[int] = 1000
"""Default L1 cache size (number of entries)."""

DEFAULT_L1_TTL: Final[int] = 3600
"""Default L1 cache TTL in seconds (1 hour)."""

DEFAULT_L2_TTL: Final[int] = 86400
"""Default L2 cache TTL in seconds (24 hours)."""

DEFAULT_L3_TTL: Final[int] = 604800
"""Default L3 cache TTL in seconds (7 days)."""

SEMANTIC_SIMILARITY_THRESHOLD: Final[float] = 0.95
"""Minimum similarity score for semantic cache hits."""

# =============================================================================
# Consensus Constants
# =============================================================================

DEFAULT_MIN_AGENTS: Final[int] = 3
"""Minimum number of agents for consensus."""

DEFAULT_MAX_AGENTS: Final[int] = 5
"""Maximum number of agents for consensus."""

DEFAULT_CONSENSUS_THRESHOLD: Final[float] = 0.7
"""Minimum agreement score for consensus."""

# =============================================================================
# Routing Constants
# =============================================================================

DEFAULT_ROUTING_THRESHOLD: Final[float] = 0.7
"""Default threshold for routing decisions."""

COMPLEXITY_THRESHOLD_SIMPLE: Final[int] = -1
"""Score threshold for SIMPLE complexity."""

COMPLEXITY_THRESHOLD_COMPLEX: Final[int] = 2
"""Score threshold for COMPLEX complexity."""

# =============================================================================
# Agent Cost Constants (per 1K tokens)
# =============================================================================

AGENT_COST_BASE: Final[float] = 0.005
"""Cost per 1K tokens for base agent."""

AGENT_COST_SWE: Final[float] = 0.015
"""Cost per 1K tokens for software engineer agent."""

AGENT_COST_ARCHITECT: Final[float] = 0.02
"""Cost per 1K tokens for architect agent."""

AGENT_COST_SQE: Final[float] = 0.01
"""Cost per 1K tokens for quality engineer agent."""

AGENT_COST_DEVOPS: Final[float] = 0.012
"""Cost per 1K tokens for DevOps agent."""

AGENT_COST_SECURITY: Final[float] = 0.018
"""Cost per 1K tokens for security specialist agent."""

# =============================================================================
# Agent Quality Scores
# =============================================================================

AGENT_QUALITY_BASE: Final[float] = 0.7
"""Quality score for base agent."""

AGENT_QUALITY_SWE: Final[float] = 0.9
"""Quality score for software engineer agent."""

AGENT_QUALITY_ARCHITECT: Final[float] = 0.95
"""Quality score for architect agent."""

AGENT_QUALITY_SQE: Final[float] = 0.85
"""Quality score for quality engineer agent."""

# =============================================================================
# Agent Speed (tokens per second)
# =============================================================================

AGENT_SPEED_BASE: Final[float] = 150.0
"""Processing speed for base agent (tokens/sec)."""

AGENT_SPEED_SWE: Final[float] = 100.0
"""Processing speed for software engineer agent (tokens/sec)."""

AGENT_SPEED_ARCHITECT: Final[float] = 80.0
"""Processing speed for architect agent (tokens/sec)."""

AGENT_SPEED_SQE: Final[float] = 120.0
"""Processing speed for quality engineer agent (tokens/sec)."""

# =============================================================================
# Streaming Constants
# =============================================================================

DEFAULT_STREAM_BUFFER_SIZE: Final[int] = 10
"""Default buffer size for streaming inference."""

DEFAULT_MAX_TOKENS: Final[int] = 2048
"""Default maximum tokens in stream."""

STREAM_QUALITY_THRESHOLD: Final[float] = 0.3
"""Minimum quality threshold for streaming."""

# =============================================================================
# Batch Processing Constants
# =============================================================================

DEFAULT_BATCH_SIZE: Final[int] = 32
"""Default batch size for processing."""

DEFAULT_BATCH_TIMEOUT_MS: Final[int] = 100
"""Default batch timeout in milliseconds."""

DEFAULT_NUM_WORKERS: Final[int] = 4
"""Default number of worker threads."""

DEFAULT_QUEUE_SIZE: Final[int] = 1000
"""Default queue size for batch processing."""

# =============================================================================
# Monitoring Constants
# =============================================================================

DEFAULT_PROMETHEUS_PORT: Final[int] = 9090
"""Default Prometheus metrics port."""

DEFAULT_METRICS_INTERVAL: Final[int] = 10
"""Default metrics collection interval in seconds."""

LATENCY_THRESHOLD_MS: Final[int] = 5000
"""Latency threshold for alerts (milliseconds)."""

ERROR_RATE_THRESHOLD: Final[float] = 0.1
"""Error rate threshold for alerts."""

# =============================================================================
# Error Handling Constants
# =============================================================================

DEFAULT_MAX_RETRIES: Final[int] = 3
"""Default maximum number of retry attempts."""

DEFAULT_BASE_DELAY: Final[float] = 1.0
"""Default base delay for exponential backoff (seconds)."""

DEFAULT_MAX_DELAY: Final[float] = 32.0
"""Default maximum delay for exponential backoff (seconds)."""

EXPONENTIAL_BASE: Final[int] = 2
"""Base for exponential backoff calculation."""

# =============================================================================
# Data Curation Constants
# =============================================================================

DATA_QUALITY_THRESHOLD: Final[float] = 0.7
"""Minimum quality score for data samples."""

AUGMENTATION_FACTOR: Final[int] = 2
"""Factor for data augmentation."""

DIVERSITY_THRESHOLD: Final[float] = 0.8
"""Minimum diversity score for dataset."""

# =============================================================================
# A/B Testing Constants
# =============================================================================

DEFAULT_TRAFFIC_SPLIT: Final[float] = 0.1
"""Default percentage of traffic to treatment variant."""

MIN_SAMPLES_SIGNIFICANCE: Final[int] = 100
"""Minimum samples for statistical significance."""

SIGNIFICANCE_LEVEL: Final[float] = 0.05
"""Alpha level for statistical tests."""

# =============================================================================
# Confidence Scoring Factor Weights
# =============================================================================

FACTOR_WEIGHT_AGENT_RELIABILITY: Final[float] = 0.3
"""Weight for agent reliability factor."""

FACTOR_WEIGHT_TASK_DIFFICULTY: Final[float] = 0.2
"""Weight for task difficulty factor."""

FACTOR_WEIGHT_TRAINING_COVERAGE: Final[float] = 0.3
"""Weight for training coverage factor."""

FACTOR_WEIGHT_OUTPUT_CONSISTENCY: Final[float] = 0.2
"""Weight for output consistency factor."""
