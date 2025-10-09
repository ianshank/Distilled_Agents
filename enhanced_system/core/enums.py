"""
Enums Module
============

Centralized enumeration types for the enhanced agent system.
Replaces magic strings with type-safe enums.

This module follows 2025 Python coding standards:
- String-based enums for JSON serialization
- Clear, self-documenting values
- Comprehensive documentation
"""

from enum import Enum


class CacheLevel(str, Enum):
    """Cache level identifiers."""
    
    L1 = "l1"
    """In-memory LRU cache."""
    
    L2 = "l2"
    """Redis distributed cache."""
    
    L3 = "l3"
    """S3 persistent cache."""


class CalibrationMethod(str, Enum):
    """Methods for confidence calibration."""
    
    ISOTONIC = "isotonic"
    """Isotonic regression calibration."""
    
    PLATT = "platt"
    """Platt scaling calibration."""
    
    LINEAR = "linear"
    """Linear calibration."""


class AgreementMethod(str, Enum):
    """Methods for calculating agreement in consensus."""
    
    EMBEDDING_SIMILARITY = "embedding_similarity"
    """Use embedding cosine similarity."""
    
    EXACT_MATCH = "exact_match"
    """Use exact string matching."""


class EnsembleMethod(str, Enum):
    """Methods for ensemble response generation."""
    
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    """Weight responses by confidence."""
    
    MAJORITY_VOTE = "majority_vote"
    """Use majority voting."""


class RoutingStrategy(str, Enum):
    """Strategies for agent routing optimization."""
    
    COST_OPTIMIZED = "cost_optimized"
    """Minimize cost."""
    
    QUALITY_OPTIMIZED = "quality_optimized"
    """Maximize quality."""
    
    BALANCED = "balanced"
    """Balance cost and quality."""


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    
    SIMPLE = "simple"
    """Simple, straightforward tasks."""
    
    MEDIUM = "medium"
    """Moderate complexity tasks."""
    
    COMPLEX = "complex"
    """Complex, multi-step tasks."""


class Priority(str, Enum):
    """Request priority levels for batch processing."""
    
    LOW = "low"
    """Low priority."""
    
    NORMAL = "normal"
    """Normal priority."""
    
    HIGH = "high"
    """High priority."""
    
    CRITICAL = "critical"
    """Critical priority."""
    
    @property
    def numeric_value(self) -> int:
        """Get numeric priority value (lower = higher priority)."""
        priority_map = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.NORMAL: 2,
            Priority.LOW: 3,
        }
        return priority_map[self]


class ErrorType(str, Enum):
    """Classification of error types."""
    
    TEMPORARY = "temporary"
    """Retryable errors (network, timeout, rate limit)."""
    
    PERMANENT = "permanent"
    """Non-retryable errors (invalid input, auth failure)."""
    
    UNKNOWN = "unknown"
    """Unclassified errors."""


class TrainingStatus(str, Enum):
    """Training job status."""
    
    PENDING = "pending"
    """Job waiting to start."""
    
    RUNNING = "running"
    """Job currently running."""
    
    COMPLETED = "completed"
    """Job completed successfully."""
    
    FAILED = "failed"
    """Job failed."""
    
    CANCELLED = "cancelled"
    """Job cancelled by user."""


class ExperimentStatus(str, Enum):
    """A/B test experiment status."""
    
    RUNNING = "running"
    """Experiment currently active."""
    
    COMPLETED = "completed"
    """Experiment completed."""
    
    CANCELLED = "cancelled"
    """Experiment cancelled."""


class MetricType(str, Enum):
    """Types of metrics to track."""
    
    LATENCY = "latency"
    """Response latency."""
    
    ACCURACY = "accuracy"
    """Prediction accuracy."""
    
    COST = "cost"
    """Processing cost."""
    
    THROUGHPUT = "throughput"
    """Requests per second."""
    
    ERROR_RATE = "error_rate"
    """Error rate."""
    
    CACHE_HIT_RATE = "cache_hit_rate"
    """Cache hit rate."""


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    
    INFO = "info"
    """Informational alert."""
    
    WARNING = "warning"
    """Warning alert."""
    
    CRITICAL = "critical"
    """Critical alert."""


class ReliabilityBand(str, Enum):
    """Reliability bands for confidence scores."""
    
    LOW = "low"
    """Low reliability (0.0-0.5)."""
    
    MEDIUM = "medium"
    """Medium reliability (0.5-0.75)."""
    
    HIGH = "high"
    """High reliability (0.75-0.9)."""
    
    VERY_HIGH = "very_high"
    """Very high reliability (0.9-1.0)."""

