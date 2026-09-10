"""
Real-Time Monitoring Module
Provides comprehensive monitoring and alerting using Prometheus
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None  # type: ignore[misc,assignment]
    logging.warning("Prometheus client not available. Monitoring will be limited.")


logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert notification"""

    alert_type: str
    message: str
    severity: str  # info, warning, critical
    timestamp: datetime
    metadata: Dict[str, Any]


class MetricsCollector:
    """Collect and expose metrics using an isolated Prometheus registry."""

    def __init__(self, enabled: bool = True, registry=None):
        self.enabled = bool(enabled and PROMETHEUS_AVAILABLE)
        self.registry = None

        if self.enabled:
            self.registry = registry or CollectorRegistry()
            self.request_counter = Counter(
                "agent_requests_total",
                "Total number of agent requests",
                ["agent", "status"],
                registry=self.registry,
            )
            self.request_latency = Histogram(
                "agent_request_latency_seconds",
                "Request latency in seconds",
                ["agent"],
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                registry=self.registry,
            )
            self.token_counter = Counter(
                "agent_tokens_total",
                "Total number of tokens processed",
                ["agent", "type"],
                registry=self.registry,
            )
            self.error_counter = Counter(
                "agent_errors_total",
                "Total number of errors",
                ["agent", "error_type"],
                registry=self.registry,
            )
            self.cache_hit_counter = Counter(
                "cache_hits_total",
                "Total cache hits",
                ["cache_level"],
                registry=self.registry,
            )
            self.cache_miss_counter = Counter(
                "cache_misses_total",
                "Total cache misses",
                ["cache_level"],
                registry=self.registry,
            )
            self.agent_utilization = Gauge(
                "agent_utilization",
                "Current agent utilization",
                ["agent"],
                registry=self.registry,
            )
            self.consensus_agreement = Histogram(
                "consensus_agreement_score",
                "Consensus agreement scores",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                registry=self.registry,
            )
            self.confidence_scores = Histogram(
                "confidence_scores",
                "Confidence scores distribution",
                ["agent"],
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                registry=self.registry,
            )
            self.queue_size = Gauge(
                "batch_queue_size",
                "Current batch queue size",
                registry=self.registry,
            )
            logger.info("Metrics collector initialized with Prometheus")
        else:
            logger.warning("Metrics collector initialized without Prometheus")

    def record_request(self, agent: str, status: str, latency: float):
        """Record a request"""
        if self.enabled:
            self.request_counter.labels(agent=agent, status=status).inc()
            self.request_latency.labels(agent=agent).observe(latency)

    def record_tokens(self, agent: str, token_type: str, count: int):
        """Record token count"""
        if self.enabled:
            self.token_counter.labels(agent=agent, type=token_type).inc(count)

    def record_error(self, agent: str, error_type: str):
        """Record an error"""
        if self.enabled:
            self.error_counter.labels(agent=agent, error_type=error_type).inc()

    def record_cache_hit(self, cache_level: str):
        """Record cache hit"""
        if self.enabled:
            self.cache_hit_counter.labels(cache_level=cache_level).inc()

    def record_cache_miss(self, cache_level: str):
        """Record cache miss"""
        if self.enabled:
            self.cache_miss_counter.labels(cache_level=cache_level).inc()

    def set_agent_utilization(self, agent: str, utilization: float):
        """Set agent utilization"""
        if self.enabled:
            self.agent_utilization.labels(agent=agent).set(utilization)

    def record_consensus_agreement(self, score: float):
        """Record consensus agreement score"""
        if self.enabled:
            self.consensus_agreement.observe(score)

    def record_confidence(self, agent: str, score: float):
        """Record confidence score"""
        if self.enabled:
            self.confidence_scores.labels(agent=agent).observe(score)

    def set_queue_size(self, size: int):
        """Set batch queue size"""
        if self.enabled:
            self.queue_size.set(size)


class AlertManager:
    """Manage and send alerts"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager

        Args:
            config: Alert configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.thresholds = {
            "latency_ms": config.get("latency_threshold_ms", 5000),
            "error_rate": config.get("error_rate_threshold", 0.1),
            "token_rate": config.get("token_rate_threshold", 1000),
        }

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Alert history (last 100 alerts)
        self.alert_history: List[Alert] = []
        self.max_history = 100

        logger.info(f"AlertManager initialized (enabled: {self.enabled})")

    def register_callback(self, callback: Callable[[Alert], None]):
        """
        Register alert callback

        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
        logger.info(f"Registered alert callback: {callback.__name__}")

    def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Send an alert

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Severity level
            metadata: Optional metadata
        """
        if not self.enabled:
            return

        alert = Alert(
            alert_type=alert_type,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # Log alert
        log_func = logger.warning if severity == "warning" else logger.critical
        log_func(f"ALERT [{severity.upper()}] {alert_type}: {message}")

        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def check_latency(self, latency_ms: float, agent: str):
        """Check latency threshold"""
        if latency_ms > self.thresholds["latency_ms"]:
            self.send_alert(
                alert_type="high_latency",
                message=f"High latency detected for {agent}: {latency_ms:.0f}ms",
                severity="warning",
                metadata={"agent": agent, "latency_ms": latency_ms},
            )

    def check_error_rate(self, error_rate: float, agent: str):
        """Check error rate threshold"""
        if error_rate > self.thresholds["error_rate"]:
            self.send_alert(
                alert_type="high_error_rate",
                message=f"High error rate for {agent}: {error_rate:.1%}",
                severity="critical",
                metadata={"agent": agent, "error_rate": error_rate},
            )

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts"""
        return self.alert_history[-limit:]


class AgentMonitor:
    """
    Comprehensive agent monitoring system

    Features:
    - Metrics collection (Prometheus)
    - Real-time alerting
    - Performance tracking
    - Usage analytics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agent monitor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.metrics_interval = config.get("metrics_interval", 10)
        self.prometheus_port = config.get("prometheus_port", 9090)

        # Initialize components
        self.metrics_collector = MetricsCollector(enabled=self.enabled)
        self.alert_manager = AlertManager(config.get("alerting", {}))

        # Tracking data
        self.request_stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

        # Start Prometheus server only when explicitly enabled
        if self.enabled and PROMETHEUS_AVAILABLE and self.metrics_collector.registry:
            try:
                start_http_server(self.prometheus_port, registry=self.metrics_collector.registry)
                logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")

        logger.info("AgentMonitor initialized")

    def track_inference(self, agent: str, task: str, result: Dict[str, Any]):
        """
        Track inference metrics

        Args:
            agent: Agent identifier
            task: Input task
            result: Inference result
        """
        latency_ms = result.get("latency_ms", 0)
        latency_s = latency_ms / 1000.0
        success = result.get("success", True)
        token_count = result.get("token_count", 0)
        confidence = result.get("confidence")

        status = "success" if success else "failure"
        if self.enabled:
            self.metrics_collector.record_request(agent, status, latency_s)
            if token_count > 0:
                self.metrics_collector.record_tokens(agent, "output", token_count)
            if confidence is not None:
                self.metrics_collector.record_confidence(agent, confidence)
            self.alert_manager.check_latency(latency_ms, agent)

        # Update stats
        with self.lock:
            if agent not in self.request_stats:
                self.request_stats[agent] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "total_latency_ms": 0,
                    "total_tokens": 0,
                }

            stats = self.request_stats[agent]
            stats["total_requests"] += 1
            if success:
                stats["successful_requests"] += 1
            else:
                stats["failed_requests"] += 1
                if self.enabled:
                    self.metrics_collector.record_error(agent, "inference_error")
            stats["total_latency_ms"] += latency_ms
            stats["total_tokens"] += token_count

            # Check error rate
            error_rate = stats["failed_requests"] / stats["total_requests"]
            if stats["total_requests"] >= 10:  # Only check after 10 requests
                self.alert_manager.check_error_rate(error_rate, agent)

    def track_cache_operation(self, cache_level: str, hit: bool):
        """
        Track cache operation

        Args:
            cache_level: Cache level (L1, L2, L3)
            hit: Whether it was a hit or miss
        """
        if not self.enabled:
            return

        if hit:
            self.metrics_collector.record_cache_hit(cache_level)
        else:
            self.metrics_collector.record_cache_miss(cache_level)

    def track_consensus(self, agreement_score: float, num_agents: int):
        """
        Track consensus operation

        Args:
            agreement_score: Consensus agreement score
            num_agents: Number of agents involved
        """
        if not self.enabled:
            return

        self.metrics_collector.record_consensus_agreement(agreement_score)

    def track_batch_queue(self, queue_size: int):
        """
        Track batch queue size

        Args:
            queue_size: Current queue size
        """
        if not self.enabled:
            return

        self.metrics_collector.set_queue_size(queue_size)

    def get_agent_stats(self, agent: str) -> Dict[str, Any]:
        """
        Get statistics for an agent

        Args:
            agent: Agent identifier

        Returns:
            Statistics dictionary
        """
        with self.lock:
            if agent not in self.request_stats:
                return {}

            stats = self.request_stats[agent].copy()

            # Calculate derived metrics
            if stats["total_requests"] > 0:
                stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
                stats["error_rate"] = stats["failed_requests"] / stats["total_requests"]
                stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
                stats["avg_tokens"] = stats["total_tokens"] / stats["total_requests"]
            else:
                stats["success_rate"] = 0.0
                stats["error_rate"] = 0.0
                stats["avg_latency_ms"] = 0.0
                stats["avg_tokens"] = 0.0

            return stats

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents"""
        with self.lock:
            return {agent: self.get_agent_stats(agent) for agent in self.request_stats.keys()}

    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register alert callback"""
        self.alert_manager.register_callback(callback)

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts"""
        return self.alert_manager.get_recent_alerts(limit)
