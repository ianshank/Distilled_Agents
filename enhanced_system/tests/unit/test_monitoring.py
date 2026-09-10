"""Unit tests for AgentMonitor without Prometheus."""

from __future__ import annotations

import pytest
from enhanced_system.core.monitoring import AgentMonitor, AlertManager


@pytest.mark.unit
class TestAgentMonitor:
    def test_track_inference_updates_stats(self):
        monitor = AgentMonitor({"enabled": False})
        monitor.track_inference(
            "swe_agent",
            "task",
            {"latency_ms": 12, "success": True, "token_count": 10, "confidence": 0.8},
        )
        stats = monitor.get_agent_stats("swe_agent")
        assert stats["total_requests"] == 1
        assert stats["success_rate"] == 1.0

    def test_alert_manager_history(self):
        manager = AlertManager({"enabled": True, "latency_threshold_ms": 1})
        manager.check_latency(5000, "swe_agent")
        alerts = manager.get_recent_alerts()
        assert alerts
        assert alerts[-1].alert_type == "high_latency"
