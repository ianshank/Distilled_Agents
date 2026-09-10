"""Unit tests for confidence calibration."""

from __future__ import annotations

import pytest
from enhanced_system.core.confidence_calibrator import ConfidenceCalibrator, ConfidenceResult
from enhanced_system.core.enums import ReliabilityBand


@pytest.mark.unit
class TestConfidenceCalibrator:
    def test_calibrate_returns_result(self):
        calibrator = ConfidenceCalibrator({"enabled": True})
        result = calibrator.calibrate_confidence(0.9, agent="swe_agent", task_type="coding")
        assert isinstance(result, ConfidenceResult)
        assert 0 <= result.confidence <= 1
        assert result.reliability_band in ReliabilityBand

    def test_disabled_passthrough(self):
        calibrator = ConfidenceCalibrator({"enabled": False})
        result = calibrator.calibrate_confidence(0.42, agent="x", task_type="y")
        assert result.confidence == 0.42
        assert result.explanation == "Calibration disabled"
