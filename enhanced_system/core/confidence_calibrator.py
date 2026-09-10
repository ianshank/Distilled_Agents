"""
Confidence Calibration Module
Provides calibrated confidence scores with explanations
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from enhanced_system.core.enums import ReliabilityBand

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Advanced calibration disabled.")


logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Result of confidence calibration"""
    confidence: float
    raw_confidence: float
    reliability_band: ReliabilityBand
    explanation: str
    factors: Dict[str, float]


class ConfidenceCalibrator:
    """
    Calibrate and explain confidence scores
    
    Features:
    - Historical accuracy-based calibration
    - Multi-factor confidence scoring
    - Reliability band calculation
    - Confidence explanation generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize confidence calibrator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.calibration_method = config.get('calibration_method', 'isotonic')
        self.min_samples = config.get('min_samples_for_calibration', 100)
        
        # Reliability bands
        bands = config.get('reliability_bands', {})
        self.reliability_bands = {
            ReliabilityBand.LOW: tuple(bands.get('low', [0.0, 0.5])),
            ReliabilityBand.MEDIUM: tuple(bands.get('medium', [0.5, 0.75])),
            ReliabilityBand.HIGH: tuple(bands.get('high', [0.75, 0.9])),
            ReliabilityBand.VERY_HIGH: tuple(bands.get('very_high', [0.9, 1.0])),
        }
        
        # Historical data for calibration
        self.calibration_data: Dict[str, Dict[str, List]] = {}
        # Format: {agent: {task_type: [(raw_conf, actual_accuracy), ...]}}
        
        # Calibration models
        self.calibration_models: Dict[str, Any] = {}
        
        logger.info(f"ConfidenceCalibrator initialized (method: {self.calibration_method})")
    
    def calibrate_confidence(
        self,
        raw_confidence: float,
        agent: str,
        task_type: str = "default",
        result: Optional[Dict[str, Any]] = None
    ) -> ConfidenceResult:
        """
        Calibrate confidence score
        
        Args:
            raw_confidence: Raw confidence score (0-1)
            agent: Agent identifier
            task_type: Type of task
            result: Optional result dictionary with additional info
        
        Returns:
            ConfidenceResult with calibrated confidence
        """
        if not self.enabled:
            return ConfidenceResult(
                confidence=raw_confidence,
                raw_confidence=raw_confidence,
                reliability_band=self._get_reliability_band(raw_confidence),
                explanation="Calibration disabled",
                factors={}
            )
        
        # Get calibration function for this agent/task_type
        calibrated = self._apply_calibration(raw_confidence, agent, task_type)
        
        # Calculate confidence factors
        factors = self._calculate_confidence_factors(
            agent, task_type, result or {}
        )
        
        # Adjust calibrated confidence based on factors
        final_confidence = self._adjust_for_factors(calibrated, factors)
        
        # Clamp to [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Get reliability band
        reliability_band = self._get_reliability_band(final_confidence)
        
        # Generate explanation
        explanation = self._generate_explanation(
            raw_confidence, final_confidence, factors, reliability_band
        )
        
        return ConfidenceResult(
            confidence=final_confidence,
            raw_confidence=raw_confidence,
            reliability_band=reliability_band,
            explanation=explanation,
            factors=factors
        )
    
    def _apply_calibration(
        self,
        raw_confidence: float,
        agent: str,
        task_type: str
    ) -> float:
        """
        Apply calibration function
        
        Args:
            raw_confidence: Raw confidence
            agent: Agent identifier
            task_type: Task type
        
        Returns:
            Calibrated confidence
        """
        model_key = f"{agent}_{task_type}"
        
        # Check if we have a calibration model
        if model_key in self.calibration_models:
            model = self.calibration_models[model_key]
            try:
                if SKLEARN_AVAILABLE and hasattr(model, 'predict'):
                    calibrated = model.predict([raw_confidence])[0]
                    return float(calibrated)
            except Exception as e:
                logger.warning(f"Calibration model prediction failed: {e}")
        
        # No calibration model available, use simple adjustment
        # Apply conservative adjustment (pull towards 0.5)
        adjustment_factor = 0.1
        calibrated = raw_confidence * (1 - adjustment_factor) + 0.5 * adjustment_factor
        
        return calibrated
    
    def _calculate_confidence_factors(
        self,
        agent: str,
        task_type: str,
        result: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate multi-factor confidence components
        
        Args:
            agent: Agent identifier
            task_type: Task type
            result: Result dictionary
        
        Returns:
            Dictionary of confidence factors
        """
        factors = {}
        
        # Agent reliability factor (based on historical performance)
        factors['agent_reliability'] = self._get_agent_reliability(agent)
        
        # Task difficulty factor
        task = result.get('task', '')
        factors['task_difficulty'] = self._estimate_task_difficulty(task)
        
        # Training data coverage factor
        factors['training_coverage'] = self._check_training_coverage(agent, task_type)
        
        # Output consistency factor
        if 'output' in result:
            factors['output_consistency'] = self._check_output_consistency(result['output'])
        else:
            factors['output_consistency'] = 0.5
        
        return factors
    
    def _get_agent_reliability(self, agent: str) -> float:
        """
        Get historical reliability for agent
        
        Args:
            agent: Agent identifier
        
        Returns:
            Reliability score (0-1)
        """
        if agent not in self.calibration_data:
            return 0.5  # Unknown agent, neutral reliability
        
        # Calculate average accuracy across all task types
        accuracies = []
        for task_type, data_points in self.calibration_data[agent].items():
            if data_points:
                # Extract actual accuracies
                accuracies.extend([acc for _, acc in data_points])
        
        if accuracies:
            return np.mean(accuracies)
        else:
            return 0.5
    
    def _estimate_task_difficulty(self, task: str) -> float:
        """
        Estimate task difficulty (higher = easier, more confident)
        
        Args:
            task: Task string
        
        Returns:
            Difficulty score (0-1, higher = easier)
        """
        # Simple heuristic based on task length and complexity indicators
        if not task:
            return 0.5
        
        difficulty_score = 0.7  # Default: moderate difficulty
        
        # Shorter tasks are often simpler
        if len(task) < 50:
            difficulty_score += 0.1
        elif len(task) > 500:
            difficulty_score -= 0.1
        
        # Check for complexity indicators
        complexity_keywords = [
            'complex', 'advanced', 'detailed', 'comprehensive',
            'analyze', 'evaluate', 'compare', 'design'
        ]
        
        task_lower = task.lower()
        complexity_count = sum(1 for keyword in complexity_keywords if keyword in task_lower)
        
        if complexity_count > 2:
            difficulty_score -= 0.15
        
        return max(0.1, min(0.9, difficulty_score))
    
    def _check_training_coverage(self, agent: str, task_type: str) -> float:
        """
        Check training data coverage for agent/task_type
        
        Args:
            agent: Agent identifier
            task_type: Task type
        
        Returns:
            Coverage score (0-1)
        """
        # Simple heuristic: return 0.7 if we have calibration data, 0.4 otherwise
        if agent in self.calibration_data:
            if task_type in self.calibration_data[agent]:
                if len(self.calibration_data[agent][task_type]) >= self.min_samples:
                    return 0.9
                else:
                    return 0.6
        
        return 0.4
    
    def _check_output_consistency(self, output: Any) -> float:
        """
        Check consistency/quality of output
        
        Args:
            output: Output to check
        
        Returns:
            Consistency score (0-1)
        """
        if isinstance(output, str):
            # Check for common quality indicators
            if len(output) == 0:
                return 0.1
            
            # Check for incomplete output markers
            incomplete_markers = ['...', '[incomplete]', '[error]', 'I cannot', 'I apologize']
            output_lower = output.lower()
            
            for marker in incomplete_markers:
                if marker in output_lower:
                    return 0.4
            
            # Check reasonable length
            if len(output) < 10:
                return 0.5
            elif len(output) > 50:
                return 0.8
            
            return 0.7
        
        return 0.5
    
    def _adjust_for_factors(self, calibrated: float, factors: Dict[str, float]) -> float:
        """
        Adjust calibrated confidence based on factors
        
        Args:
            calibrated: Calibrated confidence
            factors: Confidence factors
        
        Returns:
            Adjusted confidence
        """
        # Weighted combination of factors
        weights = {
            'agent_reliability': 0.3,
            'task_difficulty': 0.2,
            'training_coverage': 0.3,
            'output_consistency': 0.2
        }
        
        # Calculate weighted factor score
        factor_score = sum(
            factors.get(factor, 0.5) * weight
            for factor, weight in weights.items()
        )
        
        # Adjust calibrated confidence
        # If factors are positive, boost confidence; if negative, reduce
        adjustment = (factor_score - 0.5) * 0.3  # ±15% adjustment
        adjusted = calibrated + adjustment
        
        return adjusted
    
    def _get_reliability_band(self, confidence: float) -> ReliabilityBand:
        """
        Get reliability band for confidence score
        
        Args:
            confidence: Confidence score
        
        Returns:
            ReliabilityBand
        """
        for band, (low, high) in self.reliability_bands.items():
            if low <= confidence < high:
                return band
        
        # Handle edge case for exactly 1.0
        if confidence >= self.reliability_bands[ReliabilityBand.VERY_HIGH][0]:
            return ReliabilityBand.VERY_HIGH
        
        return ReliabilityBand.LOW
    
    def _generate_explanation(
        self,
        raw_confidence: float,
        final_confidence: float,
        factors: Dict[str, float],
        reliability_band: ReliabilityBand
    ) -> str:
        """
        Generate human-readable confidence explanation
        
        Args:
            raw_confidence: Raw confidence
            final_confidence: Final calibrated confidence
            factors: Confidence factors
            reliability_band: Reliability band
        
        Returns:
            Explanation string
        """
        explanation_parts = [
            f"Confidence: {final_confidence:.2%} ({reliability_band.value})"
        ]
        
        # Mention calibration adjustment
        adjustment = final_confidence - raw_confidence
        if abs(adjustment) > 0.05:
            direction = "increased" if adjustment > 0 else "decreased"
            explanation_parts.append(
                f"Calibrated from {raw_confidence:.2%} ({direction} by {abs(adjustment):.1%})"
            )
        
        # Highlight key factors
        agent_rel = factors.get('agent_reliability', 0.5)
        if agent_rel > 0.8:
            explanation_parts.append("High agent reliability")
        elif agent_rel < 0.4:
            explanation_parts.append("Limited agent reliability data")
        
        task_diff = factors.get('task_difficulty', 0.5)
        if task_diff < 0.4:
            explanation_parts.append("Complex task detected")
        elif task_diff > 0.8:
            explanation_parts.append("Straightforward task")
        
        coverage = factors.get('training_coverage', 0.5)
        if coverage < 0.5:
            explanation_parts.append("Limited training data for this task type")
        
        return "; ".join(explanation_parts)
    
    def update_calibration_data(
        self,
        agent: str,
        task_type: str,
        predicted_confidence: float,
        actual_accuracy: float
    ) -> None:
        """
        Update calibration data with new observation
        
        Args:
            agent: Agent identifier
            task_type: Task type
            predicted_confidence: Predicted confidence
            actual_accuracy: Actual accuracy achieved
        """
        if agent not in self.calibration_data:
            self.calibration_data[agent] = {}
        
        if task_type not in self.calibration_data[agent]:
            self.calibration_data[agent][task_type] = []
        
        self.calibration_data[agent][task_type].append(
            (predicted_confidence, actual_accuracy)
        )
        
        # Retrain calibration model if we have enough data
        model_key = f"{agent}_{task_type}"
        data_points = self.calibration_data[agent][task_type]
        
        if len(data_points) >= self.min_samples and SKLEARN_AVAILABLE:
            try:
                X = np.array([conf for conf, _ in data_points]).reshape(-1, 1)
                y = np.array([acc for _, acc in data_points])
                
                if self.calibration_method == 'isotonic':
                    model = IsotonicRegression(out_of_bounds='clip')
                    model.fit(X.ravel(), y)
                else:
                    # Platt scaling or other methods
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression()
                    model.fit(X, y)
                
                self.calibration_models[model_key] = model
                logger.info(f"Updated calibration model for {model_key}")
            
            except Exception as e:
                logger.error(f"Failed to train calibration model: {e}")

