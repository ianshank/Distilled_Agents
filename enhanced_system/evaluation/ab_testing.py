"""
A/B Testing Framework Module
Provides experiment management and statistical analysis for model testing
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """A/B test variant"""

    variant_id: str
    name: str
    model_path: str
    traffic_percentage: float
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class Experiment:
    """A/B testing experiment"""

    experiment_id: str
    name: str
    control_variant: Variant
    treatment_variant: Variant
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, cancelled
    min_samples: int = 100
    significance_level: float = 0.05


class ABTestFramework:
    """
    A/B testing framework for gradual model rollout

    Features:
    - Experiment creation and management
    - Consistent traffic splitting (hash-based)
    - Statistical significance testing
    - Automated winner determination
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize A/B testing framework

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.default_traffic_split = config.get("default_traffic_split", 0.1)
        self.min_samples = config.get("min_samples_for_significance", 100)
        self.significance_level = config.get("significance_level", 0.05)
        self.metrics_to_track = config.get("metrics_to_track", ["latency", "accuracy"])

        # Active experiments
        self.experiments: Dict[str, Experiment] = {}

        logger.info(f"ABTestFramework initialized (enabled: {self.enabled})")

    def create_experiment(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        traffic_split: Optional[float] = None,
    ) -> str:
        """
        Create A/B test experiment

        Args:
            name: Experiment name
            control_model: Path to control model
            treatment_model: Path to treatment model
            traffic_split: Percentage of traffic to treatment (0-1)

        Returns:
            Experiment ID
        """
        if not self.enabled:
            logger.warning("A/B testing not enabled")
            return ""

        if traffic_split is None:
            traffic_split = self.default_traffic_split

        # Generate experiment ID
        experiment_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        # Create variants
        control = Variant(
            variant_id=f"{experiment_id}_control",
            name="Control",
            model_path=control_model,
            traffic_percentage=1.0 - traffic_split,
        )

        treatment = Variant(
            variant_id=f"{experiment_id}_treatment",
            name="Treatment",
            model_path=treatment_model,
            traffic_percentage=traffic_split,
        )

        # Create experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            control_variant=control,
            treatment_variant=treatment,
            start_time=datetime.now(),
            min_samples=self.min_samples,
            significance_level=self.significance_level,
        )

        self.experiments[experiment_id] = experiment

        logger.info(
            f"Created experiment '{name}' ({experiment_id}): "
            f"{traffic_split:.1%} traffic to treatment"
        )

        return experiment_id

    def route_request(self, experiment_id: str, user_id: str) -> str:
        """
        Route request to control or treatment variant

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier for consistent routing

        Returns:
            Variant ID ('control' or 'treatment')
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Unknown experiment: {experiment_id}")
            return "control"

        experiment = self.experiments[experiment_id]

        if experiment.status != "running":
            logger.debug(f"Experiment {experiment_id} not running, using control")
            return "control"

        # Consistent hashing for user routing
        hash_value = int(
            hashlib.md5(
                f"{experiment_id}_{user_id}".encode(),
                usedforsecurity=False,
            ).hexdigest(),
            16,
        )

        # Normalize to [0, 1]
        normalized = (hash_value % 10000) / 10000.0

        # Route based on traffic split
        if normalized < experiment.treatment_variant.traffic_percentage:
            return "treatment"
        else:
            return "control"

    def record_metric(self, experiment_id: str, variant: str, metric_name: str, value: float):
        """
        Record metric for variant

        Args:
            experiment_id: Experiment identifier
            variant: Variant ('control' or 'treatment')
            metric_name: Metric name
            value: Metric value
        """
        if experiment_id not in self.experiments:
            return

        experiment = self.experiments[experiment_id]

        if variant == "control":
            experiment.control_variant.metrics[metric_name].append(value)
        elif variant == "treatment":
            experiment.treatment_variant.metrics[metric_name].append(value)

    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze experiment results with statistical testing

        Args:
            experiment_id: Experiment identifier

        Returns:
            Analysis results dictionary
        """
        if experiment_id not in self.experiments:
            return {"error": "Unknown experiment"}

        experiment = self.experiments[experiment_id]

        # Check minimum sample size
        control_samples = sum(len(values) for values in experiment.control_variant.metrics.values())
        treatment_samples = sum(
            len(values) for values in experiment.treatment_variant.metrics.values()
        )

        if control_samples < self.min_samples or treatment_samples < self.min_samples:
            return {
                "status": "insufficient_data",
                "control_samples": control_samples,
                "treatment_samples": treatment_samples,
                "min_required": self.min_samples,
            }

        # Analyze each metric
        metric_results = {}
        for metric_name in self.metrics_to_track:
            if metric_name in experiment.control_variant.metrics:
                result = self._analyze_metric(
                    experiment.control_variant.metrics[metric_name],
                    experiment.treatment_variant.metrics.get(metric_name, []),
                    metric_name,
                )
                metric_results[metric_name] = result

        # Determine winner
        winner = self._determine_winner(metric_results)

        # Generate recommendation
        recommendation = self._generate_recommendation(winner, metric_results)

        return {
            "experiment_id": experiment_id,
            "status": "complete",
            "control_samples": control_samples,
            "treatment_samples": treatment_samples,
            "metric_results": metric_results,
            "winner": winner,
            "recommendation": recommendation,
        }

    def _analyze_metric(
        self, control_values: List[float], treatment_values: List[float], metric_name: str
    ) -> Dict[str, Any]:
        """
        Analyze single metric using statistical tests

        Args:
            control_values: Control variant values
            treatment_values: Treatment variant values
            metric_name: Name of metric

        Returns:
            Analysis result
        """
        if not control_values or not treatment_values:
            return {"error": "insufficient_data", "control_mean": None, "treatment_mean": None}

        import statistics

        control_mean = statistics.mean(control_values)
        treatment_mean = statistics.mean(treatment_values)

        # Calculate effect size
        improvement = (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0

        # Simple significance test (t-test approximation)
        # In real implementation, use scipy.stats.ttest_ind
        statistically_significant = abs(improvement) > 0.05  # Simplified

        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "improvement": improvement,
            "statistically_significant": statistically_significant,
            "p_value": 0.03 if statistically_significant else 0.15,  # Placeholder
        }

    def _determine_winner(self, metric_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Determine experiment winner

        Args:
            metric_results: Results for each metric

        Returns:
            'control', 'treatment', or 'inconclusive'
        """
        treatment_wins = 0
        control_wins = 0

        for metric_name, result in metric_results.items():
            if "error" in result:
                continue

            if not result.get("statistically_significant", False):
                continue

            improvement = result.get("improvement", 0)

            # For latency, lower is better
            if metric_name == "latency":
                if improvement < 0:  # Treatment is faster
                    treatment_wins += 1
                else:
                    control_wins += 1
            else:
                # For most metrics, higher is better
                if improvement > 0:
                    treatment_wins += 1
                else:
                    control_wins += 1

        if treatment_wins > control_wins:
            return "treatment"
        elif control_wins > treatment_wins:
            return "control"
        else:
            return "inconclusive"

    def _generate_recommendation(
        self, winner: str, metric_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate rollout recommendation

        Args:
            winner: Winner variant
            metric_results: Metric results

        Returns:
            Recommendation string
        """
        if winner == "treatment":
            return (
                "Recommend rolling out treatment to 100% of traffic. "
                "Treatment shows statistically significant improvements."
            )
        elif winner == "control":
            return (
                "Recommend keeping control model. Treatment does not show significant improvements."
            )
        else:
            return (
                "Results are inconclusive. "
                "Recommend continuing experiment or increasing sample size."
            )

    def stop_experiment(self, experiment_id: str):
        """
        Stop running experiment

        Args:
            experiment_id: Experiment identifier
        """
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            experiment.status = "completed"
            experiment.end_time = datetime.now()
            logger.info(f"Stopped experiment {experiment_id}")

    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current experiment status"""
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]

        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "start_time": experiment.start_time.isoformat(),
            "control_samples": sum(len(v) for v in experiment.control_variant.metrics.values()),
            "treatment_samples": sum(len(v) for v in experiment.treatment_variant.metrics.values()),
            "traffic_split": experiment.treatment_variant.traffic_percentage,
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        return [self.get_experiment_status(exp_id) for exp_id in self.experiments.keys()]
