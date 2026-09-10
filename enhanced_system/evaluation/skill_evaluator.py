"""
Skill Evaluation Module
Provides multi-dimensional assessment of agent skills
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of skill evaluation"""

    agent_name: str
    timestamp: datetime
    overall_score: float
    dimension_scores: Dict[str, float]
    test_results: List[Dict[str, Any]]
    baseline_comparison: Optional[Dict[str, float]] = None
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class SkillEvaluator:
    """
    Comprehensive agent skill evaluation

    Features:
    - Multi-dimensional assessment
    - Baseline comparison
    - Regression detection
    - Skill coverage analysis
    """

    EVALUATION_DIMENSIONS = [
        "accuracy",
        "reliability",
        "consistency",
        "speed",
        "cost",
        "safety",
        "skill_coverage",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize skill evaluator

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.baselines: Dict[str, EvaluationResult] = {}
        self.regression_threshold = self.config.get("regression_threshold", 0.05)

        logger.info("SkillEvaluator initialized")

    def evaluate_agent(
        self, agent_name: str, test_suite: List[Dict[str, Any]], agent_func: callable
    ) -> EvaluationResult:
        """
        Evaluate agent across multiple dimensions

        Args:
            agent_name: Name of agent to evaluate
            test_suite: List of test cases
            agent_func: Agent inference function

        Returns:
            EvaluationResult with comprehensive assessment
        """
        logger.info(f"Evaluating agent: {agent_name} ({len(test_suite)} test cases)")

        # Run test suite
        test_results = []
        for test_case in test_suite:
            result = self._run_test_case(agent_func, test_case)
            test_results.append(result)

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(test_results)

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)

        # Get baseline comparison if available
        baseline_comparison = None
        if agent_name in self.baselines:
            baseline_comparison = self._compare_to_baseline(agent_name, dimension_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, baseline_comparison)

        result = EvaluationResult(
            agent_name=agent_name,
            timestamp=datetime.now(),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            test_results=test_results,
            baseline_comparison=baseline_comparison,
            recommendations=recommendations,
        )

        logger.info(f"Evaluation complete for {agent_name}: overall_score={overall_score:.3f}")

        return result

    def _run_test_case(self, agent_func: callable, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case

        Args:
            agent_func: Agent inference function
            test_case: Test case specification

        Returns:
            Test result dictionary
        """
        test_id = test_case.get("id", "unknown")
        prompt = test_case.get("prompt", "")
        expected = test_case.get("expected", "")

        start_time = time.time()

        try:
            # Run agent
            response = agent_func(prompt)

            latency_ms = (time.time() - start_time) * 1000

            # Evaluate response
            accuracy = self._evaluate_accuracy(response, expected)

            return {
                "test_id": test_id,
                "passed": accuracy > 0.7,
                "accuracy": accuracy,
                "latency_ms": latency_ms,
                "prompt": prompt,
                "response": response,
                "expected": expected,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Test case {test_id} failed: {e}")

            return {
                "test_id": test_id,
                "passed": False,
                "accuracy": 0.0,
                "latency_ms": (time.time() - start_time) * 1000,
                "prompt": prompt,
                "response": None,
                "expected": expected,
                "error": str(e),
            }

    def _evaluate_accuracy(self, response: Any, expected: str) -> float:
        """
        Evaluate accuracy of response

        Args:
            response: Agent response
            expected: Expected response

        Returns:
            Accuracy score (0-1)
        """
        if response is None:
            return 0.0

        # Convert to string if needed
        response_str = str(response) if not isinstance(response, str) else response

        # Simple exact match
        if response_str.strip().lower() == expected.strip().lower():
            return 1.0

        # Partial match based on word overlap
        response_words = set(response_str.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 0.5

        overlap = len(response_words & expected_words)
        union = len(response_words | expected_words)

        if union == 0:
            return 0.0

        # Jaccard similarity
        similarity = overlap / union

        return similarity

    def _calculate_dimension_scores(self, test_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate scores for each evaluation dimension

        Args:
            test_results: List of test results

        Returns:
            Dictionary of dimension scores
        """
        scores = {}

        # Accuracy: average accuracy across tests
        accuracies = [r["accuracy"] for r in test_results]
        scores["accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0

        # Reliability: percentage of tests that passed
        passed = sum(1 for r in test_results if r["passed"])
        scores["reliability"] = passed / len(test_results) if test_results else 0.0

        # Consistency: standard deviation of accuracies (lower is better)
        if len(accuracies) > 1:
            import statistics

            std_dev = statistics.stdev(accuracies)
            scores["consistency"] = max(0.0, 1.0 - std_dev)
        else:
            scores["consistency"] = 1.0

        # Speed: based on latency (lower is better)
        latencies = [r["latency_ms"] for r in test_results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        # Normalize: 100ms = 1.0, 1000ms = 0.5, 5000ms = 0.0
        scores["speed"] = max(0.0, 1.0 - (avg_latency / 5000.0))

        # Cost: placeholder (would need actual cost tracking)
        scores["cost"] = 0.8

        # Safety: percentage of tests without errors
        no_errors = sum(1 for r in test_results if r["error"] is None)
        scores["safety"] = no_errors / len(test_results) if test_results else 0.0

        # Skill coverage: based on test diversity (placeholder)
        scores["skill_coverage"] = 0.75

        return scores

    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall score

        Args:
            dimension_scores: Dictionary of dimension scores

        Returns:
            Overall score (0-1)
        """
        # Weights for each dimension
        weights = {
            "accuracy": 0.25,
            "reliability": 0.20,
            "consistency": 0.15,
            "speed": 0.10,
            "cost": 0.10,
            "safety": 0.15,
            "skill_coverage": 0.05,
        }

        total_score = 0.0
        total_weight = 0.0

        for dimension, weight in weights.items():
            if dimension in dimension_scores:
                total_score += dimension_scores[dimension] * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def set_baseline(self, agent_name: str, result: EvaluationResult):
        """
        Set baseline evaluation for agent

        Args:
            agent_name: Agent name
            result: Baseline evaluation result
        """
        self.baselines[agent_name] = result
        logger.info(f"Set baseline for {agent_name}: {result.overall_score:.3f}")

    def _compare_to_baseline(
        self, agent_name: str, current_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare current scores to baseline

        Args:
            agent_name: Agent name
            current_scores: Current dimension scores

        Returns:
            Dictionary of deltas from baseline
        """
        if agent_name not in self.baselines:
            return {}

        baseline = self.baselines[agent_name]
        baseline_scores = baseline.dimension_scores

        comparison = {}
        for dimension in current_scores.keys():
            if dimension in baseline_scores:
                delta = current_scores[dimension] - baseline_scores[dimension]
                comparison[dimension] = delta

        return comparison

    def detect_regression(self, agent_name: str, current_result: EvaluationResult) -> List[str]:
        """
        Detect regressions compared to baseline

        Args:
            agent_name: Agent name
            current_result: Current evaluation result

        Returns:
            List of regression warnings
        """
        if agent_name not in self.baselines:
            return []

        regressions = []
        comparison = current_result.baseline_comparison or {}

        for dimension, delta in comparison.items():
            if delta < -self.regression_threshold:
                regressions.append(
                    f"{dimension}: {delta:.2%} below baseline "
                    f"(threshold: {self.regression_threshold:.2%})"
                )

        return regressions

    def _generate_recommendations(
        self, dimension_scores: Dict[str, float], baseline_comparison: Optional[Dict[str, float]]
    ) -> List[str]:
        """
        Generate improvement recommendations

        Args:
            dimension_scores: Current dimension scores
            baseline_comparison: Comparison to baseline

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for low scores
        for dimension, score in dimension_scores.items():
            if score < 0.7:
                if dimension == "accuracy":
                    recommendations.append("Consider additional training data or fine-tuning")
                elif dimension == "reliability":
                    recommendations.append("Improve error handling and add fallback mechanisms")
                elif dimension == "speed":
                    recommendations.append("Optimize inference pipeline or use model quantization")
                elif dimension == "consistency":
                    recommendations.append(
                        "Review temperature settings and add confidence thresholds"
                    )

        # Check for regressions
        if baseline_comparison:
            for dimension, delta in baseline_comparison.items():
                if delta < -0.1:
                    recommendations.append(
                        f"REGRESSION: {dimension} decreased by {abs(delta):.1%} - "
                        f"investigate recent changes"
                    )

        return recommendations

    def generate_report(self, result: EvaluationResult) -> str:
        """
        Generate human-readable evaluation report

        Args:
            result: Evaluation result

        Returns:
            Report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                   AGENT EVALUATION REPORT                    ║
╠══════════════════════════════════════════════════════════════╣
║ Agent: {result.agent_name:50} ║
║ Date: {result.timestamp.strftime("%Y-%m-%d %H:%M:%S"):51} ║
║ Overall Score: {result.overall_score:.3f}/1.000{" " * 38} ║
╠══════════════════════════════════════════════════════════════╣
║                      DIMENSION SCORES                         ║
╠══════════════════════════════════════════════════════════════╣
"""

        for dimension, score in sorted(result.dimension_scores.items()):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            report += f"║ {dimension:20} {score:.3f} [{bar}] ║\n"

        report += "╠══════════════════════════════════════════════════════════════╣\n"
        report += "║                      TEST SUMMARY                            ║\n"
        report += "╠══════════════════════════════════════════════════════════════╣\n"

        total_tests = len(result.test_results)
        passed_tests = sum(1 for r in result.test_results if r["passed"])
        report += f"║ Total Tests: {total_tests:47} ║\n"
        report += f"║ Passed: {passed_tests:52} ║\n"
        report += f"║ Failed: {total_tests - passed_tests:52} ║\n"

        if result.recommendations:
            report += "╠══════════════════════════════════════════════════════════════╣\n"
            report += "║                     RECOMMENDATIONS                          ║\n"
            report += "╠══════════════════════════════════════════════════════════════╣\n"
            for rec in result.recommendations:
                # Wrap long recommendations
                words = rec.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 <= 58:
                        line += " " + word if line else word
                    else:
                        report += f"║ • {line:57} ║\n"
                        line = word
                if line:
                    report += f"║ • {line:57} ║\n"

        report += "╚══════════════════════════════════════════════════════════════╝\n"

        return report
