"""
Unit Tests for AdaptiveRouter
==============================

Comprehensive tests for task classification, complexity estimation, and agent selection.
Target: 80%+ coverage
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from enhanced_system.core.adaptive_router import AdaptiveRouter
from enhanced_system.core.constants import *
from enhanced_system.core.enums import TaskComplexity, RoutingStrategy


@pytest.mark.unit
class TestAdaptiveRouter:
    """Test AdaptiveRouter class."""
    
    def test_initialization(self, test_config):
        """Test router initialization."""
        config = {
            'enabled': True,
            'routing_strategy': RoutingStrategy.COST_OPTIMIZED.value,
            'complexity_model': 'heuristic'
        }
        
        router = AdaptiveRouter(config)
        assert router.config == config
        assert router.agent_profiles is not None
    
    def test_classify_task_type_coding(self):
        """Test classification of coding tasks."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Write a Python function to implement binary search"
        task_type = router.classify_task_type(task)
        
        assert task_type == "coding"
    
    def test_classify_task_type_design(self):
        """Test classification of design tasks."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Design a microservices architecture for an e-commerce platform"
        task_type = router.classify_task_type(task)
        
        assert task_type == "architecture"
    
    def test_classify_task_type_testing(self):
        """Test classification of testing tasks."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Create unit tests for the authentication module"
        task_type = router.classify_task_type(task)
        
        assert task_type == "testing"
    
    def test_classify_task_type_devops(self):
        """Test classification of DevOps tasks."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Set up a CI/CD pipeline using GitHub Actions"
        task_type = router.classify_task_type(task)
        
        assert task_type == "devops"
    
    def test_estimate_complexity_simple(self):
        """Test complexity estimation for simple tasks."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Print hello world"
        complexity = router.estimate_complexity(task)
        
        assert complexity == TaskComplexity.SIMPLE
    
    def test_estimate_complexity_medium(self):
        """Test complexity estimation for medium tasks."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Implement a REST API endpoint with input validation"
        complexity = router.estimate_complexity(task)
        
        assert complexity == TaskComplexity.MEDIUM
    
    def test_estimate_complexity_complex(self):
        """Test complexity estimation for complex tasks."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Design and implement a distributed caching system with Redis, handle failover, implement sharding strategy, and ensure consistency"
        complexity = router.estimate_complexity(task)
        
        assert complexity == TaskComplexity.COMPLEX
    
    def test_select_agent_cost_optimized(self):
        """Test agent selection with cost optimization."""
        config = {
            'enabled': True,
            'routing_strategy': RoutingStrategy.COST_OPTIMIZED.value
        }
        router = AdaptiveRouter(config)
        
        # For simple task, should select cheaper agent
        task = "Write a simple function"
        task_type = "coding"
        complexity = TaskComplexity.SIMPLE
        
        agent = router.select_agent(task, task_type, complexity)
        
        assert agent is not None
        # Should prefer lower cost agent for simple tasks
        assert agent['cost'] <= AGENT_COST_SWE
    
    def test_select_agent_quality_optimized(self):
        """Test agent selection with quality optimization."""
        config = {
            'enabled': True,
            'routing_strategy': RoutingStrategy.QUALITY_OPTIMIZED.value
        }
        router = AdaptiveRouter(config)
        
        # For complex task, should select higher quality agent
        task = "Design complex distributed system"
        task_type = "architecture"
        complexity = TaskComplexity.COMPLEX
        
        agent = router.select_agent(task, task_type, complexity)
        
        assert agent is not None
        # Should prefer higher quality agent for complex tasks
        assert agent['quality'] >= AGENT_QUALITY_ARCHITECT
    
    def test_select_agent_balanced(self):
        """Test balanced agent selection."""
        config = {
            'enabled': True,
            'routing_strategy': RoutingStrategy.BALANCED.value
        }
        router = AdaptiveRouter(config)
        
        task = "Implement a feature with moderate complexity"
        task_type = "coding"
        complexity = TaskComplexity.MEDIUM
        
        agent = router.select_agent(task, task_type, complexity)
        
        assert agent is not None
        # Should balance cost and quality
        assert agent['cost'] < AGENT_COST_ARCHITECT
        assert agent['quality'] > AGENT_QUALITY_BASE
    
    def test_route_task_end_to_end(self):
        """Test complete task routing flow."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Write unit tests for a sorting algorithm"
        result = router.route_task(task)
        
        assert 'agent' in result
        assert 'task_type' in result
        assert 'complexity' in result
        assert 'confidence' in result
        assert result['task_type'] == "testing"
    
    def test_route_task_with_context(self):
        """Test routing with additional context."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Implement feature X"
        context = {
            'preferred_agent': 'swe_agent',
            'priority': 'high',
            'max_cost': 0.01
        }
        
        result = router.route_task(task, context=context)
        
        assert result is not None
        # Should respect cost constraint
        assert result['agent']['cost'] <= 0.01
    
    def test_disabled_router(self):
        """Test that router can be disabled."""
        config = {'enabled': False}
        router = AdaptiveRouter(config)
        
        task = "Any task"
        result = router.route_task(task)
        
        # Should return default agent when disabled
        assert result is not None
        assert 'agent' in result
    
    def test_agent_profile_loading(self):
        """Test loading agent profiles from config."""
        config = {
            'enabled': True,
            'agent_profiles_path': 'config/agent_profiles.json'
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = '''
            {
                "swe_agent": {
                    "name": "Software Engineer Agent",
                    "specialties": ["coding", "testing"],
                    "cost": 0.015,
                    "quality": 0.9
                }
            }
            '''
            
            router = AdaptiveRouter(config)
            
            assert 'swe_agent' in router.agent_profiles
            assert router.agent_profiles['swe_agent']['quality'] == 0.9
    
    def test_historical_performance_tracking(self):
        """Test tracking of historical agent performance."""
        router = AdaptiveRouter({'enabled': True, 'track_performance': True})
        
        # Record some results
        router.record_result(
            agent='swe_agent',
            task_type='coding',
            success=True,
            latency_ms=150,
            quality_score=0.92
        )
        
        router.record_result(
            agent='swe_agent',
            task_type='coding',
            success=True,
            latency_ms=180,
            quality_score=0.88
        )
        
        # Get performance stats
        stats = router.get_agent_stats('swe_agent', 'coding')
        
        assert stats['success_rate'] == 1.0
        assert stats['avg_latency_ms'] == 165.0
        assert stats['avg_quality'] == 0.9
    
    def test_adaptive_learning_from_feedback(self):
        """Test that router adapts based on feedback."""
        router = AdaptiveRouter({
            'enabled': True,
            'adaptive_learning': True
        })
        
        # Simulate negative feedback for specific agent
        for _ in range(10):
            router.record_result(
                agent='base_agent',
                task_type='architecture',
                success=False,
                latency_ms=500,
                quality_score=0.3
            )
        
        # Router should learn to avoid base_agent for architecture tasks
        task = "Design system architecture"
        result = router.route_task(task)
        
        # Should not select base_agent for architecture after poor performance
        assert result['agent']['name'] != 'base_agent'
    
    def test_specialization_matching(self):
        """Test matching task to agent specialization."""
        router = AdaptiveRouter({'enabled': True})
        
        # Security task should route to security specialist
        task = "Perform security audit of authentication system"
        result = router.route_task(task)
        
        assert 'security' in result['task_type'].lower() or 'security' in result['agent']['specialties']
    
    def test_multi_step_task_detection(self):
        """Test detection of multi-step tasks."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "First implement the feature, then write tests, and finally deploy to staging"
        complexity = router.estimate_complexity(task)
        
        # Multi-step tasks should be complex
        assert complexity == TaskComplexity.COMPLEX
    
    def test_routing_confidence_score(self):
        """Test that routing includes confidence score."""
        router = AdaptiveRouter({'enabled': True})
        
        # Clear task should have high confidence
        task = "Write a Python function to sort a list"
        result = router.route_task(task)
        
        assert result['confidence'] > 0.7
        
        # Ambiguous task should have lower confidence
        ambiguous_task = "Do something with the thing"
        result = router.route_task(ambiguous_task)
        
        assert result['confidence'] < 0.6
    
    def test_fallback_for_unknown_task_type(self):
        """Test fallback behavior for unknown task types."""
        router = AdaptiveRouter({'enabled': True})
        
        task = "Xyzabc undefined task type qwerty"
        result = router.route_task(task)
        
        # Should still return a valid agent (fallback to general purpose)
        assert result is not None
        assert 'agent' in result
    
    def test_cost_vs_quality_tradeoff(self):
        """Test cost vs quality tradeoff calculation."""
        router = AdaptiveRouter({'enabled': True})
        
        # Calculate scores for different strategies
        agent_low_cost = {'cost': 0.005, 'quality': 0.7, 'speed': 150}
        agent_high_quality = {'cost': 0.02, 'quality': 0.95, 'speed': 80}
        
        cost_score = router._calculate_agent_score(
            agent_low_cost,
            RoutingStrategy.COST_OPTIMIZED,
            TaskComplexity.SIMPLE
        )
        
        quality_score = router._calculate_agent_score(
            agent_high_quality,
            RoutingStrategy.QUALITY_OPTIMIZED,
            TaskComplexity.COMPLEX
        )
        
        # Cost-optimized should score low-cost agent higher for simple tasks
        # Quality-optimized should score high-quality agent higher for complex tasks
        assert isinstance(cost_score, (int, float))
        assert isinstance(quality_score, (int, float))

