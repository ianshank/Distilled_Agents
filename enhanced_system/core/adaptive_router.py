"""
Adaptive Agent Router Module
Provides dynamic agent selection based on task characteristics
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class RoutingStrategy(Enum):
    """Routing optimization strategies"""
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"


@dataclass
class AgentProfile:
    """Profile for an agent"""
    agent_id: str
    name: str
    specializations: List[str]
    cost: float  # Cost per 1K tokens
    quality_score: float  # Historical quality (0-1)
    speed: float  # Tokens per second
    capabilities: List[str]
    max_complexity: TaskComplexity


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_agents: List[str]
    reasoning: str
    task_complexity: TaskComplexity
    estimated_cost: float
    estimated_quality: float


class AdaptiveRouter:
    """
    Dynamic agent routing based on task characteristics
    
    Features:
    - Task classification
    - Complexity estimation
    - Agent profile management
    - Multi-objective optimization (cost, quality, speed)
    """
    
    # Task type keywords for classification
    TASK_TYPE_KEYWORDS = {
        'coding': ['code', 'program', 'function', 'bug', 'debug', 'implement', 'algorithm'],
        'analysis': ['analyze', 'evaluate', 'assess', 'compare', 'review'],
        'design': ['design', 'architect', 'plan', 'structure', 'model'],
        'documentation': ['document', 'explain', 'describe', 'write', 'readme'],
        'testing': ['test', 'qa', 'quality', 'verify', 'validate'],
        'security': ['security', 'vulnerability', 'auth', 'encryption', 'secure'],
        'deployment': ['deploy', 'release', 'cicd', 'pipeline', 'infrastructure'],
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        'high': [
            'comprehensive', 'detailed', 'advanced', 'complex', 'sophisticated',
            'enterprise', 'scalable', 'distributed', 'multi-step', 'intricate'
        ],
        'low': [
            'simple', 'basic', 'straightforward', 'quick', 'easy',
            'trivial', 'minimal', 'small'
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adaptive router
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.complexity_model = config.get('complexity_model', 'heuristic')
        self.routing_strategy = RoutingStrategy(
            config.get('routing_strategy', 'cost_optimized')
        )
        
        # Load agent profiles
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self._load_agent_profiles(config.get('agent_profiles_path'))
        
        logger.info(
            f"AdaptiveRouter initialized (strategy: {self.routing_strategy.value})"
        )
    
    def _load_agent_profiles(self, profiles_path: Optional[str]) -> None:
        """
        Load agent profiles from file
        
        Args:
            profiles_path: Path to profiles JSON file
        """
        if not profiles_path:
            # Use default profiles
            self._create_default_profiles()
            return
        
        try:
            with open(profiles_path, 'r') as f:
                profiles_data = json.load(f)
            
            for profile_data in profiles_data.get('agents', []):
                profile = AgentProfile(
                    agent_id=profile_data['agent_id'],
                    name=profile_data['name'],
                    specializations=profile_data.get('specializations', []),
                    cost=profile_data.get('cost', 0.01),
                    quality_score=profile_data.get('quality_score', 0.7),
                    speed=profile_data.get('speed', 100.0),
                    capabilities=profile_data.get('capabilities', []),
                    max_complexity=TaskComplexity(
                        profile_data.get('max_complexity', 'complex')
                    )
                )
                self.agent_profiles[profile.agent_id] = profile
            
            logger.info(f"Loaded {len(self.agent_profiles)} agent profiles")
        
        except Exception as e:
            logger.warning(f"Failed to load agent profiles: {e}. Using defaults.")
            self._create_default_profiles()
    
    def _create_default_profiles(self) -> None:
        """Create default agent profiles"""
        default_profiles = [
            AgentProfile(
                agent_id='base_agent',
                name='Base Agent',
                specializations=['general'],
                cost=0.005,
                quality_score=0.7,
                speed=150.0,
                capabilities=['general_tasks'],
                max_complexity=TaskComplexity.MEDIUM
            ),
            AgentProfile(
                agent_id='swe_agent',
                name='Software Engineer Agent',
                specializations=['coding', 'debugging', 'implementation'],
                cost=0.015,
                quality_score=0.9,
                speed=100.0,
                capabilities=['code_generation', 'debugging', 'refactoring'],
                max_complexity=TaskComplexity.COMPLEX
            ),
            AgentProfile(
                agent_id='architect_agent',
                name='Principal Architect Agent',
                specializations=['design', 'architecture', 'planning'],
                cost=0.02,
                quality_score=0.95,
                speed=80.0,
                capabilities=['system_design', 'architecture', 'planning'],
                max_complexity=TaskComplexity.COMPLEX
            ),
            AgentProfile(
                agent_id='sqe_agent',
                name='Quality Engineer Agent',
                specializations=['testing', 'qa', 'validation'],
                cost=0.01,
                quality_score=0.85,
                speed=120.0,
                capabilities=['test_design', 'qa', 'validation'],
                max_complexity=TaskComplexity.COMPLEX
            ),
        ]
        
        for profile in default_profiles:
            self.agent_profiles[profile.agent_id] = profile
        
        logger.info(f"Created {len(default_profiles)} default agent profiles")
    
    def route_task(
        self,
        task: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route task to appropriate agents
        
        Args:
            task: Input task
            constraints: Optional constraints (max_cost, min_quality, etc.)
        
        Returns:
            RoutingDecision with selected agents
        """
        if not self.enabled:
            # Routing disabled, use base agent
            return RoutingDecision(
                selected_agents=['base_agent'],
                reasoning="Routing disabled, using base agent",
                task_complexity=TaskComplexity.MEDIUM,
                estimated_cost=0.01,
                estimated_quality=0.7
            )
        
        # Classify task type
        task_type = self.classify_task(task)
        
        # Estimate complexity
        complexity = self.estimate_complexity(task)
        
        # Select agents based on strategy
        if complexity == TaskComplexity.SIMPLE:
            selected_agents = self._select_for_simple_task(task_type, constraints)
        elif complexity == TaskComplexity.MEDIUM:
            selected_agents = self._select_for_medium_task(task_type, constraints)
        else:  # COMPLEX
            selected_agents = self._select_for_complex_task(task_type, constraints)
        
        # Calculate estimates
        estimated_cost = self._estimate_cost(selected_agents)
        estimated_quality = self._estimate_quality(selected_agents)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            task_type, complexity, selected_agents
        )
        
        return RoutingDecision(
            selected_agents=selected_agents,
            reasoning=reasoning,
            task_complexity=complexity,
            estimated_cost=estimated_cost,
            estimated_quality=estimated_quality
        )
    
    def classify_task(self, task: str) -> str:
        """
        Classify task type based on keywords
        
        Args:
            task: Input task
        
        Returns:
            Task type string
        """
        task_lower = task.lower()
        
        # Count keywords for each task type
        type_scores = {}
        for task_type, keywords in self.TASK_TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            type_scores[task_type] = score
        
        # Return type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return 'general'
    
    def estimate_complexity(self, task: str) -> TaskComplexity:
        """
        Estimate task complexity
        
        Args:
            task: Input task
        
        Returns:
            TaskComplexity level
        """
        if self.complexity_model == 'ml_model':
            # Use ML model (not implemented, fallback to heuristic)
            return self._estimate_complexity_heuristic(task)
        else:
            return self._estimate_complexity_heuristic(task)
    
    def _estimate_complexity_heuristic(self, task: str) -> TaskComplexity:
        """
        Estimate complexity using heuristic rules
        
        Args:
            task: Input task
        
        Returns:
            TaskComplexity level
        """
        task_lower = task.lower()
        
        # Count complexity indicators
        high_indicators = sum(
            1 for indicator in self.COMPLEXITY_INDICATORS['high']
            if indicator in task_lower
        )
        
        low_indicators = sum(
            1 for indicator in self.COMPLEXITY_INDICATORS['low']
            if indicator in task_lower
        )
        
        # Task length factor
        length_score = 0
        if len(task) < 50:
            length_score = -1
        elif len(task) > 500:
            length_score = 1
        
        # Multi-step indicator (bullet points, numbered lists)
        multi_step = len(re.findall(r'[\n\r][-*•\d]+[.)]\s', task))
        
        # Calculate final score
        complexity_score = (
            high_indicators - low_indicators + length_score + multi_step * 0.5
        )
        
        if complexity_score >= 2:
            return TaskComplexity.COMPLEX
        elif complexity_score <= -1:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MEDIUM
    
    def _select_for_simple_task(
        self,
        task_type: str,
        constraints: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Select agents for simple task"""
        # Use cheapest agent that can handle the task
        if self.routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            return ['base_agent']
        else:
            # Even for simple tasks, use specialist if available
            specialist = self._find_specialist(task_type)
            return [specialist] if specialist else ['base_agent']
    
    def _select_for_medium_task(
        self,
        task_type: str,
        constraints: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Select agents for medium complexity task"""
        specialist = self._find_specialist(task_type)
        
        if self.routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            return [specialist] if specialist else ['base_agent']
        elif self.routing_strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            # Use specialist + base for verification
            agents = [specialist] if specialist else []
            if 'base_agent' not in agents:
                agents.append('base_agent')
            return agents
        else:  # BALANCED
            return [specialist] if specialist else ['base_agent']
    
    def _select_for_complex_task(
        self,
        task_type: str,
        constraints: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Select agents for complex task"""
        specialist = self._find_specialist(task_type)
        
        if self.routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            # Use specialist + one support agent
            agents = [specialist] if specialist else ['base_agent']
            if len(agents) < 2:
                agents.append('base_agent')
            return agents
        elif self.routing_strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            # Use multiple specialists for consensus
            agents = []
            if specialist:
                agents.append(specialist)
            # Add architect for complex tasks
            if 'architect_agent' in self.agent_profiles:
                agents.append('architect_agent')
            # Add base agent
            if 'base_agent' not in agents:
                agents.append('base_agent')
            return agents[:3]  # Limit to 3 agents
        else:  # BALANCED
            agents = [specialist] if specialist else []
            if 'base_agent' not in agents:
                agents.append('base_agent')
            return agents
    
    def _find_specialist(self, task_type: str) -> Optional[str]:
        """
        Find specialist agent for task type
        
        Args:
            task_type: Type of task
        
        Returns:
            Agent ID or None
        """
        # Map task types to agents
        task_agent_map = {
            'coding': 'swe_agent',
            'design': 'architect_agent',
            'testing': 'sqe_agent',
            'analysis': 'architect_agent',
            'security': 'swe_agent',
        }
        
        specialist_id = task_agent_map.get(task_type)
        
        if specialist_id and specialist_id in self.agent_profiles:
            return specialist_id
        
        return None
    
    def _estimate_cost(self, agent_ids: List[str]) -> float:
        """Estimate cost for selected agents"""
        total_cost = 0.0
        for agent_id in agent_ids:
            if agent_id in self.agent_profiles:
                total_cost += self.agent_profiles[agent_id].cost
        return total_cost
    
    def _estimate_quality(self, agent_ids: List[str]) -> float:
        """Estimate quality for selected agents"""
        if not agent_ids:
            return 0.5
        
        qualities = []
        for agent_id in agent_ids:
            if agent_id in self.agent_profiles:
                qualities.append(self.agent_profiles[agent_id].quality_score)
        
        if not qualities:
            return 0.5
        
        # Use max quality if multiple agents (assuming consensus improves quality)
        return max(qualities) if len(qualities) > 1 else qualities[0]
    
    def _generate_reasoning(
        self,
        task_type: str,
        complexity: TaskComplexity,
        agents: List[str]
    ) -> str:
        """Generate reasoning for routing decision"""
        agent_names = [
            self.agent_profiles[aid].name
            for aid in agents
            if aid in self.agent_profiles
        ]
        
        reasoning_parts = [
            f"Task type: {task_type}",
            f"Complexity: {complexity.value}",
            f"Selected: {', '.join(agent_names)}",
            f"Strategy: {self.routing_strategy.value}"
        ]
        
        return " | ".join(reasoning_parts)

