"""
Multi-Agent Consensus Inference Module
Provides consensus-based inference with multiple agents for improved reliability
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence-transformers not available. Embedding-based agreement disabled.")


logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from a single agent"""
    agent_id: str
    response: str
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any]


@dataclass
class ConsensusResult:
    """Result of consensus inference"""
    consensus_response: str
    confidence: float
    agreement_score: float
    individual_responses: List[AgentResponse]
    method: str
    metadata: Dict[str, Any]


class ConsensusInference:
    """
    Multi-agent consensus inference system
    
    Features:
    - Parallel agent execution
    - Agreement calculation using semantic similarity
    - Consensus merging strategies
    - Ensemble fallback with confidence weighting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize consensus inference
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self.min_agents = config.get('min_agents', 3)
        self.max_agents = config.get('max_agents', 5)
        self.threshold = config.get('threshold', 0.7)
        self.agreement_method = config.get('agreement_method', 'embedding_similarity')
        self.ensemble_method = config.get('ensemble_method', 'confidence_weighted')
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = None
        if self.agreement_method == 'embedding_similarity' and EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded for consensus agreement")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.agreement_method = 'exact_match'
        
        logger.info(f"ConsensusInference initialized (method: {self.agreement_method})")
    
    async def infer_with_consensus(
        self,
        task: str,
        agent_funcs: List[Callable],
        agent_ids: Optional[List[str]] = None,
        **kwargs
    ) -> ConsensusResult:
        """
        Run inference with multiple agents and get consensus
        
        Args:
            task: Input task
            agent_funcs: List of agent inference functions
            agent_ids: Optional list of agent identifiers
            **kwargs: Additional arguments for agent functions
        
        Returns:
            ConsensusResult with consensus response
        """
        if not self.enabled:
            # Consensus disabled, use first agent only
            logger.info("Consensus disabled, using single agent")
            if agent_funcs:
                result = await self._execute_agent(agent_funcs[0], task, "agent_0", **kwargs)
                return ConsensusResult(
                    consensus_response=result.response,
                    confidence=result.confidence,
                    agreement_score=1.0,
                    individual_responses=[result],
                    method="single",
                    metadata={"consensus_disabled": True}
                )
        
        # Select agents to use
        num_agents = min(len(agent_funcs), self.max_agents)
        num_agents = max(num_agents, self.min_agents)
        
        if len(agent_funcs) < num_agents:
            logger.warning(
                f"Requested {num_agents} agents but only {len(agent_funcs)} available"
            )
            num_agents = len(agent_funcs)
        
        # Generate agent IDs if not provided
        if agent_ids is None:
            agent_ids = [f"agent_{i}" for i in range(num_agents)]
        
        # Execute all agents in parallel
        logger.info(f"Executing {num_agents} agents in parallel for consensus")
        
        tasks = [
            self._execute_agent(agent_funcs[i], task, agent_ids[i], **kwargs)
            for i in range(num_agents)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed responses
        valid_responses = [
            r for r in responses
            if isinstance(r, AgentResponse) and not isinstance(r, Exception)
        ]
        
        if not valid_responses:
            raise Exception("All agents failed to produce valid responses")
        
        if len(valid_responses) < 2:
            logger.warning("Only one valid response, cannot calculate consensus")
            return ConsensusResult(
                consensus_response=valid_responses[0].response,
                confidence=valid_responses[0].confidence,
                agreement_score=1.0,
                individual_responses=valid_responses,
                method="single",
                metadata={"insufficient_responses": True}
            )
        
        # Calculate agreement
        agreement_score = self.calculate_agreement(valid_responses)
        
        logger.info(f"Agreement score: {agreement_score:.3f} (threshold: {self.threshold})")
        
        # Determine consensus method based on agreement
        if agreement_score >= self.threshold:
            # High agreement - merge responses
            consensus_response = self.merge_responses(valid_responses)
            method = "consensus"
        else:
            # Low agreement - use ensemble
            consensus_response = self.ensemble_responses(valid_responses)
            method = "ensemble"
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            valid_responses, agreement_score
        )
        
        return ConsensusResult(
            consensus_response=consensus_response,
            confidence=confidence,
            agreement_score=agreement_score,
            individual_responses=valid_responses,
            method=method,
            metadata={
                "num_agents": len(valid_responses),
                "threshold": self.threshold,
                "agreement_method": self.agreement_method
            }
        )
    
    async def _execute_agent(
        self,
        agent_func: Callable,
        task: str,
        agent_id: str,
        **kwargs
    ) -> AgentResponse:
        """
        Execute a single agent
        
        Args:
            agent_func: Agent inference function
            task: Input task
            agent_id: Agent identifier
            **kwargs: Additional arguments
        
        Returns:
            AgentResponse
        """
        import time
        start_time = time.time()
        
        try:
            # Execute agent function
            if asyncio.iscoroutinefunction(agent_func):
                result = await agent_func(task, **kwargs)
            else:
                result = agent_func(task, **kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse result
            if isinstance(result, dict):
                response = result.get('response', str(result))
                confidence = result.get('confidence', 0.5)
                metadata = result.get('metadata', {})
            elif isinstance(result, str):
                response = result
                confidence = 0.5
                metadata = {}
            else:
                response = str(result)
                confidence = 0.5
                metadata = {}
            
            return AgentResponse(
                agent_id=agent_id,
                response=response,
                confidence=confidence,
                latency_ms=latency_ms,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}")
            raise
    
    def calculate_agreement(self, responses: List[AgentResponse]) -> float:
        """
        Calculate agreement score among responses
        
        Args:
            responses: List of agent responses
        
        Returns:
            Agreement score (0-1)
        """
        if len(responses) < 2:
            return 1.0
        
        if self.agreement_method == 'embedding_similarity' and self.embedding_model:
            return self._calculate_embedding_agreement(responses)
        else:
            return self._calculate_exact_match_agreement(responses)
    
    def _calculate_embedding_agreement(
        self,
        responses: List[AgentResponse]
    ) -> float:
        """
        Calculate agreement using embedding similarity
        
        Args:
            responses: List of agent responses
        
        Returns:
            Agreement score (0-1)
        """
        if not self.embedding_model:
            return self._calculate_exact_match_agreement(responses)
        
        try:
            # Get embeddings for all responses
            texts = [r.response for r in responses]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(similarity)
            
            # Return average similarity
            if similarities:
                return float(np.mean(similarities))
            else:
                return 1.0
        
        except Exception as e:
            logger.error(f"Embedding agreement calculation failed: {e}")
            return self._calculate_exact_match_agreement(responses)
    
    def _calculate_exact_match_agreement(
        self,
        responses: List[AgentResponse]
    ) -> float:
        """
        Calculate agreement using exact string matching
        
        Args:
            responses: List of agent responses
        
        Returns:
            Agreement score (0-1)
        """
        # Count unique responses
        unique_responses = set(r.response.strip().lower() for r in responses)
        
        # Agreement is inverse of diversity
        # If all responses are the same, agreement = 1.0
        # If all responses are different, agreement = 0.0
        max_unique = len(responses)
        agreement = 1.0 - (len(unique_responses) - 1) / max(max_unique - 1, 1)
        
        return agreement
    
    def merge_responses(self, responses: List[AgentResponse]) -> str:
        """
        Merge responses with high agreement
        
        Args:
            responses: List of agent responses
        
        Returns:
            Merged response string
        """
        # Use the response from the most confident agent
        most_confident = max(responses, key=lambda r: r.confidence)
        return most_confident.response
    
    def ensemble_responses(self, responses: List[AgentResponse]) -> str:
        """
        Create ensemble response from disagreeing agents
        
        Args:
            responses: List of agent responses
        
        Returns:
            Ensemble response string
        """
        if self.ensemble_method == 'confidence_weighted':
            return self._confidence_weighted_ensemble(responses)
        else:
            return self._majority_vote_ensemble(responses)
    
    def _confidence_weighted_ensemble(
        self,
        responses: List[AgentResponse]
    ) -> str:
        """
        Create ensemble using confidence-weighted selection
        
        Args:
            responses: List of agent responses
        
        Returns:
            Selected response
        """
        # Select response based on confidence weights
        total_confidence = sum(r.confidence for r in responses)
        
        if total_confidence == 0:
            # All have zero confidence, use first response
            return responses[0].response
        
        # Use weighted random selection (deterministic by using max)
        # In a real implementation, might want to use actual weighted sampling
        weighted_responses = [
            (r.response, r.confidence / total_confidence)
            for r in responses
        ]
        
        # Return highest weighted response
        best_response = max(weighted_responses, key=lambda x: x[1])
        return best_response[0]
    
    def _majority_vote_ensemble(self, responses: List[AgentResponse]) -> str:
        """
        Create ensemble using majority vote
        
        Args:
            responses: List of agent responses
        
        Returns:
            Majority response
        """
        # Count response occurrences
        response_counts: Dict[str, int] = {}
        response_map: Dict[str, str] = {}
        
        for r in responses:
            # Use normalized version as key
            normalized = r.response.strip().lower()
            response_counts[normalized] = response_counts.get(normalized, 0) + 1
            response_map[normalized] = r.response  # Keep original
        
        # Get most common response
        if response_counts:
            most_common = max(response_counts.items(), key=lambda x: x[1])
            return response_map[most_common[0]]
        else:
            return responses[0].response
    
    def _calculate_overall_confidence(
        self,
        responses: List[AgentResponse],
        agreement_score: float
    ) -> float:
        """
        Calculate overall confidence score
        
        Args:
            responses: List of agent responses
            agreement_score: Agreement score
        
        Returns:
            Overall confidence (0-1)
        """
        # Average individual confidences
        avg_confidence = np.mean([r.confidence for r in responses])
        
        # Boost confidence if high agreement, reduce if low agreement
        # agreement_score acts as a multiplier
        overall_confidence = avg_confidence * (0.5 + 0.5 * agreement_score)
        
        return float(np.clip(overall_confidence, 0.0, 1.0))


async def run_consensus_inference(
    task: str,
    agents: List[Callable],
    config: Dict[str, Any],
    **kwargs
) -> ConsensusResult:
    """
    Convenience function to run consensus inference
    
    Args:
        task: Input task
        agents: List of agent functions
        config: Configuration
        **kwargs: Additional arguments
    
    Returns:
        ConsensusResult
    """
    consensus = ConsensusInference(config)
    return await consensus.infer_with_consensus(task, agents, **kwargs)

