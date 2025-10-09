"""
Unit Tests for ConsensusInference
==================================

Comprehensive tests for multi-agent consensus and ensemble methods.
Target: 80%+ coverage
"""

from __future__ import annotations

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from enhanced_system.core.consensus_inference import ConsensusInference
from enhanced_system.core.constants import *
from enhanced_system.core.enums import AgreementMethod, EnsembleMethod


@pytest.mark.unit
class TestConsensusInference:
    """Test ConsensusInference class."""
    
    def test_initialization(self, test_config):
        """Test consensus inference initialization."""
        config = {
            'enabled': True,
            'min_agents': 3,
            'max_agents': 5,
            'threshold': 0.7,
            'agreement_method': AgreementMethod.EMBEDDING_SIMILARITY.value
        }
        
        consensus = ConsensusInference(config)
        assert consensus.min_agents == 3
        assert consensus.max_agents == 5
        assert consensus.threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_run_consensus_basic(self, test_config, mock_agents):
        """Test basic consensus with multiple agents."""
        config = {
            'enabled': True,
            'min_agents': 2,
            'threshold': 0.5,
            'agreement_method': AgreementMethod.EXACT_MATCH.value,
            'ensemble_method': EnsembleMethod.CONFIDENCE_WEIGHTED.value
        }
        
        consensus = ConsensusInference(config)
        result = await consensus.run_consensus("Test task", mock_agents[:2])
        
        assert result is not None
        assert 'response' in result
        assert 'confidence' in result
        assert 'agreement_score' in result
    
    @pytest.mark.asyncio
    async def test_run_consensus_high_agreement(self, test_config):
        """Test consensus with high agreement among agents."""
        config = {
            'enabled': True,
            'min_agents': 3,
            'threshold': 0.8,
            'agreement_method': AgreementMethod.EXACT_MATCH.value
        }
        
        # Create agents that return similar responses
        async def agent1(task):
            return {"response": "The answer is 42", "confidence": 0.9}
        
        async def agent2(task):
            return {"response": "The answer is 42", "confidence": 0.85}
        
        async def agent3(task):
            return {"response": "The answer is 42", "confidence": 0.88}
        
        consensus = ConsensusInference(config)
        result = await consensus.run_consensus("What is the answer?", [agent1, agent2, agent3])
        
        assert result['agreement_score'] > 0.8
        assert "42" in result['response']
    
    @pytest.mark.asyncio
    async def test_run_consensus_low_agreement(self, test_config):
        """Test consensus with low agreement among agents."""
        config = {
            'enabled': True,
            'min_agents': 3,
            'threshold': 0.9,
            'agreement_method': AgreementMethod.EXACT_MATCH.value
        }
        
        # Create agents with different responses
        async def agent1(task):
            return {"response": "Answer A", "confidence": 0.9}
        
        async def agent2(task):
            return {"response": "Answer B", "confidence": 0.85}
        
        async def agent3(task):
            return {"response": "Answer C", "confidence": 0.8}
        
        consensus = ConsensusInference(config)
        result = await consensus.run_consensus("Ambiguous question", [agent1, agent2, agent3])
        
        # Should indicate low agreement
        assert result['agreement_score'] < 0.5
        assert 'low_agreement_warning' in result or result['confidence'] < 0.7
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, test_config):
        """Test that agents run in parallel."""
        config = {
            'enabled': True,
            'min_agents': 3,
            'parallel': True
        }
        
        execution_times = []
        
        async def slow_agent(task):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)
            execution_times.append(asyncio.get_event_loop().time() - start)
            return {"response": "Slow response", "confidence": 0.8}
        
        agents = [slow_agent, slow_agent, slow_agent]
        
        consensus = ConsensusInference(config)
        start_time = asyncio.get_event_loop().time()
        await consensus.run_consensus("Test task", agents)
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Parallel execution should be faster than sequential (3 * 0.1 = 0.3s)
        assert total_time < 0.25  # Should be ~0.1s, not 0.3s
    
    @pytest.mark.asyncio
    async def test_min_agents_requirement(self, test_config):
        """Test minimum agent requirement validation."""
        config = {
            'enabled': True,
            'min_agents': 3
        }
        
        consensus = ConsensusInference(config)
        
        # Only 2 agents provided, should fail
        async def agent(task):
            return {"response": "Response", "confidence": 0.8}
        
        with pytest.raises(ValueError, match="minimum.*agent"):
            await consensus.run_consensus("Test", [agent, agent])
    
    @pytest.mark.asyncio
    async def test_max_agents_limit(self, test_config):
        """Test maximum agent limit enforcement."""
        config = {
            'enabled': True,
            'min_agents': 2,
            'max_agents': 3
        }
        
        async def agent(task):
            return {"response": "Response", "confidence": 0.8}
        
        # Provide 5 agents, should use only first 3
        agents = [agent] * 5
        
        consensus = ConsensusInference(config)
        with patch.object(consensus, '_run_agents_parallel') as mock_run:
            mock_run.return_value = [
                {"response": "R1", "confidence": 0.8},
                {"response": "R2", "confidence": 0.8},
                {"response": "R3", "confidence": 0.8}
            ]
            
            await consensus.run_consensus("Test", agents)
            
            # Should only run 3 agents
            called_agents = mock_run.call_args[0][1]
            assert len(called_agents) == 3
    
    @pytest.mark.asyncio
    @patch('enhanced_system.core.consensus_inference.SentenceTransformer')
    async def test_embedding_similarity_agreement(self, mock_transformer_class, test_config):
        """Test agreement calculation using embedding similarity."""
        config = {
            'enabled': True,
            'min_agents': 2,
            'agreement_method': AgreementMethod.EMBEDDING_SIMILARITY.value
        }
        
        mock_model = MagicMock()
        mock_transformer_class.return_value = mock_model
        
        # Mock similar embeddings
        mock_model.encode.return_value = [
            [0.1, 0.2, 0.3],
            [0.11, 0.21, 0.31]
        ]
        
        async def agent1(task):
            return {"response": "The answer is Python", "confidence": 0.9}
        
        async def agent2(task):
            return {"response": "Python is the answer", "confidence": 0.85}
        
        consensus = ConsensusInference(config)
        result = await consensus.run_consensus("What is it?", [agent1, agent2])
        
        # Should have high agreement due to similar embeddings
        assert result['agreement_score'] > 0.9
    
    @pytest.mark.asyncio
    async def test_confidence_weighted_ensemble(self, test_config):
        """Test confidence-weighted ensemble method."""
        config = {
            'enabled': True,
            'min_agents': 3,
            'ensemble_method': EnsembleMethod.CONFIDENCE_WEIGHTED.value
        }
        
        async def high_conf_agent(task):
            return {"response": "Answer A", "confidence": 0.95}
        
        async def low_conf_agent(task):
            return {"response": "Answer B", "confidence": 0.3}
        
        async def med_conf_agent(task):
            return {"response": "Answer B", "confidence": 0.4}
        
        consensus = ConsensusInference(config)
        result = await consensus.run_consensus("Test", [high_conf_agent, low_conf_agent, med_conf_agent])
        
        # Should prefer high-confidence answer
        assert "Answer A" in result['response']
    
    @pytest.mark.asyncio
    async def test_majority_vote_ensemble(self, test_config):
        """Test majority vote ensemble method."""
        config = {
            'enabled': True,
            'min_agents': 3,
            'ensemble_method': EnsembleMethod.MAJORITY_VOTE.value,
            'agreement_method': AgreementMethod.EXACT_MATCH.value
        }
        
        async def agent_a(task):
            return {"response": "Answer A", "confidence": 0.7}
        
        async def agent_b(task):
            return {"response": "Answer B", "confidence": 0.9}
        
        async def agent_a2(task):
            return {"response": "Answer A", "confidence": 0.8}
        
        consensus = ConsensusInference(config)
        result = await consensus.run_consensus("Test", [agent_a, agent_b, agent_a2])
        
        # Should pick "Answer A" (2 votes vs 1)
        assert "Answer A" in result['response']
    
    @pytest.mark.asyncio
    async def test_handle_agent_failure(self, test_config):
        """Test handling of agent failures during consensus."""
        config = {
            'enabled': True,
            'min_agents': 2
        }
        
        async def working_agent(task):
            return {"response": "Success", "confidence": 0.8}
        
        async def failing_agent(task):
            raise Exception("Agent failed")
        
        consensus = ConsensusInference(config)
        
        # Should handle failure gracefully and continue with working agents
        result = await consensus.run_consensus("Test", [working_agent, failing_agent, working_agent])
        
        assert result is not None
        assert 'response' in result
    
    @pytest.mark.asyncio
    async def test_disabled_consensus(self, test_config):
        """Test that consensus can be disabled."""
        config = {
            'enabled': False,
            'min_agents': 3
        }
        
        async def agent(task):
            return {"response": "Single response", "confidence": 0.8}
        
        consensus = ConsensusInference(config)
        
        # When disabled, should just return first agent's response
        result = await consensus.run_consensus("Test", [agent, agent, agent])
        
        assert result['response'] == "Single response"
    
    @pytest.mark.asyncio
    async def test_consensus_metadata(self, test_config):
        """Test that consensus includes metadata."""
        config = {
            'enabled': True,
            'min_agents': 2
        }
        
        async def agent(task):
            return {"response": "Response", "confidence": 0.8}
        
        consensus = ConsensusInference(config)
        result = await consensus.run_consensus("Test", [agent, agent])
        
        assert 'num_agents' in result
        assert 'agreement_score' in result
        assert 'ensemble_method' in result
        assert result['num_agents'] == 2
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, test_config):
        """Test timeout handling for slow agents."""
        config = {
            'enabled': True,
            'min_agents': 2,
            'timeout': 0.5
        }
        
        async def fast_agent(task):
            return {"response": "Fast", "confidence": 0.8}
        
        async def slow_agent(task):
            await asyncio.sleep(2.0)
            return {"response": "Slow", "confidence": 0.9}
        
        consensus = ConsensusInference(config)
        
        # Should timeout slow agent and continue with fast agent
        result = await consensus.run_consensus("Test", [fast_agent, slow_agent])
        
        assert result is not None
        assert "Fast" in result['response']

