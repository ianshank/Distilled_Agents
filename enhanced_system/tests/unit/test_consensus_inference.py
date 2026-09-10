"""Unit tests for ConsensusInference against infer_with_consensus."""

from __future__ import annotations

import asyncio

import pytest

from enhanced_system.core.consensus_inference import ConsensusInference, ConsensusResult
from enhanced_system.core.enums import AgreementMethod, EnsembleMethod


@pytest.mark.unit
class TestConsensusInference:
    def test_initialization(self):
        consensus = ConsensusInference(
            {
                "enabled": True,
                "min_agents": 3,
                "max_agents": 5,
                "threshold": 0.7,
                "agreement_method": AgreementMethod.EXACT_MATCH.value,
            }
        )
        assert consensus.min_agents == 3
        assert consensus.max_agents == 5
        assert consensus.threshold == 0.7

    @pytest.mark.asyncio
    async def test_infer_with_consensus_basic(self, mock_agents):
        consensus = ConsensusInference(
            {
                "enabled": True,
                "min_agents": 2,
                "max_agents": 5,
                "threshold": 0.5,
                "agreement_method": AgreementMethod.EXACT_MATCH.value,
                "ensemble_method": EnsembleMethod.CONFIDENCE_WEIGHTED.value,
            }
        )
        result = await consensus.infer_with_consensus("Test task", mock_agents[:2])
        assert isinstance(result, ConsensusResult)
        assert result.consensus_response
        assert 0 <= result.confidence <= 1

    @pytest.mark.asyncio
    async def test_high_agreement(self):
        consensus = ConsensusInference(
            {
                "enabled": True,
                "min_agents": 3,
                "max_agents": 5,
                "threshold": 0.8,
                "agreement_method": AgreementMethod.EXACT_MATCH.value,
            }
        )

        async def agent1(task):
            return {"response": "The answer is 42", "confidence": 0.9}

        async def agent2(task):
            return {"response": "The answer is 42", "confidence": 0.85}

        async def agent3(task):
            return {"response": "The answer is 42", "confidence": 0.88}

        result = await consensus.infer_with_consensus(
            "What is the answer?", [agent1, agent2, agent3]
        )
        assert result.agreement_score > 0.8
        assert "42" in result.consensus_response

    @pytest.mark.asyncio
    async def test_low_agreement_uses_ensemble(self):
        consensus = ConsensusInference(
            {
                "enabled": True,
                "min_agents": 3,
                "max_agents": 5,
                "threshold": 0.9,
                "agreement_method": AgreementMethod.EXACT_MATCH.value,
                "ensemble_method": EnsembleMethod.CONFIDENCE_WEIGHTED.value,
            }
        )

        async def agent1(task):
            return {"response": "Answer A", "confidence": 0.9}

        async def agent2(task):
            return {"response": "Answer B", "confidence": 0.85}

        async def agent3(task):
            return {"response": "Answer C", "confidence": 0.8}

        result = await consensus.infer_with_consensus(
            "Ambiguous question", [agent1, agent2, agent3]
        )
        assert result.agreement_score < 0.5
        assert result.method == "ensemble"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        consensus = ConsensusInference(
            {
                "enabled": True,
                "min_agents": 3,
                "max_agents": 5,
                "agreement_method": AgreementMethod.EXACT_MATCH.value,
            }
        )

        async def slow_agent(task):
            await asyncio.sleep(0.05)
            return {"response": "Slow response", "confidence": 0.8}

        start = asyncio.get_event_loop().time()
        await consensus.infer_with_consensus("Test task", [slow_agent, slow_agent, slow_agent])
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.2

    @pytest.mark.asyncio
    async def test_majority_vote_ensemble(self):
        consensus = ConsensusInference(
            {
                "enabled": True,
                "min_agents": 3,
                "max_agents": 5,
                "threshold": 0.99,
                "ensemble_method": EnsembleMethod.MAJORITY_VOTE.value,
                "agreement_method": AgreementMethod.EXACT_MATCH.value,
            }
        )

        async def agent_a(task):
            return {"response": "Answer A", "confidence": 0.7}

        async def agent_b(task):
            return {"response": "Answer B", "confidence": 0.9}

        async def agent_a2(task):
            return {"response": "Answer A", "confidence": 0.8}

        result = await consensus.infer_with_consensus("Test", [agent_a, agent_b, agent_a2])
        assert "Answer A" in result.consensus_response

    @pytest.mark.asyncio
    async def test_handle_agent_failure(self):
        consensus = ConsensusInference(
            {
                "enabled": True,
                "min_agents": 2,
                "max_agents": 5,
                "agreement_method": AgreementMethod.EXACT_MATCH.value,
            }
        )

        async def working_agent(task):
            return {"response": "Success", "confidence": 0.8}

        async def failing_agent(task):
            raise Exception("Agent failed")

        result = await consensus.infer_with_consensus(
            "Test", [working_agent, failing_agent, working_agent]
        )
        assert result.consensus_response

    @pytest.mark.asyncio
    async def test_disabled_consensus(self):
        consensus = ConsensusInference({"enabled": False, "min_agents": 3})

        async def agent(task):
            return {"response": "Single response", "confidence": 0.8}

        result = await consensus.infer_with_consensus("Test", [agent, agent, agent])
        assert result.consensus_response == "Single response"
        assert result.method == "single"

    @pytest.mark.asyncio
    async def test_consensus_metadata(self):
        consensus = ConsensusInference(
            {
                "enabled": True,
                "min_agents": 2,
                "max_agents": 5,
                "agreement_method": AgreementMethod.EXACT_MATCH.value,
            }
        )

        async def agent(task):
            return {"response": "Response", "confidence": 0.8}

        result = await consensus.infer_with_consensus("Test", [agent, agent])
        assert result.metadata.get("num_agents") == 2
        assert result.agreement_score is not None
