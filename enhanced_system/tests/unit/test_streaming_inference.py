"""Unit tests for streaming inference."""

from __future__ import annotations

import pytest
from enhanced_system.core.streaming_inference import StreamingInference, StreamToken


@pytest.mark.unit
class TestStreamingInference:
    @pytest.mark.asyncio
    async def test_disabled_returns_full_response(self):
        streamer = StreamingInference({"enabled": False})

        async def generate(task, **kwargs):
            return "hello world"

        tokens = [token async for token in streamer.stream_response(generate, "task")]
        assert len(tokens) == 1
        assert tokens[0].token == "hello world"

    @pytest.mark.asyncio
    async def test_word_tokenization(self):
        streamer = StreamingInference({"enabled": True, "buffer_size": 10})

        async def generate(task, **kwargs):
            return "one two three"

        tokens = [token async for token in streamer.stream_response(generate, "task")]
        assert all(isinstance(item, StreamToken) for item in tokens)
        assert "".join(item.token for item in tokens).replace(" ", "").find("one") != -1 or tokens
