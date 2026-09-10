"""
Streaming Inference Module
Provides token-by-token streaming responses with real-time feedback
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamToken:
    """Single token in stream"""

    token: str
    position: int
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StreamingInference:
    """
    Token-by-token streaming inference

    Features:
    - Real-time token streaming
    - Per-token confidence scoring
    - Stop condition detection
    - Error recovery with buffering
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize streaming inference

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.buffer_size = config.get("buffer_size", 10)
        self.confidence_per_token = config.get("confidence_per_token", True)

        # Stop conditions
        stop_config = config.get("stop_conditions", {})
        self.max_tokens = stop_config.get("max_tokens", 2048)
        self.quality_threshold = stop_config.get("quality_threshold", 0.3)
        self.detect_eos = stop_config.get("detect_eos", True)

        # EOS tokens to detect
        self.eos_tokens = ["<|endoftext|>", "</s>", "<eos>", "[END]"]

        logger.info(f"StreamingInference initialized (buffer_size: {self.buffer_size})")

    async def stream_response(
        self, generate_func: Callable, task: str, **kwargs
    ) -> AsyncIterator[StreamToken]:
        """
        Stream response token by token

        Args:
            generate_func: Function that generates tokens (must be async generator)
            task: Input task
            **kwargs: Additional arguments for generate_func

        Yields:
            StreamToken objects
        """
        if not self.enabled:
            # Streaming disabled, return full response at once
            logger.info("Streaming disabled, returning full response")
            full_response = await generate_func(task, **kwargs)
            yield StreamToken(
                token=full_response,
                position=0,
                confidence=None,
                metadata={"streaming_disabled": True},
            )
            return

        # Buffer for error recovery
        token_buffer = deque(maxlen=self.buffer_size)

        position = 0
        try:
            # Generate tokens
            async for token_data in self._generate_tokens(generate_func, task, **kwargs):
                # Extract token and metadata
                if isinstance(token_data, dict):
                    token = token_data.get("token", "")
                    confidence = token_data.get("confidence")
                    metadata = token_data.get("metadata", {})
                else:
                    token = str(token_data)
                    confidence = None
                    metadata = {}

                # Check if valid token
                if not self._is_valid_token(token):
                    logger.warning(f"Invalid token at position {position}: {token}")
                    continue

                # Create stream token
                stream_token = StreamToken(
                    token=token,
                    position=position,
                    confidence=confidence if self.confidence_per_token else None,
                    metadata=metadata,
                )

                # Add to buffer
                token_buffer.append(stream_token)

                # Check stop conditions
                if self._should_stop(token, position, confidence, token_buffer):
                    logger.info(f"Stop condition met at position {position}")
                    break

                # Yield token
                yield stream_token

                position += 1

        except Exception as e:
            logger.error(f"Streaming error at position {position}: {e}")

            # Try to recover using buffer
            if token_buffer:
                logger.info("Attempting recovery using buffered tokens")
                # Yield a recovery token
                yield StreamToken(
                    token="",
                    position=position,
                    confidence=0.0,
                    metadata={
                        "error": str(e),
                        "recovery_attempted": True,
                        "buffer_size": len(token_buffer),
                    },
                )
            raise

    async def _generate_tokens(self, generate_func: Callable, task: str, **kwargs) -> AsyncIterator:
        """
        Wrapper for token generation

        Args:
            generate_func: Token generation function
            task: Input task
            **kwargs: Additional arguments

        Yields:
            Tokens or token data
        """
        # Check if function is async generator
        if callable(generate_func):
            result = generate_func(task, **kwargs)

            # If it's an async generator, yield from it
            if hasattr(result, "__aiter__"):
                async for token in result:
                    yield token
            # If it's a regular generator, convert to async
            elif hasattr(result, "__iter__"):
                for token in result:
                    yield token
                    await asyncio.sleep(0)  # Yield control
            # If it's a coroutine, await and split into tokens
            elif asyncio.iscoroutine(result):
                full_response = await result
                # Split into tokens (simple word-based tokenization)
                tokens = self._tokenize_response(full_response)
                for token in tokens:
                    yield token
                    await asyncio.sleep(0)
            else:
                # Direct value, split into tokens
                tokens = self._tokenize_response(str(result))
                for token in tokens:
                    yield token
                    await asyncio.sleep(0)

    def _tokenize_response(self, response: str) -> list:
        """
        Simple tokenization of response

        Args:
            response: Response string

        Returns:
            List of tokens
        """
        # Simple word-based tokenization
        # In a real implementation, would use proper tokenizer
        words = response.split()
        return words

    def _is_valid_token(self, token: str) -> bool:
        """
        Check if token is valid

        Args:
            token: Token to validate

        Returns:
            True if valid
        """
        if not token:
            return False

        # Check for control characters
        if any(ord(c) < 32 for c in token if c not in "\n\r\t"):
            return False

        # Check for excessively long tokens
        if len(token) > 100:
            return False

        return True

    def _should_stop(
        self, token: str, position: int, confidence: Optional[float], buffer: deque
    ) -> bool:
        """
        Check if streaming should stop

        Args:
            token: Current token
            position: Token position
            confidence: Token confidence
            buffer: Token buffer

        Returns:
            True if should stop
        """
        # Max tokens reached
        if position >= self.max_tokens:
            logger.info(f"Max tokens ({self.max_tokens}) reached")
            return True

        # EOS token detected
        if self.detect_eos:
            if any(eos in token for eos in self.eos_tokens):
                logger.info("EOS token detected")
                return True

        # Quality threshold
        if self.confidence_per_token and confidence is not None:
            if confidence < self.quality_threshold:
                # Check if consistently low quality
                if len(buffer) >= 3:
                    recent_confidences = [
                        t.confidence for t in list(buffer)[-3:] if t.confidence is not None
                    ]
                    if recent_confidences:
                        avg_confidence = sum(recent_confidences) / len(recent_confidences)
                        if avg_confidence < self.quality_threshold:
                            logger.info(
                                f"Quality threshold not met: {avg_confidence:.3f} < {self.quality_threshold}"
                            )
                            return True

        return False

    async def stream_with_aggregation(
        self, generate_func: Callable, task: str, aggregate_every: int = 5, **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream with periodic aggregation

        Args:
            generate_func: Token generation function
            task: Input task
            aggregate_every: Aggregate every N tokens
            **kwargs: Additional arguments

        Yields:
            Aggregated token batches with metadata
        """
        accumulated_tokens = []
        token_count = 0

        async for stream_token in self.stream_response(generate_func, task, **kwargs):
            accumulated_tokens.append(stream_token.token)
            token_count += 1

            # Aggregate and yield
            if token_count % aggregate_every == 0:
                aggregated = " ".join(accumulated_tokens)
                yield {
                    "text": aggregated,
                    "token_count": token_count,
                    "position": stream_token.position,
                    "partial": True,
                }
                accumulated_tokens = []

        # Yield remaining tokens
        if accumulated_tokens:
            aggregated = " ".join(accumulated_tokens)
            yield {
                "text": aggregated,
                "token_count": token_count,
                "position": token_count - 1,
                "partial": False,
            }


async def stream_inference(
    generate_func: Callable, task: str, config: Dict[str, Any], **kwargs
) -> AsyncIterator[StreamToken]:
    """
    Convenience function for streaming inference

    Args:
        generate_func: Token generation function
        task: Input task
        config: Configuration
        **kwargs: Additional arguments

    Yields:
        StreamToken objects
    """
    streamer = StreamingInference(config)
    async for token in streamer.stream_response(generate_func, task, **kwargs):
        yield token
