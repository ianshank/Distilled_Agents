"""Semantic similarity cache keys."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    np = None  # type: ignore[assignment]
    logging.warning(
        "Sentence-transformers not available. Semantic caching will be disabled."
    )


class SemanticCacheManager:
    """Semantic similarity-based cache key lookup."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True) and EMBEDDINGS_AVAILABLE
        self.threshold = config.get("threshold", 0.95)
        self.model_name = config.get("model", "all-MiniLM-L6-v2")
        self.model = None
        self.cache_embeddings: dict[str, Any] = {}

        if self.enabled:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info("Semantic cache initialized with model: %s", self.model_name)
            except Exception as exc:
                logger.warning("Failed to initialize semantic cache: %s", exc)
                self.enabled = False

    def get_embedding(self, text: str) -> Optional[Any]:
        if not self.enabled or not self.model:
            return None
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as exc:
            logger.error("Embedding generation error: %s", exc)
            return None

    def find_similar_cached(self, text: str) -> Optional[str]:
        if not self.enabled or not self.model:
            return None
        embedding = self.get_embedding(text)
        if embedding is None:
            return None
        for cache_key, cached_embedding in self.cache_embeddings.items():
            similarity = np.dot(embedding, cached_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(cached_embedding)
            )
            if similarity >= self.threshold:
                logger.info("Found similar cached text with similarity: %.3f", similarity)
                return cache_key
        return None

    def register_embedding(self, cache_key: str, text: str) -> None:
        if not self.enabled:
            return
        embedding = self.get_embedding(text)
        if embedding is not None:
            self.cache_embeddings[cache_key] = embedding
