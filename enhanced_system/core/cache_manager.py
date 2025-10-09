"""
Intelligent Multi-Level Caching System
Provides L1 (memory), L2 (Redis), and L3 (S3) caching with semantic similarity
"""

import hashlib
import logging
import pickle
from typing import Any, Optional, Callable, Dict
from functools import lru_cache
from datetime import datetime
import asyncio

try:
    import redis
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. L2 caching will be disabled.")

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("Boto3 not available. L3 caching will be disabled.")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence-transformers not available. Semantic caching will be disabled.")


logger = logging.getLogger(__name__)


class L1Cache:
    """In-memory LRU cache"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize L1 cache
        
        Args:
            max_size: Maximum number of entries
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            # Check if expired
            if (datetime.now().timestamp() - timestamp) < self.ttl:
                self._hits += 1
                logger.debug(f"L1 cache hit for key: {key[:20]}...")
                return value
            else:
                # Remove expired entry
                del self.cache[key]
        
        self._misses += 1
        logger.debug(f"L1 cache miss for key: {key[:20]}...")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        # Implement simple LRU eviction
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now().timestamp())
        logger.debug(f"L1 cache set for key: {key[:20]}...")
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        logger.info("L1 cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }


class L2Cache:
    """Redis distributed cache"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize L2 cache
        
        Args:
            config: Redis configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True) and REDIS_AVAILABLE
        self.ttl = config.get('ttl', 86400)
        self.redis_client = None
        self._hits = 0
        self._misses = 0
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=config.get('host', 'localhost'),
                    port=config.get('port', 6379),
                    db=config.get('db', 0),
                    socket_timeout=config.get('socket_timeout', 5),
                    socket_connect_timeout=config.get('socket_connect_timeout', 5),
                    decode_responses=False,
                    max_connections=config.get('pool_size', 10)
                )
                # Test connection
                self.redis_client.ping()
                logger.info("L2 Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}. L2 caching disabled.")
                self.enabled = False
                self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                self._hits += 1
                logger.debug(f"L2 cache hit for key: {key[:20]}...")
                return pickle.loads(value)
            else:
                self._misses += 1
                logger.debug(f"L2 cache miss for key: {key[:20]}...")
                return None
        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
            return None
    
    async def get_async(self, key: str) -> Optional[Any]:
        """Get value from cache asynchronously"""
        # For now, use sync version wrapped in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(key, self.ttl, serialized)
            logger.debug(f"L2 cache set for key: {key[:20]}...")
        except Exception as e:
            logger.error(f"L2 cache set error: {e}")
    
    async def set_async(self, key: str, value: Any) -> None:
        """Set value in cache asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.set, key, value)
    
    def clear(self) -> None:
        """Clear cache"""
        if self.enabled and self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("L2 cache cleared")
            except Exception as e:
                logger.error(f"L2 cache clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        stats = {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'enabled': self.enabled
        }
        
        if self.enabled and self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'total_commands_processed': info.get('total_commands_processed')
                })
            except Exception as e:
                logger.error(f"Failed to get Redis stats: {e}")
        
        return stats


class L3Cache:
    """S3 persistent cache"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize L3 cache
        
        Args:
            config: S3 configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True) and S3_AVAILABLE
        self.bucket = config.get('bucket', 'agent-cache')
        self.prefix = config.get('prefix', 'enhanced-agents/')
        self.region = config.get('region', 'us-east-1')
        self.s3_client = None
        self._hits = 0
        self._misses = 0
        
        if self.enabled:
            try:
                self.s3_client = boto3.client('s3', region_name=self.region)
                # Test connection
                self.s3_client.head_bucket(Bucket=self.bucket)
                logger.info(f"L3 S3 cache initialized successfully (bucket: {self.bucket})")
            except Exception as e:
                logger.warning(f"Failed to initialize S3: {e}. L3 caching disabled.")
                self.enabled = False
                self.s3_client = None
    
    def _get_s3_key(self, key: str) -> str:
        """Get full S3 key with prefix"""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.s3_client:
            return None
        
        try:
            s3_key = self._get_s3_key(key)
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            value = pickle.loads(response['Body'].read())
            self._hits += 1
            logger.debug(f"L3 cache hit for key: {key[:20]}...")
            return value
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                self._misses += 1
                logger.debug(f"L3 cache miss for key: {key[:20]}...")
            else:
                logger.error(f"L3 cache get error: {e}")
            return None
        except Exception as e:
            logger.error(f"L3 cache get error: {e}")
            return None
    
    async def get_async(self, key: str) -> Optional[Any]:
        """Get value from cache asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        if not self.enabled or not self.s3_client:
            return
        
        try:
            s3_key = self._get_s3_key(key)
            serialized = pickle.dumps(value)
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=serialized
            )
            logger.debug(f"L3 cache set for key: {key[:20]}...")
        except Exception as e:
            logger.error(f"L3 cache set error: {e}")
    
    async def set_async(self, key: str, value: Any) -> None:
        """Set value in cache asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.set, key, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'enabled': self.enabled,
            'bucket': self.bucket if self.enabled else None
        }


class SemanticCacheManager:
    """Semantic similarity-based cache key generation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize semantic cache manager
        
        Args:
            config: Semantic similarity configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True) and EMBEDDINGS_AVAILABLE
        self.threshold = config.get('threshold', 0.95)
        self.model_name = config.get('model', 'all-MiniLM-L6-v2')
        self.model = None
        self.cache_embeddings: Dict[str, np.ndarray] = {}
        
        if self.enabled:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Semantic cache initialized with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic cache: {e}")
                self.enabled = False
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text"""
        if not self.enabled or not self.model:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None
    
    def find_similar_cached(self, text: str) -> Optional[str]:
        """
        Find similar cached text using semantic similarity
        
        Args:
            text: Text to find similar cache for
        
        Returns:
            Cache key of similar text if found
        """
        if not self.enabled or not self.model:
            return None
        
        embedding = self.get_embedding(text)
        if embedding is None:
            return None
        
        # Check similarity with cached embeddings
        for cache_key, cached_embedding in self.cache_embeddings.items():
            similarity = np.dot(embedding, cached_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity >= self.threshold:
                logger.info(f"Found similar cached text with similarity: {similarity:.3f}")
                return cache_key
        
        return None
    
    def register_embedding(self, cache_key: str, text: str) -> None:
        """Register embedding for cache key"""
        if not self.enabled:
            return
        
        embedding = self.get_embedding(text)
        if embedding is not None:
            self.cache_embeddings[cache_key] = embedding


class IntelligentCacheManager:
    """
    Multi-level intelligent caching system
    
    Provides L1 (memory), L2 (Redis), and L3 (S3) caching
    with semantic similarity for cache key generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager
        
        Args:
            config: Cache configuration
        """
        self.config = config
        
        # Initialize cache levels
        self.l1_cache = L1Cache(
            max_size=config.get('l1', {}).get('max_size', 1000),
            ttl=config.get('l1', {}).get('ttl', 3600)
        )
        
        self.l2_cache = L2Cache(config.get('l2', {}))
        self.l3_cache = L3Cache(config.get('l3', {}))
        
        # Initialize semantic cache
        self.semantic_cache = SemanticCacheManager(
            config.get('semantic_similarity', {})
        )
        
        logger.info("IntelligentCacheManager initialized")
    
    def compute_cache_key(self, task: str, agent: str, params: Optional[Dict] = None) -> str:
        """
        Compute cache key for task
        
        Args:
            task: Task string
            agent: Agent identifier
            params: Optional parameters
        
        Returns:
            Cache key
        """
        # Check for semantic similarity if enabled
        similar_key = self.semantic_cache.find_similar_cached(task)
        if similar_key:
            return similar_key
        
        # Generate new cache key
        key_components = [task, agent]
        if params:
            key_components.append(str(sorted(params.items())))
        
        key_string = "|".join(key_components)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()
        
        # Register embedding for future lookups
        self.semantic_cache.register_embedding(cache_key, task)
        
        return cache_key
    
    async def get_or_compute(
        self, 
        key: str, 
        compute_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Get from cache or compute
        
        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            *args: Arguments for compute function
            **kwargs: Keyword arguments for compute function
        
        Returns:
            Cached or computed value
        """
        # Check L1 (fastest)
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Check L2 (fast)
        value = await self.l2_cache.get_async(key)
        if value is not None:
            # Populate L1
            self.l1_cache.set(key, value)
            return value
        
        # Check L3 (slow)
        value = await self.l3_cache.get_async(key)
        if value is not None:
            # Populate L2 and L1
            await self.l2_cache.set_async(key, value)
            self.l1_cache.set(key, value)
            return value
        
        # Compute value
        logger.info("Cache miss on all levels, computing value...")
        if asyncio.iscoroutinefunction(compute_func):
            value = await compute_func(*args, **kwargs)
        else:
            value = compute_func(*args, **kwargs)
        
        # Cache at all levels
        self.l1_cache.set(key, value)
        await self.l2_cache.set_async(key, value)
        await self.l3_cache.set_async(key, value)
        
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'l1': self.l1_cache.get_stats(),
            'l2': self.l2_cache.get_stats(),
            'l3': self.l3_cache.get_stats(),
            'semantic_enabled': self.semantic_cache.enabled
        }
    
    def clear_all(self) -> None:
        """Clear all cache levels"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        # Note: L3 cache (S3) is not cleared by default for persistence
        logger.info("All cache levels cleared (except L3)")

