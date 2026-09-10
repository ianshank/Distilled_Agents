"""S3 L3 cache."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from enhanced_system.core.base.cache import BaseCache
from enhanced_system.core.cache.serialize import dumps, loads

logger = logging.getLogger(__name__)


def _default_region() -> str:
    env_region = os.getenv("AWS_REGION") or os.getenv("MANGOMAS_AWS_REGION")
    if env_region:
        return env_region
    try:
        from enhanced_system.ops.settings import get_settings

        return get_settings().aws_region
    except Exception:
        return "us-east-1"


try:
    import boto3
    from botocore.exceptions import ClientError

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    ClientError = Exception  # type: ignore[misc,assignment]
    logging.warning("Boto3 not available. L3 caching will be disabled.")


class L3Cache(BaseCache):
    """S3 persistent cache using JSON serialization."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True) and S3_AVAILABLE
        self.bucket = config.get("bucket", "agent-cache")
        self.prefix = config.get("prefix", "enhanced-agents/")
        self.region = config.get("region") or _default_region()
        self.s3_client = None
        self._hits = 0
        self._misses = 0

        if self.enabled:
            try:
                self.s3_client = boto3.client("s3", region_name=self.region)
                self.s3_client.head_bucket(Bucket=self.bucket)
                logger.info("L3 S3 cache initialized successfully (bucket: %s)", self.bucket)
            except Exception as exc:
                logger.warning("Failed to initialize S3: %s. L3 caching disabled.", exc)
                self.enabled = False
                self.s3_client = None

    def _get_s3_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled or not self.s3_client:
            return None
        try:
            s3_key = self._get_s3_key(key)
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            self._hits += 1
            return loads(response["Body"].read())
        except ClientError as exc:
            error_code = ""
            if hasattr(exc, "response"):
                error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                self._misses += 1
            else:
                logger.error("L3 cache get error: %s", exc)
            return None
        except Exception as exc:
            logger.error("L3 cache get error: %s", exc)
            return None

    async def get_async(self, key: str) -> Optional[Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self.enabled or not self.s3_client:
            return
        try:
            s3_key = self._get_s3_key(key)
            self.s3_client.put_object(Bucket=self.bucket, Key=s3_key, Body=dumps(value))
        except Exception as exc:
            logger.error("L3 cache set error: %s", exc)

    async def set_async(self, key: str, value: Any) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.set, key, value)

    def delete(self, key: str) -> bool:
        if not self.enabled or not self.s3_client:
            return False
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=self._get_s3_key(key))
            return True
        except Exception as exc:
            logger.error("L3 cache delete error: %s", exc)
            return False

    def clear(self) -> None:
        logger.info("L3 cache clear skipped (S3 persistence)")

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "enabled": self.enabled,
            "bucket": self.bucket if self.enabled else None,
        }
