"""L2/L3 cache JSON roundtrip with mocked Redis and S3."""

from __future__ import annotations

import pytest
from enhanced_system.core.cache.l2_redis import L2Cache
from enhanced_system.core.cache.l3_s3 import L3Cache
from enhanced_system.core.cache.serialize import dumps, loads


class _Body:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _MockS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def head_bucket(self, Bucket: str):
        return {}

    def put_object(self, Bucket: str, Key: str, Body: bytes):
        self.objects[Key] = Body

    def get_object(self, Bucket: str, Key: str):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": _Body(self.objects[Key])}


@pytest.mark.unit
def test_json_serialize_roundtrip_bytes():
    payload = {"hits": 3, "nested": ["a", "b"], "ok": True}
    assert loads(dumps(payload)) == payload


@pytest.mark.unit
def test_l2_cache_mocked_redis_roundtrip(mock_redis):
    cache = L2Cache({"enabled": False, "ttl": 60})
    cache.enabled = True
    cache.redis_client = mock_redis
    cache.set("agent:1", {"response": "ok", "score": 0.9})
    assert cache.get("agent:1") == {"response": "ok", "score": 0.9}
    assert cache.get("missing") is None
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.unit
def test_l3_cache_mocked_s3_roundtrip():
    cache = L3Cache(
        {"enabled": False, "bucket": "test-bucket", "prefix": "cache/", "region": "us-west-2"}
    )
    cache.enabled = True
    cache.s3_client = _MockS3()
    cache.set("k1", {"v": 2})
    assert cache.get("k1") == {"v": 2}
    assert cache.region == "us-west-2"
    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["bucket"] == "test-bucket"
