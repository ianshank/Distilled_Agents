"""Unit tests for batch processor."""

from __future__ import annotations

import pytest

from enhanced_system.core.batch_processor import BatchProcessor
from enhanced_system.core.enums import Priority


@pytest.mark.unit
class TestBatchProcessor:
    @pytest.mark.asyncio
    async def test_disabled_processes_immediately(self):
        processor = BatchProcessor({"enabled": False})

        async def _process_single(task):
            return {"task": task}

        processor._process_single = _process_single  # type: ignore[method-assign]
        result = await processor.submit_request("hello", priority=Priority.HIGH)
        assert result["task"] == "hello"

    def test_initialization(self):
        processor = BatchProcessor({"enabled": True, "max_batch_size": 8})
        assert processor.max_batch_size == 8
        assert processor.enabled is True

    @pytest.mark.asyncio
    async def test_start_stop_and_stats(self):
        processor = BatchProcessor({"enabled": True, "num_workers": 1, "max_batch_size": 2})
        await processor.start()
        stats = processor.get_stats()
        assert stats["running"] is True
        await processor.stop()
        assert processor.get_stats()["running"] is False

    @pytest.mark.asyncio
    async def test_process_batch_sync(self):
        processor = BatchProcessor({"enabled": True})

        async def _process_single(task):
            return task.upper()

        processor._process_single = _process_single  # type: ignore[method-assign]
        results = await processor.process_batch_sync(["a", "b"])
        assert results == ["A", "B"]
