"""
Batch Processing Module
Provides intelligent batching for improved throughput
"""

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from enhanced_system.core.enums import Priority

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Individual batch request"""

    request_id: str
    task: str
    priority: Priority
    future: asyncio.Future
    timestamp: float
    metadata: Dict[str, Any]

    def __lt__(self, other):
        """For priority queue comparison"""
        if self.priority.numeric_value != other.priority.numeric_value:
            return self.priority.numeric_value < other.priority.numeric_value
        return self.timestamp < other.timestamp


class BatchProcessor:
    """
    Intelligent batch processing system

    Features:
    - Priority-based queuing
    - Dynamic batching (time + size based)
    - Multiple worker threads
    - Request routing and result distribution
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize batch processor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.max_batch_size = config.get("max_batch_size", 32)
        self.batch_timeout_ms = config.get("batch_timeout_ms", 100)
        self.num_workers = config.get("num_workers", 4)
        self.queue_size = config.get("queue_size", 1000)
        self.priority_levels = config.get("priority_levels", 3)

        # Request queue (priority queue)
        self.request_queue: List[BatchRequest] = []
        self.queue_lock = asyncio.Lock()

        # Workers
        self.workers: List[asyncio.Task] = []
        self.running = False

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "queue_length": 0,
        }

        logger.info(
            f"BatchProcessor initialized (max_batch_size: {self.max_batch_size}, "
            f"workers: {self.num_workers})"
        )

    async def start(self):
        """Start batch processing workers"""
        if self.running:
            logger.warning("Batch processor already running")
            return

        self.running = True

        # Start worker tasks
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._batch_worker(i))
            self.workers.append(worker)

        logger.info(f"Started {self.num_workers} batch workers")

    async def stop(self):
        """Stop batch processing workers"""
        if not self.running:
            return

        self.running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()
        logger.info("Stopped all batch workers")

    async def submit_request(
        self,
        task: str,
        priority: Priority = Priority.NORMAL,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Submit request for batch processing

        Args:
            task: Input task
            priority: Request priority
            request_id: Optional request ID
            metadata: Optional metadata

        Returns:
            Processing result
        """
        if not self.enabled:
            # Batching disabled, process immediately
            logger.debug("Batching disabled, processing immediately")
            return await self._process_single(task)

        # Check queue size
        async with self.queue_lock:
            if len(self.request_queue) >= self.queue_size:
                raise Exception(f"Queue full ({self.queue_size} requests)")

        # Create request
        future = asyncio.Future()
        request = BatchRequest(
            request_id=request_id or f"req_{time.time()}_{id(future)}",
            task=task,
            priority=priority,
            future=future,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Add to queue
        async with self.queue_lock:
            heapq.heappush(self.request_queue, request)
            self.stats["total_requests"] += 1
            self.stats["queue_length"] = len(self.request_queue)

        logger.debug(f"Request {request.request_id} queued (priority: {priority.name})")

        # Wait for result
        return await future

    async def _batch_worker(self, worker_id: int):
        """
        Batch processing worker

        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Batch worker {worker_id} started")

        while self.running:
            try:
                # Collect batch
                batch = await self._collect_batch()

                if not batch:
                    # No requests, wait a bit
                    await asyncio.sleep(0.01)
                    continue

                logger.debug(f"Worker {worker_id} processing batch of {len(batch)} requests")

                # Process batch
                try:
                    results = await self._process_batch(batch)

                    # Distribute results
                    for request, result in zip(batch, results):
                        if not request.future.done():
                            request.future.set_result(result)

                    # Update stats
                    self.stats["total_batches"] += 1
                    total_batch_sizes = self.stats["avg_batch_size"] * (
                        self.stats["total_batches"] - 1
                    ) + len(batch)
                    self.stats["avg_batch_size"] = total_batch_sizes / self.stats["total_batches"]

                except Exception as e:
                    logger.error(f"Batch processing error: {e}")

                    # Set exception for all requests in batch
                    for request in batch:
                        if not request.future.done():
                            request.future.set_exception(e)

            except asyncio.CancelledError:
                logger.info(f"Batch worker {worker_id} cancelled")
                break

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

        logger.info(f"Batch worker {worker_id} stopped")

    async def _collect_batch(self) -> List[BatchRequest]:
        """
        Collect a batch of requests

        Returns:
            List of batch requests
        """
        batch = []
        deadline = time.time() + (self.batch_timeout_ms / 1000.0)

        while len(batch) < self.max_batch_size and time.time() < deadline:
            async with self.queue_lock:
                if self.request_queue:
                    # Pop highest priority request
                    request = heapq.heappop(self.request_queue)
                    batch.append(request)
                    self.stats["queue_length"] = len(self.request_queue)
                else:
                    # Queue empty
                    break

            # Small delay to allow more requests to accumulate
            if len(batch) < self.max_batch_size:
                await asyncio.sleep(0.001)

        return batch

    async def _process_batch(self, batch: List[BatchRequest]) -> List[Any]:
        """
        Process a batch of requests

        Args:
            batch: List of requests

        Returns:
            List of results
        """
        # Extract tasks
        tasks = [req.task for req in batch]

        # Process all tasks
        # In a real implementation, this would call the actual model/agent
        # For now, simulate processing
        results = []
        for task in tasks:
            result = await self._process_single(task)
            results.append(result)

        return results

    async def _process_single(self, task: str) -> Dict[str, Any]:
        """
        Process a single task (placeholder)

        Args:
            task: Input task

        Returns:
            Processing result
        """
        # Simulate processing
        await asyncio.sleep(0.01)

        return {"response": f"Processed: {task[:50]}...", "success": True, "processing_time_ms": 10}

    def set_batch_processor(self, processor_func: Callable):
        """
        Set custom batch processing function

        Args:
            processor_func: Function to process batches
                           Should accept List[str] and return List[Any]
        """
        self._custom_processor = processor_func
        logger.info("Custom batch processor registered")

    async def _process_batch_custom(self, batch: List[BatchRequest]) -> List[Any]:
        """
        Process batch using custom processor

        Args:
            batch: List of requests

        Returns:
            List of results
        """
        if not hasattr(self, "_custom_processor"):
            return await self._process_batch(batch)

        tasks = [req.task for req in batch]

        if asyncio.iscoroutinefunction(self._custom_processor):
            results = await self._custom_processor(tasks)
        else:
            results = self._custom_processor(tasks)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        stats = self.stats.copy()
        stats["running"] = self.running
        stats["num_workers"] = len(self.workers)
        return stats

    async def process_batch_sync(self, tasks: List[str], **kwargs) -> List[Any]:
        """
        Process a batch synchronously (no queuing)

        Args:
            tasks: List of tasks
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        results = []
        for task in tasks:
            result = await self._process_single(task)
            results.append(result)

        return results


async def process_in_batches(
    tasks: List[str], config: Dict[str, Any], processor_func: Optional[Callable] = None
) -> List[Any]:
    """
    Convenience function to process tasks in batches

    Args:
        tasks: List of tasks
        config: Configuration
        processor_func: Optional custom processor

    Returns:
        List of results
    """
    batch_processor = BatchProcessor(config)

    if processor_func:
        batch_processor.set_batch_processor(processor_func)

    await batch_processor.start()

    try:
        # Submit all tasks
        futures = [batch_processor.submit_request(task) for task in tasks]

        # Wait for all results
        results = await asyncio.gather(*futures)

        return results

    finally:
        await batch_processor.stop()
