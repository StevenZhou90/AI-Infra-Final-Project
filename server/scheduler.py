from __future__ import annotations

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from common.types import InferenceRequest, Priority
from server.config import SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _QueueEntry:
    """Heap entry. Lower sort key = higher scheduling priority."""
    sort_key: tuple[int, float, int] = field(compare=True)
    request: InferenceRequest = field(compare=False)
    gpu_id: int = field(compare=False, default=-1)
    cancelled: bool = field(compare=False, default=False)

    @staticmethod
    def make(request: InferenceRequest, seq: int, gpu_id: int = -1) -> _QueueEntry:
        effective_priority = -int(request.priority)
        enqueue_sec = request.enqueue_time_ns / 1e9
        return _QueueEntry(
            sort_key=(effective_priority, enqueue_sec, seq),
            request=request,
            gpu_id=gpu_id,
        )


class PriorityScheduler:
    """Priority-heap scheduler with age-boosting and optional preemption."""

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self._heap: list[_QueueEntry] = []
        self._seq = 0
        self._event = asyncio.Event()
        self._depth = 0
        self._metrics = _SchedulerMetrics()

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, request: InferenceRequest, gpu_id: int = -1) -> bool:
        """Add request to queue. Returns False if queue is full."""
        if self._depth >= self._config.max_queue_depth:
            logger.warning("Queue full (%d), rejecting %s", self._depth, request.request_id)
            self._metrics.rejected += 1
            return False

        request._seq = self._seq
        entry = _QueueEntry.make(request, self._seq, gpu_id)
        self._seq += 1
        heapq.heappush(self._heap, entry)
        self._depth += 1
        self._metrics.enqueued += 1
        self._metrics.per_priority[request.priority] = (
            self._metrics.per_priority.get(request.priority, 0) + 1
        )
        self._event.set()
        return True

    # ------------------------------------------------------------------
    # Dequeue
    # ------------------------------------------------------------------

    async def dequeue(self, timeout: float | None = None) -> InferenceRequest | None:
        """Wait for and return the highest-priority request."""
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            self._apply_age_boost()

            while self._heap:
                entry = heapq.heappop(self._heap)
                self._depth -= 1
                if entry.cancelled:
                    continue
                wait_ns = time.time_ns() - entry.request.enqueue_time_ns
                self._metrics.total_wait_ns += wait_ns
                self._metrics.dequeued += 1
                return entry.request

            self._event.clear()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._event.wait(), remaining)
                except asyncio.TimeoutError:
                    return None
            else:
                await self._event.wait()

    def dequeue_nowait(self) -> InferenceRequest | None:
        """Non-blocking dequeue."""
        self._apply_age_boost()
        while self._heap:
            entry = heapq.heappop(self._heap)
            self._depth -= 1
            if entry.cancelled:
                continue
            wait_ns = time.time_ns() - entry.request.enqueue_time_ns
            self._metrics.total_wait_ns += wait_ns
            self._metrics.dequeued += 1
            return entry.request
        return None

    # ------------------------------------------------------------------
    # Preemption
    # ------------------------------------------------------------------

    def should_preempt(self, running_priority: Priority) -> bool:
        """Check if head-of-queue should preempt the currently running request."""
        if not self._config.preempt_enabled:
            return False
        if not self._heap:
            return False
        top = self._heap[0]
        if top.cancelled:
            return False
        return top.request.priority == Priority.CRITICAL and running_priority < Priority.CRITICAL

    def peek_priority(self) -> Priority | None:
        for entry in self._heap:
            if not entry.cancelled:
                return entry.request.priority
        return None

    # ------------------------------------------------------------------
    # Age boost
    # ------------------------------------------------------------------

    def _apply_age_boost(self) -> None:
        """Promote requests that have waited too long."""
        if self._config.age_boost_ms <= 0:
            return

        now_ns = time.time_ns()
        boost_ns = int(self._config.age_boost_ms * 1e6)
        rebuilt = False

        for entry in self._heap:
            if entry.cancelled:
                continue
            age_ns = now_ns - entry.request.enqueue_time_ns
            if age_ns > boost_ns and entry.request.priority < Priority.CRITICAL:
                new_pri = Priority(min(entry.request.priority + 1, Priority.CRITICAL))
                if new_pri != entry.request.priority:
                    entry.request.priority = new_pri
                    entry.sort_key = (
                        -int(new_pri),
                        entry.sort_key[1],
                        entry.sort_key[2],
                    )
                    rebuilt = True
                    self._metrics.age_boosts += 1

        if rebuilt:
            heapq.heapify(self._heap)

    # ------------------------------------------------------------------
    # Cancellation / stats
    # ------------------------------------------------------------------

    def cancel(self, request_id: str) -> bool:
        for entry in self._heap:
            if entry.request.request_id == request_id and not entry.cancelled:
                entry.cancelled = True
                self._depth -= 1
                return True
        return False

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def is_empty(self) -> bool:
        return self._depth <= 0

    def stats(self) -> dict[str, Any]:
        avg_wait_ms = 0.0
        if self._metrics.dequeued > 0:
            avg_wait_ms = (self._metrics.total_wait_ns / self._metrics.dequeued) / 1e6
        return {
            "depth": self._depth,
            "enqueued": self._metrics.enqueued,
            "dequeued": self._metrics.dequeued,
            "rejected": self._metrics.rejected,
            "age_boosts": self._metrics.age_boosts,
            "avg_wait_ms": round(avg_wait_ms, 2),
            "per_priority": {p.name: c for p, c in self._metrics.per_priority.items()},
        }


@dataclass
class _SchedulerMetrics:
    enqueued: int = 0
    dequeued: int = 0
    rejected: int = 0
    age_boosts: int = 0
    total_wait_ns: int = 0
    per_priority: dict[Priority, int] = field(default_factory=dict)
