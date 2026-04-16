"""Priority scheduler — scores and orders inference requests.

Pluggable: swap the scoring function or replace the whole scheduler
without touching the gRPC server. The server just calls submit() and
wait_next().

Scoring:
    score = base_priority * priority_weight + age_ms * age_weight + cache_bonus
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Callable
from heapq import heappush, heappop

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    priority_weight: float = 1000.0
    age_weight: float = 1.0
    age_boost_after_ms: float = 500.0
    cache_bonus: float = 200.0
    max_queue_depth: int = 1024


@dataclass(order=True)
class QueuedRequest:
    """Wrapper that makes requests sortable by negative score (highest first)."""
    neg_score: float
    item: dict = field(compare=False)
    submitted_ns: int = field(compare=False, default_factory=time.time_ns)
    event: threading.Event = field(compare=False, default_factory=threading.Event)
    result: Any = field(compare=False, default=None)
    error: Exception | None = field(compare=False, default=None)


class PriorityScheduler:
    """Thread-safe priority queue with pluggable scoring.

    Usage:
        scheduler = PriorityScheduler(config)
        # Server handler submits a request and blocks until it's processed:
        result = scheduler.submit(request_dict)

        # Worker thread pulls the next request:
        queued = scheduler.wait_next(timeout=1.0)
        queued.result = do_inference(queued.item)
        queued.event.set()  # unblocks the submitter
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        score_fn: Callable[[dict, SchedulerConfig], float] | None = None,
    ) -> None:
        self._config = config or SchedulerConfig()
        self._score_fn = score_fn or default_score
        self._heap: list[QueuedRequest] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._total_submitted = 0
        self._total_processed = 0

    def submit(self, request: dict, timeout: float | None = None) -> Any:
        """Submit a request and block until it's processed. Returns the result."""
        score = self._score_fn(request, self._config)
        entry = QueuedRequest(neg_score=-score, item=request)

        with self._lock:
            if len(self._heap) >= self._config.max_queue_depth:
                raise RuntimeError(f"Scheduler queue full ({self._config.max_queue_depth})")
            heappush(self._heap, entry)
            self._total_submitted += 1
            self._not_empty.notify()

        entry.event.wait(timeout=timeout)
        if entry.error:
            raise entry.error
        return entry.result

    def wait_next(self, timeout: float = 1.0) -> QueuedRequest | None:
        """Block until a request is available, then pop the highest-priority one."""
        with self._not_empty:
            while not self._heap:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            self._rescore_heap()
            entry = heappop(self._heap)
            self._total_processed += 1
        return entry

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._heap)

    def stats(self) -> dict:
        with self._lock:
            return {
                "queue_depth": len(self._heap),
                "total_submitted": self._total_submitted,
                "total_processed": self._total_processed,
            }

    def _rescore_heap(self) -> None:
        """Re-score all entries (age changes over time) and re-heapify."""
        now_ns = time.time_ns()
        for entry in self._heap:
            age_ms = (now_ns - entry.submitted_ns) / 1e6
            entry.item["_age_ms"] = age_ms
            score = self._score_fn(entry.item, self._config)
            entry.neg_score = -score
        self._heap.sort()


def default_score(request: dict, config: SchedulerConfig) -> float:
    """Default scoring: base priority + age boost + optional cache bonus."""
    priority = request.get("priority", 1)
    age_ms = request.get("_age_ms", 0.0)
    has_warm_cache = request.get("_has_warm_cache", False)

    score = priority * config.priority_weight
    score += age_ms * config.age_weight
    if has_warm_cache:
        score += config.cache_bonus

    return score
