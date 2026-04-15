import asyncio
import time

import pytest

from common.types import InferenceRequest, Priority
from server.config import SchedulerConfig
from server.scheduler import PriorityScheduler


@pytest.fixture
def scheduler():
    config = SchedulerConfig(age_boost_ms=200, max_queue_depth=10, preempt_enabled=True)
    return PriorityScheduler(config)


def _req(priority: Priority = Priority.NORMAL, rid: str = "") -> InferenceRequest:
    return InferenceRequest(
        request_id=rid or f"r-{time.time_ns()}",
        model_id="test-model",
        priority=priority,
    )


class TestPriorityScheduler:
    def test_enqueue_dequeue_order(self, scheduler: PriorityScheduler):
        """Higher priority dequeued first, not FIFO."""
        scheduler.enqueue(_req(Priority.LOW, "low"))
        scheduler.enqueue(_req(Priority.HIGH, "high"))
        scheduler.enqueue(_req(Priority.NORMAL, "normal"))

        r1 = scheduler.dequeue_nowait()
        r2 = scheduler.dequeue_nowait()
        r3 = scheduler.dequeue_nowait()

        assert r1.request_id == "high"
        assert r2.request_id == "normal"
        assert r3.request_id == "low"

    def test_same_priority_fifo(self, scheduler: PriorityScheduler):
        """Same priority requests come out in FIFO order."""
        scheduler.enqueue(_req(Priority.NORMAL, "a"))
        scheduler.enqueue(_req(Priority.NORMAL, "b"))
        scheduler.enqueue(_req(Priority.NORMAL, "c"))

        assert scheduler.dequeue_nowait().request_id == "a"
        assert scheduler.dequeue_nowait().request_id == "b"
        assert scheduler.dequeue_nowait().request_id == "c"

    def test_queue_full_rejects(self, scheduler: PriorityScheduler):
        for i in range(10):
            assert scheduler.enqueue(_req(rid=str(i)))
        assert not scheduler.enqueue(_req(rid="overflow"))
        assert scheduler.depth == 10

    def test_cancel(self, scheduler: PriorityScheduler):
        scheduler.enqueue(_req(Priority.NORMAL, "to-cancel"))
        scheduler.enqueue(_req(Priority.NORMAL, "keep"))

        assert scheduler.cancel("to-cancel")
        assert scheduler.depth == 1

        r = scheduler.dequeue_nowait()
        assert r.request_id == "keep"

    def test_preemption_check(self, scheduler: PriorityScheduler):
        scheduler.enqueue(_req(Priority.CRITICAL, "urgent"))
        assert scheduler.should_preempt(Priority.NORMAL)
        assert not scheduler.should_preempt(Priority.CRITICAL)

    def test_empty_dequeue(self, scheduler: PriorityScheduler):
        assert scheduler.dequeue_nowait() is None

    @pytest.mark.asyncio
    async def test_async_dequeue_timeout(self, scheduler: PriorityScheduler):
        result = await scheduler.dequeue(timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_dequeue_wakes(self, scheduler: PriorityScheduler):
        async def delayed_enqueue():
            await asyncio.sleep(0.05)
            scheduler.enqueue(_req(Priority.HIGH, "delayed"))

        asyncio.create_task(delayed_enqueue())
        result = await scheduler.dequeue(timeout=1.0)
        assert result is not None
        assert result.request_id == "delayed"

    def test_stats(self, scheduler: PriorityScheduler):
        scheduler.enqueue(_req(Priority.HIGH, "s1"))
        scheduler.dequeue_nowait()
        stats = scheduler.stats()
        assert stats["enqueued"] == 1
        assert stats["dequeued"] == 1
        assert stats["depth"] == 0
