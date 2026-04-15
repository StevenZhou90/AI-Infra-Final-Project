import pytest

from common.types import ActionChunk, ActionStep
from client.action_buffer import ActionBuffer


def _make_chunk(n: int = 10, base: float = 0.0) -> ActionChunk:
    steps = [
        ActionStep(joint_targets=[base + i * 0.1] * 6, gripper_targets=[0.0, 0.0])
        for i in range(n)
    ]
    return ActionChunk(steps=steps, frequency_hz=50.0)


@pytest.fixture
def buffer():
    return ActionBuffer(frequency_hz=50.0, ensemble_weight=0.01, low_water_mark=3)


class TestActionBuffer:
    def test_push_and_pop(self, buffer: ActionBuffer):
        chunk = _make_chunk(5)
        buffer.push_chunk(chunk)
        assert buffer.buffered_count == 5

        action = buffer.pop_action()
        assert len(action.joint_targets) == 6
        assert buffer.buffered_count == 4

    def test_safe_stop_on_empty(self, buffer: ActionBuffer):
        action = buffer.pop_action()
        assert action.joint_targets == [0.0] * 14

    def test_temporal_ensembling(self, buffer: ActionBuffer):
        chunk1 = _make_chunk(5, base=1.0)
        chunk2 = _make_chunk(5, base=2.0)
        buffer.push_chunk(chunk1)
        buffer.push_chunk(chunk2)

        action = buffer.pop_action()
        assert action.joint_targets[0] != 1.0
        assert action.joint_targets[0] != 2.0

    def test_needs_refill(self, buffer: ActionBuffer):
        assert buffer.needs_refill
        chunk = _make_chunk(20)
        buffer.push_chunk(chunk)
        assert not buffer.needs_refill

    def test_refill_callback(self, buffer: ActionBuffer):
        called = [False]

        def on_refill():
            called[0] = True

        buffer.set_refill_callback(on_refill)
        buffer.push_chunk(_make_chunk(5))

        for _ in range(3):
            buffer.pop_action()

        assert called[0]

    def test_clear(self, buffer: ActionBuffer):
        buffer.push_chunk(_make_chunk(10))
        buffer.clear()
        assert buffer.buffered_count == 0

    def test_peek(self, buffer: ActionBuffer):
        buffer.push_chunk(_make_chunk(5))
        peeked = buffer.peek(3)
        assert len(peeked) == 3
        assert buffer.buffered_count == 5  # peek doesn't consume

    def test_buffered_seconds(self, buffer: ActionBuffer):
        buffer.push_chunk(_make_chunk(50))
        assert buffer.buffered_seconds == pytest.approx(1.0, rel=0.01)

    def test_stats(self, buffer: ActionBuffer):
        buffer.push_chunk(_make_chunk(10))
        buffer.pop_action()
        stats = buffer.stats()
        assert stats["buffered"] == 9
        assert stats["consumed"] == 1
        assert stats["chunks_received"] == 1
