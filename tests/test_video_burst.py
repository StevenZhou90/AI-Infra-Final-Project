import time

import pytest

from common.types import Codec, EncodedFrame
from client.video_burst import VideoBurstBatcher


def _frame(camera_id: str = "cam0") -> EncodedFrame:
    return EncodedFrame(
        data=b"\x00" * 100,
        codec=Codec.JPEG,
        width=640,
        height=480,
        camera_id=camera_id,
        timestamp_ns=time.time_ns(),
    )


class TestVideoBurstBatcher:
    def test_burst_after_n_frames(self):
        batcher = VideoBurstBatcher(burst_size=3)
        assert batcher.add_frame(_frame()) is None
        assert batcher.add_frame(_frame()) is None
        burst = batcher.add_frame(_frame())
        assert burst is not None
        assert len(burst.frames) == 3

    def test_separate_cameras(self):
        batcher = VideoBurstBatcher(burst_size=2)
        batcher.add_frame(_frame("left"))
        batcher.add_frame(_frame("right"))

        assert batcher.pending_count("left") == 1
        assert batcher.pending_count("right") == 1

        burst = batcher.add_frame(_frame("left"))
        assert burst is not None
        assert burst.camera_id == "left"
        assert batcher.pending_count("right") == 1

    def test_flush_all(self):
        batcher = VideoBurstBatcher(burst_size=10)
        for _ in range(3):
            batcher.add_frame(_frame("cam0"))
        for _ in range(2):
            batcher.add_frame(_frame("cam1"))

        bursts = batcher.flush_all()
        assert len(bursts) == 2
        sizes = sorted(len(b.frames) for b in bursts)
        assert sizes == [2, 3]

    def test_timestamps(self):
        batcher = VideoBurstBatcher(burst_size=2)
        batcher.add_frame(_frame())
        burst = batcher.add_frame(_frame())
        assert burst.start_timestamp_ns > 0
        assert burst.end_timestamp_ns >= burst.start_timestamp_ns

    def test_pending_count_total(self):
        batcher = VideoBurstBatcher(burst_size=10)
        for _ in range(5):
            batcher.add_frame(_frame())
        assert batcher.pending_count() == 5
