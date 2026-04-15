from __future__ import annotations

import logging
import time
from collections import deque

from common.types import EncodedFrame, VideoBurst

logger = logging.getLogger(__name__)


class VideoBurstBatcher:
    """Batches consecutive frames into burst packets to reduce gRPC overhead.

    Instead of sending 50 individual frames/sec per camera, batch them into
    bursts of N frames (e.g., 4 frames at 50Hz = 80ms per burst).
    """

    def __init__(self, burst_size: int = 4, max_latency_ms: float = 100.0) -> None:
        self._burst_size = burst_size
        self._max_latency_ms = max_latency_ms
        self._buffers: dict[str, deque[EncodedFrame]] = {}
        self._first_frame_time: dict[str, int] = {}

    @property
    def burst_size(self) -> int:
        return self._burst_size

    @burst_size.setter
    def burst_size(self, value: int) -> None:
        self._burst_size = max(1, value)

    def add_frame(self, frame: EncodedFrame) -> VideoBurst | None:
        """Add a frame. Returns a VideoBurst when the batch is full or max latency is hit."""
        cam = frame.camera_id or "default"

        if cam not in self._buffers:
            self._buffers[cam] = deque()
            self._first_frame_time[cam] = frame.timestamp_ns or time.time_ns()

        self._buffers[cam].append(frame)

        if self._should_flush(cam):
            return self._flush(cam)
        return None

    def flush_all(self) -> list[VideoBurst]:
        """Force-flush all buffered cameras. Call before shutdown or on timeout."""
        bursts = []
        for cam in list(self._buffers.keys()):
            if self._buffers[cam]:
                bursts.append(self._flush(cam))
        return bursts

    def flush_camera(self, camera_id: str) -> VideoBurst | None:
        cam = camera_id or "default"
        if cam in self._buffers and self._buffers[cam]:
            return self._flush(cam)
        return None

    def pending_count(self, camera_id: str | None = None) -> int:
        if camera_id:
            return len(self._buffers.get(camera_id, []))
        return sum(len(buf) for buf in self._buffers.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _should_flush(self, camera_id: str) -> bool:
        buf = self._buffers[camera_id]
        if len(buf) >= self._burst_size:
            return True

        first_ns = self._first_frame_time.get(camera_id, 0)
        now = time.time_ns()
        elapsed_ms = (now - first_ns) / 1e6
        if elapsed_ms >= self._max_latency_ms and len(buf) > 0:
            return True

        return False

    def _flush(self, camera_id: str) -> VideoBurst:
        buf = self._buffers[camera_id]
        frames = list(buf)
        buf.clear()

        start_ns = frames[0].timestamp_ns if frames else 0
        end_ns = frames[-1].timestamp_ns if frames else 0
        self._first_frame_time[camera_id] = time.time_ns()

        return VideoBurst(
            frames=frames,
            camera_id=camera_id,
            start_timestamp_ns=start_ns,
            end_timestamp_ns=end_ns,
        )
