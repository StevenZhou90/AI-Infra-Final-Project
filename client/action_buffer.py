from __future__ import annotations

import logging
import math
import time
import threading
from collections import deque

from common.types import ActionChunk, ActionStep

logger = logging.getLogger(__name__)


class ActionBuffer:
    """Client-side buffer for smooth action execution with temporal ensembling.

    Receives action chunks (k future actions) from the server and blends
    overlapping predictions using exponential weighting (per ACT paper).
    Provides a safe stop if the buffer runs dry.
    """

    def __init__(
        self,
        frequency_hz: float = 50.0,
        ensemble_weight: float = 0.01,
        low_water_mark: int = 10,
        safe_stop_action: ActionStep | None = None,
    ) -> None:
        self._frequency_hz = frequency_hz
        self._dt_ns = int(1e9 / frequency_hz)
        self._ensemble_w = ensemble_weight
        self._low_water_mark = low_water_mark
        self._safe_stop = safe_stop_action or ActionStep(joint_targets=[0.0] * 14)

        self._buffer: deque[ActionStep] = deque()
        self._lock = threading.Lock()
        self._total_consumed = 0
        self._total_received_chunks = 0
        self._last_consume_ns = 0
        self._needs_refill_callback = None

    # ------------------------------------------------------------------
    # Receive chunks from server
    # ------------------------------------------------------------------

    def push_chunk(self, chunk: ActionChunk) -> None:
        """Add an action chunk. Overlapping steps are blended via temporal ensembling."""
        with self._lock:
            new_steps = chunk.steps
            overlap = len(self._buffer)

            if overlap > 0 and overlap <= len(new_steps):
                for i in range(overlap):
                    existing = self._buffer[i]
                    incoming = new_steps[i]
                    blended = self._temporal_ensemble(existing, incoming, i, overlap)
                    self._buffer[i] = blended
                for step in new_steps[overlap:]:
                    self._buffer.append(step)
            else:
                for step in new_steps:
                    self._buffer.append(step)

            self._total_received_chunks += 1

    # ------------------------------------------------------------------
    # Consume actions at control frequency
    # ------------------------------------------------------------------

    def pop_action(self) -> ActionStep:
        """Get the next action to execute. Returns safe-stop if buffer is empty."""
        with self._lock:
            if not self._buffer:
                logger.warning("Action buffer empty -- executing safe stop")
                return self._safe_stop

            action = self._buffer.popleft()
            self._total_consumed += 1
            self._last_consume_ns = time.time_ns()

            if len(self._buffer) <= self._low_water_mark and self._needs_refill_callback:
                self._needs_refill_callback()

            return action

    def peek(self, n: int = 1) -> list[ActionStep]:
        """Preview next N actions without consuming them."""
        with self._lock:
            return list(self._buffer)[:n]

    # ------------------------------------------------------------------
    # Temporal ensembling (ACT paper)
    # ------------------------------------------------------------------

    def _temporal_ensemble(
        self,
        existing: ActionStep,
        incoming: ActionStep,
        position: int,
        total_overlap: int,
    ) -> ActionStep:
        """Exponential-weighted blend: newer predictions get exponentially more weight.

        w = exp(-ensemble_weight * (total_overlap - position))
        blended = w * incoming + (1 - w) * existing
        """
        w = math.exp(-self._ensemble_w * (total_overlap - position))
        blended_joints = [
            w * inc + (1 - w) * ext
            for ext, inc in zip(existing.joint_targets, incoming.joint_targets)
        ]
        blended_gripper = [
            w * inc + (1 - w) * ext
            for ext, inc in zip(
                existing.gripper_targets or [],
                incoming.gripper_targets or [],
            )
        ]
        return ActionStep(joint_targets=blended_joints, gripper_targets=blended_gripper)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def buffered_seconds(self) -> float:
        return self.buffered_count / self._frequency_hz

    @property
    def needs_refill(self) -> bool:
        return self.buffered_count <= self._low_water_mark

    @property
    def is_empty(self) -> bool:
        return self.buffered_count == 0

    def set_refill_callback(self, callback) -> None:
        self._needs_refill_callback = callback

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def stats(self) -> dict:
        return {
            "buffered": self.buffered_count,
            "buffered_seconds": round(self.buffered_seconds, 2),
            "consumed": self._total_consumed,
            "chunks_received": self._total_received_chunks,
            "needs_refill": self.needs_refill,
            "frequency_hz": self._frequency_hz,
        }
