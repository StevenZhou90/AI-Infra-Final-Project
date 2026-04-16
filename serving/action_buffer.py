"""Client-side action buffer — smooths latency between server and sim.

The server returns action chunks (e.g. 100 actions at once).
The buffer stores them and feeds one action per sim step.
While the buffer drains, the client can pre-fetch the next chunk
so the sim never stalls waiting for the network.

Pluggable: swap interpolation strategy without touching the client.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class ActionBuffer:
    """Thread-safe action buffer with optional prefetch callback.

    Usage:
        buffer = ActionBuffer(action_dim=14, prefetch_fn=my_fetch)
        buffer.push(actions_array)  # push a chunk from server

        for step in sim_loop:
            action = buffer.pop()   # get one action per step
            if action is None:
                action = fallback    # buffer empty, use last action or zero
    """

    def __init__(
        self,
        action_dim: int,
        prefetch_fn: Callable[[], np.ndarray | None] | None = None,
        prefetch_threshold: int = 10,
        max_buffer_size: int = 500,
    ) -> None:
        self._action_dim = action_dim
        self._queue: deque[np.ndarray] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._prefetch_fn = prefetch_fn
        self._prefetch_threshold = prefetch_threshold
        self._prefetching = False
        self._last_action: np.ndarray = np.zeros(action_dim, dtype=np.float32)

    def push(self, actions: np.ndarray) -> int:
        """Push an action chunk into the buffer. Returns new buffer size."""
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        elif actions.ndim == 1 and actions.shape[0] == self._action_dim:
            actions = actions.reshape(1, -1)

        with self._lock:
            for i in range(actions.shape[0]):
                self._queue.append(actions[i])
            size = len(self._queue)

        logger.debug("Pushed %d actions, buffer size: %d", actions.shape[0], size)
        return size

    def pop(self) -> np.ndarray:
        """Pop one action. Returns zeros if empty. Triggers prefetch if low."""
        with self._lock:
            if self._queue:
                action = self._queue.popleft()
                self._last_action = action.copy()
                remaining = len(self._queue)
            else:
                action = self._last_action.copy()
                remaining = 0

        if remaining <= self._prefetch_threshold and not self._prefetching:
            self._maybe_prefetch()

        return action

    def pop_or_none(self) -> np.ndarray | None:
        """Pop one action or return None if buffer is empty."""
        with self._lock:
            if self._queue:
                action = self._queue.popleft()
                self._last_action = action.copy()
                return action
            return None

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._last_action = np.zeros(self._action_dim, dtype=np.float32)

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def _maybe_prefetch(self) -> None:
        """Trigger async prefetch if a callback is registered."""
        if self._prefetch_fn is None:
            return
        self._prefetching = True

        def _do_prefetch():
            try:
                result = self._prefetch_fn()
                if result is not None:
                    self.push(result)
            except Exception as e:
                logger.warning("Prefetch failed: %s", e)
            finally:
                self._prefetching = False

        thread = threading.Thread(target=_do_prefetch, daemon=True)
        thread.start()
