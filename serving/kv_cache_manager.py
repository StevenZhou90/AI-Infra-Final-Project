"""KV cache manager — memory-budgeted caching with prefix reuse and sliding window.

Manages GPU memory for transformer KV caches across concurrent requests.
Supports:
- Per-sequence cache allocation and tracking
- Cross-timestep prefix caching (reuse instruction KV across robot control steps)
- Sliding window to bound per-sequence memory
- LRU eviction to stay within memory budget (target: 90% utilization)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch

logger = logging.getLogger(__name__)

PastKV = Any  # tuple[tuple[Tensor, Tensor], ...] or transformers.DynamicCache


# ---------------------------------------------------------------------------
# KV cache utilities — work with both legacy tuple and DynamicCache formats
# ---------------------------------------------------------------------------

def is_dynamic_cache(kv: PastKV) -> bool:
    return hasattr(kv, "key_cache") and hasattr(kv, "value_cache")


def kv_seq_len(kv: PastKV) -> int:
    if kv is None:
        return 0
    if is_dynamic_cache(kv):
        return kv.get_seq_length()
    return kv[0][0].shape[2]


def kv_num_layers(kv: PastKV) -> int:
    if kv is None:
        return 0
    if is_dynamic_cache(kv):
        return len(kv.key_cache)
    return len(kv)


def measure_kv_bytes(kv: PastKV) -> int:
    if kv is None:
        return 0
    if is_dynamic_cache(kv):
        total = sum(k.nelement() * k.element_size() for k in kv.key_cache)
        total += sum(v.nelement() * v.element_size() for v in kv.value_cache)
        return total
    return sum(
        k.nelement() * k.element_size() + v.nelement() * v.element_size()
        for k, v in kv
    )


def trim_kv(kv: PastKV, max_len: int) -> PastKV:
    """Trim every layer's KV cache to *max_len* tokens along the sequence dim."""
    if kv is None:
        return None
    if is_dynamic_cache(kv):
        for i in range(len(kv.key_cache)):
            kv.key_cache[i] = kv.key_cache[i][:, :, :max_len, :].contiguous()
            kv.value_cache[i] = kv.value_cache[i][:, :, :max_len, :].contiguous()
        return kv
    return tuple(
        (k[:, :, :max_len, :].contiguous(), v[:, :, :max_len, :].contiguous())
        for k, v in kv
    )


def extract_kv_layers(kv: PastKV, num_layers: int) -> PastKV:
    """Return a new KV object containing only the first *num_layers* layers."""
    if kv is None:
        return None
    if is_dynamic_cache(kv):
        from transformers import DynamicCache

        new = DynamicCache()
        new.key_cache = list(kv.key_cache[:num_layers])
        new.value_cache = list(kv.value_cache[:num_layers])
        return new
    return kv[:num_layers]


def clone_kv(kv: PastKV) -> PastKV:
    if kv is None:
        return None
    if is_dynamic_cache(kv):
        from transformers import DynamicCache

        new = DynamicCache()
        new.key_cache = [k.clone() for k in kv.key_cache]
        new.value_cache = [v.clone() for v in kv.value_cache]
        return new
    return tuple((k.clone(), v.clone()) for k, v in kv)


# ---------------------------------------------------------------------------
# Cache entry and statistics
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    sequence_id: str
    past_key_values: PastKV
    num_tokens: int
    memory_bytes: int
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    prefix_hash: str = ""


@dataclass
class CacheStats:
    total_entries: int = 0
    total_memory_mb: float = 0.0
    budget_mb: float = 0.0
    utilization: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class KVCacheManager:
    """Thread-safe KV cache manager with memory budgeting and prefix reuse."""

    def __init__(
        self,
        max_memory_mb: float = 4096,
        target_utilization: float = 0.90,
        sliding_window: int | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self._max_bytes = int(max_memory_mb * 1024 * 1024)
        self._target_util = target_utilization
        self._sliding_window = sliding_window
        self._device = torch.device(device) if isinstance(device, str) else device

        self._caches: dict[str, CacheEntry] = {}
        self._prefix_index: dict[str, str] = {}  # prefix_hash → sequence_id
        self._current_bytes = 0
        self._lock = Lock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info(
            "KVCacheManager: budget=%.0f MB  target_util=%.0f%%  window=%s  device=%s",
            max_memory_mb, target_utilization * 100,
            sliding_window or "unlimited", self._device,
        )

    # ---- public API -------------------------------------------------------

    def get(self, sequence_id: str) -> PastKV | None:
        with self._lock:
            entry = self._caches.get(sequence_id)
            if entry is None:
                self._misses += 1
                return None
            entry.last_access = time.time()
            entry.access_count += 1
            self._hits += 1
            return entry.past_key_values

    def get_by_prefix(self, prefix_hash: str) -> PastKV | None:
        """Retrieve a cache that was stored under a given prompt prefix hash."""
        with self._lock:
            seq_id = self._prefix_index.get(prefix_hash)
            if seq_id is None:
                self._misses += 1
                return None
            entry = self._caches.get(seq_id)
            if entry is None:
                del self._prefix_index[prefix_hash]
                self._misses += 1
                return None
            entry.last_access = time.time()
            entry.access_count += 1
            self._hits += 1
            return entry.past_key_values

    def put(
        self,
        sequence_id: str,
        past_key_values: PastKV,
        prefix_hash: str = "",
    ) -> bool:
        """Store or update a cache entry. Returns False if budget cannot accommodate."""
        if self._sliding_window is not None:
            past_key_values = trim_kv(past_key_values, self._sliding_window)

        mem = measure_kv_bytes(past_key_values)
        ntok = kv_seq_len(past_key_values)

        with self._lock:
            # Remove old entry for this sequence
            self._remove_locked(sequence_id)

            # Evict until we have room (within target utilization)
            target = int(self._max_bytes * self._target_util)
            while self._current_bytes + mem > target:
                if not self._evict_lru_locked():
                    logger.warning(
                        "KV cache full — cannot store %d bytes (used %d / %d)",
                        mem, self._current_bytes, self._max_bytes,
                    )
                    return False

            self._caches[sequence_id] = CacheEntry(
                sequence_id=sequence_id,
                past_key_values=past_key_values,
                num_tokens=ntok,
                memory_bytes=mem,
                prefix_hash=prefix_hash,
            )
            self._current_bytes += mem
            if prefix_hash:
                self._prefix_index[prefix_hash] = sequence_id

        return True

    def evict(self, sequence_id: str) -> bool:
        with self._lock:
            return self._remove_locked(sequence_id)

    def clear_all(self) -> None:
        with self._lock:
            for entry in self._caches.values():
                self._free(entry.past_key_values)
            self._caches.clear()
            self._prefix_index.clear()
            self._current_bytes = 0

    def stats(self) -> CacheStats:
        with self._lock:
            budget = self._max_bytes / (1024 * 1024)
            used = self._current_bytes / (1024 * 1024)
            return CacheStats(
                total_entries=len(self._caches),
                total_memory_mb=round(used, 1),
                budget_mb=round(budget, 1),
                utilization=round(used / budget, 3) if budget > 0 else 0,
                hit_count=self._hits,
                miss_count=self._misses,
                eviction_count=self._evictions,
            )

    @staticmethod
    def hash_prefix(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ---- internals --------------------------------------------------------

    def _remove_locked(self, sequence_id: str) -> bool:
        """Remove an entry (caller must hold self._lock)."""
        entry = self._caches.pop(sequence_id, None)
        if entry is None:
            return False
        self._current_bytes -= entry.memory_bytes
        if entry.prefix_hash:
            self._prefix_index.pop(entry.prefix_hash, None)
        self._free(entry.past_key_values)
        self._evictions += 1
        return True

    def _evict_lru_locked(self) -> bool:
        if not self._caches:
            return False
        lru_id = min(self._caches, key=lambda k: self._caches[k].last_access)
        freed = self._caches[lru_id].memory_bytes
        self._remove_locked(lru_id)
        logger.debug("Evicted '%s' (freed %.1f MB)", lru_id, freed / 1e6)
        return True

    @staticmethod
    def _free(kv: PastKV) -> None:
        if kv is None:
            return
        if is_dynamic_cache(kv):
            kv.key_cache.clear()
            kv.value_cache.clear()
        else:
            for pair in kv:
                del pair
