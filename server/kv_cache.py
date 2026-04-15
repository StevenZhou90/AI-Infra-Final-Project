from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from common.types import Priority
from server.config import CacheConfig
from server.gpu_manager import GpuManager

logger = logging.getLogger(__name__)


@dataclass
class CachePage:
    page_id: int
    gpu_id: int
    block_size: int
    key_buffer: torch.Tensor | None = None
    value_buffer: torch.Tensor | None = None
    request_id: str = ""
    layer_idx: int = 0
    seq_start: int = 0
    seq_end: int = 0
    last_access_ns: int = 0
    is_free: bool = True

    def mark_used(self, request_id: str, layer_idx: int, seq_start: int, seq_end: int) -> None:
        self.request_id = request_id
        self.layer_idx = layer_idx
        self.seq_start = seq_start
        self.seq_end = seq_end
        self.last_access_ns = time.time_ns()
        self.is_free = False

    def mark_free(self) -> None:
        self.request_id = ""
        self.is_free = True
        self.seq_start = 0
        self.seq_end = 0


@dataclass
class RequestCache:
    request_id: str
    priority: Priority
    pages: list[CachePage] = field(default_factory=list)
    context_length: int = 0
    created_ns: int = field(default_factory=time.time_ns)
    last_access_ns: int = field(default_factory=time.time_ns)


class KVCacheManager:
    """Paged KV cache targeting 90% GPU memory utilization with sliding window eviction."""

    def __init__(self, config: CacheConfig, gpu_manager: GpuManager) -> None:
        self._config = config
        self._gpu = gpu_manager
        self._pages: dict[int, list[CachePage]] = {}  # gpu_id -> pages
        self._request_caches: dict[str, RequestCache] = {}
        self._page_counter = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization -- allocate page pools per GPU
    # ------------------------------------------------------------------

    def initialize(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Pre-allocate KV cache pages across all GPUs."""
        page_kv_bytes = self._page_bytes(num_heads, head_dim, dtype)

        for gpu_id in self._gpu.gpu_ids:
            budget = self._gpu.available_for_kv_cache(gpu_id)
            num_pages = budget // page_kv_bytes
            if num_pages <= 0:
                logger.warning("GPU %d: no budget for KV cache pages", gpu_id)
                continue

            device = torch.device(f"cuda:{gpu_id}")
            pages: list[CachePage] = []
            for _ in range(num_pages):
                pid = self._page_counter
                self._page_counter += 1
                page = CachePage(
                    page_id=pid,
                    gpu_id=gpu_id,
                    block_size=self._config.page_size,
                    key_buffer=torch.zeros(
                        num_layers, self._config.page_size, num_heads, head_dim,
                        dtype=dtype, device=device,
                    ),
                    value_buffer=torch.zeros(
                        num_layers, self._config.page_size, num_heads, head_dim,
                        dtype=dtype, device=device,
                    ),
                )
                pages.append(page)

            self._pages[gpu_id] = pages
            alloc_gb = (num_pages * page_kv_bytes) / (1 << 30)
            total_gb = self._gpu.get_info(gpu_id).total_memory / (1 << 30)
            logger.info(
                "GPU %d: allocated %d KV cache pages (%.2f GB / %.2f GB total = %.1f%%)",
                gpu_id, num_pages, alloc_gb, total_gb, 100 * alloc_gb / total_gb,
            )

        self._initialized = True

    # ------------------------------------------------------------------
    # Page allocation
    # ------------------------------------------------------------------

    def allocate_pages(
        self, request_id: str, gpu_id: int, num_pages: int, priority: Priority
    ) -> list[CachePage]:
        """Allocate pages for a request. Evicts if necessary."""
        free = self._get_free_pages(gpu_id)

        if len(free) < num_pages:
            self._evict(gpu_id, num_pages - len(free), priority)
            free = self._get_free_pages(gpu_id)

        if len(free) < num_pages:
            raise RuntimeError(
                f"GPU {gpu_id}: cannot allocate {num_pages} pages "
                f"({len(free)} free after eviction)"
            )

        allocated = free[:num_pages]
        for page in allocated:
            page.is_free = False
            page.request_id = request_id
            page.last_access_ns = time.time_ns()

        if request_id not in self._request_caches:
            self._request_caches[request_id] = RequestCache(
                request_id=request_id, priority=priority
            )
        self._request_caches[request_id].pages.extend(allocated)
        return allocated

    def release_request(self, request_id: str) -> int:
        """Free all pages held by a request. Returns number freed."""
        rc = self._request_caches.pop(request_id, None)
        if rc is None:
            return 0
        for page in rc.pages:
            page.mark_free()
        return len(rc.pages)

    # ------------------------------------------------------------------
    # Sliding window
    # ------------------------------------------------------------------

    def apply_sliding_window(self, request_id: str) -> int:
        """Evict pages outside the sliding window for this request. Returns pages freed."""
        rc = self._request_caches.get(request_id)
        if rc is None:
            return 0
        max_pages = self._config.sliding_window_size // self._config.page_size
        if len(rc.pages) <= max_pages:
            return 0

        evict_count = len(rc.pages) - max_pages
        to_evict = sorted(rc.pages, key=lambda p: p.seq_start)[:evict_count]
        for page in to_evict:
            page.mark_free()
            rc.pages.remove(page)
        rc.context_length = max_pages * self._config.page_size
        return evict_count

    # ------------------------------------------------------------------
    # Write / Read KV
    # ------------------------------------------------------------------

    def write_kv(
        self,
        page: CachePage,
        layer_idx: int,
        seq_offset: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write KV tensors into a page at a given layer and sequence offset."""
        seq_len = keys.shape[0]
        end = seq_offset + seq_len
        page.key_buffer[layer_idx, seq_offset:end] = keys  # type: ignore[index]
        page.value_buffer[layer_idx, seq_offset:end] = values  # type: ignore[index]
        page.last_access_ns = time.time_ns()
        page.seq_end = max(page.seq_end, end)

    def read_kv(
        self, pages: list[CachePage], layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate KV from multiple pages for a given layer."""
        keys = []
        values = []
        for page in sorted(pages, key=lambda p: p.seq_start):
            length = page.seq_end - page.seq_start
            if length > 0:
                keys.append(page.key_buffer[layer_idx, :length])  # type: ignore[index]
                values.append(page.value_buffer[layer_idx, :length])  # type: ignore[index]
        if not keys:
            device = pages[0].key_buffer.device if pages and pages[0].key_buffer is not None else "cpu"  # type: ignore[union-attr]
            return torch.empty(0, device=device), torch.empty(0, device=device)
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    # ------------------------------------------------------------------
    # Utilization
    # ------------------------------------------------------------------

    def utilization(self, gpu_id: int) -> float:
        """Fraction of pages in use on this GPU."""
        pages = self._pages.get(gpu_id, [])
        if not pages:
            return 0.0
        used = sum(1 for p in pages if not p.is_free)
        return used / len(pages)

    def stats(self) -> dict[int, dict]:
        out = {}
        for gpu_id, pages in self._pages.items():
            total = len(pages)
            used = sum(1 for p in pages if not p.is_free)
            out[gpu_id] = {
                "total_pages": total,
                "used_pages": used,
                "free_pages": total - used,
                "utilization": used / total if total else 0.0,
                "active_requests": len(
                    {p.request_id for p in pages if not p.is_free and p.request_id}
                ),
            }
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_free_pages(self, gpu_id: int) -> list[CachePage]:
        return [p for p in self._pages.get(gpu_id, []) if p.is_free]

    def _evict(self, gpu_id: int, needed: int, requesting_priority: Priority) -> int:
        """Evict pages to free space. Prefer LOW priority, then LRU."""
        freed = 0

        if self._config.evict_low_priority_first:
            candidates = self._eviction_candidates_by_priority(gpu_id, requesting_priority)
        else:
            candidates = self._eviction_candidates_lru(gpu_id)

        for page in candidates:
            if freed >= needed:
                break
            req_cache = self._request_caches.get(page.request_id)
            if req_cache:
                req_cache.pages.remove(page)
                if not req_cache.pages:
                    self._request_caches.pop(page.request_id, None)
            page.mark_free()
            freed += 1

        if freed > 0:
            logger.debug("Evicted %d pages on GPU %d", freed, gpu_id)
        return freed

    def _eviction_candidates_by_priority(
        self, gpu_id: int, min_priority: Priority
    ) -> list[CachePage]:
        """Return used pages sorted: lowest priority first, then oldest access."""
        used = [p for p in self._pages.get(gpu_id, []) if not p.is_free]

        def sort_key(page: CachePage) -> tuple[int, int]:
            rc = self._request_caches.get(page.request_id)
            pri = rc.priority if rc else Priority.LOW
            if pri >= min_priority:
                return (999, page.last_access_ns)
            return (pri, page.last_access_ns)

        return sorted(used, key=sort_key)

    def _eviction_candidates_lru(self, gpu_id: int) -> list[CachePage]:
        used = [p for p in self._pages.get(gpu_id, []) if not p.is_free]
        return sorted(used, key=lambda p: p.last_access_ns)

    def _page_bytes(
        self, num_heads: int, head_dim: int, dtype: torch.dtype
    ) -> int:
        elem = torch.tensor([], dtype=dtype).element_size()
        per_layer = self._config.page_size * num_heads * head_dim * elem
        return per_layer * 2  # key + value
