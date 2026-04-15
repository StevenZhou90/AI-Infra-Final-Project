import pytest
import torch

from common.types import Priority
from server.config import CacheConfig, GpuConfig
from server.kv_cache import KVCacheManager


class MockGpuManager:
    """Minimal GPU manager stub for testing without real GPUs."""

    def __init__(self, num_gpus: int = 1, memory_per_gpu: int = 1 * (1 << 30)):
        self._num_gpus = num_gpus
        self._memory = memory_per_gpu

    @property
    def gpu_ids(self):
        return list(range(self._num_gpus))

    @property
    def gpu_count(self):
        return self._num_gpus

    def get_info(self, gpu_id):
        from common.types import GpuInfo
        return GpuInfo(
            gpu_id=gpu_id,
            name="MockGPU",
            total_memory=self._memory,
            compute_capability=(8, 0),
            supports_bf16=True,
        )

    def available_for_kv_cache(self, gpu_id):
        return self._memory // 2


@pytest.fixture
def cache_manager():
    config = CacheConfig(page_size=4, sliding_window_size=16, evict_low_priority_first=True)
    gpu = MockGpuManager(num_gpus=1, memory_per_gpu=1 * (1 << 30))
    mgr = KVCacheManager(config, gpu)
    mgr.initialize(num_layers=2, num_heads=4, head_dim=8, dtype=torch.float32)
    return mgr


class TestKVCache:
    def test_initialization(self, cache_manager: KVCacheManager):
        stats = cache_manager.stats()
        assert 0 in stats
        assert stats[0]["total_pages"] > 0
        assert stats[0]["free_pages"] == stats[0]["total_pages"]

    def test_allocate_and_release(self, cache_manager: KVCacheManager):
        pages = cache_manager.allocate_pages("req-1", gpu_id=0, num_pages=2, priority=Priority.NORMAL)
        assert len(pages) == 2
        assert cache_manager.utilization(0) > 0

        freed = cache_manager.release_request("req-1")
        assert freed == 2
        assert cache_manager.utilization(0) == 0.0

    def test_sliding_window(self, cache_manager: KVCacheManager):
        pages = cache_manager.allocate_pages("req-sw", gpu_id=0, num_pages=8, priority=Priority.NORMAL)
        for i, page in enumerate(pages):
            page.seq_start = i * 4
            page.seq_end = (i + 1) * 4

        evicted = cache_manager.apply_sliding_window("req-sw")
        assert evicted == 4  # 16 / 4 = 4 max pages, 8 - 4 = 4 evicted

    def test_eviction_by_priority(self, cache_manager: KVCacheManager):
        stats = cache_manager.stats()
        total = stats[0]["total_pages"]
        cache_manager.allocate_pages("low-req", gpu_id=0, num_pages=total, priority=Priority.LOW)

        pages = cache_manager.allocate_pages("high-req", gpu_id=0, num_pages=2, priority=Priority.HIGH)
        assert len(pages) == 2

    def test_write_read_kv(self, cache_manager: KVCacheManager):
        pages = cache_manager.allocate_pages("rw-req", gpu_id=0, num_pages=1, priority=Priority.NORMAL)
        page = pages[0]

        keys = torch.randn(2, 4, 8)
        values = torch.randn(2, 4, 8)
        cache_manager.write_kv(page, layer_idx=0, seq_offset=0, keys=keys, values=values)

        k_out, v_out = cache_manager.read_kv([page], layer_idx=0)
        assert k_out.shape[0] == 2
        assert v_out.shape[0] == 2

    def test_utilization(self, cache_manager: KVCacheManager):
        stats = cache_manager.stats()
        total = stats[0]["total_pages"]
        half = total // 2

        cache_manager.allocate_pages("util-req", gpu_id=0, num_pages=half, priority=Priority.NORMAL)
        util = cache_manager.utilization(0)
        assert 0.4 <= util <= 0.6
