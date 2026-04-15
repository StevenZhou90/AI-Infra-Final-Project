from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from common.types import GpuInfo, GpuMemoryState, ModelPlacement, Precision
from server.config import GpuConfig

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1 << 20
_BF16_MIN_COMPUTE_CAP = (8, 0)  # Ampere+


@dataclass
class GpuSlot:
    info: GpuInfo
    memory: GpuMemoryState
    models: dict[str, ModelPlacement] = field(default_factory=dict)


class GpuManager:
    """Discovers GPUs at startup, tracks memory, and decides model placement."""

    def __init__(self, config: GpuConfig) -> None:
        self._config = config
        self._slots: dict[int, GpuSlot] = {}
        self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        if not torch.cuda.is_available():
            logger.warning("No CUDA GPUs detected – running in CPU-only mode")
            return

        count = torch.cuda.device_count()
        target_ids = self._config.target_gpu_ids or list(range(count))

        for gid in target_ids:
            if gid >= count:
                logger.warning("GPU %d requested but only %d GPUs available", gid, count)
                continue
            props = torch.cuda.get_device_properties(gid)
            cc = (props.major, props.minor)
            info = GpuInfo(
                gpu_id=gid,
                name=props.name,
                total_memory=props.total_mem,
                compute_capability=cc,
                supports_bf16=cc >= _BF16_MIN_COMPUTE_CAP,
            )
            mem = GpuMemoryState(
                gpu_id=gid,
                total=props.total_mem,
                used=0,
                overhead=self._config.memory_overhead_mb * _BYTES_PER_MB,
            )
            self._slots[gid] = GpuSlot(info=info, memory=mem)
            logger.info(
                "Discovered GPU %d: %s  %.1f GB  cc=%d.%d  bf16=%s",
                gid,
                info.name,
                info.total_memory / (1 << 30),
                cc[0],
                cc[1],
                info.supports_bf16,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def gpu_count(self) -> int:
        return len(self._slots)

    @property
    def gpu_ids(self) -> list[int]:
        return sorted(self._slots.keys())

    def get_info(self, gpu_id: int) -> GpuInfo:
        return self._slots[gpu_id].info

    def get_memory(self, gpu_id: int) -> GpuMemoryState:
        return self._slots[gpu_id].memory

    def all_gpu_info(self) -> list[GpuInfo]:
        return [s.info for s in self._slots.values()]

    def preferred_precision(self, gpu_id: int) -> Precision:
        return self._slots[gpu_id].info.preferred_precision

    def refresh_memory(self, gpu_id: int) -> GpuMemoryState:
        """Re-read actual GPU memory usage from CUDA runtime."""
        if gpu_id not in self._slots:
            raise ValueError(f"Unknown GPU {gpu_id}")
        allocated = torch.cuda.memory_allocated(gpu_id)
        reserved = torch.cuda.memory_reserved(gpu_id)
        slot = self._slots[gpu_id]
        slot.memory.used = max(allocated, reserved)
        return slot.memory

    def available_for_kv_cache(self, gpu_id: int) -> int:
        """Bytes available for KV cache on this GPU (applying 90% target)."""
        slot = self._slots[gpu_id]
        free_after_models = (
            slot.memory.total - slot.memory.model_weights - slot.memory.overhead
        )
        return max(0, int(free_after_models * self._config.kv_cache_target))

    # ------------------------------------------------------------------
    # Model placement
    # ------------------------------------------------------------------

    def find_best_gpu(self, weight_bytes: int, drafter_bytes: int = 0) -> int | None:
        """Bin-pack: pick the GPU with the most free memory that can fit the model pair."""
        required = weight_bytes + drafter_bytes
        best_id: int | None = None
        best_free = -1
        for gid, slot in self._slots.items():
            free = slot.memory.total - slot.memory.used - slot.memory.overhead
            if free >= required and free > best_free:
                best_free = free
                best_id = gid
        return best_id

    def register_model(self, placement: ModelPlacement) -> None:
        gid = placement.gpu_id
        slot = self._slots[gid]
        slot.models[placement.model_id] = placement
        slot.memory.model_weights += placement.memory_bytes
        slot.memory.used += placement.memory_bytes
        logger.info(
            "Registered model %s on GPU %d  (%.1f MB)",
            placement.model_id,
            gid,
            placement.memory_bytes / _BYTES_PER_MB,
        )

    def unregister_model(self, model_id: str, gpu_id: int) -> None:
        slot = self._slots[gpu_id]
        placement = slot.models.pop(model_id, None)
        if placement:
            slot.memory.model_weights -= placement.memory_bytes
            slot.memory.used -= placement.memory_bytes
            logger.info("Unregistered model %s from GPU %d", model_id, gpu_id)

    def get_model_gpu(self, model_id: str) -> int | None:
        for gid, slot in self._slots.items():
            if model_id in slot.models:
                return gid
        return None

    def models_on_gpu(self, gpu_id: int) -> list[ModelPlacement]:
        return list(self._slots[gpu_id].models.values())

    def summary(self) -> list[dict]:
        out = []
        for gid, slot in self._slots.items():
            self.refresh_memory(gid)
            out.append(
                {
                    "gpu_id": gid,
                    "name": slot.info.name,
                    "total_gb": round(slot.info.total_memory / (1 << 30), 2),
                    "used_gb": round(slot.memory.used / (1 << 30), 2),
                    "models": list(slot.models.keys()),
                    "kv_budget_gb": round(
                        self.available_for_kv_cache(gid) / (1 << 30), 2
                    ),
                    "precision": slot.info.preferred_precision.value,
                }
            )
        return out
