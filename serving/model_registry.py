"""Model registry: load, unload, and route inference to multiple models across GPUs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

import torch

from policies.act_policy import ACTPolicyWrapper, ACTPolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model_id: str
    pretrained_path: str
    gpu_id: int
    policy: ACTPolicyWrapper
    memory_mb: float
    loaded_at: float = field(default_factory=time.time)
    total_requests: int = 0


class ModelRegistry:
    """Thread-safe registry for loading/unloading models and routing inference."""

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._lock = Lock()

    def load_model(self, model_id: str, pretrained_path: str, gpu_id: int = -1) -> LoadedModel:
        with self._lock:
            if model_id in self._models:
                raise ValueError(f"Model '{model_id}' is already loaded")

        if gpu_id < 0:
            gpu_id = self._pick_gpu()

        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        mem_before = torch.cuda.memory_allocated(gpu_id) if torch.cuda.is_available() else 0

        config = ACTPolicyConfig(pretrained_path=pretrained_path, device=device)
        policy = ACTPolicyWrapper(config)

        mem_after = torch.cuda.memory_allocated(gpu_id) if torch.cuda.is_available() else 0
        memory_mb = (mem_after - mem_before) / (1 << 20)

        entry = LoadedModel(
            model_id=model_id, pretrained_path=pretrained_path,
            gpu_id=gpu_id, policy=policy, memory_mb=memory_mb,
        )

        with self._lock:
            self._models[model_id] = entry

        logger.info("Loaded model '%s' from %s on GPU %d (%.1f MB)", model_id, pretrained_path, gpu_id, memory_mb)
        return entry

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            entry = self._models.pop(model_id, None)
        if entry is None:
            raise ValueError(f"Model '{model_id}' not found")
        del entry.policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded model '%s' from GPU %d", model_id, entry.gpu_id)

    def get_model(self, model_id: str) -> LoadedModel:
        with self._lock:
            entry = self._models.get(model_id)
        if entry is None:
            raise ValueError(f"Model '{model_id}' not loaded")
        return entry

    def list_models(self) -> list[LoadedModel]:
        with self._lock:
            return list(self._models.values())

    def predict(self, model_id: str, observation: dict[str, torch.Tensor]) -> tuple[list[float], float]:
        """Run inference on a loaded model. Returns (actions_list, inference_time_ms)."""
        entry = self.get_model(model_id)
        t0 = time.perf_counter()
        action = entry.policy.predict(observation)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            entry.total_requests += 1
        return action.tolist(), elapsed_ms

    def _pick_gpu(self) -> int:
        """Pick GPU with most free memory."""
        if not torch.cuda.is_available():
            return 0
        best_id = 0
        best_free = -1
        for i in range(torch.cuda.device_count()):
            free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            if free > best_free:
                best_free = free
                best_id = i
        return best_id

    def gpu_status(self) -> list[dict]:
        """Return per-GPU memory and model info."""
        gpus = []
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            models_on_gpu = [m.model_id for m in self._models.values() if m.gpu_id == i]
            gpus.append({
                "gpu_id": i,
                "name": props.name,
                "total_memory_mb": props.total_memory / (1 << 20),
                "used_memory_mb": torch.cuda.memory_allocated(i) / (1 << 20),
                "loaded_models": models_on_gpu,
            })
        return gpus
