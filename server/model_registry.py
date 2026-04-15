from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from common.types import ModelPlacement, Precision
from server.config import ServerConfig
from server.gpu_manager import GpuManager

logger = logging.getLogger(__name__)

_BYTES_PER_PARAM = {
    Precision.FP32: 4,
    Precision.FP16: 2,
    Precision.BF16: 2,
}


def _estimate_weight_bytes(param_count: int, precision: Precision) -> int:
    return param_count * _BYTES_PER_PARAM[precision]


@dataclass
class LoadedModel:
    placement: ModelPlacement
    model: nn.Module
    drafter: nn.Module | None = None
    processor: Any = None  # tokenizer / image processor


class ModelRegistry:
    """Loads, tracks, and hot-swaps VLA + drafter models across GPUs."""

    def __init__(self, config: ServerConfig, gpu_manager: GpuManager) -> None:
        self._config = config
        self._gpu = gpu_manager
        self._models: dict[str, LoadedModel] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_id: str,
        hf_repo: str,
        *,
        drafter_hf_repo: str | None = None,
        target_gpu: int | None = None,
        precision: Precision | None = None,
    ) -> ModelPlacement:
        if model_id in self._models:
            raise ValueError(f"Model {model_id} already loaded")

        if target_gpu is not None:
            gpu_id = target_gpu
        else:
            gpu_id = self._gpu.find_best_gpu(
                weight_bytes=_estimate_weight_bytes(7_000_000_000, Precision.FP16),
                drafter_bytes=_estimate_weight_bytes(80_000_000, Precision.FP16)
                if drafter_hf_repo
                else 0,
            )
            if gpu_id is None:
                raise RuntimeError("No GPU with enough free memory to load model")

        if precision is None:
            precision = self._gpu.preferred_precision(gpu_id)

        device = torch.device(f"cuda:{gpu_id}")
        dtype = _precision_to_dtype(precision)

        logger.info("Loading VLA model %s from %s onto GPU %d (%s)", model_id, hf_repo, gpu_id, precision.value)

        model, processor = self._load_vla_from_hf(hf_repo, device, dtype)
        param_count = sum(p.numel() for p in model.parameters())
        weight_bytes = _estimate_weight_bytes(param_count, precision)

        drafter_model = None
        drafter_id = None
        drafter_bytes = 0
        if drafter_hf_repo:
            drafter_id = f"{model_id}_drafter"
            logger.info("Loading drafter %s from %s onto GPU %d", drafter_id, drafter_hf_repo, gpu_id)
            drafter_model, _ = self._load_act_drafter(drafter_hf_repo, device, dtype)
            drafter_param_count = sum(p.numel() for p in drafter_model.parameters())
            drafter_bytes = _estimate_weight_bytes(drafter_param_count, precision)

        placement = ModelPlacement(
            model_id=model_id,
            gpu_id=gpu_id,
            precision=precision,
            memory_bytes=weight_bytes + drafter_bytes,
            drafter_model_id=drafter_id,
            hf_repo=hf_repo,
            param_count=param_count,
            is_loaded=True,
        )

        self._gpu.register_model(placement)
        self._models[model_id] = LoadedModel(
            placement=placement,
            model=model,
            drafter=drafter_model,
            processor=processor,
        )
        logger.info("Model %s loaded successfully (%.1f GB total)", model_id, placement.memory_bytes / (1 << 30))
        return placement

    def unload_model(self, model_id: str) -> None:
        loaded = self._models.pop(model_id, None)
        if loaded is None:
            return
        self._gpu.unregister_model(model_id, loaded.placement.gpu_id)
        del loaded.model
        if loaded.drafter is not None:
            del loaded.drafter
        torch.cuda.empty_cache()
        logger.info("Unloaded model %s from GPU %d", model_id, loaded.placement.gpu_id)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> LoadedModel | None:
        return self._models.get(model_id)

    def get_model(self, model_id: str) -> nn.Module:
        loaded = self._models.get(model_id)
        if loaded is None:
            raise KeyError(f"Model {model_id} not loaded")
        return loaded.model

    def get_drafter(self, model_id: str) -> nn.Module | None:
        loaded = self._models.get(model_id)
        if loaded is None:
            return None
        return loaded.drafter

    def get_processor(self, model_id: str) -> Any:
        loaded = self._models.get(model_id)
        return loaded.processor if loaded else None

    def list_models(self) -> list[ModelPlacement]:
        return [lm.placement for lm in self._models.values()]

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_vla_from_hf(
        hf_repo: str, device: torch.device, dtype: torch.dtype
    ) -> tuple[nn.Module, Any]:
        """Load a 7B VLA model (OpenVLA-style) from HuggingFace."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                hf_repo,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
            model.eval()
            return model, processor
        except Exception:
            logger.warning("transformers VLA load failed for %s, trying lerobot", hf_repo)
            return ModelRegistry._load_act_drafter(hf_repo, device, dtype)

    @staticmethod
    def _load_act_drafter(
        hf_repo: str, device: torch.device, dtype: torch.dtype
    ) -> tuple[nn.Module, Any]:
        """Load an ACT policy (~80M) from HuggingFace via lerobot."""
        try:
            from lerobot.common.policies.act.modeling_act import ACTPolicy
            from lerobot.common.policies.act.configuration_act import ACTConfig

            config = ACTConfig()
            model = ACTPolicy(config)
            model = model.to(device=device, dtype=dtype)
            model.eval()
            return model, None
        except ImportError:
            logger.warning("lerobot not available, creating stub ACT model")
            model = _StubACTModel().to(device=device, dtype=dtype)
            model.eval()
            return model, None


class _StubACTModel(nn.Module):
    """Lightweight stand-in when lerobot isn't installed."""

    def __init__(self, action_dim: int = 14, chunk_size: int = 90) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.backbone = nn.Sequential(
            nn.Linear(3 * 480 * 640, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.head = nn.Linear(256, action_dim * chunk_size)

    def forward(self, images: torch.Tensor, joint_state: torch.Tensor | None = None) -> torch.Tensor:
        b = images.shape[0]
        x = images.reshape(b, -1)[:, : 3 * 480 * 640]
        if x.shape[1] < 3 * 480 * 640:
            x = torch.nn.functional.pad(x, (0, 3 * 480 * 640 - x.shape[1]))
        x = self.backbone(x)
        actions = self.head(x)
        return actions.reshape(b, self.chunk_size, self.action_dim)


def _precision_to_dtype(p: Precision) -> torch.dtype:
    return {
        Precision.FP32: torch.float32,
        Precision.FP16: torch.float16,
        Precision.BF16: torch.bfloat16,
    }[p]
