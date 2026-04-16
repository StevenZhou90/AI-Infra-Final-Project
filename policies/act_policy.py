"""ACT policy wrapper with manual normalization for LeRobot version compatibility."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


@dataclass
class ACTPolicyConfig:
    pretrained_path: str = "lerobot/act_aloha_sim_transfer_cube_human"
    device: str = "cuda"
    use_amp: bool = False


class _NormStats:

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> None:
        self.mean = mean.to(device)
        self.std = std.to(device).clamp(min=1e-8)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class ACTPolicyWrapper:
    """Loads ACT from HuggingFace, handles normalization, provides predict() interface."""

    def __init__(self, config: ACTPolicyConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._policy = self._load_policy(config.pretrained_path)
        self._norm_stats = self._load_norm_stats(config.pretrained_path)

    def _load_policy(self, pretrained_path: str) -> torch.nn.Module:
        logger.info("Loading ACT policy from: %s", pretrained_path)
        try:
            from lerobot.common.policies.act.modeling_act import ACTPolicy
        except ImportError:
            from lerobot.policies.act.modeling_act import ACTPolicy

        policy = ACTPolicy.from_pretrained(pretrained_path)
        policy.eval()
        policy.to(self._device)
        logger.info("ACT policy loaded on %s  params=%.1fM", self._device, self._param_count(policy))
        return policy

    def _load_norm_stats(self, pretrained_path: str) -> dict[str, _NormStats]:
        """Extract normalization mean/std from the checkpoint's extra buffers."""
        safetensors_path = hf_hub_download(pretrained_path, "model.safetensors")
        state = safetensors.torch.load_file(safetensors_path)

        stats: dict[str, _NormStats] = {}
        prefix_map = {
            "normalize_inputs.buffer_observation_images_top": "observation.images.top",
            "normalize_inputs.buffer_observation_state": "observation.state",
            "unnormalize_outputs.buffer_action": "action",
        }

        for prefix, key in prefix_map.items():
            mean_key = f"{prefix}.mean"
            std_key = f"{prefix}.std"
            if mean_key in state and std_key in state:
                stats[key] = _NormStats(state[mean_key], state[std_key], self._device)
                logger.info("Loaded norm stats for '%s'  mean_shape=%s", key, list(state[mean_key].shape))

        if not stats:
            logger.warning("No normalization stats found — policy may produce bad actions")

        return stats

    def reset(self) -> None:
        """Clear internal state at the start of each episode."""
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def predict(self, observation: dict[str, torch.Tensor]) -> np.ndarray:
        """Return a single action given the current observation."""
        normed_obs = self._normalize_obs(observation)
        batched_obs = {k: v.unsqueeze(0).to(self._device) for k, v in normed_obs.items()}

        with torch.no_grad():
            if self._config.use_amp:
                with torch.autocast(device_type="cuda"):
                    action = self._policy.select_action(batched_obs)
            else:
                action = self._policy.select_action(batched_obs)

        action = action.squeeze(0)
        if "action" in self._norm_stats:
            action = self._norm_stats["action"].unnormalize(action)
        return action.cpu().numpy()

    def _normalize_obs(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normed: dict[str, torch.Tensor] = {}
        for key, val in observation.items():
            if key in self._norm_stats:
                normed[key] = self._norm_stats[key].normalize(val)
            else:
                normed[key] = val
        return normed

    @staticmethod
    def _param_count(model: torch.nn.Module) -> float:
        return sum(p.numel() for p in model.parameters()) / 1e6
