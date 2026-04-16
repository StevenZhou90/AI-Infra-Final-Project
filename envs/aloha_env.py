"""Wraps gym-aloha and converts observations to the format LeRobot policies expect."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AlohaEnvConfig:
    task: str = "AlohaTransferCube-v0"
    obs_type: str = "pixels_agent_pos"
    max_episode_steps: int = 400
    render_width: int = 640
    render_height: int = 480


class AlohaEnv:

    def __init__(self, config: AlohaEnvConfig, device: str = "cpu") -> None:
        self._config = config
        self._device = device

        import gym_aloha  # noqa: F401

        self._env = gym.make(
            f"gym_aloha/{config.task}",
            obs_type=config.obs_type,
            max_episode_steps=config.max_episode_steps,
        )
        logger.info("Created gym-aloha env: %s  obs_type=%s", config.task, config.obs_type)

    @property
    def action_dim(self) -> int:
        return int(self._env.action_space.shape[0])

    @property
    def fps(self) -> int:
        return int(getattr(self._env.unwrapped, "metadata", {}).get("render_fps", 50))

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        obs, _info = self._env.reset(seed=seed)
        return self._convert_obs(obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._convert_obs(obs), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def _convert_obs(self, obs: dict | np.ndarray) -> dict[str, torch.Tensor]:
        """Convert gym obs to dict[str, Tensor].

        gym-aloha returns: {"pixels": {"top": ndarray}, "agent_pos": ndarray}
        LeRobot expects: {"observation.images.top": Tensor[C,H,W], "observation.state": Tensor[D]}
        """
        if isinstance(obs, np.ndarray):
            return {"observation.state": torch.from_numpy(obs).float().to(self._device)}

        converted: dict[str, torch.Tensor] = {}

        if "pixels" in obs:
            for cam_name, img in obs["pixels"].items():
                t = torch.from_numpy(img).float()
                if t.ndim == 3 and t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1)
                t = t / 255.0
                converted[f"observation.images.{cam_name}"] = t.to(self._device)

        if "agent_pos" in obs:
            converted["observation.state"] = (
                torch.from_numpy(obs["agent_pos"]).float().to(self._device)
            )

        return converted
