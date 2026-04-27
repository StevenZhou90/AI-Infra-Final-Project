"""Adapter for SimplerEnv robot manipulation tasks.

SimplerEnv tasks are a better zero-shot fit for OpenVLA than gym-aloha because
they expose the same 7-DOF end-effector action convention used by OpenVLA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SimplerEnvConfig:
    task: str = "google_robot_pick_coke_can"
    max_episode_steps: int | None = None


class SimplerEnv:
    """Thin wrapper around `simpler_env.make` with LeRobot-style observations."""

    def __init__(self, config: SimplerEnvConfig, device: str = "cpu") -> None:
        self._config = config
        self._device = device
        self._steps = 0
        self._last_image: np.ndarray | None = None

        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        self._image_from_obs = get_image_from_maniskill2_obs_dict
        self._env = simpler_env.make(config.task)
        logger.info("Created SimplerEnv task: %s", config.task)

    @property
    def action_dim(self) -> int:
        return int(self._env.action_space.shape[0])

    @property
    def fps(self) -> int:
        # SimplerEnv defaults: Google Robot tasks at 3 Hz, Bridge/WidowX at 5 Hz.
        return 3 if self._config.task.startswith("google_robot") else 5

    def get_language_instruction(self) -> str:
        return str(self._env.get_language_instruction())

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        self._steps = 0
        try:
            obs, _info = self._env.reset(seed=seed)
        except TypeError:
            obs, _info = self._env.reset()
        return self._convert_obs(obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._steps += 1
        if self._config.max_episode_steps is not None and self._steps >= self._config.max_episode_steps:
            truncated = True
        return self._convert_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        try:
            frame = self._env.render()
        except Exception:
            frame = None
        return frame if frame is not None else self._last_image

    def close(self) -> None:
        self._env.close()

    def _convert_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = self._image_from_obs(self._env, obs)
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        self._last_image = image

        image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        converted = {"observation.images.primary": image_t.to(self._device)}

        # OpenVLA ignores proprioception here; keep state length aligned with
        # the 7-DOF action space so the gRPC client can infer action_dim.
        state = np.zeros(self.action_dim, dtype=np.float32)
        converted["observation.state"] = torch.as_tensor(state, dtype=torch.float32, device=self._device)

        return converted
