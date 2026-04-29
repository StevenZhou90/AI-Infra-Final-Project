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
    use_published_eval_setup: bool = False
    env_name: str | None = None
    scene_name: str = "google_pick_coke_can_1_v4"
    robot: str = "google_robot_static"
    control_freq: int = 3
    sim_freq: int = 513
    obj_init_x: float | None = None
    obj_init_y: float | None = None
    robot_init_x: float = 0.35
    robot_init_y: float = 0.20
    robot_init_rot_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    additional_env_build_kwargs: dict[str, Any] | None = None


class SimplerEnv:
    """Thin wrapper around `simpler_env.make` with LeRobot-style observations."""

    def __init__(self, config: SimplerEnvConfig, device: str = "cpu") -> None:
        self._config = config
        self._device = device
        self._steps = 0
        self._last_image: np.ndarray | None = None

        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        self._image_from_obs = get_image_from_maniskill2_obs_dict
        self._env = self._make_env(config)
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
        if self._config.use_published_eval_setup:
            obs, _info = self._env.reset(options=self._reset_options())
            return self._convert_obs(obs)

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

    def _make_env(self, config: SimplerEnvConfig):
        if not config.use_published_eval_setup:
            import simpler_env

            return simpler_env.make(config.task)

        from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode

        env_name = config.env_name or self._published_env_name(config.task)
        control_mode = get_robot_control_mode(config.robot, None)
        kwargs = {
            "obs_mode": "rgbd",
            "robot": config.robot,
            "sim_freq": config.sim_freq,
            "control_mode": control_mode,
            "control_freq": config.control_freq,
            "max_episode_steps": config.max_episode_steps or 80,
            "scene_name": config.scene_name,
            "camera_cfgs": {"add_segmentation": True},
            "rgb_overlay_path": None,
        }
        additional = config.additional_env_build_kwargs or {}
        return build_maniskill2_env(env_name, **additional, **kwargs)

    @staticmethod
    def _published_env_name(task: str) -> str:
        if task.startswith("google_robot_pick"):
            return "GraspSingleOpenedCokeCanInScene-v0"
        raise ValueError(f"No published eval env_name mapping for task: {task}")

    def _reset_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "robot_init_options": {
                "init_xy": np.array([self._config.robot_init_x, self._config.robot_init_y]),
                "init_rot_quat": np.array(self._config.robot_init_rot_quat),
            },
        }
        if self._config.obj_init_x is not None:
            if self._config.obj_init_y is None:
                raise ValueError("obj_init_y must be provided when obj_init_x is set")
            options["obj_init_options"] = {
                "init_xy": np.array([self._config.obj_init_x, self._config.obj_init_y]),
            }
        return options

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
