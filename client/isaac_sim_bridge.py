from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import numpy as np

from common.types import ActionStep, JointState

logger = logging.getLogger(__name__)


class SimulatorAPI(Protocol):
    """Interface that Isaac Sim (or any simulator) must implement."""

    def get_camera_images(self) -> dict[str, np.ndarray]:
        """Return {camera_id: (H, W, 3) uint8 RGB image} for all cameras."""
        ...

    def get_joint_state(self) -> JointState:
        """Return current joint positions, velocities, efforts."""
        ...

    def apply_action(self, action: ActionStep) -> None:
        """Apply joint targets + gripper targets to the robot."""
        ...

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        ...

    def reset(self) -> None:
        """Reset the environment."""
        ...


class IsaacSimBridge:
    """Bridge between Isaac Sim and the VLA serving client.

    Captures camera images and joint states from the simulator,
    applies action steps received from the inference server.
    """

    def __init__(
        self,
        sim: SimulatorAPI | None = None,
        camera_ids: list[str] | None = None,
        control_frequency_hz: float = 50.0,
    ) -> None:
        self._sim = sim
        self._camera_ids = camera_ids or ["cam_left", "cam_right", "cam_wrist_left", "cam_wrist_right"]
        self._frequency_hz = control_frequency_hz
        self._dt = 1.0 / control_frequency_hz
        self._step_count = 0
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Camera capture
    # ------------------------------------------------------------------

    def capture_cameras(self) -> dict[str, np.ndarray]:
        """Get images from all cameras. Returns {camera_id: (H, W, 3) array}."""
        if self._sim is not None:
            images = self._sim.get_camera_images()
            filtered = {}
            for cid in self._camera_ids:
                if cid in images:
                    filtered[cid] = images[cid]
                else:
                    logger.warning("Camera %s not found in simulator", cid)
            return filtered

        return {cid: self._dummy_image() for cid in self._camera_ids}

    def get_joint_state(self) -> JointState:
        if self._sim is not None:
            return self._sim.get_joint_state()

        return JointState(
            positions=[0.0] * 14,
            velocities=[0.0] * 14,
            timestamp_ns=time.time_ns(),
        )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def apply_action(self, action: ActionStep) -> None:
        """Apply a single action step to the robot in simulation."""
        if self._sim is not None:
            self._sim.apply_action(action)
            self._sim.step()
        self._step_count += 1

    def execute_chunk(
        self,
        actions: list[ActionStep],
        realtime: bool = True,
    ) -> None:
        """Execute an entire action chunk at the control frequency."""
        for action in actions:
            t0 = time.perf_counter()
            self.apply_action(action)
            if realtime:
                elapsed = time.perf_counter() - t0
                sleep_time = self._dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        """Reset the environment and return initial observations."""
        if self._sim is not None:
            self._sim.reset()
        self._step_count = 0
        self._episode_count += 1
        return self.capture_cameras()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dummy_image() -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def stats(self) -> dict:
        return {
            "cameras": self._camera_ids,
            "frequency_hz": self._frequency_hz,
            "step_count": self._step_count,
            "episode_count": self._episode_count,
        }


class IsaacSimAdapter:
    """Concrete adapter for NVIDIA Isaac Sim via omni.isaac APIs.

    Requires Isaac Sim to be running and the omni.isaac modules to be importable.
    """

    def __init__(
        self,
        robot_prim_path: str = "/World/Robot",
        camera_prims: dict[str, str] | None = None,
    ) -> None:
        self._robot_path = robot_prim_path
        self._camera_prims = camera_prims or {
            "cam_left": "/World/Camera_Left",
            "cam_right": "/World/Camera_Right",
            "cam_wrist_left": "/World/Camera_WristLeft",
            "cam_wrist_right": "/World/Camera_WristRight",
        }
        self._world = None
        self._robot = None
        self._cameras: dict[str, Any] = {}

    def initialize(self) -> None:
        """Initialize Isaac Sim world and robot."""
        try:
            from omni.isaac.core import World
            from omni.isaac.core.robots import Robot
            from omni.isaac.sensor import Camera

            self._world = World()
            self._world.scene.add_default_ground_plane()
            self._robot = self._world.scene.add(
                Robot(prim_path=self._robot_path, name="vla_robot")
            )
            for cam_id, prim_path in self._camera_prims.items():
                self._cameras[cam_id] = Camera(
                    prim_path=prim_path,
                    resolution=(640, 480),
                )
            self._world.reset()
            logger.info("Isaac Sim initialized: robot=%s, cameras=%s", self._robot_path, list(self._cameras.keys()))
        except ImportError:
            logger.error("Isaac Sim (omni.isaac) not available")
            raise

    def get_camera_images(self) -> dict[str, np.ndarray]:
        images = {}
        for cam_id, camera in self._cameras.items():
            rgba = camera.get_rgba()
            if rgba is not None:
                images[cam_id] = rgba[:, :, :3]
            else:
                images[cam_id] = np.zeros((480, 640, 3), dtype=np.uint8)
        return images

    def get_joint_state(self) -> JointState:
        if self._robot is None:
            return JointState(positions=[], timestamp_ns=time.time_ns())

        pos = self._robot.get_joint_positions()
        vel = self._robot.get_joint_velocities()
        return JointState(
            positions=pos.tolist() if pos is not None else [],
            velocities=vel.tolist() if vel is not None else [],
            timestamp_ns=time.time_ns(),
        )

    def apply_action(self, action: ActionStep) -> None:
        if self._robot is None:
            return
        from omni.isaac.core.utils.types import ArticulationAction

        targets = action.joint_targets + action.gripper_targets
        self._robot.apply_action(
            ArticulationAction(joint_positions=np.array(targets, dtype=np.float32))
        )

    def step(self) -> None:
        if self._world is not None:
            self._world.step(render=True)

    def reset(self) -> None:
        if self._world is not None:
            self._world.reset()
