from __future__ import annotations

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation

from scripts.benchmark_pi0fast_system_components import (
    fast_libero_preprocess_observation,
    pi05_recommendation,
    pi0fast_recommendation,
)


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif torch.is_tensor(left):
        torch.testing.assert_close(left, right)
    else:
        assert left == right


def test_fast_libero_preprocess_matches_lerobot_observation_conversion() -> None:
    image = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    wrist = np.flip(image, axis=2).copy()
    observation = {
        "pixels": {"image": image, "image2": wrist},
        "robot_state": {
            "eef": {
                "pos": np.ones((2, 3), dtype=np.float32),
                "quat": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
            },
            "gripper": {"qpos": np.zeros((2, 2), dtype=np.float32)},
        },
        "agent_pos": np.arange(16, dtype=np.float32).reshape(2, 8),
    }

    reference = preprocess_observation(observation)
    fast = fast_libero_preprocess_observation(observation, device=torch.device("cpu"))

    _assert_nested_equal(reference, fast)


def test_pi0fast_recommendation_requires_single_request_target() -> None:
    rows = [
        {
            "decode_path": "action_end",
            "errors": [],
            "actions_per_request": 10,
            "single_inference": {
                "mean_ms": 500.0,
                "per_action_mean_ms": 50.0,
            },
            "batch_inference": {
                "8": {
                    "mean_ms": 720.0,
                    "per_request_mean_ms": 90.0,
                    "per_action_mean_ms": 9.0,
                }
            },
        }
    ]

    assert pi0fast_recommendation(rows, target_latency_ms=100.0) is None


def test_pi0fast_recommendation_reports_true_single_request_pass() -> None:
    rows = [
        {
            "decode_path": "action_end",
            "errors": [],
            "actions_per_request": 10,
            "single_inference": {
                "mean_ms": 95.0,
                "per_action_mean_ms": 9.5,
            },
            "batch_inference": {},
        }
    ]

    recommendation = pi0fast_recommendation(rows, target_latency_ms=100.0)

    assert recommendation is not None
    assert recommendation["path"] == "single_request"
    assert recommendation["per_request_mean_ms"] == 95.0
    assert recommendation["meets_target"] is True


def test_pi0fast_recommendation_uses_full_request_when_present() -> None:
    rows = [
        {
            "decode_path": "action_end",
            "errors": [],
            "actions_per_request": 10,
            "single_inference": {
                "mean_ms": 95.0,
                "per_action_mean_ms": 9.5,
            },
            "full_request": {
                "mean_ms": 105.0,
            },
            "batch_inference": {},
        }
    ]

    assert pi0fast_recommendation(rows, target_latency_ms=100.0) is None


def test_pi05_recommendation_reports_full_request_pass() -> None:
    rows = [
        {
            "decode_path": "public",
            "errors": [],
            "num_inference_steps": 10,
            "single_inference": {"mean_ms": 84.0},
            "full_request": {"mean_ms": 86.0},
        }
    ]

    recommendation = pi05_recommendation(rows, target_latency_ms=100.0)

    assert recommendation is not None
    assert recommendation["single_mean_ms"] == 84.0
    assert recommendation["full_request_mean_ms"] == 86.0
