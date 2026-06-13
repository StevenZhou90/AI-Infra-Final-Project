from __future__ import annotations

import torch

from serving.pi0fast_trajectory_head import PI0FastTrajectoryTailConfig, PI0FastTrajectoryTailHead


def test_trajectory_tail_head_extends_chunk_shape() -> None:
    config = PI0FastTrajectoryTailConfig(input_horizon=10, tail_horizon=8, action_dim=7, hidden_dim=32, num_layers=2)
    model = PI0FastTrajectoryTailHead(config)
    chunk = torch.zeros(2, 10, 7)

    extended = model.extend_chunk(chunk, total_horizon=18)

    assert tuple(extended.shape) == (2, 18, 7)
    torch.testing.assert_close(extended[:, :10], chunk)


def test_trajectory_tail_head_clamps_actions() -> None:
    config = PI0FastTrajectoryTailConfig(input_horizon=2, tail_horizon=2, action_dim=3, hidden_dim=16, num_layers=2)
    model = PI0FastTrajectoryTailHead(config)
    chunk = torch.ones(1, 2, 3) * 10.0

    tail = model(chunk)

    assert float(tail.max()) <= 1.0
    assert float(tail.min()) >= -1.0
