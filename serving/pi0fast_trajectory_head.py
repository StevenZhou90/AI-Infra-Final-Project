"""Trajectory-tail draft heads for PI0-FAST chunk execution.

The head is intentionally small and policy-agnostic: it consumes a verified
continuous PI0-FAST action chunk and predicts a short future tail. In rollout,
PI0-FAST remains the verifier for the near-horizon chunk while the head supplies
the speculative actions used to reduce refresh frequency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PI0FastTrajectoryTailConfig:
    input_horizon: int = 10
    tail_horizon: int = 8
    action_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.05
    residual: bool = True
    residual_from_damped: bool = False
    residual_scale: float = 0.04
    damping: float = 0.65
    freeze_gripper: bool = True


class PI0FastTrajectoryTailHead(nn.Module):
    """Predict future continuous actions from a verified PI0-FAST chunk."""

    def __init__(self, config: PI0FastTrajectoryTailConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.input_horizon * config.action_dim
        out_dim = config.tail_horizon * config.action_dim
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        dim = in_dim
        for _ in range(max(config.num_layers - 1, 1)):
            layers.extend(
                [
                    nn.Linear(dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 3:
            raise ValueError(f"Expected chunk [batch, horizon, action_dim], got {tuple(chunk.shape)}")
        cfg = self.config
        if chunk.shape[1] < cfg.input_horizon or chunk.shape[2] < cfg.action_dim:
            raise ValueError(
                "Chunk is too small for trajectory head: "
                f"got {tuple(chunk.shape)}, need horizon>={cfg.input_horizon}, action_dim>={cfg.action_dim}"
            )
        x = chunk[:, : cfg.input_horizon, : cfg.action_dim].reshape(chunk.shape[0], -1)
        raw = self.net(x).reshape(chunk.shape[0], cfg.tail_horizon, cfg.action_dim)
        if cfg.residual_from_damped:
            pred = self._damped_tail(chunk[:, : cfg.input_horizon, : cfg.action_dim])
            pred = pred + cfg.residual_scale * torch.tanh(raw)
            if cfg.freeze_gripper and cfg.action_dim >= 7:
                pred[:, :, 6] = chunk[:, cfg.input_horizon - 1 : cfg.input_horizon, 6]
            return pred.clamp(-1.0, 1.0)
        pred = raw
        if cfg.residual:
            pred = pred + chunk[:, cfg.input_horizon - 1 : cfg.input_horizon, : cfg.action_dim]
        if cfg.freeze_gripper and cfg.action_dim >= 7:
            pred[:, :, 6] = chunk[:, cfg.input_horizon - 1 : cfg.input_horizon, 6]
        return pred.clamp(-1.0, 1.0)

    def _damped_tail(self, chunk: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        last = chunk[:, cfg.input_horizon - 1].clone()
        delta = chunk[:, cfg.input_horizon - 1] - chunk[:, cfg.input_horizon - 2]
        if cfg.action_dim >= 7:
            delta[:, 6] = 0.0
        outs = []
        for _ in range(cfg.tail_horizon):
            delta = delta * cfg.damping
            last = (last + delta).clamp(-1.0, 1.0)
            if cfg.action_dim >= 7:
                last[:, 6] = chunk[:, cfg.input_horizon - 1, 6]
            outs.append(last.clone())
        return torch.stack(outs, dim=1)

    @torch.inference_mode()
    def extend_chunk(self, chunk: torch.Tensor, total_horizon: int | None = None) -> torch.Tensor:
        cfg = self.config
        total = total_horizon or (cfg.input_horizon + cfg.tail_horizon)
        base = chunk[:, : cfg.input_horizon, : cfg.action_dim]
        if total <= base.shape[1]:
            return base[:, :total]
        tail = self.forward(chunk)[:, : max(total - cfg.input_horizon, 0)]
        return torch.cat([base, tail], dim=1)


def save_trajectory_tail_checkpoint(
    path: str | Path,
    model: PI0FastTrajectoryTailHead,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        out,
    )


def load_trajectory_tail_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PI0FastTrajectoryTailHead, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = PI0FastTrajectoryTailConfig(**checkpoint["config"])
    model = PI0FastTrajectoryTailHead(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, dict(checkpoint.get("extra", {}))
