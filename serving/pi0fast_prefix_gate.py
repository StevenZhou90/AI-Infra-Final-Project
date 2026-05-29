"""Learned safety gate for PI0-FAST action-prefix cutoff candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


PREFIX_GATE_FEATURES = [
    "cutoff_norm",
    "token_count_norm",
    "forced_eos",
    "logprob_mean",
    "logprob_min",
    "entropy_mean",
    "entropy_max",
    "action_abs_max",
    "max_step_delta",
    "max_jerk",
    "gripper_change",
    "position_span",
    "rotation_span",
]


@dataclass
class PI0FastPrefixGateConfig:
    input_dim: int = len(PREFIX_GATE_FEATURES)
    hidden_dim: int = 64
    dropout: float = 0.05


class PI0FastPrefixGate(nn.Module):
    """Small MLP that predicts whether a cutoff candidate is safe to execute."""

    def __init__(self, config: PI0FastPrefixGateConfig | None = None) -> None:
        super().__init__()
        self.config = config or PI0FastPrefixGateConfig()
        self.net = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def probability(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(features))


def action_feature_values(actions: np.ndarray) -> dict[str, float]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "action_abs_max": 0.0,
            "max_step_delta": 0.0,
            "max_jerk": 0.0,
            "gripper_change": 0.0,
            "position_span": 0.0,
            "rotation_span": 0.0,
        }

    diffs = np.diff(arr, axis=0) if arr.shape[0] > 1 else np.zeros((0, arr.shape[1]), dtype=np.float32)
    second = np.diff(arr, n=2, axis=0) if arr.shape[0] > 2 else np.zeros((0, arr.shape[1]), dtype=np.float32)
    pos = arr[:, : min(3, arr.shape[1])]
    rot = arr[:, 3: min(6, arr.shape[1])] if arr.shape[1] > 3 else np.zeros((arr.shape[0], 0), dtype=np.float32)
    gripper_change = 0.0
    if arr.shape[1] > 6 and arr.shape[0] > 1:
        gripper_change = float(np.max(np.abs(np.diff(arr[:, 6]))))
    return {
        "action_abs_max": float(np.max(np.abs(arr))),
        "max_step_delta": float(np.max(np.linalg.norm(diffs, axis=1))) if diffs.size else 0.0,
        "max_jerk": float(np.max(np.linalg.norm(second, axis=1))) if second.size else 0.0,
        "gripper_change": gripper_change,
        "position_span": float(np.max(np.ptp(pos, axis=0))) if pos.size else 0.0,
        "rotation_span": float(np.max(np.ptp(rot, axis=0))) if rot.size else 0.0,
    }


def vectorize_prefix_gate_row(row: dict[str, Any]) -> list[float]:
    return [float(row.get(name, 0.0)) for name in PREFIX_GATE_FEATURES]


def save_prefix_gate(
    path: str | Path,
    model: PI0FastPrefixGate,
    *,
    threshold: float,
    summary: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "features": PREFIX_GATE_FEATURES,
            "threshold": float(threshold),
            "summary": summary,
        },
        path,
    )


def load_prefix_gate(path: str | Path, device: str | torch.device = "cpu") -> tuple[PI0FastPrefixGate, float, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = PI0FastPrefixGateConfig(**ckpt["model_config"])
    model = PI0FastPrefixGate(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, float(ckpt.get("threshold", 0.95)), ckpt.get("summary", {})
