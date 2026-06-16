"""Learned accept-length gate for direct trajectory chunks."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AcceptLengthGateConfig:
    feature_dim: int = 29
    max_accept: int = 3
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class AcceptLengthGate(nn.Module):
    """Predict how many direct-head actions to accept: 0..max_accept."""

    def __init__(self, config: AcceptLengthGateConfig | None = None) -> None:
        super().__init__()
        self.config = config or AcceptLengthGateConfig()
        c = self.config
        layers: list[nn.Module] = []
        dim = c.feature_dim
        for _ in range(c.num_layers):
            layers.extend([nn.Linear(dim, c.hidden_dim), nn.GELU(), nn.Dropout(c.dropout)])
            dim = c.hidden_dim
        layers.append(nn.Linear(dim, c.max_accept + 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.net(features.float())

    @torch.no_grad()
    def predict_length(self, features: torch.Tensor) -> int:
        logits = self(features)
        return int(logits.argmax(dim=-1).item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config.__dict__, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "AcceptLengthGate":
        ckpt = torch.load(path, map_location="cpu")
        cfg = AcceptLengthGateConfig()
        valid = {f.name for f in fields(AcceptLengthGateConfig)}
        for key, value in ckpt["config"].items():
            if key in valid:
                setattr(cfg, key, value)
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model.to(device).eval()


def build_accept_length_features(
    *,
    history_bins: np.ndarray,
    predicted_bins: np.ndarray,
    max_probs: np.ndarray,
    phase: str,
    timestep: float = 0.0,
) -> np.ndarray:
    """Build stable scalar features for accept-length prediction.

    Shapes:
    - history_bins: [history, 7]
    - predicted_bins: [horizon, 7]
    - max_probs: [horizon, 7]
    """
    hist = np.asarray(history_bins, dtype=np.float32)
    pred = np.asarray(predicted_bins, dtype=np.float32)
    probs = np.asarray(max_probs, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[None, :]
    if probs.ndim == 1:
        probs = probs[None, :]

    last = hist[-1] if hist.size else pred[0]
    motion = hist[:, :6] if hist.size else np.zeros((1, 6), dtype=np.float32)
    if motion.shape[0] >= 2:
        velocity = motion[-1] - motion[-2]
        displacement = float(np.mean(np.abs(velocity)))
    else:
        velocity = np.zeros(6, dtype=np.float32)
        displacement = 0.0
    if motion.shape[0] >= 3:
        acceleration = motion[-1] - 2.0 * motion[-2] + motion[-3]
        curvature = float(np.mean(np.abs(acceleration)))
    else:
        acceleration = np.zeros(6, dtype=np.float32)
        curvature = 0.0

    deltas = np.abs(pred[:, :6] - last[:6])
    block_deltas = np.abs(np.diff(pred[:, :6], axis=0)) if pred.shape[0] > 1 else np.zeros((1, 6), dtype=np.float32)
    gripper_flips = pred[:, 6] != last[6]

    feature_values = [
        1.0 if phase == "smooth" else 0.0,
        float(timestep) / 300.0,
        displacement / 32.0,
        curvature / 32.0,
        float(np.mean(np.abs(velocity))) / 32.0,
        float(np.mean(np.abs(acceleration))) / 32.0,
        float(np.mean(probs)),
        float(np.min(probs)),
        float(np.mean(probs[:, :6])),
        float(np.min(probs[:, :6])),
        float(np.mean(probs[:, 6])),
        float(np.min(probs[:, 6])),
        float(np.mean(deltas)) / 64.0,
        float(np.max(deltas)) / 64.0,
        float(np.mean(block_deltas)) / 64.0,
        float(np.max(block_deltas)) / 64.0,
        float(np.sum(gripper_flips)),
    ]
    for idx in range(pred.shape[0]):
        step_probs = probs[idx]
        step_delta = np.abs(pred[idx, :6] - last[:6])
        feature_values.extend(
            [
                float(np.mean(step_probs)),
                float(np.min(step_probs)),
                float(np.mean(step_delta)) / 64.0,
                float(np.max(step_delta)) / 64.0,
            ]
        )
    min_feature_dim = 17 + 4 * min(3, pred.shape[0])
    while len(feature_values) < min_feature_dim:
        feature_values.append(0.0)
    return np.asarray(feature_values, dtype=np.float32)
