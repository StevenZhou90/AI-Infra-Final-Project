"""Learned gate for PI0-FAST full-block speculative verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PI0FastBlockGateConfig:
    hidden_dim: int
    lookahead: int
    feature_dim: int
    model_dim: int = 256
    dropout: float = 0.05


class PI0FastBlockGate(nn.Module):
    """Predict whether a drafted FAST-token block will fully verify."""

    def __init__(self, config: PI0FastBlockGateConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.net = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 2, 1),
        )

    def forward(self, hidden: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h = self.hidden_proj(hidden)
        f = self.feature_proj(features)
        return self.net(torch.cat([h, f], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def probability(self, hidden: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(hidden, features))


def block_gate_features(
    *,
    position: int,
    max_decoding_steps: int,
    confidences: list[float] | torch.Tensor,
    lookahead: int,
) -> torch.Tensor:
    if isinstance(confidences, torch.Tensor):
        conf = confidences.detach().float().flatten().cpu()
    else:
        conf = torch.tensor([float(v) for v in confidences], dtype=torch.float32)
    if conf.numel() < lookahead:
        conf = torch.cat([conf, torch.zeros(lookahead - conf.numel(), dtype=torch.float32)])
    conf = conf[:lookahead]
    if conf.numel() == 0:
        conf = torch.zeros(lookahead, dtype=torch.float32)
    pos_norm = float(position) / max(float(max_decoding_steps), 1.0)
    stats = torch.tensor(
        [
            pos_norm,
            float(lookahead) / 16.0,
            float(conf.min().item()),
            float(conf.max().item()),
            float(conf.mean().item()),
            float(conf.std(unbiased=False).item()),
        ],
        dtype=torch.float32,
    )
    return torch.cat([stats, conf], dim=0)


def save_block_gate(path: str | Path, model: PI0FastBlockGate, *, threshold: float, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "threshold": float(threshold),
            "summary": summary,
        },
        path,
    )


def load_block_gate(path: str | Path, device: str | torch.device = "cpu") -> tuple[PI0FastBlockGate, float, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = PI0FastBlockGateConfig(**ckpt["model_config"])
    model = PI0FastBlockGate(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, float(ckpt.get("threshold", 0.95)), ckpt.get("summary", {})
