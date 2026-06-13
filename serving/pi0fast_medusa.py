"""PI0-FAST Medusa-style future FAST-token heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from serving.pi0fast_eagle import CompactTokenMap


@dataclass
class PI0FastMedusaConfig:
    hidden_dim: int
    vocab_size: int
    lookahead: int
    hidden_proj_dim: int = 0
    dropout: float = 0.0


class PI0FastMedusaHead(nn.Module):
    """Parallel future-token classifiers over target PI0-FAST hidden states."""

    def __init__(self, config: PI0FastMedusaConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.hidden_dim
        if config.hidden_proj_dim > 0:
            self.backbone = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_proj_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            in_dim = config.hidden_proj_dim
        else:
            self.backbone = nn.LayerNorm(config.hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(in_dim, config.vocab_size) for _ in range(config.lookahead)])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        x = self.backbone(hidden)
        return [head(x) for head in self.heads]


def load_medusa_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PI0FastMedusaHead, CompactTokenMap, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = PI0FastMedusaConfig(**ckpt["model_config"])
    model = PI0FastMedusaHead(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    token_map = CompactTokenMap.from_dict(ckpt["token_map"])
    return model, token_map, ckpt.get("summary", {})
