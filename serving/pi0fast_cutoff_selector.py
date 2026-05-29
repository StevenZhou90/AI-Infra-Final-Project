"""Prefix-hidden cutoff selector for PI0-FAST block SD experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PI0FastCutoffSelectorConfig:
    hidden_dim: int = 2048
    cutoffs: tuple[int, ...] = (52, 56, 60, 64)
    model_dim: int = 256
    dropout: float = 0.05


class PI0FastCutoffSelector(nn.Module):
    """Predict the earliest safe cutoff from the target prefix hidden state."""

    def __init__(self, config: PI0FastCutoffSelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, len(config.cutoffs)),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        return self.net(hidden.float())

    @torch.no_grad()
    def choose(self, hidden: torch.Tensor, *, max_risk: float = 0.35) -> tuple[int, torch.Tensor]:
        """Return the earliest cutoff whose predicted unsafe probability is below max_risk."""

        self.eval()
        probs = torch.softmax(self(hidden), dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        risk = 1.0 - cdf
        index = torch.full((probs.shape[0],), len(self.config.cutoffs) - 1, dtype=torch.long, device=probs.device)
        ok = risk <= float(max_risk)
        for idx in range(ok.shape[1]):
            take = ok[:, idx] & (index == len(self.config.cutoffs) - 1)
            index = torch.where(take, torch.full_like(index, idx), index)
        cutoff = int(self.config.cutoffs[int(index[0].item())])
        return cutoff, probs


def save_cutoff_selector(
    path: str | Path,
    model: PI0FastCutoffSelector,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "pi0fast_cutoff_selector",
            "config": asdict(model.config),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "extra": extra or {},
        },
        out,
    )


def load_cutoff_selector(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PI0FastCutoffSelector, dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    config = dict(ckpt["config"])
    if isinstance(config.get("cutoffs"), list):
        config["cutoffs"] = tuple(int(v) for v in config["cutoffs"])
    model = PI0FastCutoffSelector(PI0FastCutoffSelectorConfig(**config))
    model.load_state_dict(ckpt["state_dict"])
    return model, dict(ckpt.get("extra", {}))
