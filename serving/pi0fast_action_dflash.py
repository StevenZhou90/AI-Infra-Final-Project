"""Action-space DFlash-style chunk drafter for PI0-FAST experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PI0FastActionDFlashConfig:
    hidden_dim: int = 2048
    action_horizon: int = 10
    action_dim: int = 7
    model_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.08
    action_residual_scale: float = 1.0


class PI0FastActionDFlashHead(nn.Module):
    """Denoise/refine a continuous action block from a PI0-FAST prefix hidden state."""

    def __init__(self, config: PI0FastActionDFlashConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, config.model_dim),
        )
        self.noise_embed = nn.Sequential(
            nn.Linear(1, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.pos_embed = nn.Embedding(1 + config.action_horizon, config.model_dim)
        self.type_embed = nn.Embedding(2, config.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.model_dim)
        self.action_head = nn.Linear(config.model_dim, config.action_dim)
        self.confidence_head = nn.Linear(config.model_dim, 1)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        action_block: torch.Tensor,
        noise_level: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if hidden.ndim != 2:
            raise ValueError(f"Expected hidden [B, D], got {tuple(hidden.shape)}")
        if action_block.shape != (hidden.shape[0], cfg.action_horizon, cfg.action_dim):
            raise ValueError(
                "Expected action block "
                f"{(hidden.shape[0], cfg.action_horizon, cfg.action_dim)}, got {tuple(action_block.shape)}"
            )
        if not torch.is_tensor(noise_level):
            noise = torch.full((hidden.shape[0], 1), float(noise_level), dtype=hidden.dtype, device=hidden.device)
        else:
            noise = noise_level.to(device=hidden.device, dtype=hidden.dtype)
            if noise.ndim == 0:
                noise = noise.expand(hidden.shape[0]).unsqueeze(-1)
            elif noise.ndim == 1:
                noise = noise.unsqueeze(-1)
        cond = self.hidden_proj(hidden).unsqueeze(1)
        actions = self.action_proj(action_block)
        actions = actions + self.noise_embed(noise).unsqueeze(1)
        x = torch.cat([cond, actions], dim=1)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        type_ids = torch.cat(
            [
                torch.zeros((hidden.shape[0], 1), dtype=torch.long, device=x.device),
                torch.ones((hidden.shape[0], cfg.action_horizon), dtype=torch.long, device=x.device),
            ],
            dim=1,
        )
        x = x + self.pos_embed(positions) + self.type_embed(type_ids)
        out = self.transformer(x)
        block_out = self.norm(out[:, 1:])
        residual = self.action_head(block_out)
        clean = action_block + float(cfg.action_residual_scale) * residual
        confidence = torch.sigmoid(self.confidence_head(block_out)).squeeze(-1)
        return clean, confidence

    @torch.no_grad()
    def draft(
        self,
        hidden: torch.Tensor,
        *,
        refine_steps: int = 2,
        init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.eval()
        cfg = self.config
        if init is None:
            block = torch.zeros((hidden.shape[0], cfg.action_horizon, cfg.action_dim), device=hidden.device)
        else:
            block = init.to(device=hidden.device, dtype=hidden.dtype)
        confidences: list[torch.Tensor] = []
        deltas: list[float] = []
        steps = int(refine_steps)
        if steps <= 0:
            conf = torch.ones((hidden.shape[0], cfg.action_horizon), dtype=block.dtype, device=block.device)
            return block, conf, {
                "refine_steps": 0,
                "mean_confidence": 1.0,
                "min_confidence": 1.0,
                "mean_refine_delta": 0.0,
                "last_refine_delta": 0.0,
            }
        for step in range(steps):
            noise = 1.0 - (float(step) / max(steps - 1, 1))
            clean, confidence = self(hidden, block, noise)
            deltas.append(float(torch.mean(torch.abs(clean - block)).detach().cpu().item()))
            block = clean
            confidences.append(confidence)
        conf = confidences[-1] if confidences else torch.ones((hidden.shape[0], cfg.action_horizon), device=hidden.device)
        stats = {
            "refine_steps": steps,
            "mean_confidence": float(conf.mean().detach().cpu().item()),
            "min_confidence": float(conf.min().detach().cpu().item()),
            "mean_refine_delta": float(sum(deltas) / max(len(deltas), 1)),
            "last_refine_delta": float(deltas[-1]) if deltas else 0.0,
        }
        return block, conf, stats


def save_action_dflash_checkpoint(
    path: str | Path,
    model: PI0FastActionDFlashHead,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "action_dflash_head",
            "config": asdict(model.config),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "extra": extra or {},
        },
        out,
    )


def load_action_dflash_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PI0FastActionDFlashHead, dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    config = PI0FastActionDFlashConfig(**ckpt["config"])
    model = PI0FastActionDFlashHead(config)
    model.load_state_dict(ckpt["state_dict"])
    return model, dict(ckpt.get("extra", {}))
