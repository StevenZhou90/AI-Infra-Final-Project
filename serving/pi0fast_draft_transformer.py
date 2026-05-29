"""Small causal transformer drafter for PI0-FAST FAST-token speculation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from serving.pi0fast_eagle import CompactTokenMap


@dataclass
class PI0FastDraftTransformerConfig:
    hidden_dim: int
    vocab_size: int
    context_len: int = 32
    model_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1


class PI0FastDraftTransformer(nn.Module):
    """Token-history drafter conditioned on the current target hidden state."""

    def __init__(self, config: PI0FastDraftTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.token_embed = nn.Embedding(config.vocab_size + 1, config.model_dim)
        self.pos_embed = nn.Embedding(config.context_len + 1, config.model_dim)
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
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    @property
    def pad_class_id(self) -> int:
        return self.config.vocab_size

    def forward(self, hidden: torch.Tensor, context_class_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for the next compact FAST token.

        ``hidden`` is ``[B, hidden_dim]``. ``context_class_ids`` is
        ``[B, context_len]`` and should be left-padded with ``pad_class_id``.
        """

        bsz, ctx_len = context_class_ids.shape
        if ctx_len != self.config.context_len:
            raise ValueError(f"Expected context_len={self.config.context_len}, got {ctx_len}")
        cond = self.hidden_proj(hidden).unsqueeze(1)
        token_x = self.token_embed(context_class_ids.clamp_min(0))
        positions = torch.arange(ctx_len + 1, device=context_class_ids.device).unsqueeze(0)
        x = torch.cat([cond, token_x], dim=1) + self.pos_embed(positions)
        mask = torch.triu(
            torch.ones(ctx_len + 1, ctx_len + 1, device=context_class_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        out = self.transformer(x, mask=mask)
        return self.lm_head(self.norm(out[:, -1, :]))

    @torch.no_grad()
    def draft(
        self,
        hidden: torch.Tensor,
        context_class_ids: torch.Tensor,
        *,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively draft compact token classes and probabilities."""

        self.eval()
        drafted: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        context = context_class_ids
        for _ in range(steps):
            logits = self(hidden, context)
            probs = torch.softmax(logits.float(), dim=-1)
            confidence, cls = torch.max(probs, dim=-1)
            drafted.append(cls)
            confidences.append(confidence)
            context = torch.cat([context[:, 1:], cls[:, None]], dim=1)
        return torch.stack(drafted, dim=1), torch.stack(confidences, dim=1)


@dataclass
class PI0FastParallelDraftTransformerConfig(PI0FastDraftTransformerConfig):
    parallel_lookahead: int = 6


class PI0FastParallelDraftTransformer(PI0FastDraftTransformer):
    """One-forward multi-token drafter conditioned on the current target state."""

    def __init__(self, config: PI0FastParallelDraftTransformerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.lm_head = nn.Linear(config.model_dim, config.parallel_lookahead * config.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor, context_class_ids: torch.Tensor) -> torch.Tensor:
        logits = super().forward(hidden, context_class_ids)
        return logits.view(hidden.shape[0], self.config.parallel_lookahead, self.config.vocab_size)

    @torch.no_grad()
    def draft(
        self,
        hidden: torch.Tensor,
        context_class_ids: torch.Tensor,
        *,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logits = self(hidden, context_class_ids)[:, :steps, :]
        probs = torch.softmax(logits.float(), dim=-1)
        confidences, classes = torch.max(probs, dim=-1)
        return classes, confidences


def load_draft_transformer_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PI0FastDraftTransformer | PI0FastParallelDraftTransformer, CompactTokenMap, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("model_kind") == "parallel_draft_transformer":
        config = PI0FastParallelDraftTransformerConfig(**ckpt["model_config"])
        model = PI0FastParallelDraftTransformer(config)
    else:
        config = PI0FastDraftTransformerConfig(**ckpt["model_config"])
        model = PI0FastDraftTransformer(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    token_map = CompactTokenMap.from_dict(ckpt["token_map"])
    return model, token_map, ckpt.get("summary", {})
