"""Masked-block FAST-token drafter for PI0-FAST speculation.

The drafter predicts a block of future FAST tokens in one transformer forward.
Unlike the earlier parallel head, draft slots can attend to each other, which
lets the model learn local trajectory structure inside the proposed block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from serving.pi0fast_eagle import CompactTokenMap


@dataclass
class PI0FastBlockDrafterConfig:
    hidden_dim: int
    vocab_size: int
    context_len: int = 64
    block_len: int = 7
    model_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1


class PI0FastBlockDrafter(nn.Module):
    """Bidirectional masked-block drafter conditioned on target hidden state."""

    def __init__(self, config: PI0FastBlockDrafterConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.token_embed = nn.Embedding(config.vocab_size + 2, config.model_dim)
        self.pos_embed = nn.Embedding(1 + config.context_len + config.block_len, config.model_dim)
        self.type_embed = nn.Embedding(3, config.model_dim)
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

    @property
    def mask_class_id(self) -> int:
        return self.config.vocab_size + 1

    def forward(
        self,
        hidden: torch.Tensor,
        context_class_ids: torch.Tensor,
        block_class_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for every draft block slot.

        ``hidden`` is ``[B, hidden_dim]``. ``context_class_ids`` is
        ``[B, context_len]`` left-padded with ``pad_class_id``. Optional
        ``block_class_ids`` is ``[B, block_len]`` and normally contains only
        ``mask_class_id`` at inference.
        """

        bsz, ctx_len = context_class_ids.shape
        if ctx_len != self.config.context_len:
            raise ValueError(f"Expected context_len={self.config.context_len}, got {ctx_len}")
        if block_class_ids is None:
            block_class_ids = torch.full(
                (bsz, self.config.block_len),
                self.mask_class_id,
                dtype=torch.long,
                device=context_class_ids.device,
            )
        if block_class_ids.shape != (bsz, self.config.block_len):
            raise ValueError(f"Expected block shape {(bsz, self.config.block_len)}, got {tuple(block_class_ids.shape)}")

        cond = self.hidden_proj(hidden).unsqueeze(1)
        context = self.token_embed(context_class_ids.clamp(0, self.mask_class_id))
        block = self.token_embed(block_class_ids.clamp(0, self.mask_class_id))
        x = torch.cat([cond, context, block], dim=1)
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        type_ids = torch.cat(
            [
                torch.zeros((bsz, 1), dtype=torch.long, device=x.device),
                torch.ones((bsz, ctx_len), dtype=torch.long, device=x.device),
                torch.full((bsz, self.config.block_len), 2, dtype=torch.long, device=x.device),
            ],
            dim=1,
        )
        x = x + self.pos_embed(positions) + self.type_embed(type_ids)
        key_padding_mask = torch.cat(
            [
                torch.zeros((bsz, 1), dtype=torch.bool, device=x.device),
                context_class_ids.eq(self.pad_class_id),
                torch.zeros((bsz, self.config.block_len), dtype=torch.bool, device=x.device),
            ],
            dim=1,
        )
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        block_out = out[:, -self.config.block_len :, :]
        return self.lm_head(self.norm(block_out))

    @torch.no_grad()
    def draft(
        self,
        hidden: torch.Tensor,
        context_class_ids: torch.Tensor,
        *,
        steps: int,
        refine_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft compact token classes and per-slot probabilities."""

        self.eval()
        if steps <= 0:
            empty = torch.empty((hidden.shape[0], 0), dtype=torch.long, device=hidden.device)
            return empty, empty.to(dtype=torch.float32)
        steps = min(int(steps), self.config.block_len)
        block = torch.full(
            (hidden.shape[0], self.config.block_len),
            self.mask_class_id,
            dtype=torch.long,
            device=hidden.device,
        )
        logits = None
        for refine_idx in range(max(1, int(refine_steps))):
            logits = self(hidden, context_class_ids, block)
            probs = torch.softmax(logits.float(), dim=-1)
            confidence, classes = torch.max(probs, dim=-1)
            if refine_idx < refine_steps - 1:
                block[:, :steps] = classes[:, :steps]
        if logits is None:
            raise RuntimeError("Block drafter produced no logits")
        probs = torch.softmax(logits.float(), dim=-1)
        confidence, classes = torch.max(probs, dim=-1)
        return classes[:, :steps], confidence[:, :steps]


def load_block_drafter_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PI0FastBlockDrafter, CompactTokenMap, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = PI0FastBlockDrafterConfig(**ckpt["model_config"])
    model = PI0FastBlockDrafter(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    token_map = CompactTokenMap.from_dict(ckpt["token_map"])
    return model, token_map, ckpt.get("summary", {})
