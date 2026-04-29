"""Small learned draft head for OpenVLA action-token trajectories."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class TrajectoryHeadConfig:
    history_size: int = 4
    action_dim: int = 7
    n_bins: int = 256
    embed_dim: int = 64
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    # When True, fuse a projected OpenVLA prefill hidden (last prompt position, layer -2).
    use_prefill_hidden: bool = False
    llm_hidden_size: int = 4096
    hidden_fusion_dim: int = 256


class TinyTrajectoryHead(nn.Module):
    """Predict next OpenVLA action-bin distributions from recent action bins.

    Optionally conditions on the vision-language prefill hidden state at the last
    prompt position (second-to-last transformer layer), matching the EAGLE data
    regime for temporal / task context.
    """

    def __init__(self, config: TrajectoryHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or TrajectoryHeadConfig()
        c = self.config
        self.embed = nn.Embedding(c.n_bins, c.embed_dim)
        hist_flat = c.history_size * c.action_dim * c.embed_dim

        self.hidden_proj: nn.Linear | None
        if c.use_prefill_hidden:
            self.hidden_proj = nn.Linear(c.llm_hidden_size, c.hidden_fusion_dim)
            in_dim = hist_flat + c.hidden_fusion_dim
        else:
            self.hidden_proj = None
            in_dim = hist_flat

        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(c.num_layers):
            layers.extend(
                [
                    nn.Linear(dim, c.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(c.dropout),
                ]
            )
            dim = c.hidden_dim
        layers.append(nn.Linear(dim, c.action_dim * c.n_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, history_bins: torch.Tensor, prefill_hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits with shape [batch, action_dim, n_bins].

        If config.use_prefill_hidden, prefill_hidden must be [batch, llm_hidden_size]
        (typically bfloat16/float32 upcast for matmul).
        """
        c = self.config
        if history_bins.dim() == 2:
            history_bins = history_bins.unsqueeze(0)
        history_bins = history_bins.clamp(0, c.n_bins - 1).long()
        emb = self.embed(history_bins).flatten(start_dim=1)

        if c.use_prefill_hidden:
            assert self.hidden_proj is not None
            if prefill_hidden is None:
                fused = torch.zeros(
                    emb.shape[0],
                    c.hidden_fusion_dim,
                    device=emb.device,
                    dtype=emb.dtype,
                )
            else:
                ph = prefill_hidden.to(device=emb.device, dtype=self.hidden_proj.weight.dtype)
                if ph.dim() == 1:
                    ph = ph.unsqueeze(0)
                fused = self.hidden_proj(ph.float()).to(dtype=emb.dtype)
            emb = torch.cat([emb, fused], dim=-1)
        logits = self.net(emb)
        return logits.view(-1, c.action_dim, c.n_bins)

    @torch.no_grad()
    def predict(
        self,
        history_bins: torch.Tensor,
        top_k: int = 3,
        prefill_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (top_bins, top_probs, max_probs) per action dimension."""
        logits = self(history_bins, prefill_hidden=prefill_hidden)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_bins = probs.topk(k=top_k, dim=-1)
        max_probs = top_probs[..., 0]
        return top_bins, top_probs, max_probs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.__dict__,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "TinyTrajectoryHead":
        ckpt = torch.load(path, map_location="cpu")
        raw = ckpt["config"]
        cfg = TrajectoryHeadConfig()
        valid = {f.name for f in fields(TrajectoryHeadConfig)}
        for k, v in raw.items():
            if k in valid:
                setattr(cfg, k, v)
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model.to(device).eval()


def token_ids_to_bins(token_ids, vocab_size: int) -> torch.Tensor:
    return (vocab_size - torch.as_tensor(token_ids).long() - 1).clamp(0, 255)


def bins_to_token_ids(bin_ids, vocab_size: int) -> torch.Tensor:
    return (vocab_size - torch.as_tensor(bin_ids).long() - 1)
