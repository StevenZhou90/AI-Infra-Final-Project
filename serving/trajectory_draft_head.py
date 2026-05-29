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
    action_horizon: int = 1
    n_bins: int = 256
    embed_dim: int = 64
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    # When True, fuse a projected OpenVLA prefill hidden (last prompt position, layer -2).
    use_prefill_hidden: bool = False
    llm_hidden_size: int = 4096
    hidden_fusion_dim: int = 256
    # When True, predict chunk steps sequentially, conditioning each future
    # action on the previous predicted action rather than independent heads.
    dependent_horizon: bool = False


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

        if c.dependent_horizon:
            context_layers: list[nn.Module] = []
            dim = in_dim
            for _ in range(c.num_layers):
                context_layers.extend([nn.Linear(dim, c.hidden_dim), nn.GELU(), nn.Dropout(c.dropout)])
                dim = c.hidden_dim
            self.context_net: nn.Module | None = nn.Sequential(*context_layers)
            self.prev_proj: nn.Linear | None = nn.Linear(c.action_dim * c.embed_dim, c.hidden_dim)
            self.step_cell: nn.GRUCell | None = nn.GRUCell(c.hidden_dim, c.hidden_dim)
            self.step_out: nn.Linear | None = nn.Linear(c.hidden_dim, c.action_dim * c.n_bins)
            self.net: nn.Module | None = None
        else:
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
            layers.append(nn.Linear(dim, c.action_horizon * c.action_dim * c.n_bins))
            self.net = nn.Sequential(*layers)
            self.context_net = None
            self.prev_proj = None
            self.step_cell = None
            self.step_out = None

    def forward(
        self,
        history_bins: torch.Tensor,
        prefill_hidden: torch.Tensor | None = None,
        target_bins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return action-bin logits.

        For legacy one-step heads, returns ``[batch, action_dim, n_bins]``.
        For direct chunk heads with ``action_horizon > 1``, returns
        ``[batch, action_horizon, action_dim, n_bins]``.

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
        if c.dependent_horizon:
            assert self.context_net is not None
            assert self.prev_proj is not None
            assert self.step_cell is not None
            assert self.step_out is not None
            context = self.context_net(emb)
            state = context
            prev_bins = history_bins[:, -1, :]
            if target_bins is not None and target_bins.dim() == 2:
                target_bins = target_bins.unsqueeze(1)
            outputs = []
            for step_idx in range(c.action_horizon):
                prev_emb = self.embed(prev_bins.clamp(0, c.n_bins - 1).long()).flatten(start_dim=1)
                state = self.step_cell(self.prev_proj(prev_emb), state)
                step_logits = self.step_out(state).view(-1, c.action_dim, c.n_bins)
                outputs.append(step_logits)
                if self.training and target_bins is not None and step_idx < target_bins.shape[1]:
                    prev_bins = target_bins[:, step_idx, :]
                else:
                    prev_bins = step_logits.argmax(dim=-1).detach()
            logits = torch.stack(outputs, dim=1)
            if c.action_horizon == 1:
                return logits[:, 0]
            return logits

        assert self.net is not None
        logits = self.net(emb)
        logits = logits.view(-1, c.action_horizon, c.action_dim, c.n_bins)
        if c.action_horizon == 1:
            return logits[:, 0]
        return logits

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
