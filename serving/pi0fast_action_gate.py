"""Learned action-space gate for PI0-FAST speculative chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from serving.pi0fast_prefix_gate import action_feature_values


ACTION_GATE_FEATURES = [
    "token_count_norm",
    "action_abs_max",
    "max_step_delta",
    "max_jerk",
    "gripper_change",
    "position_span",
    "rotation_span",
    "target_forwards_norm",
    "verify_forwards_norm",
    "fallback_forwards_norm",
    "drafted_tokens_norm",
    "accepted_tokens_norm",
    "accepted_future_tokens_norm",
    "unknown_context_tokens_norm",
    "verify_margin_rejects_norm",
    "short_accept_rejects_norm",
    "acceptance_rate",
    "future_acceptance_rate",
    "tokens_per_target_forward",
    "min_verify_margin_norm",
    "max_future_accept_norm",
    "min_future_accept_norm",
    "allow_unknown_context",
    "full_block_only",
    "forced_action_end",
    "forced_cutoff_norm",
    "emitted_tokens_norm",
]


@dataclass
class PI0FastActionGateConfig:
    input_dim: int = len(ACTION_GATE_FEATURES)
    hidden_dim: int = 96
    dropout: float = 0.05


class PI0FastActionGate(nn.Module):
    """Small MLP that predicts whether a speculative chunk is action-safe."""

    def __init__(self, config: PI0FastActionGateConfig | None = None) -> None:
        super().__init__()
        self.config = config or PI0FastActionGateConfig()
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


def action_gate_feature_values(
    actions: np.ndarray,
    *,
    token_count: int | None,
    stats: dict[str, Any] | None,
    max_decoding_steps: int = 256,
) -> dict[str, float]:
    stats = stats or {}
    denom = max(float(max_decoding_steps), 1.0)

    def norm_count(name: str) -> float:
        value = stats.get(name, 0.0)
        if value is None:
            return 0.0
        return float(value) / denom

    max_future_accept = stats.get("max_future_accept", 0.0)
    if max_future_accept is None:
        max_future_accept = max_decoding_steps

    values = {
        "token_count_norm": float(token_count or 0) / denom,
        **action_feature_values(actions),
        "target_forwards_norm": norm_count("target_forwards"),
        "verify_forwards_norm": norm_count("verify_forwards"),
        "fallback_forwards_norm": norm_count("fallback_forwards"),
        "drafted_tokens_norm": norm_count("drafted_tokens"),
        "accepted_tokens_norm": norm_count("accepted_tokens"),
        "accepted_future_tokens_norm": norm_count("accepted_future_tokens"),
        "unknown_context_tokens_norm": norm_count("unknown_context_tokens"),
        "verify_margin_rejects_norm": norm_count("verify_margin_rejects"),
        "short_accept_rejects_norm": norm_count("short_accept_rejects"),
        "acceptance_rate": float(stats.get("acceptance_rate", 0.0) or 0.0),
        "future_acceptance_rate": float(stats.get("future_acceptance_rate", 0.0) or 0.0),
        "tokens_per_target_forward": float(stats.get("tokens_per_target_forward", 0.0) or 0.0),
        "min_verify_margin_norm": float(stats.get("min_verify_margin", 0.0) or 0.0) / 10.0,
        "max_future_accept_norm": float(max_future_accept) / denom,
        "min_future_accept_norm": float(stats.get("min_future_accept", 0.0) or 0.0) / 16.0,
        "allow_unknown_context": float(bool(stats.get("allow_unknown_context", False))),
        "full_block_only": float(bool(stats.get("full_block_only", False))),
        "forced_action_end": float(bool(stats.get("forced_action_end", False))),
        "forced_cutoff_norm": float(stats.get("forced_cutoff_tokens", 0.0) or 0.0) / denom,
        "emitted_tokens_norm": float(stats.get("emitted_tokens", token_count or 0) or 0.0) / denom,
    }
    return values


def vectorize_action_gate_row(row: dict[str, Any]) -> list[float]:
    return [float(row.get(name, 0.0)) for name in ACTION_GATE_FEATURES]


def save_action_gate(
    path: str | Path,
    model: PI0FastActionGate,
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
            "features": ACTION_GATE_FEATURES,
            "threshold": float(threshold),
            "summary": summary,
        },
        path,
    )


def load_action_gate(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PI0FastActionGate, float, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = PI0FastActionGateConfig(**ckpt["model_config"])
    model = PI0FastActionGate(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, float(ckpt.get("threshold", 0.95)), ckpt.get("summary", {})
