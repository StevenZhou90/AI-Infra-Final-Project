"""EAGLE draft head for speculative decoding.

Loads a pre-trained EAGLE-1 autoregression head (a lightweight single-layer
transformer that takes second-to-last-layer hidden states from the target
model and predicts next tokens).  Works as a drop-in draft model alongside
our existing SpeculativeDecoder.

Architecture (EAGLE-1):
    1. embed_tokens(token_id) → embedding [hidden_size]
    2. fc(concat(embedding, target_hidden_state)) → [hidden_size]
    3. One Llama decoder layer (attention + MLP) with its own KV cache
    4. target_model.norm → target_model.lm_head → logits

Reference: Li et al., "EAGLE: Speculative Sampling Requires Rethinking
Feature Uncertainty", ICML 2024.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal Llama building blocks (matching EAGLE-1 weight format)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin):
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class EagleAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 max_pos: int = 4096, rope_base: float = 10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None
        self._cache_len = 0
        self._build_rope_cache(max_pos)

    def _build_rope_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cache = emb.cos()[None, None, :, :]
        self._sin_cache = emb.sin()[None, None, :, :]
        self._cache_len = seq_len

    def _rope(self, q, k, position_ids):
        seq_len = position_ids.max().item() + 1
        if seq_len > self._cache_len:
            self._build_rope_cache(seq_len)
        cos = self._cos_cache.to(q.device)[:, :, position_ids.squeeze(0), :].to(q.dtype)
        sin = self._sin_cache.to(q.device)[:, :, position_ids.squeeze(0), :].to(q.dtype)
        return _apply_rotary(q, k, cos, sin)

    def forward(self, hidden_states, position_ids, past_kv=None, use_cache=True):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self._rope(q, k, position_ids)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        kv_len = k.shape[2]
        if q_len > 1:
            causal = torch.triu(torch.full((q_len, kv_len), float("-inf"), device=q.device, dtype=q.dtype),
                                diagonal=kv_len - q_len + 1)
            attn = attn + causal[None, None, :, :]

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(out), new_kv


class EagleMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EagleDecoderLayer(nn.Module):
    """Single Llama-style decoder layer used in the EAGLE head.

    EAGLE-1's layer 0 has no input_layernorm (skip pre-norm for first layer),
    only post_attention_layernorm.
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size,
                 rms_eps=1e-6, layer_idx=0):
        super().__init__()
        self.self_attn = EagleAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp = EagleMLP(hidden_size, intermediate_size)
        self.layer_idx = layer_idx
        if layer_idx != 0:
            self.input_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_eps)

    def forward(self, hidden_states, position_ids, past_kv=None, use_cache=True):
        residual = hidden_states
        if self.layer_idx != 0:
            hidden_states = self.input_layernorm(hidden_states)

        attn_out, new_kv = self.self_attn(hidden_states, position_ids, past_kv, use_cache)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, new_kv


# ---------------------------------------------------------------------------
# EAGLE Draft Head
# ---------------------------------------------------------------------------

class EagleDraftHead(nn.Module):
    """Standalone EAGLE-1 draft head that can be loaded from HuggingFace."""

    def __init__(self, config: dict):
        super().__init__()
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkv = config.get("num_key_value_heads", nh)
        inter = config["intermediate_size"]
        vocab = config["vocab_size"]
        eps = config.get("rms_norm_eps", 1e-6)
        bias = config.get("bias", True)

        self.embed_tokens = nn.Embedding(vocab, hs)
        self.fc = nn.Linear(2 * hs, hs, bias=bias)

        n_layers = config.get("num_hidden_layers", 1)
        self.layers = nn.ModuleList([
            EagleDecoderLayer(hs, nh, nkv, inter, eps, layer_idx=i)
            for i in range(n_layers)
        ])

        self._kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    def reset_kv(self):
        self._kv_cache = [None] * len(self.layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden_size] — from target model's
                second-to-last layer output.
            input_ids: [batch, seq] — token IDs for embedding lookup.
            position_ids: [batch, seq] — position indices for RoPE.

        Returns:
            output hidden states [batch, seq, hidden_size] (apply target's
            norm + lm_head externally to get logits).
        """
        with torch.no_grad():
            embeds = self.embed_tokens(input_ids).to(hidden_states.dtype)

        seq_len = hidden_states.shape[1]

        if position_ids is None:
            past_len = self._kv_cache[0][0].shape[2] if self._kv_cache[0] is not None else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=hidden_states.device
            ).unsqueeze(0)

        h = self.fc(torch.cat((embeds, hidden_states), dim=-1))

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            h, new_kv = layer(h, position_ids, past_kv=self._kv_cache[i], use_cache=use_cache)
            new_kv_cache.append(new_kv)

        if use_cache:
            self._kv_cache = new_kv_cache

        return h

    @classmethod
    def from_pretrained(cls, ea_model_path: str, device: str | torch.device = "cpu",
                        dtype: torch.dtype = torch.bfloat16) -> "EagleDraftHead":
        """Load an EAGLE-1 draft head from a HuggingFace repo or local path."""
        import os

        config_path = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(ea_model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        try:
            weight_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(weight_path):
                weight_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            state_dict = torch.load(weight_path, map_location="cpu")
        except Exception:
            from safetensors.torch import load_file
            weight_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(weight_path):
                weight_path = hf_hub_download(ea_model_path, "model.safetensors")
            state_dict = load_file(weight_path)

        model = cls(config)

        key_map = {}
        for k in state_dict:
            new_k = k
            if k.startswith("layers."):
                parts = k.split(".")
                layer_idx = int(parts[1])
                rest = ".".join(parts[2:])
                new_k = f"layers.{layer_idx}.{rest}"
            key_map[k] = new_k

        mapped = {key_map[k]: v for k, v in state_dict.items()}
        model.load_state_dict(mapped, strict=False)
        model = model.to(dtype).to(device).eval()

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("EAGLE draft head loaded: %.1fM params on %s", total_params, device)
        return model


def extract_hidden_states(model_output, target_model) -> torch.Tensor:
    """Extract second-to-last layer hidden states from a target model forward.

    For OpenVLA (Prismatic), the language model is accessed via
    model.language_model.model, and we need to hook into the decoder layers
    to capture the output of the second-to-last layer.
    """
    if hasattr(model_output, "hidden_states") and model_output.hidden_states is not None:
        return model_output.hidden_states[-2]
    return model_output[0]
