"""Speculative decoding — draft-verify loop for faster autoregressive generation.

Two modes:
1. **Self-speculative (layer-skip)**: reuse the target model with only the
   first N decoder layers as the draft.  No extra parameters, no extra memory.
2. **External draft model**: plug in any smaller HF model for drafting.

For greedy decoding (``do_sample=False``, standard in VLA inference) the output
is *identical* to naive autoregressive generation — speculative decoding only
changes the speed, never the result.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from serving.kv_cache_manager import (
    PastKV,
    extract_kv_layers,
    kv_num_layers,
    kv_seq_len,
    trim_kv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class SpecStats:
    total_steps: int = 0
    total_draft_tokens: int = 0
    total_accepted: int = 0
    total_generated: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / max(self.total_draft_tokens, 1)

    @property
    def tokens_per_step(self) -> float:
        return self.total_generated / max(self.total_steps, 1)

    def summary(self) -> dict:
        return {
            "steps": self.total_steps,
            "generated": self.total_generated,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "tokens_per_step": round(self.tokens_per_step, 2),
            "draft_ms": round(self.draft_time_ms, 1),
            "verify_ms": round(self.verify_time_ms, 1),
        }


# ---------------------------------------------------------------------------
# Layer-skip wrapper for self-speculative drafting
# ---------------------------------------------------------------------------

_LAYER_PATHS = [
    # (language-model attr, layers attr inside it)
    ("language_model", "model.layers"),   # Prismatic / OpenVLA
    (None, "model.layers"),               # LlamaForCausalLM
    (None, "transformer.h"),              # GPT-2 / GPT-NeoX style
]


def _resolve_attr(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _set_attr(obj, dotted: str, value):
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class _LayerSkipDraft:
    """Context-manager that temporarily swaps the decoder layer list to use
    only the first *n* layers, turning the target model into its own draft."""

    def __init__(self, model: nn.Module, num_draft_layers: int | None = None):
        self._model = model
        self._lm_root, self._layers_attr = self._find_layers(model)
        self._all_layers: nn.ModuleList = _resolve_attr(self._lm_root, self._layers_attr)
        total = len(self._all_layers)
        self.num_layers = num_draft_layers or max(total // 2, 1)
        self._draft_layers = nn.ModuleList(list(self._all_layers)[: self.num_layers])
        logger.info(
            "Self-speculative draft: %d / %d layers (%.0f%% of target)",
            self.num_layers, total, 100 * self.num_layers / total,
        )

    @staticmethod
    def _find_layers(model: nn.Module):
        for lm_attr, layers_attr in _LAYER_PATHS:
            root = getattr(model, lm_attr, model) if lm_attr else model
            layers = _resolve_attr(root, layers_attr)
            if layers is not None and isinstance(layers, nn.ModuleList):
                return root, layers_attr
        raise AttributeError(
            f"Cannot locate decoder layers in {type(model).__name__}. "
            "Add an entry to _LAYER_PATHS for this architecture."
        )

    @contextmanager
    def active(self):
        """While inside this context the model runs with fewer layers."""
        _set_attr(self._lm_root, self._layers_attr, self._draft_layers)
        try:
            yield
        finally:
            _set_attr(self._lm_root, self._layers_attr, self._all_layers)


# ---------------------------------------------------------------------------
# Speculative decoder
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """Draft-verify speculative decoding engine.

    Parameters
    ----------
    model : nn.Module
        The *target* (full) model used for prefill.
    decode_model : nn.Module | None
        The model component used for decode-phase forward passes (draft &
        verify).  For VLMs like OpenVLA whose top-level ``forward()`` only
        supports single-token decode, pass ``model.language_model`` here so
        that multi-token verify batches go through the underlying LLM directly.
        When *None*, ``model`` itself is used for all phases.
    draft_model : nn.Module | None
        An external draft model.  When *None*, self-speculative layer-skip
        is used instead.
    lookahead : int
        Maximum number of tokens to draft per step.  For OpenVLA (7 action
        tokens) a lookahead of 7 tries to draft all actions at once.
    draft_layers : int | None
        Number of layers to keep for self-speculative draft.  Ignored when
        *draft_model* is provided.
    device : str | torch.device
        Device for generated tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        decode_model: nn.Module | None = None,
        draft_model: nn.Module | None = None,
        lookahead: int = 7,
        draft_layers: int | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self._target = model
        self._decode = decode_model or model
        self._device = torch.device(device) if isinstance(device, str) else device
        self._lookahead = lookahead

        if draft_model is not None:
            self._external_draft = draft_model
            self._layer_skip: _LayerSkipDraft | None = None
            self._self_spec = False
        else:
            self._external_draft = None
            self._layer_skip = _LayerSkipDraft(model, num_draft_layers=draft_layers)
            self._self_spec = True

        self.stats = SpecStats()
        logger.info(
            "SpeculativeDecoder: lookahead=%d  mode=%s  decode_model=%s",
            lookahead,
            "self-speculative" if self._self_spec else "external-draft",
            type(self._decode).__name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 7,
        prefill_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, PastKV]:
        """Generate *max_new_tokens* using speculative decoding.

        Parameters
        ----------
        input_ids : Tensor  [batch, seq_len]
        max_new_tokens : int
        prefill_kwargs : dict
            Extra kwargs for the first (prefill) forward pass only
            (``pixel_values``, ``attention_mask``, …).

        Returns
        -------
        (all_ids [batch, seq_len + max_new_tokens], target_past_kv)
        """
        prefill_kwargs = prefill_kwargs or {}
        batch = input_ids.shape[0]
        assert batch == 1, "Speculative decoding currently supports batch_size=1"

        # ---- 1. Prefill with full target model ----------------------------
        prefill_out = self._target(
            input_ids, use_cache=True, **prefill_kwargs,
        )
        target_kv = prefill_out.past_key_values
        # logits[:, -1, :] → distribution over the *first* new token
        last_logits = prefill_out.logits[:, -1:, :]

        generated: list[torch.Tensor] = []

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            K = min(self._lookahead, remaining)

            # Token from the target's last logits (always correct)
            first_token = last_logits.argmax(dim=-1)  # [1, 1]

            if K == 1:
                # No room to speculate — just emit this token
                generated.append(first_token)
                out = self._decode(first_token, past_key_values=target_kv, use_cache=True)
                target_kv = out.past_key_values
                last_logits = out.logits[:, -1:, :]
                self.stats.total_generated += 1
                continue

            # ---- 2. Draft K-1 more tokens ---------------------------------
            t0 = time.perf_counter()
            draft_tokens = self._draft_tokens(first_token, target_kv, K - 1)
            self.stats.draft_time_ms += (time.perf_counter() - t0) * 1000
            self.stats.total_draft_tokens += len(draft_tokens)

            # ---- 3. Verify all K tokens with full target model ------------
            # Concatenate: [first_token, draft_0, draft_1, ..., draft_{K-2}]
            verify_input = torch.cat(
                [first_token] + [t.view(1, 1) for t in draft_tokens], dim=1,
            )  # [1, K]

            t0 = time.perf_counter()
            verify_out = self._decode(
                verify_input, past_key_values=target_kv, use_cache=True,
            )
            self.stats.verify_time_ms += (time.perf_counter() - t0) * 1000
            verify_logits = verify_out.logits  # [1, K, vocab]

            # ---- 4. Greedy accept / reject --------------------------------
            # verify_logits[:, i, :] predicts the token *after* position i
            # So verify_logits[:, 0, :] should predict draft_tokens[0], etc.
            accepted = [first_token]
            n_draft_ok = 0

            for i, dtok in enumerate(draft_tokens):
                target_pred = verify_logits[:, i, :].argmax(dim=-1)  # [1]
                if target_pred.item() == dtok.item():
                    accepted.append(target_pred.view(1, 1))
                    n_draft_ok += 1
                else:
                    # Draft was wrong — use the target's correction
                    accepted.append(target_pred.view(1, 1))
                    break
            else:
                # All draft tokens matched — collect a bonus token
                bonus = verify_logits[:, K - 1, :].argmax(dim=-1).view(1, 1)
                accepted.append(bonus)

            self.stats.total_accepted += n_draft_ok
            self.stats.total_steps += 1

            n_accepted = len(accepted)
            generated.extend(accepted)

            # ---- 5. Update target KV cache --------------------------------
            # The verify pass cached KV for all K tokens; trim to accepted count
            prefix_len = kv_seq_len(target_kv)
            keep_len = prefix_len + n_accepted
            target_kv = trim_kv(verify_out.past_key_values, keep_len)

            # Logits for the next iteration
            last_logits = verify_logits[:, n_accepted - 1: n_accepted, :]

        self.stats.total_generated += max_new_tokens

        # Assemble output ids
        gen_ids = torch.cat(generated[:max_new_tokens], dim=1)
        all_ids = torch.cat([input_ids, gen_ids], dim=1)
        return all_ids, target_kv

    # ------------------------------------------------------------------
    # Draft helpers
    # ------------------------------------------------------------------

    def _draft_tokens(
        self,
        first_token: torch.Tensor,
        target_kv: PastKV,
        count: int,
    ) -> list[torch.Tensor]:
        """Produce *count* draft tokens starting after *first_token*.

        Uses self-speculative (layer-skip) when available, otherwise the
        external draft model.
        """
        if self._self_spec:
            return self._draft_self_speculative(first_token, target_kv, count)
        return self._draft_external(first_token, count)

    def _draft_self_speculative(
        self,
        first_token: torch.Tensor,
        target_kv: PastKV,
        count: int,
    ) -> list[torch.Tensor]:
        assert self._layer_skip is not None
        n_draft = self._layer_skip.num_layers
        # Seed the draft's KV with the first n layers of the target's cache
        draft_kv = extract_kv_layers(target_kv, n_draft)

        tokens: list[torch.Tensor] = []
        current = first_token  # [1, 1]

        with self._layer_skip.active():
            # First draft step: process first_token to update draft KV
            out = self._decode(current, past_key_values=draft_kv, use_cache=True)
            draft_kv = out.past_key_values

            for _ in range(count):
                next_tok = out.logits[:, -1, :].argmax(dim=-1)  # [1]
                tokens.append(next_tok)
                out = self._decode(
                    next_tok.unsqueeze(0), past_key_values=draft_kv, use_cache=True,
                )
                draft_kv = out.past_key_values

        return tokens

    def _draft_external(
        self,
        first_token: torch.Tensor,
        count: int,
    ) -> list[torch.Tensor]:
        tokens: list[torch.Tensor] = []
        current = first_token
        draft_kv = None

        for _ in range(count):
            out = self._external_draft(current, past_key_values=draft_kv, use_cache=True)
            draft_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1)
            tokens.append(next_tok)
            current = next_tok.unsqueeze(0)

        return tokens
