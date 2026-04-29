"""Trajectory-prior speculative decoding for OpenVLA action tokens.

The draft is domain-specific: recent continuous robot actions are extrapolated
per action dimension, converted back into OpenVLA action token IDs, then
verified by the target OpenVLA language model. Verification keeps greedy target
outputs exact; the trajectory prior only changes how many target forwards are
needed to obtain the same action tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from serving.kv_cache_manager import PastKV, kv_seq_len, trim_kv
from serving.trajectory_draft_head import TinyTrajectoryHead, bins_to_token_ids, token_ids_to_bins


@torch.no_grad()
def capture_prefill_hidden_openvla(
    model,
    *,
    input_ids: torch.Tensor,
    device: torch.device,
    **prefill_kwargs: torch.Tensor | None,
) -> tuple[Any, torch.Tensor]:
    """Run one OpenVLA prefill forward and return (prefill_outputs, hidden_vec).

    ``hidden_vec`` is the output of ``language_model.model.layers[-2]`` at the last
    sequence position — same signal used for EAGLE-style draft training.
    """
    decode_model = getattr(model, "language_model", model)
    layers = decode_model.model.layers
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inp, out) -> None:
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    handle = layers[-2].register_forward_hook(hook_fn)
    try:
        prefill = model(input_ids=input_ids, use_cache=True, **prefill_kwargs)
    finally:
        handle.remove()

    hs = captured["h"][:, -1:, :]  # [batch, 1, hidden]
    vec = hs.squeeze(1).to(device=device)
    return prefill, vec


@dataclass
class TrajectorySpecStats:
    calls: int = 0
    fallback_calls: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    generated_tokens: int = 0
    target_forwards: int = 0
    exact_hits: int = 0
    band_hits: int = 0
    band_misses: int = 0
    conservative_fallbacks: int = 0
    token_fallbacks: int = 0
    tree_verifies: int = 0
    tree_candidates: int = 0
    max_tree_depth_used: int = 0
    fast_draft_calls: int = 0
    fast_draft_tokens: int = 0
    fast_draft_rejects: int = 0
    learned_head_calls: int = 0
    learned_head_confident_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.drafted_tokens, 1)

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "fallback_calls": self.fallback_calls,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "generated_tokens": self.generated_tokens,
            "target_forwards": self.target_forwards,
            "exact_hits": self.exact_hits,
            "band_hits": self.band_hits,
            "band_misses": self.band_misses,
            "conservative_fallbacks": self.conservative_fallbacks,
            "token_fallbacks": self.token_fallbacks,
            "tree_verifies": self.tree_verifies,
            "tree_candidates": self.tree_candidates,
            "max_tree_depth_used": self.max_tree_depth_used,
            "fast_draft_calls": self.fast_draft_calls,
            "fast_draft_tokens": self.fast_draft_tokens,
            "fast_draft_rejects": self.fast_draft_rejects,
            "learned_head_calls": self.learned_head_calls,
            "learned_head_confident_tokens": self.learned_head_confident_tokens,
            "draft_ms": round(self.draft_time_ms, 1),
            "verify_ms": round(self.verify_time_ms, 1),
        }


class TrajectorySpeculativeDecoder:
    """Speculative decoder that drafts OpenVLA action tokens from action history."""

    def __init__(
        self,
        model,
        device: str | torch.device = "cuda",
        history_size: int = 4,
        min_history: int = 2,
        band_radius: int = 2,
        max_residual_bins: float = 8.0,
        tree_width: int = 8,
        max_tree_depth: int = 1,
        allow_approx_tree: bool = False,
        fast_draft_only: bool = False,
        fast_min_confident_tokens: int = 7,
        draft_head: TinyTrajectoryHead | None = None,
        head_threshold: float = 0.5,
        head_top_k: int = 3,
    ) -> None:
        self.model = model
        self.decode_model = getattr(model, "language_model", model)
        self.device = torch.device(device)
        self.history_size = history_size
        self.min_history = min_history
        self.band_radius = band_radius
        self.max_residual_bins = max_residual_bins
        self.tree_width = tree_width
        self.max_tree_depth = max_tree_depth
        self.allow_approx_tree = allow_approx_tree
        self.fast_draft_only = fast_draft_only
        self.fast_min_confident_tokens = fast_min_confident_tokens
        self.draft_head = draft_head
        self.head_threshold = head_threshold
        self.head_top_k = head_top_k
        self.history: list[np.ndarray] = []
        self.token_history: list[np.ndarray] = []
        self._last_residual_std = np.zeros(7, dtype=np.float32)
        self.stats = TrajectorySpecStats()

    def reset(self) -> None:
        self.history.clear()
        self.token_history.clear()
        self.stats = TrajectorySpecStats()

    @torch.no_grad()
    def predict_action(
        self,
        inputs,
        unnorm_key: str,
        max_new_tokens: int = 7,
    ) -> np.ndarray:
        """Return exact greedy OpenVLA raw action using trajectory speculation."""
        self.stats.calls += 1

        if len(self.history) < self.min_history:
            self.stats.fallback_calls += 1
            self.stats.target_forwards += max_new_tokens + 1
            out = self.model.generate(
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            token_ids_t = out[0, -max_new_tokens:].detach().cpu()
            action = self.decode_action_ids(token_ids_t, unnorm_key)
            token_ids = token_ids_t.numpy()
            self.update_history(action, token_ids)
            return action

        t0 = time.perf_counter()
        prefill_out: Any | None = None

        hist_ready = (
            self.draft_head is not None
            and len(self.token_history) >= self.draft_head.config.history_size
        )
        if hist_ready and self.draft_head is not None and self.draft_head.config.use_prefill_hidden:
            prefill_kw = {
                key: value
                for key, value in {
                    "pixel_values": inputs.get("pixel_values"),
                    "attention_mask": inputs.get("attention_mask"),
                }.items()
                if value is not None
            }
            prefill_out, prefill_h = capture_prefill_hidden_openvla(
                self.model,
                input_ids=inputs["input_ids"],
                device=self.device,
                **prefill_kw,
            )
            self.stats.target_forwards += 1
            draft_ids, draft_bands, residual_bins, confidence_mask = self._draft_learned_head(
                prefill_hidden=prefill_h
            )
            self.stats.learned_head_calls += 1
            self.stats.learned_head_confident_tokens += int(np.sum(confidence_mask))
        elif hist_ready:
            draft_ids, draft_bands, residual_bins, confidence_mask = self._draft_learned_head()
            self.stats.learned_head_calls += 1
            self.stats.learned_head_confident_tokens += int(np.sum(confidence_mask))
        elif len(self.token_history) >= self.min_history:
            draft_ids, draft_bands, residual_bins = self._draft_token_bands()
            confidence_mask = residual_bins <= self.max_residual_bins
        else:
            draft_action = self._draft_action()
            draft_ids, draft_bands, residual_bins = self.action_to_token_bands(draft_action, unnorm_key)
            confidence_mask = residual_bins <= self.max_residual_bins
        draft_ids = draft_ids.to(self.device)
        self.stats.draft_time_ms += (time.perf_counter() - t0) * 1000
        if self.draft_head is None:
            # Gripper is discrete and task-phase dependent; let the target model
            # generate it until we add a phase-aware gripper predictor.
            confidence_mask[6] = False

        if self.fast_draft_only and self.draft_head is not None:
            if self._fast_draft_allowed(draft_ids, confidence_mask, max_new_tokens):
                self.stats.fast_draft_calls += 1
                self.stats.fast_draft_tokens += max_new_tokens
                self.stats.generated_tokens += max_new_tokens
                token_ids = draft_ids[0, :max_new_tokens].detach().cpu()
                action = self.decode_action_ids(token_ids, unnorm_key)
                self.update_history(action, token_ids.numpy())
                return action
            self.stats.fast_draft_rejects += 1
            generated_ids = self._generate_baseline_from_prefill(
                input_ids=inputs["input_ids"],
                prefill_out=prefill_out,
                prefill_kwargs={
                    key: value
                    for key, value in {
                        "pixel_values": inputs.get("pixel_values"),
                        "attention_mask": inputs.get("attention_mask"),
                    }.items()
                    if value is not None
                },
                max_new_tokens=max_new_tokens,
            )
            verified_token_ids = generated_ids[0, -max_new_tokens:].detach().cpu().numpy()
            action = self.decode_action_ids(generated_ids[0, -max_new_tokens:], unnorm_key)
            self.update_history(action, verified_token_ids)
            return action

        generated_ids = self._generate_verified(
            input_ids=inputs["input_ids"],
            draft_ids=draft_ids,
            draft_bands=draft_bands,
            confidence_mask=confidence_mask,
            prefill_kwargs={
                key: value
                for key, value in {
                    "pixel_values": inputs.get("pixel_values"),
                    "attention_mask": inputs.get("attention_mask"),
                }.items()
                if value is not None
            },
            max_new_tokens=max_new_tokens,
            prefill_out=prefill_out,
        )

        verified_token_ids = generated_ids[0, -max_new_tokens:].detach().cpu().numpy()
        action = self.decode_action_ids(generated_ids[0, -max_new_tokens:], unnorm_key)
        self.update_history(action, verified_token_ids)
        return action

    @torch.no_grad()
    def generate_action_ids(
        self,
        inputs,
        unnorm_key: str,
        max_new_tokens: int = 7,
        update_history: bool = True,
    ) -> torch.Tensor:
        """Generate action token IDs through the speculative path for exactness checks."""
        self.stats.calls += 1

        if len(self.history) < self.min_history:
            self.stats.fallback_calls += 1
            out = self.model.generate(
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            action_ids = out[0, -max_new_tokens:].detach().cpu()
            if update_history:
                action = self.decode_action_ids(action_ids, unnorm_key)
                self.update_history(action, action_ids.numpy())
            return action_ids

        prefill_out: Any | None = None

        hist_ready = (
            self.draft_head is not None
            and len(self.token_history) >= self.draft_head.config.history_size
        )
        if hist_ready and self.draft_head is not None and self.draft_head.config.use_prefill_hidden:
            prefill_kw = {
                key: value
                for key, value in {
                    "pixel_values": inputs.get("pixel_values"),
                    "attention_mask": inputs.get("attention_mask"),
                }.items()
                if value is not None
            }
            prefill_out, prefill_h = capture_prefill_hidden_openvla(
                self.model,
                input_ids=inputs["input_ids"],
                device=self.device,
                **prefill_kw,
            )
            self.stats.target_forwards += 1
            draft_ids, draft_bands, residual_bins, confidence_mask = self._draft_learned_head(
                prefill_hidden=prefill_h
            )
            self.stats.learned_head_calls += 1
            self.stats.learned_head_confident_tokens += int(np.sum(confidence_mask))
        elif hist_ready:
            draft_ids, draft_bands, residual_bins, confidence_mask = self._draft_learned_head()
            self.stats.learned_head_calls += 1
            self.stats.learned_head_confident_tokens += int(np.sum(confidence_mask))
        elif len(self.token_history) >= self.min_history:
            draft_ids, draft_bands, residual_bins = self._draft_token_bands()
            confidence_mask = residual_bins <= self.max_residual_bins
        else:
            draft_action = self._draft_action()
            draft_ids, draft_bands, residual_bins = self.action_to_token_bands(draft_action, unnorm_key)
            confidence_mask = residual_bins <= self.max_residual_bins
        draft_ids = draft_ids.to(self.device)
        if self.draft_head is None:
            confidence_mask[6] = False

        if self.fast_draft_only and self.draft_head is not None:
            if self._fast_draft_allowed(draft_ids, confidence_mask, max_new_tokens):
                self.stats.fast_draft_calls += 1
                self.stats.fast_draft_tokens += max_new_tokens
                self.stats.generated_tokens += max_new_tokens
                action_ids = draft_ids[0, :max_new_tokens].detach().cpu()
                if update_history:
                    action = self.decode_action_ids(action_ids, unnorm_key)
                    self.update_history(action, action_ids.numpy())
                return action_ids
            self.stats.fast_draft_rejects += 1
            generated_ids = self._generate_baseline_from_prefill(
                input_ids=inputs["input_ids"],
                prefill_out=prefill_out,
                prefill_kwargs={
                    key: value
                    for key, value in {
                        "pixel_values": inputs.get("pixel_values"),
                        "attention_mask": inputs.get("attention_mask"),
                    }.items()
                    if value is not None
                },
                max_new_tokens=max_new_tokens,
            )
            action_ids = generated_ids[0, -max_new_tokens:].detach().cpu()
            if update_history:
                action = self.decode_action_ids(action_ids, unnorm_key)
                self.update_history(action, action_ids.numpy())
            return action_ids

        generated_ids = self._generate_verified(
            input_ids=inputs["input_ids"],
            draft_ids=draft_ids,
            draft_bands=draft_bands,
            confidence_mask=confidence_mask,
            prefill_kwargs={
                key: value
                for key, value in {
                    "pixel_values": inputs.get("pixel_values"),
                    "attention_mask": inputs.get("attention_mask"),
                }.items()
                if value is not None
            },
            max_new_tokens=max_new_tokens,
            prefill_out=prefill_out,
        )
        action_ids = generated_ids[0, -max_new_tokens:].detach().cpu()
        if update_history:
            action = self.decode_action_ids(action_ids, unnorm_key)
            self.update_history(action, action_ids.numpy())
        return action_ids

    def _generate_baseline_from_prefill(
        self,
        input_ids: torch.Tensor,
        prefill_out: Any | None,
        prefill_kwargs: dict,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if prefill_out is None:
            prefill = self.model(input_ids=input_ids, use_cache=True, **prefill_kwargs)
            self.stats.target_forwards += 1
        else:
            prefill = prefill_out

        target_kv: PastKV = prefill.past_key_values
        last_logits = prefill.logits[:, -1:, :]
        generated: list[torch.Tensor] = []
        while len(generated) < max_new_tokens:
            next_id = last_logits.argmax(dim=-1)
            generated.append(next_id)
            out = self.decode_model(
                next_id,
                past_key_values=target_kv,
                attention_mask=self._decode_attention_mask(target_kv, new_tokens=1, batch_size=1),
                position_ids=self._decode_position_ids(target_kv, new_tokens=1, batch_size=1),
                cache_position=self._decode_cache_position(target_kv, new_tokens=1),
                use_cache=True,
            )
            target_kv = out.past_key_values
            last_logits = out.logits[:, -1:, :]
            self.stats.target_forwards += 1
        self.stats.generated_tokens += max_new_tokens
        return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)

    def _fast_draft_allowed(
        self,
        draft_ids: torch.Tensor,
        confidence_mask: np.ndarray,
        max_new_tokens: int,
    ) -> bool:
        if int(np.sum(confidence_mask)) < self.fast_min_confident_tokens:
            return False
        if len(self.token_history) == 0:
            return False
        draft = draft_ids[0, :max_new_tokens].detach().cpu().numpy()
        prev = np.asarray(self.token_history[-1], dtype=np.int64)

        # Gripper/phase changes are where direct draft errors are most costly.
        if draft.shape[0] >= 7 and int(draft[6]) != int(prev[6]):
            return False

        # Near-stationary drafts often mean the head has collapsed to a settle
        # action; let OpenVLA handle those delicate contact/termination phases.
        if draft.shape[0] >= 6 and float(np.mean(np.abs(draft[:6] - prev[:6]))) < 2.0:
            return False

        return True

    def update_history(self, action: np.ndarray, token_ids: np.ndarray | None = None) -> None:
        self.history.append(np.asarray(action, dtype=np.float32))
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size :]
        if token_ids is not None:
            self.token_history.append(np.asarray(token_ids, dtype=np.int64))
            if len(self.token_history) > self.history_size:
                self.token_history = self.token_history[-self.history_size :]

    def _draft_action(self) -> np.ndarray:
        hist = np.stack(self.history, axis=0).astype(np.float32)
        n = hist.shape[0]
        if n < 2:
            pred = hist[-1].copy()
            self._last_residual_std = np.zeros(hist.shape[1], dtype=np.float32)
        else:
            x = np.arange(n, dtype=np.float32)
            pred = np.empty(hist.shape[1], dtype=np.float32)
            residual_std = np.empty(hist.shape[1], dtype=np.float32)
            for dim in range(hist.shape[1]):
                slope, intercept = np.polyfit(x, hist[:, dim], deg=1)
                pred[dim] = slope * n + intercept
                fitted = slope * x + intercept
                residual_std[dim] = float(np.std(hist[:, dim] - fitted))
            self._last_residual_std = residual_std

        # Gripper is an absolute sticky/categorical value in OpenVLA raw action
        # space. Extrapolation is unstable, so carry the recent mode.
        recent_gripper = hist[-min(n, 3) :, 6]
        pred[6] = 1.0 if float(np.mean(recent_gripper)) > 0.5 else 0.0
        return pred

    def action_to_token_ids(self, action: np.ndarray, unnorm_key: str) -> torch.Tensor:
        stats = self.model.get_action_stats(unnorm_key)
        mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        normalized = action.copy()
        denom = np.maximum(high - low, 1e-6)
        normalized[mask] = 2.0 * (action[mask] - low[mask]) / denom[mask] - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        bin_centers = np.asarray(self.model.bin_centers, dtype=np.float32)
        bin_ids = np.abs(bin_centers[:, None] - normalized[None, :]).argmin(axis=0)
        token_ids = self.model.vocab_size - (bin_ids + 1)
        return torch.as_tensor(token_ids, dtype=torch.long).view(1, -1)

    def action_to_token_bands(
        self,
        action: np.ndarray,
        unnorm_key: str,
    ) -> tuple[torch.Tensor, list[set[int]], np.ndarray]:
        stats = self.model.get_action_stats(unnorm_key)
        mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        normalized = action.copy()
        denom = np.maximum(high - low, 1e-6)
        normalized[mask] = 2.0 * (action[mask] - low[mask]) / denom[mask] - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        bin_centers = np.asarray(self.model.bin_centers, dtype=np.float32)
        bin_ids = np.abs(bin_centers[:, None] - normalized[None, :]).argmin(axis=0)
        residual_norm = np.zeros_like(self._last_residual_std, dtype=np.float32)
        residual_norm[mask] = 2.0 * self._last_residual_std[mask] / denom[mask]
        bin_width = 2.0 / max(len(bin_centers) - 1, 1)
        residual_bins = residual_norm / max(bin_width, 1e-6)

        token_ids = self.model.vocab_size - (bin_ids + 1)
        bands: list[set[int]] = []
        max_bin = len(bin_centers) - 1
        for dim, center_bin in enumerate(bin_ids):
            radius = int(self.band_radius + np.ceil(residual_bins[dim]))
            if dim == 6:
                radius = max(radius, 1)
            lo = max(int(center_bin) - radius, 0)
            hi = min(int(center_bin) + radius, max_bin)
            bands.append({int(self.model.vocab_size - (bin_id + 1)) for bin_id in range(lo, hi + 1)})

        return torch.as_tensor(token_ids, dtype=torch.long).view(1, -1), bands, residual_bins

    def _draft_token_bands(self) -> tuple[torch.Tensor, list[set[int]], np.ndarray]:
        hist = np.stack(self.token_history, axis=0).astype(np.float32)
        n = hist.shape[0]
        x = np.arange(n, dtype=np.float32)
        pred_tokens = np.empty(hist.shape[1], dtype=np.int64)
        residual_bins = np.empty(hist.shape[1], dtype=np.float32)

        for dim in range(hist.shape[1]):
            if n < 2:
                pred = hist[-1, dim]
                residual_bins[dim] = 0.0
            else:
                slope, intercept = np.polyfit(x, hist[:, dim], deg=1)
                pred = slope * n + intercept
                fitted = slope * x + intercept
                residual_bins[dim] = float(np.std(hist[:, dim] - fitted))
            pred_tokens[dim] = int(round(pred))

        # Gripper token history is categorical; carry recent mode rather than
        # extrapolating token IDs.
        gripper_vals = hist[-min(n, 3) :, 6].astype(np.int64)
        values, counts = np.unique(gripper_vals, return_counts=True)
        pred_tokens[6] = int(values[np.argmax(counts)])
        residual_bins[6] = 0.0

        min_token = int(self.model.vocab_size - len(self.model.bin_centers))
        max_token = int(self.model.vocab_size - 1)
        pred_tokens = np.clip(pred_tokens, min_token, max_token)

        bands: list[set[int]] = []
        for dim, token in enumerate(pred_tokens):
            radius = int(self.band_radius + np.ceil(residual_bins[dim]))
            if dim == 6:
                radius = max(radius, 1)
            lo = max(int(token) - radius, min_token)
            hi = min(int(token) + radius, max_token)
            bands.append(set(range(lo, hi + 1)))

        return torch.as_tensor(pred_tokens, dtype=torch.long).view(1, -1), bands, residual_bins

    def _draft_learned_head(
        self,
        prefill_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[set[int]], np.ndarray, np.ndarray]:
        assert self.draft_head is not None
        token_hist = np.stack(self.token_history[-self.draft_head.config.history_size :], axis=0)
        bins = token_ids_to_bins(token_hist, self.model.vocab_size).to(self.device)
        if self.draft_head.config.use_prefill_hidden:
            if prefill_hidden is None:
                raise ValueError("prefill_hidden required when use_prefill_hidden=True")
            top_bins, top_probs, max_probs = self.draft_head.predict(
                bins.unsqueeze(0),
                top_k=self.head_top_k,
                prefill_hidden=prefill_hidden,
            )
        else:
            top_bins, top_probs, max_probs = self.draft_head.predict(bins.unsqueeze(0), top_k=self.head_top_k)
        top_bins = top_bins[0].detach().cpu()
        max_probs_np = max_probs[0].detach().cpu().numpy()
        center_bins = top_bins[:, 0]
        token_ids = bins_to_token_ids(center_bins, self.model.vocab_size)
        bands = []
        for dim in range(top_bins.shape[0]):
            bands.append({int(tok) for tok in bins_to_token_ids(top_bins[dim], self.model.vocab_size).tolist()})
        residual_bins = 1.0 - max_probs_np
        confidence_mask = max_probs_np >= self.head_threshold
        return token_ids.view(1, -1).long(), bands, residual_bins.astype(np.float32), confidence_mask

    def decode_action_ids(self, action_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
        predicted_action_token_ids = action_ids.detach().cpu().numpy()
        discretized_actions = self.model.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.model.bin_centers.shape[0] - 1,
        )
        normalized_actions = self.model.bin_centers[discretized_actions]

        stats = self.model.get_action_stats(unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        high = np.asarray(stats["q99"], dtype=np.float32)
        low = np.asarray(stats["q01"], dtype=np.float32)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (high - low) + low,
            normalized_actions,
        )

    def _generate_verified(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_bands: list[set[int]],
        confidence_mask: np.ndarray,
        prefill_kwargs: dict,
        max_new_tokens: int,
        prefill_out: Any | None = None,
    ) -> torch.Tensor:
        assert input_ids.shape[0] == 1, "Trajectory speculation supports batch_size=1"

        if prefill_out is not None:
            prefill = prefill_out
        else:
            prefill = self.model(input_ids=input_ids, use_cache=True, **prefill_kwargs)
            self.stats.target_forwards += 1
        target_kv: PastKV = prefill.past_key_values
        last_logits = prefill.logits[:, -1:, :]

        generated: list[torch.Tensor] = []
        while len(generated) < max_new_tokens:
            pos = len(generated)
            target_next = last_logits.argmax(dim=-1)
            draft_next = draft_ids[:, pos : pos + 1]

            if not bool(confidence_mask[pos]):
                self.stats.token_fallbacks += 1
                generated.append(target_next)
                out = self.decode_model(
                    target_next,
                    past_key_values=target_kv,
                    attention_mask=self._decode_attention_mask(target_kv, new_tokens=1, batch_size=1),
                    position_ids=self._decode_position_ids(target_kv, new_tokens=1, batch_size=1),
                    cache_position=self._decode_cache_position(target_kv, new_tokens=1),
                    use_cache=True,
                )
                target_kv = out.past_key_values
                last_logits = out.logits[:, -1:, :]
                self.stats.target_forwards += 1
                continue

            self.stats.drafted_tokens += 1
            target_item = int(target_next.item())
            draft_item = int(draft_next.item())
            in_band = target_item in draft_bands[pos]
            if target_item == draft_item:
                self.stats.exact_hits += 1
            elif in_band:
                self.stats.band_hits += 1
            else:
                self.stats.band_misses += 1
                generated.append(target_next)
                out = self.decode_model(
                    target_next,
                    past_key_values=target_kv,
                    attention_mask=self._decode_attention_mask(target_kv, new_tokens=1, batch_size=1),
                    position_ids=self._decode_position_ids(target_kv, new_tokens=1, batch_size=1),
                    cache_position=self._decode_cache_position(target_kv, new_tokens=1),
                    use_cache=True,
                )
                target_kv = out.past_key_values
                last_logits = out.logits[:, -1:, :]
                self.stats.target_forwards += 1
                continue

            run_len = self._confident_run_length(pos, max_new_tokens, confidence_mask)
            self.stats.max_tree_depth_used = max(self.stats.max_tree_depth_used, run_len)
            candidate_input = self._candidate_sequences(
                first_token=target_item,
                start_pos=pos,
                run_len=run_len,
                draft_ids=draft_ids,
                draft_bands=draft_bands,
            ).to(self.device)
            self.stats.tree_verifies += 1
            self.stats.tree_candidates += int(candidate_input.shape[0])

            t0 = time.perf_counter()
            verify_kv = self._repeat_kv_batch(target_kv, candidate_input.shape[0])
            verify_out = self.decode_model(
                candidate_input,
                past_key_values=verify_kv,
                attention_mask=self._decode_attention_mask(
                    target_kv,
                    new_tokens=candidate_input.shape[1],
                    batch_size=candidate_input.shape[0],
                ),
                position_ids=self._decode_position_ids(
                    target_kv,
                    new_tokens=candidate_input.shape[1],
                    batch_size=candidate_input.shape[0],
                ),
                cache_position=self._decode_cache_position(target_kv, new_tokens=candidate_input.shape[1]),
                use_cache=True,
            )
            self.stats.verify_time_ms += (time.perf_counter() - t0) * 1000
            self.stats.target_forwards += 1

            best_idx, accepted_count = self._best_candidate_prefix(candidate_input, verify_out.logits)
            self.stats.drafted_tokens += max(accepted_count - 1, 0)
            self.stats.accepted_tokens += accepted_count
            selected = candidate_input[best_idx : best_idx + 1]
            for offset in range(accepted_count):
                generated.append(selected[:, offset : offset + 1])

            prefix_len = kv_seq_len(target_kv)
            selected_kv = self._select_kv_batch(verify_out.past_key_values, best_idx)
            target_kv = trim_kv(selected_kv, prefix_len + accepted_count)
            last_logits = verify_out.logits[best_idx : best_idx + 1, accepted_count - 1 : accepted_count, :]

        self.stats.generated_tokens += max_new_tokens
        return torch.cat([input_ids, torch.cat(generated[:max_new_tokens], dim=1)], dim=1)

    def _confident_run_length(self, pos: int, max_new_tokens: int, confidence_mask: np.ndarray) -> int:
        # NOTE: OpenVLA's current Llama/Transformers cached multi-token decode
        # is not equivalent to sequential decode for action prompts. Default to
        # policy-preserving depth 1; allow deeper trees only for explicit
        # approximate robot-success/speed experiments.
        if self.max_tree_depth > 1 and not self.allow_approx_tree:
            return 1
        run_len = 0
        while (
            pos + run_len < max_new_tokens
            and bool(confidence_mask[pos + run_len])
            and run_len < self.max_tree_depth
        ):
            run_len += 1
        return max(run_len, 1)

    def _candidate_sequences(
        self,
        first_token: int,
        start_pos: int,
        run_len: int,
        draft_ids: torch.Tensor,
        draft_bands: list[set[int]],
    ) -> torch.Tensor:
        seqs: list[tuple[list[int], int]] = [([first_token], abs(first_token - int(draft_ids[0, start_pos].item())))]
        for offset in range(1, run_len):
            pos = start_pos + offset
            center = int(draft_ids[0, pos].item())
            values = sorted(draft_bands[pos], key=lambda tok: (abs(tok - center), tok))[: self.tree_width]
            expanded: list[tuple[list[int], int]] = []
            for seq, score in seqs:
                for tok in values:
                    expanded.append((seq + [int(tok)], score + abs(int(tok) - center)))
            expanded.sort(key=lambda item: item[1])
            seqs = expanded[: self.tree_width]
        return torch.as_tensor([seq for seq, _score in seqs], dtype=torch.long)

    @staticmethod
    def _best_candidate_prefix(candidate_input: torch.Tensor, verify_logits: torch.Tensor) -> tuple[int, int]:
        best_idx = 0
        best_len = 1
        for idx in range(candidate_input.shape[0]):
            accepted = 1
            for offset in range(1, candidate_input.shape[1]):
                pred = verify_logits[idx, offset - 1, :].argmax(dim=-1)
                if int(pred.item()) != int(candidate_input[idx, offset].item()):
                    break
                accepted += 1
            if accepted > best_len:
                best_idx = idx
                best_len = accepted
                if best_len == candidate_input.shape[1]:
                    break
        return best_idx, best_len

    @staticmethod
    def _repeat_kv_batch(past_kv: PastKV, batch_size: int) -> PastKV:
        if batch_size == 1:
            return past_kv
        repeated = []
        for layer in past_kv:
            repeated_layer = []
            for tensor in layer:
                repeated_layer.append(tensor.repeat_interleave(batch_size, dim=0))
            repeated.append(tuple(repeated_layer))
        return tuple(repeated)

    @staticmethod
    def _select_kv_batch(past_kv: PastKV, batch_idx: int) -> PastKV:
        selected = []
        for layer in past_kv:
            selected_layer = []
            for tensor in layer:
                selected_layer.append(tensor[batch_idx : batch_idx + 1])
            selected.append(tuple(selected_layer))
        return tuple(selected)

    def _decode_attention_mask(self, past_kv: PastKV, new_tokens: int, batch_size: int) -> torch.Tensor:
        total_len = kv_seq_len(past_kv) + new_tokens
        return torch.ones(batch_size, total_len, dtype=torch.long, device=self.device)

    def _decode_position_ids(self, past_kv: PastKV, new_tokens: int, batch_size: int) -> torch.Tensor:
        start = kv_seq_len(past_kv)
        pos = torch.arange(start, start + new_tokens, dtype=torch.long, device=self.device)
        return pos.unsqueeze(0).expand(batch_size, -1)

    def _decode_cache_position(self, past_kv: PastKV, new_tokens: int) -> torch.Tensor:
        start = kv_seq_len(past_kv)
        return torch.arange(start, start + new_tokens, dtype=torch.long, device=self.device)
