"""Trajectory-prior speculative decoding for OpenVLA action tokens.

The draft is domain-specific: recent continuous robot actions are extrapolated
per action dimension, converted back into OpenVLA action token IDs, then
verified by the target OpenVLA language model. Verification keeps greedy target
outputs exact; the trajectory prior only changes how many target forwards are
needed to obtain the same action tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from serving.kv_cache_manager import PastKV, kv_seq_len, trim_kv
from serving.trajectory_draft_head import TinyTrajectoryHead, bins_to_token_ids, token_ids_to_bins
from serving.trajectory_phase import PhaseThresholds, label_phase


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
    retrieval_calls: int = 0
    retrieval_hits: int = 0
    retrieval_tokens: int = 0
    retrieval_rejects: int = 0
    hybrid_head_switches: int = 0
    hybrid_retrieval_switches: int = 0
    two_head_smooth_calls: int = 0
    two_head_complex_calls: int = 0
    two_head_fallback_calls: int = 0
    relaxed_group_accepts: int = 0
    relaxed_group_rejects: int = 0
    chunk_anchor_calls: int = 0
    chunk_buffer_hits: int = 0
    chunk_buffered_actions: int = 0
    chunk_refreshes: int = 0
    chunk_refresh_reasons: dict[str, int] = field(default_factory=dict)
    chunk_smooth_actions: int = 0
    chunk_complex_actions: int = 0
    chunk_guard_rejects: int = 0
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
            "retrieval_calls": self.retrieval_calls,
            "retrieval_hits": self.retrieval_hits,
            "retrieval_tokens": self.retrieval_tokens,
            "retrieval_rejects": self.retrieval_rejects,
            "hybrid_head_switches": self.hybrid_head_switches,
            "hybrid_retrieval_switches": self.hybrid_retrieval_switches,
            "two_head_smooth_calls": self.two_head_smooth_calls,
            "two_head_complex_calls": self.two_head_complex_calls,
            "two_head_fallback_calls": self.two_head_fallback_calls,
            "relaxed_group_accepts": self.relaxed_group_accepts,
            "relaxed_group_rejects": self.relaxed_group_rejects,
            "chunk_anchor_calls": self.chunk_anchor_calls,
            "chunk_buffer_hits": self.chunk_buffer_hits,
            "chunk_buffered_actions": self.chunk_buffered_actions,
            "chunk_refreshes": self.chunk_refreshes,
            "chunk_refresh_reasons": dict(sorted(self.chunk_refresh_reasons.items())),
            "chunk_smooth_actions": self.chunk_smooth_actions,
            "chunk_complex_actions": self.chunk_complex_actions,
            "chunk_guard_rejects": self.chunk_guard_rejects,
            "draft_ms": round(self.draft_time_ms, 1),
            "verify_ms": round(self.verify_time_ms, 1),
        }


@dataclass
class RetrievalEntry:
    task_key: str
    timestep_bucket: int
    prefix: np.ndarray
    next_tokens: np.ndarray


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
        smooth_draft_head: TinyTrajectoryHead | None = None,
        complex_draft_head: TinyTrajectoryHead | None = None,
        direct_chunk_head: TinyTrajectoryHead | None = None,
        smooth_direct_chunk_head: TinyTrajectoryHead | None = None,
        complex_direct_chunk_head: TinyTrajectoryHead | None = None,
        head_threshold: float = 0.5,
        head_top_k: int = 3,
        decoder_mode: str = "trajectory-spec",
        use_hybrid_spec: bool | None = None,
        retrieval_top_k: int = 4,
        retrieval_min_confidence: float = 0.55,
        retrieval_context_steps: int = 2,
        retrieval_history_size: int = 512,
        kinematic_threshold: float = 4.0,
        smooth_phase_curvature: float = 6.0,
        smooth_phase_acceleration: float = 8.0,
        smooth_phase_min_displacement: float = 1.5,
        relaxed_tolerance: float | tuple[float, float, float] = (2.0, 2.0, 0.0),
        hybrid_max_draft_length: int = 7,
        use_draft_prefill_hidden: bool | None = None,
        fast_max_draft_calls: int | None = None,
        fast_max_action_step: int | None = None,
        fast_stop_after_gripper_change: bool = False,
        fast_stationary_token_delta: float = 2.0,
        fast_stationary_patience: int | None = None,
        chunk_smooth_len: int = 4,
        chunk_complex_len: int = 2,
        chunk_heartbeat: int = 6,
        chunk_min_confident_tokens: int = 6,
        chunk_max_token_delta: float = 32.0,
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
        self.smooth_draft_head = smooth_draft_head
        self.complex_draft_head = complex_draft_head
        self.direct_chunk_head = direct_chunk_head
        self.smooth_direct_chunk_head = smooth_direct_chunk_head
        self.complex_direct_chunk_head = complex_direct_chunk_head
        self.head_threshold = head_threshold
        self.head_top_k = head_top_k
        self.decoder_mode = decoder_mode
        self.use_hybrid_spec = (decoder_mode == "trajectory-hybrid-spec") if use_hybrid_spec is None else use_hybrid_spec
        self.use_two_head_spec = decoder_mode == "trajectory-two-head-spec"
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_min_confidence = retrieval_min_confidence
        self.retrieval_context_steps = retrieval_context_steps
        self.retrieval_history_size = retrieval_history_size
        self.kinematic_threshold = kinematic_threshold
        self.phase_thresholds = PhaseThresholds(
            smooth_curvature=smooth_phase_curvature,
            smooth_acceleration=smooth_phase_acceleration,
            min_displacement=smooth_phase_min_displacement,
        )
        if isinstance(relaxed_tolerance, tuple):
            self.relaxed_tolerance = relaxed_tolerance
        else:
            self.relaxed_tolerance = (float(relaxed_tolerance), float(relaxed_tolerance), 0.0)
        self.hybrid_max_draft_length = hybrid_max_draft_length
        self.use_draft_prefill_hidden = use_draft_prefill_hidden
        self.fast_max_draft_calls = fast_max_draft_calls
        self.fast_max_action_step = fast_max_action_step
        self.fast_stop_after_gripper_change = fast_stop_after_gripper_change
        self.fast_stationary_token_delta = fast_stationary_token_delta
        self.fast_stationary_patience = fast_stationary_patience
        self.use_chunk_spec = decoder_mode == "trajectory-chunk-spec"
        self.use_direct_chunk_spec = decoder_mode == "trajectory-direct-chunk-spec"
        self.chunk_smooth_len = max(1, int(chunk_smooth_len))
        self.chunk_complex_len = max(1, int(chunk_complex_len))
        self.chunk_heartbeat = max(1, int(chunk_heartbeat))
        self.chunk_min_confident_tokens = max(1, int(chunk_min_confident_tokens))
        self.chunk_max_token_delta = float(chunk_max_token_delta)
        self.history: list[np.ndarray] = []
        self.token_history: list[np.ndarray] = []
        self._chunk_buffer: list[dict[str, Any]] = []
        self.retrieval_entries: list[RetrievalEntry] = []
        self._current_task_key = "default"
        self._action_step = 0
        self._fast_draft_calls_used = 0
        self._steps_since_vla_refresh = 0
        self._stationary_steps = 0
        self._last_residual_std = np.zeros(7, dtype=np.float32)
        self.stats = TrajectorySpecStats()

    def set_decoder_mode(self, decoder_mode: str) -> None:
        self.decoder_mode = decoder_mode
        self.use_hybrid_spec = decoder_mode == "trajectory-hybrid-spec"
        self.use_two_head_spec = decoder_mode == "trajectory-two-head-spec"
        self.use_chunk_spec = decoder_mode == "trajectory-chunk-spec"
        self.use_direct_chunk_spec = decoder_mode == "trajectory-direct-chunk-spec"
        self._chunk_buffer.clear()

    def reset(self) -> None:
        self.history.clear()
        self.token_history.clear()
        self._chunk_buffer.clear()
        self.retrieval_entries.clear()
        self._current_task_key = "default"
        self._action_step = 0
        self._fast_draft_calls_used = 0
        self._steps_since_vla_refresh = 0
        self._stationary_steps = 0
        self.stats = TrajectorySpecStats()

    @torch.no_grad()
    def predict_action(
        self,
        inputs,
        unnorm_key: str,
        max_new_tokens: int = 7,
        task_key: str | None = None,
    ) -> np.ndarray:
        """Return exact greedy OpenVLA raw action using trajectory speculation."""
        self.stats.calls += 1
        self._current_task_key = task_key or unnorm_key

        if self.use_chunk_spec or self.use_direct_chunk_spec:
            return self._predict_action_chunked(inputs, unnorm_key, max_new_tokens)

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

        hist_ready = self._any_head_ready()
        needs_prefill_hidden = self._needs_draft_prefill_hidden()
        if needs_prefill_hidden:
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
            draft_ids, draft_bands, residual_bins, confidence_mask = self._select_draft(
                unnorm_key=unnorm_key,
                hist_ready=hist_ready,
                prefill_hidden=prefill_h,
                max_new_tokens=max_new_tokens,
            )
        else:
            draft_ids, draft_bands, residual_bins, confidence_mask = self._select_draft(
                unnorm_key=unnorm_key,
                hist_ready=hist_ready,
                prefill_hidden=None,
                max_new_tokens=max_new_tokens,
            )
        draft_ids = draft_ids.to(self.device)
        self.stats.draft_time_ms += (time.perf_counter() - t0) * 1000
        if not self._has_any_head() and not self.use_hybrid_spec:
            # Gripper is discrete and task-phase dependent; let the target model
            # generate it until we add a phase-aware gripper predictor.
            confidence_mask[6] = False

        if self.fast_draft_only and (self._has_any_head() or self.use_hybrid_spec):
            if self._fast_draft_allowed(draft_ids, confidence_mask, max_new_tokens):
                self.stats.fast_draft_calls += 1
                self._fast_draft_calls_used += 1
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
        task_key: str | None = None,
    ) -> torch.Tensor:
        """Generate action token IDs through the speculative path for exactness checks."""
        self.stats.calls += 1
        self._current_task_key = task_key or unnorm_key

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

        hist_ready = self._any_head_ready()
        needs_prefill_hidden = self._needs_draft_prefill_hidden()
        if needs_prefill_hidden:
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
            draft_ids, draft_bands, residual_bins, confidence_mask = self._select_draft(
                unnorm_key=unnorm_key,
                hist_ready=hist_ready,
                prefill_hidden=prefill_h,
                max_new_tokens=max_new_tokens,
            )
        else:
            draft_ids, draft_bands, residual_bins, confidence_mask = self._select_draft(
                unnorm_key=unnorm_key,
                hist_ready=hist_ready,
                prefill_hidden=None,
                max_new_tokens=max_new_tokens,
            )
        draft_ids = draft_ids.to(self.device)
        if not self._has_any_head() and not self.use_hybrid_spec:
            confidence_mask[6] = False

        if self.fast_draft_only and (self._has_any_head() or self.use_hybrid_spec):
            if self._fast_draft_allowed(draft_ids, confidence_mask, max_new_tokens):
                self.stats.fast_draft_calls += 1
                self._fast_draft_calls_used += 1
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

    def _heads(self) -> list[TinyTrajectoryHead]:
        return [
            head
            for head in (
                self.draft_head,
                self.smooth_draft_head,
                self.complex_draft_head,
                self.direct_chunk_head,
                self.smooth_direct_chunk_head,
                self.complex_direct_chunk_head,
            )
            if head is not None
        ]

    def _has_any_head(self) -> bool:
        return bool(self._heads())

    def _any_head_ready(self) -> bool:
        return any(len(self.token_history) >= head.config.history_size for head in self._heads())

    def _needs_draft_prefill_hidden(self) -> bool:
        ready_heads = [head for head in self._heads() if len(self.token_history) >= head.config.history_size]
        if not ready_heads:
            return False
        if not any(head.config.use_prefill_hidden for head in ready_heads):
            return False
        return True if self.use_draft_prefill_hidden is None else bool(self.use_draft_prefill_hidden)

    def _select_draft(
        self,
        *,
        unnorm_key: str,
        hist_ready: bool,
        prefill_hidden: torch.Tensor | None,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, list[set[int]], np.ndarray, np.ndarray]:
        if self.use_hybrid_spec:
            retrieval = self._draft_retrieval(max_new_tokens=max_new_tokens)
            if retrieval is not None and self._kinematic_retrieval_allowed():
                self.stats.hybrid_retrieval_switches += 1
                return retrieval
            if retrieval is not None:
                self.stats.retrieval_rejects += 1
            self.stats.hybrid_head_switches += 1

        if self.use_two_head_spec:
            routed_head, phase = self._select_phase_head()
            if routed_head is not None:
                if phase == "smooth":
                    self.stats.two_head_smooth_calls += 1
                else:
                    self.stats.two_head_complex_calls += 1
                draft = self._draft_learned_head(head=routed_head, prefill_hidden=prefill_hidden)
                self.stats.learned_head_calls += 1
                self.stats.learned_head_confident_tokens += int(np.sum(draft[3]))
                return self._cap_draft_length(*draft, max_new_tokens=max_new_tokens)
            self.stats.two_head_fallback_calls += 1

        if hist_ready and self.draft_head is not None:
            draft = self._draft_learned_head(head=self.draft_head, prefill_hidden=prefill_hidden)
            self.stats.learned_head_calls += 1
            self.stats.learned_head_confident_tokens += int(np.sum(draft[3]))
            return self._cap_draft_length(*draft, max_new_tokens=max_new_tokens)
        if len(self.token_history) >= self.min_history:
            draft_ids, draft_bands, residual_bins = self._draft_token_bands()
            confidence_mask = residual_bins <= self.max_residual_bins
            return self._cap_draft_length(draft_ids, draft_bands, residual_bins, confidence_mask, max_new_tokens)

        draft_action = self._draft_action()
        draft_ids, draft_bands, residual_bins = self.action_to_token_bands(draft_action, unnorm_key)
        confidence_mask = residual_bins <= self.max_residual_bins
        return self._cap_draft_length(draft_ids, draft_bands, residual_bins, confidence_mask, max_new_tokens)

    def _cap_draft_length(
        self,
        draft_ids: torch.Tensor,
        draft_bands: list[set[int]],
        residual_bins: np.ndarray,
        confidence_mask: np.ndarray,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, list[set[int]], np.ndarray, np.ndarray]:
        cap = min(max_new_tokens, max(1, int(self.hybrid_max_draft_length)))
        confidence = np.asarray(confidence_mask, dtype=bool).copy()
        if cap < len(confidence):
            confidence[cap:] = False
        return draft_ids, draft_bands, np.asarray(residual_bins, dtype=np.float32), confidence

    def _retrieval_prefix(self) -> np.ndarray | None:
        if len(self.token_history) < self.retrieval_context_steps:
            return None
        return np.stack(self.token_history[-self.retrieval_context_steps :], axis=0).astype(np.int64)

    def _draft_retrieval(
        self,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, list[set[int]], np.ndarray, np.ndarray] | None:
        self.stats.retrieval_calls += 1
        prefix = self._retrieval_prefix()
        if prefix is None or not self.retrieval_entries:
            return None

        current_bucket = self._timestep_bucket(self._action_step)
        scored: list[tuple[float, RetrievalEntry]] = []
        for entry in self.retrieval_entries:
            if entry.task_key != self._current_task_key or entry.prefix.shape != prefix.shape:
                continue
            mean_abs = float(np.mean(np.abs(entry.prefix.astype(np.float32) - prefix.astype(np.float32))))
            bucket_penalty = 0.25 * abs(entry.timestep_bucket - current_bucket)
            scored.append((mean_abs + bucket_penalty, entry))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        top = scored[: max(1, self.retrieval_top_k)]
        best_score, _best = top[0]
        confidence = float(np.exp(-best_score / max(self.kinematic_threshold, 1e-6)))
        if confidence < self.retrieval_min_confidence:
            return None

        candidates = np.stack([entry.next_tokens for _score, entry in top], axis=0).astype(np.int64)
        center = np.median(candidates, axis=0).round().astype(np.int64)
        min_token = int(self.model.vocab_size - len(self.model.bin_centers))
        max_token = int(self.model.vocab_size - 1)
        center = np.clip(center, min_token, max_token)
        spread = np.maximum(np.std(candidates.astype(np.float32), axis=0), 0.0)

        bands: list[set[int]] = []
        for dim, token in enumerate(center):
            radius = int(self.band_radius + np.ceil(spread[dim]))
            if dim == 6:
                radius = max(radius, 1)
            lo = max(int(token) - radius, min_token)
            hi = min(int(token) + radius, max_token)
            bands.append(set(range(lo, hi + 1)))

        residual_bins = spread.astype(np.float32)
        confidence_mask = residual_bins <= max(self.max_residual_bins, self.kinematic_threshold)
        confidence_mask[:max_new_tokens] &= confidence >= self.retrieval_min_confidence
        self.stats.retrieval_hits += 1
        self.stats.retrieval_tokens += int(np.sum(confidence_mask[:max_new_tokens]))
        return self._cap_draft_length(
            torch.as_tensor(center, dtype=torch.long).view(1, -1),
            bands,
            residual_bins,
            confidence_mask,
            max_new_tokens,
        )

    def _kinematic_retrieval_allowed(self) -> bool:
        if len(self.token_history) < 3:
            return True
        hist = np.stack(self.token_history[-3:], axis=0).astype(np.float32)
        curvature = float(np.mean(np.abs(hist[2, :6] - 2.0 * hist[1, :6] + hist[0, :6])))
        gripper_change = bool(hist[-1, 6] != hist[-2, 6])
        return curvature <= self.kinematic_threshold and not gripper_change

    def _select_phase_head(self) -> tuple[TinyTrajectoryHead | None, str]:
        if len(self.token_history) < 2:
            return None, "complex"
        token_hist = np.stack(self.token_history, axis=0)
        bin_hist = token_ids_to_bins(token_hist, self.model.vocab_size).numpy()
        phase = label_phase(bin_hist, thresholds=self.phase_thresholds)
        preferred = self.smooth_draft_head if phase == "smooth" else self.complex_draft_head
        fallback = self.complex_draft_head if phase == "smooth" else self.smooth_draft_head
        if preferred is not None and len(self.token_history) >= preferred.config.history_size:
            return preferred, phase
        if fallback is not None and len(self.token_history) >= fallback.config.history_size:
            return fallback, "complex" if phase == "smooth" else "smooth"
        if self.draft_head is not None and len(self.token_history) >= self.draft_head.config.history_size:
            return self.draft_head, phase
        return None, phase

    @torch.no_grad()
    def _predict_action_chunked(self, inputs, unnorm_key: str, max_new_tokens: int) -> np.ndarray:
        refresh_reason = self._chunk_refresh_reason()
        if refresh_reason is None:
            item = self._chunk_buffer.pop(0)
            token_ids = np.asarray(item["token_ids"], dtype=np.int64)
            action = self.decode_action_ids(torch.as_tensor(token_ids, dtype=torch.long), unnorm_key)
            self.stats.chunk_buffer_hits += 1
            if item.get("phase") == "smooth":
                self.stats.chunk_smooth_actions += 1
            else:
                self.stats.chunk_complex_actions += 1
            self.update_history(action, token_ids)
            self._steps_since_vla_refresh += 1
            return action

        self._record_chunk_refresh(refresh_reason)
        self._chunk_buffer.clear()
        self.stats.chunk_anchor_calls += 1
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
        token_ids = token_ids_t.numpy()
        action = self.decode_action_ids(token_ids_t, unnorm_key)
        self.update_history(action, token_ids)
        self._steps_since_vla_refresh = 0
        self._fill_chunk_buffer(max_new_tokens=max_new_tokens)
        return action

    def _record_chunk_refresh(self, reason: str) -> None:
        self.stats.chunk_refreshes += 1
        self.stats.chunk_refresh_reasons[reason] = self.stats.chunk_refresh_reasons.get(reason, 0) + 1

    def _chunk_refresh_reason(self) -> str | None:
        if not self._chunk_buffer:
            return "empty_buffer"
        if self._steps_since_vla_refresh >= self.chunk_heartbeat:
            return "heartbeat"
        if not self.token_history:
            return "no_history"
        next_ids = np.asarray(self._chunk_buffer[0]["token_ids"], dtype=np.int64)
        prev = np.asarray(self.token_history[-1], dtype=np.int64)
        if next_ids.shape[0] >= 7 and int(next_ids[6]) != int(prev[6]):
            self.stats.chunk_guard_rejects += 1
            return "gripper_change"
        if next_ids.shape[0] >= 6 and float(np.mean(np.abs(next_ids[:6] - prev[:6]))) > self.chunk_max_token_delta:
            self.stats.chunk_guard_rejects += 1
            return "large_delta"
        if self._buffer_phase_switches_to_complex():
            self.stats.chunk_guard_rejects += 1
            return "phase_switch"
        return None

    def _buffer_phase_switches_to_complex(self) -> bool:
        if len(self.token_history) < 2 or not self._chunk_buffer:
            return False
        history = [np.asarray(row, dtype=np.int64) for row in self.token_history[-2:]]
        history.append(np.asarray(self._chunk_buffer[0]["token_ids"], dtype=np.int64))
        bin_hist = token_ids_to_bins(np.stack(history, axis=0), self.model.vocab_size).numpy()
        return label_phase(bin_hist, thresholds=self.phase_thresholds) == "complex" and self._chunk_buffer[0].get("phase") == "smooth"

    def _fill_chunk_buffer(self, max_new_tokens: int) -> None:
        if not self._has_any_head() or len(self.token_history) < self.min_history:
            return
        token_history = [np.asarray(row, dtype=np.int64).copy() for row in self.token_history]
        if self.use_direct_chunk_spec and (
            self.direct_chunk_head is not None
            or self.smooth_direct_chunk_head is not None
            or self.complex_direct_chunk_head is not None
        ):
            self._fill_direct_chunk_buffer(token_history=token_history, max_new_tokens=max_new_tokens)
            return
        head, phase = self._select_phase_head_for_history(token_history)
        if head is None:
            self.stats.two_head_fallback_calls += 1
            return
        total_len = self.chunk_smooth_len if phase == "smooth" else self.chunk_complex_len
        draft_count = max(0, total_len - 1)
        for _ in range(draft_count):
            if len(token_history) < head.config.history_size:
                break
            draft_ids, _bands, _residual_bins, confidence = self._draft_learned_head_from_history(
                head=head,
                token_history=token_history,
                prefill_hidden=None,
            )
            if int(np.sum(confidence[:max_new_tokens])) < self.chunk_min_confident_tokens:
                break
            token_ids = draft_ids[0, :max_new_tokens].detach().cpu().numpy()
            if token_history and token_ids.shape[0] >= 7 and int(token_ids[6]) != int(token_history[-1][6]):
                break
            if token_history and token_ids.shape[0] >= 6:
                delta = float(np.mean(np.abs(token_ids[:6] - token_history[-1][:6])))
                if delta > self.chunk_max_token_delta:
                    break
            self._chunk_buffer.append(
                {
                    "token_ids": token_ids.copy(),
                    "phase": phase,
                    "confidence": int(np.sum(confidence[:max_new_tokens])),
                }
            )
            token_history.append(token_ids.copy())
            token_history = token_history[-self.history_size :]
            self.stats.chunk_buffered_actions += 1
            if phase == "smooth":
                self.stats.two_head_smooth_calls += 1
            else:
                self.stats.two_head_complex_calls += 1

    def _fill_direct_chunk_buffer(self, token_history: list[np.ndarray], max_new_tokens: int) -> None:
        phase = label_phase(
            token_ids_to_bins(np.stack(token_history, axis=0), self.model.vocab_size).numpy(),
            thresholds=self.phase_thresholds,
        )
        head = self._select_direct_chunk_head_for_phase(phase)
        if head is None or len(token_history) < head.config.history_size:
            self.stats.two_head_fallback_calls += 1
            return

        total_len = self.chunk_smooth_len if phase == "smooth" else self.chunk_complex_len
        max_buffer = max(0, min(total_len - 1, int(head.config.action_horizon)))
        if max_buffer <= 0:
            return

        top_bins, _top_probs, max_probs = self._predict_head_topk_from_history(
            head=head,
            token_history=token_history,
            prefill_hidden=None,
        )
        if top_bins.dim() == 3:
            top_bins = top_bins.unsqueeze(0)
            max_probs = max_probs.unsqueeze(0)
        top_bins = top_bins[0].detach().cpu()
        max_probs_np = max_probs[0].detach().cpu().numpy()

        for step_idx in range(min(max_buffer, top_bins.shape[0])):
            confidence = max_probs_np[step_idx] >= self.head_threshold
            if int(np.sum(confidence[:max_new_tokens])) < self.chunk_min_confident_tokens:
                break
            center_bins = top_bins[step_idx, :, 0]
            token_ids = bins_to_token_ids(center_bins, self.model.vocab_size).long().numpy()[:max_new_tokens]
            if token_history and token_ids.shape[0] >= 7 and int(token_ids[6]) != int(token_history[-1][6]):
                break
            if token_history and token_ids.shape[0] >= 6:
                delta = float(np.mean(np.abs(token_ids[:6] - token_history[-1][:6])))
                if delta > self.chunk_max_token_delta:
                    break
            self._chunk_buffer.append(
                {
                    "token_ids": token_ids.copy(),
                    "phase": phase,
                    "confidence": int(np.sum(confidence[:max_new_tokens])),
                    "direct_step": step_idx,
                }
            )
            token_history.append(token_ids.copy())
            token_history = token_history[-self.history_size :]
            self.stats.chunk_buffered_actions += 1
            if phase == "smooth":
                self.stats.two_head_smooth_calls += 1
            else:
                self.stats.two_head_complex_calls += 1

    def _select_direct_chunk_head_for_phase(self, phase: str) -> TinyTrajectoryHead | None:
        if phase == "smooth":
            return self.smooth_direct_chunk_head or self.direct_chunk_head
        return self.complex_direct_chunk_head or self.direct_chunk_head

    def _select_phase_head_for_history(self, token_history: list[np.ndarray]) -> tuple[TinyTrajectoryHead | None, str]:
        if len(token_history) < 2:
            return None, "complex"
        token_hist = np.stack(token_history, axis=0)
        bin_hist = token_ids_to_bins(token_hist, self.model.vocab_size).numpy()
        phase = label_phase(bin_hist, thresholds=self.phase_thresholds)
        preferred = self.smooth_draft_head if phase == "smooth" else self.complex_draft_head
        fallback = self.complex_draft_head if phase == "smooth" else self.smooth_draft_head
        if preferred is not None and len(token_history) >= preferred.config.history_size:
            return preferred, phase
        if fallback is not None and len(token_history) >= fallback.config.history_size:
            return fallback, "complex" if phase == "smooth" else "smooth"
        if self.draft_head is not None and len(token_history) >= self.draft_head.config.history_size:
            return self.draft_head, phase
        return None, phase

    def _draft_learned_head_from_history(
        self,
        head: TinyTrajectoryHead,
        token_history: list[np.ndarray],
        prefill_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[set[int]], np.ndarray, np.ndarray]:
        token_hist = np.stack(token_history[-head.config.history_size :], axis=0)
        bins = token_ids_to_bins(token_hist, self.model.vocab_size).to(self.device)
        top_bins, _top_probs, max_probs = self._predict_head_topk(
            head=head,
            bins=bins,
            prefill_hidden=prefill_hidden,
        )
        top_bins = top_bins[0].detach().cpu()
        max_probs_np = max_probs[0].detach().cpu().numpy()
        if top_bins.dim() == 3:
            top_bins = top_bins[0]
            max_probs_np = max_probs_np[0]
        center_bins = top_bins[:, 0]
        token_ids = bins_to_token_ids(center_bins, self.model.vocab_size)
        bands = []
        for dim in range(top_bins.shape[0]):
            bands.append({int(tok) for tok in bins_to_token_ids(top_bins[dim], self.model.vocab_size).tolist()})
        residual_bins = 1.0 - max_probs_np
        confidence_mask = max_probs_np >= self.head_threshold
        return token_ids.view(1, -1).long(), bands, residual_bins.astype(np.float32), confidence_mask

    def _predict_head_topk_from_history(
        self,
        head: TinyTrajectoryHead,
        token_history: list[np.ndarray],
        prefill_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_hist = np.stack(token_history[-head.config.history_size :], axis=0)
        bins = token_ids_to_bins(token_hist, self.model.vocab_size).to(self.device)
        return self._predict_head_topk(head=head, bins=bins, prefill_hidden=prefill_hidden)

    def _predict_head_topk(
        self,
        head: TinyTrajectoryHead,
        bins: torch.Tensor,
        prefill_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if head.config.use_prefill_hidden:
            return head.predict(
                bins.unsqueeze(0),
                top_k=self.head_top_k,
                prefill_hidden=prefill_hidden,
            )
        return head.predict(bins.unsqueeze(0), top_k=self.head_top_k)

    @staticmethod
    def _timestep_bucket(step: int) -> int:
        return int(step // 10)

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
        if self.fast_max_draft_calls is not None and self._fast_draft_calls_used >= self.fast_max_draft_calls:
            return False
        if self.fast_max_action_step is not None and self._action_step > self.fast_max_action_step:
            return False
        if self.fast_stationary_patience is not None and self._stationary_steps >= self.fast_stationary_patience:
            return False
        if int(np.sum(confidence_mask)) < self.fast_min_confident_tokens:
            return False
        if len(self.token_history) == 0:
            return False
        draft = draft_ids[0, :max_new_tokens].detach().cpu().numpy()
        prev = np.asarray(self.token_history[-1], dtype=np.int64)

        # Gripper/phase changes are where direct draft errors are most costly.
        if draft.shape[0] >= 7 and int(draft[6]) != int(prev[6]):
            if self.fast_stop_after_gripper_change:
                self.fast_max_action_step = min(self.fast_max_action_step or self._action_step, self._action_step)
            return False

        # Near-stationary drafts often mean the head has collapsed to a settle
        # action; let OpenVLA handle those delicate contact/termination phases.
        if draft.shape[0] >= 6 and float(np.mean(np.abs(draft[:6] - prev[:6]))) < self.fast_stationary_token_delta:
            return False

        if self.use_hybrid_spec and not self._relaxed_group_accepts(draft):
            return False

        return True

    def _relaxed_group_accepts(self, draft: np.ndarray) -> bool:
        if len(self.token_history) < 2:
            self.stats.relaxed_group_accepts += 1
            return True
        prev2 = np.asarray(self.token_history[-2], dtype=np.float32)
        prev1 = np.asarray(self.token_history[-1], dtype=np.float32)
        expected = prev1 + (prev1 - prev2)
        pos_tol, rot_tol, grip_tol = self.relaxed_tolerance
        checks = (
            float(np.mean(np.abs(draft[:3].astype(np.float32) - expected[:3]))) <= pos_tol,
            float(np.mean(np.abs(draft[3:6].astype(np.float32) - expected[3:6]))) <= rot_tol,
            abs(float(draft[6]) - float(expected[6])) <= grip_tol,
        )
        if all(checks):
            self.stats.relaxed_group_accepts += 1
            return True
        self.stats.relaxed_group_rejects += 1
        return False

    def update_history(self, action: np.ndarray, token_ids: np.ndarray | None = None) -> None:
        self.history.append(np.asarray(action, dtype=np.float32))
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size :]
        if token_ids is not None:
            token_arr = np.asarray(token_ids, dtype=np.int64)
            prefix = self._retrieval_prefix()
            if prefix is not None:
                self.retrieval_entries.append(
                    RetrievalEntry(
                        task_key=self._current_task_key,
                        timestep_bucket=self._timestep_bucket(self._action_step),
                        prefix=prefix.copy(),
                        next_tokens=token_arr.copy(),
                    )
                )
                if len(self.retrieval_entries) > self.retrieval_history_size:
                    self.retrieval_entries = self.retrieval_entries[-self.retrieval_history_size :]
            if self.token_history:
                delta = float(np.mean(np.abs(token_arr[:6] - self.token_history[-1][:6])))
                self._stationary_steps = self._stationary_steps + 1 if delta < self.fast_stationary_token_delta else 0
            self.token_history.append(token_arr)
            if len(self.token_history) > self.history_size:
                self.token_history = self.token_history[-self.history_size :]
        self._action_step += 1

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
        head: TinyTrajectoryHead,
        prefill_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[set[int]], np.ndarray, np.ndarray]:
        token_hist = np.stack(self.token_history[-head.config.history_size :], axis=0)
        bins = token_ids_to_bins(token_hist, self.model.vocab_size).to(self.device)
        top_bins, top_probs, max_probs = self._predict_head_topk(
            head=head,
            bins=bins,
            prefill_hidden=prefill_hidden,
        )
        top_bins = top_bins[0].detach().cpu()
        max_probs_np = max_probs[0].detach().cpu().numpy()
        if top_bins.dim() == 3:
            top_bins = top_bins[0]
            max_probs_np = max_probs_np[0]
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
