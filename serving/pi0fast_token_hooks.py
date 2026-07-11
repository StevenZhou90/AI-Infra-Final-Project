"""Token/logit accessors for LeRobot π0-FAST policies.

LeRobot's public ``PI0FastPolicy.predict_action_chunk`` returns continuous
actions. Internally, π0-FAST autoregressively emits PaliGemma-token-space FAST
action tokens and computes logits at every decode step. This adapter mirrors the
upstream decode paths so experiments can log tokens/logits and run exact
draft/verify checks without vendoring or forking LeRobot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from serving.kv_cache_manager import clone_kv, repeat_kv_batch, select_kv_batch, trim_kv
from serving.pi0fast_block_gate import block_gate_features
from serving.pi0fast_prefix_gate import PREFIX_GATE_FEATURES, action_feature_values


@dataclass
class PI0FastGenerationTrace:
    """Decoded π0-FAST action chunk plus FAST-token generation internals."""

    actions: torch.Tensor
    token_ids: torch.Tensor
    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None
    stats: dict[str, Any] | None = None

    @property
    def token_count(self) -> int:
        return int(self.token_ids.shape[-1])


@dataclass
class PI0FastVerifyResult:
    """Target-model verification result for drafted FAST tokens."""

    draft_token_ids: torch.Tensor
    target_token_ids: torch.Tensor
    logits: torch.Tensor
    accepted_prefix: int


class PI0FastTokenLogitAdapter:
    """Expose π0-FAST generated token IDs and target logits.

    The adapter expects a LeRobot ``PI0FastPolicy``-like object. It intentionally
    uses the same private helpers as ``predict_action_chunk`` because those are
    where image preprocessing, tokenization, and FAST detokenization live.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.model = policy.model

    def _canonical_lookup_device(self, device: torch.device | str | None) -> torch.device:
        if device is None:
            return torch.device("cpu")
        lookup_device = torch.device(device)
        if lookup_device.type == "cuda" and lookup_device.index is None and torch.cuda.is_available():
            lookup_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return lookup_device

    def prepare_action_char_length_lookup(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Return a cached PaliGemma-token -> FAST decoded-character length table."""

        lookup_device = self._canonical_lookup_device(device)
        cpu_lookup = getattr(self, "_action_char_length_lookup_cpu", None)
        if cpu_lookup is None:
            paligemma_vocab_size = int(self.model._paligemma_tokenizer.vocab_size)
            tokenizer_vocab_size = int(getattr(self.policy.action_tokenizer, "vocab_size", 0) or 0)
            raw_action_limit = min(paligemma_vocab_size, max(tokenizer_vocab_size, 65_536))

            raw_lengths = torch.zeros((raw_action_limit,), dtype=torch.int16)
            bpe_tokenizer = self.policy.action_tokenizer.bpe_tokenizer
            for raw_id in range(raw_action_limit):
                try:
                    raw_lengths[raw_id] = len(bpe_tokenizer.decode(raw_id))
                except Exception:
                    raw_lengths[raw_id] = 0

            paligemma_ids = torch.arange(paligemma_vocab_size, dtype=torch.long)
            try:
                raw_ids = self.policy._paligemma_tokens_to_act_tokens(paligemma_ids).detach().cpu().long()
            except Exception:
                raw_ids = (
                    paligemma_vocab_size
                    - 1
                    - int(getattr(self.policy.config, "fast_skip_tokens", 0))
                    - paligemma_ids
                )
            valid = (raw_ids >= 0) & (raw_ids < raw_action_limit)
            cpu_lookup = torch.zeros((paligemma_vocab_size,), dtype=torch.int16)
            cpu_lookup[valid] = raw_lengths.index_select(0, raw_ids[valid])

            action_end_token_id = self._action_end_token_id()
            if 0 <= action_end_token_id < paligemma_vocab_size:
                cpu_lookup[action_end_token_id] = 0
            for special_id in (
                getattr(self.model._paligemma_tokenizer, "eos_token_id", None),
                getattr(self.model._paligemma_tokenizer, "bos_token_id", None),
            ):
                if special_id is not None and 0 <= int(special_id) < paligemma_vocab_size:
                    cpu_lookup[int(special_id)] = 0
            for token_id in self.model._paligemma_tokenizer.encode("Action: ", add_special_tokens=False):
                if 0 <= int(token_id) < paligemma_vocab_size:
                    cpu_lookup[int(token_id)] = 0

            self._action_char_length_lookup_cpu = cpu_lookup
            self._action_char_length_raw_limit = int(raw_action_limit)

        if lookup_device.type == "cpu":
            return cpu_lookup

        cache = getattr(self, "_action_char_length_lookup_device_cache", None)
        if cache is None:
            cache = {}
            self._action_char_length_lookup_device_cache = cache
        key = str(lookup_device)
        device_lookup = cache.get(key)
        if device_lookup is None or device_lookup.device != lookup_device:
            device_lookup = cpu_lookup.to(device=lookup_device, non_blocking=True)
            cache[key] = device_lookup
        return device_lookup

    @torch.no_grad()
    def predict_action_chunk_with_trace(
        self,
        batch: dict[str, torch.Tensor],
        temperature: float | None = None,
        return_hidden_states: bool = False,
        early_stop_action_end: bool = False,
    ) -> PI0FastGenerationTrace:
        """Return continuous actions, generated FAST token IDs, and per-step logits."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        decode_temperature = self.policy.config.temperature if temperature is None else temperature
        max_steps = self.policy.config.max_decoding_steps

        if self.policy.config.use_kv_cache:
            token_ids, logits, hidden_states = self.sample_actions_fast_kv_cache_with_logits(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_steps,
                temperature=decode_temperature,
                return_hidden_states=return_hidden_states,
                early_stop_action_end=early_stop_action_end,
            )
        else:
            token_ids, logits, hidden_states = self.sample_actions_fast_with_logits(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_steps,
                temperature=decode_temperature,
                return_hidden_states=return_hidden_states,
                early_stop_action_end=early_stop_action_end,
            )

        actions = self._detokenize_generated_actions(token_ids)
        return PI0FastGenerationTrace(
            actions=actions,
            token_ids=token_ids,
            logits=logits,
            hidden_states=hidden_states,
            stats={
                "action_end_token_id": self._action_end_token_id(),
                "early_stop_action_end": int(bool(early_stop_action_end)),
            },
        )

    @torch.no_grad()
    def predict_action_chunk_action_end(
        self,
        batch: dict[str, torch.Tensor],
        temperature: float | None = None,
        max_decoding_steps: int | None = None,
        constrained_action_vocab: bool = False,
        constrained_action_vocab_size: int | None = None,
        constrained_text_vocab_size: int | None = None,
        constrained_full_head_margin: float | None = None,
        constrained_force_action_prefix: bool = True,
        constrained_structural_token_radius: int = 512,
        constrained_full_head_prefix_tokens: int = 0,
        constrained_extra_token_ids: torch.Tensor | list[int] | None = None,
        force_action_prefix: bool = False,
        stop_on_action_chars: bool = False,
        action_char_min_chars: int | None = None,
        action_char_plateau_tokens: int = 0,
        action_char_stable_checks: int = 0,
        action_char_stable_tolerance: float = 0.0,
        action_char_stable_max_chars: int = 0,
        action_char_plateau_reject_eos_restart: bool = False,
        action_char_restart_continue: bool = False,
        action_char_restart_reset: bool = False,
        action_char_restart_reset_low_text_tokens: int = 0,
        action_char_plateau_low_text_tail_tokens: int = 0,
        action_char_strict_target_stop: bool = False,
        action_char_reset_requires_action_end: bool = False,
        action_char_target_confirm_tokens: int = 0,
    ) -> PI0FastGenerationTrace:
        """Serving-oriented greedy decode that stops on the FAST action-end token.

        Unlike ``predict_action_chunk_with_trace``, this path does not retain
        per-step vocabulary logits.  It is intended for latency measurement and
        serving, not offline drafter training.
        """

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        decode_temperature = self.policy.config.temperature if temperature is None else temperature
        max_steps = max_decoding_steps or self.policy.config.max_decoding_steps

        if constrained_action_vocab and not self.policy.config.use_kv_cache:
            raise ValueError("Constrained PI0-FAST action-vocab decode requires KV cache")

        if constrained_action_vocab:
            token_ids = self.sample_actions_fast_kv_cache_action_end_constrained(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_steps,
                temperature=decode_temperature,
                action_vocab_size=constrained_action_vocab_size,
                text_vocab_size=constrained_text_vocab_size,
                full_head_margin=constrained_full_head_margin,
                force_action_prefix=constrained_force_action_prefix,
                structural_token_radius=constrained_structural_token_radius,
                full_head_prefix_tokens=constrained_full_head_prefix_tokens,
                extra_token_ids=constrained_extra_token_ids,
                stop_on_action_chars=stop_on_action_chars,
                action_char_min_chars=action_char_min_chars,
                action_char_plateau_tokens=action_char_plateau_tokens,
                action_char_stable_checks=action_char_stable_checks,
                action_char_stable_tolerance=action_char_stable_tolerance,
                action_char_stable_max_chars=action_char_stable_max_chars,
                action_char_plateau_reject_eos_restart=action_char_plateau_reject_eos_restart,
                action_char_restart_continue=action_char_restart_continue,
                action_char_restart_reset=action_char_restart_reset,
                action_char_restart_reset_low_text_tokens=action_char_restart_reset_low_text_tokens,
                action_char_plateau_low_text_tail_tokens=action_char_plateau_low_text_tail_tokens,
                action_char_strict_target_stop=action_char_strict_target_stop,
                action_char_reset_requires_action_end=action_char_reset_requires_action_end,
                action_char_target_confirm_tokens=action_char_target_confirm_tokens,
            )
            mode = "action_end_constrained_charstop_no_logits" if stop_on_action_chars else "action_end_constrained_no_logits"
        elif self.policy.config.use_kv_cache:
            token_ids = self.sample_actions_fast_kv_cache_action_end(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_steps,
                temperature=decode_temperature,
                force_action_prefix=force_action_prefix,
                stop_on_action_chars=stop_on_action_chars,
                action_char_min_chars=action_char_min_chars,
                action_char_plateau_tokens=action_char_plateau_tokens,
                action_char_stable_checks=action_char_stable_checks,
                action_char_stable_tolerance=action_char_stable_tolerance,
                action_char_stable_max_chars=action_char_stable_max_chars,
                action_char_plateau_reject_eos_restart=action_char_plateau_reject_eos_restart,
                action_char_restart_continue=action_char_restart_continue,
                action_char_restart_reset=action_char_restart_reset,
                action_char_restart_reset_low_text_tokens=action_char_restart_reset_low_text_tokens,
                action_char_plateau_low_text_tail_tokens=action_char_plateau_low_text_tail_tokens,
                action_char_strict_target_stop=action_char_strict_target_stop,
                action_char_reset_requires_action_end=action_char_reset_requires_action_end,
                action_char_target_confirm_tokens=action_char_target_confirm_tokens,
            )
            mode = "action_end_prefix_no_logits" if force_action_prefix else "action_end_no_logits"
        else:
            token_ids, _logits, _hidden = self.sample_actions_fast_with_logits(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_steps,
                temperature=decode_temperature,
                early_stop_action_end=True,
            )
            mode = "action_end_no_logits"

        actions = self._detokenize_generated_actions(token_ids)
        empty_logits = torch.empty((token_ids.shape[0], 0, 0), dtype=torch.float32, device=token_ids.device)
        return PI0FastGenerationTrace(
            actions=actions,
            token_ids=token_ids,
            logits=empty_logits,
            stats={
                "mode": mode,
                "emitted_tokens": int(token_ids.shape[1]),
                "row_token_counts": getattr(self, "_last_action_end_row_token_counts", None),
                "stopped_on_action_end": int(token_ids.shape[1]) < int(max_steps),
                "action_end_token_id": self._action_end_token_id(),
                "constrained_action_vocab": int(bool(constrained_action_vocab)),
                "constrained_candidate_size": int(getattr(self, "_last_constrained_candidate_size", 0)),
                "constrained_action_vocab_size": int(getattr(self, "_last_constrained_action_vocab_size", 0)),
                "constrained_text_vocab_size": int(getattr(self, "_last_constrained_text_vocab_size", 0)),
                "constrained_structural_token_radius": int(
                    getattr(self, "_last_constrained_structural_token_radius", 0)
                ),
                "constrained_force_action_prefix": int(bool(getattr(self, "_last_constrained_force_action_prefix", False))),
                "constrained_full_head_margin": float(getattr(self, "_last_constrained_full_head_margin", -1.0)),
                "constrained_full_head_prefix_tokens": int(
                    getattr(self, "_last_constrained_full_head_prefix_tokens", 0)
                ),
                "constrained_extra_token_count": int(getattr(self, "_last_constrained_extra_token_count", 0)),
                "constrained_full_head_prefix_calls": int(
                    getattr(self, "_last_constrained_full_head_prefix_calls", 0)
                ),
                "constrained_full_head_fallbacks": int(getattr(self, "_last_constrained_full_head_fallbacks", 0)),
                "constrained_restricted_head_calls": int(getattr(self, "_last_constrained_restricted_head_calls", 0)),
                "constrained_full_head_fallback_rate": float(
                    getattr(self, "_last_constrained_full_head_fallback_rate", 0.0)
                ),
                "constrained_margin_min": float(getattr(self, "_last_constrained_margin_min", 0.0)),
                "constrained_margin_mean": float(getattr(self, "_last_constrained_margin_mean", 0.0)),
                "forced_action_prefix": int(bool(force_action_prefix)),
                "forced_action_prefix_tokens": int(getattr(self, "_last_forced_action_prefix_token_count", 0)),
                "stop_on_action_chars": int(bool(stop_on_action_chars)),
                "action_char_target": int(getattr(self, "_last_action_char_target", 0)),
                "action_char_min_chars": int(getattr(self, "_last_action_char_min_chars", 0)),
                "action_char_plateau_tokens": int(getattr(self, "_last_action_char_plateau_tokens", 0)),
                "action_char_stable_checks": int(getattr(self, "_last_action_char_stable_checks", 0)),
                "action_char_stable_tolerance": float(getattr(self, "_last_action_char_stable_tolerance", 0.0)),
                "action_char_stable_max_chars": int(getattr(self, "_last_action_char_stable_max_chars", 0)),
                "action_char_plateau_reject_eos_restart": int(
                    bool(getattr(self, "_last_action_char_plateau_reject_eos_restart", False))
                ),
                "action_char_restart_continue": int(
                    bool(getattr(self, "_last_action_char_restart_continue", False))
                ),
                "action_char_restart_reset": int(bool(getattr(self, "_last_action_char_restart_reset", False))),
                "action_char_restart_reset_low_text_tokens": int(
                    getattr(self, "_last_action_char_restart_reset_low_text_tokens", 0)
                ),
                "action_char_plateau_low_text_tail_tokens": int(
                    getattr(self, "_last_action_char_plateau_low_text_tail_tokens", 0)
                ),
                "action_char_strict_target_stop": int(
                    bool(getattr(self, "_last_action_char_strict_target_stop", False))
                ),
                "action_char_reset_requires_action_end": int(
                    bool(getattr(self, "_last_action_char_reset_requires_action_end", False))
                ),
                "action_char_target_confirm_tokens": int(
                    getattr(self, "_last_action_char_target_confirm_tokens", 0)
                ),
                "action_char_stop_count": int(getattr(self, "_last_action_char_stop_count", 0)),
                "action_char_plateau_stop_count": int(getattr(self, "_last_action_char_plateau_stop_count", 0)),
                "action_char_plateau_stability_block_count": int(
                    getattr(self, "_last_action_char_plateau_stability_block_count", 0)
                ),
                "action_char_plateau_eos_restart_block_count": int(
                    getattr(self, "_last_action_char_plateau_eos_restart_block_count", 0)
                ),
                "action_char_restart_fallback_count": int(
                    getattr(self, "_last_action_char_restart_fallback_count", 0)
                ),
                "action_char_restart_continue_count": int(
                    getattr(self, "_last_action_char_restart_continue_count", 0)
                ),
                "action_char_restart_reset_count": int(getattr(self, "_last_action_char_restart_reset_count", 0)),
                "action_char_reset_requires_action_end_block_count": int(
                    getattr(self, "_last_action_char_reset_requires_action_end_block_count", 0)
                ),
                "action_char_target_confirm_block_count": int(
                    getattr(self, "_last_action_char_target_confirm_block_count", 0)
                ),
                "action_char_plateau_low_text_tail_block_count": int(
                    getattr(self, "_last_action_char_plateau_low_text_tail_block_count", 0)
                ),
                "action_char_count_mean": float(getattr(self, "_last_action_char_count_mean", 0.0)),
                "action_char_stable_snapshot_count_mean": float(
                    getattr(self, "_last_action_char_stable_snapshot_count_mean", 0.0)
                ),
                "action_char_stable_count_mean": float(getattr(self, "_last_action_char_stable_count_mean", 0.0)),
                "action_char_lookup_raw_limit": int(getattr(self, "_action_char_length_raw_limit", 0)),
            },
        )

    @torch.no_grad()
    def fast_prefix_hidden(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the BOS FAST-action hidden state without decoding action tokens."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        bsize = tokens.shape[0]
        device = tokens.device
        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens_in,
            masks_in,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), _past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        return prefix_out[:, -1, :].float()

    @torch.no_grad()
    def verify_draft_tokens(self, batch: dict[str, torch.Tensor], draft_token_ids: torch.Tensor) -> PI0FastVerifyResult:
        """Teacher-force a FAST-token block and return greedy target agreement.

        ``draft_token_ids`` must be in the PaliGemma token space produced by
        ``sample_actions_fast*``. The returned logits align one-to-one with the
        draft positions: logits[:, i] predicts ``draft_token_ids[:, i]``.
        """

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        draft = self._ensure_2d_long(draft_token_ids, tokens.device)
        bsize, draft_len = draft.shape
        if draft_len == 0:
            empty_logits = torch.empty((bsize, 0, 0), device=tokens.device)
            return PI0FastVerifyResult(draft, draft.clone(), empty_logits, 0)

        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=tokens.device)], dim=1)

        # Include draft[:K-1] as context. The BOS position predicts draft[0];
        # draft[i-1] positions predict draft[i].
        context_tokens = draft[:, :-1]
        context_masks = torch.ones_like(context_tokens, dtype=torch.bool)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, num_fast_embs = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens_in,
            masks_in,
            fast_action_tokens=context_tokens if draft_len > 1 else None,
            fast_action_masks=context_masks if draft_len > 1 else None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), _past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=[None, None],
        )

        start = prefix_out.shape[1] - num_fast_embs - 1
        hidden_for_pred = prefix_out[:, start : start + draft_len, :]
        logits = lm_head(hidden_for_pred)
        target_token_ids = logits.argmax(dim=-1)
        accepted = self._accepted_prefix(target_token_ids, draft)
        return PI0FastVerifyResult(
            draft_token_ids=draft,
            target_token_ids=target_token_ids,
            logits=logits,
            accepted_prefix=accepted,
        )

    @torch.no_grad()
    def tokenize_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode raw continuous actions into generated PI0-FAST token ids.

        The returned ids are in PaliGemma token space and match the generated
        sequence shape used by ``predict_action_chunk_with_trace``: no BOS,
        includes the ``Action: `` prefix, action tokens, and the ``|`` marker.
        """

        if self.policy.action_tokenizer is None or self.model._paligemma_tokenizer is None:
            raise ValueError("PI0-FAST action tokenizers are not initialized")
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,H,D] or [H,D], got {tuple(actions.shape)}")
        device = actions.device
        paligemma = self.model._paligemma_tokenizer
        prefix = torch.tensor(paligemma.encode("Action: ", add_special_tokens=False), dtype=torch.long, device=device)
        end = torch.tensor(paligemma.encode("|", add_special_tokens=False), dtype=torch.long, device=device)
        rows: list[torch.Tensor] = []
        for idx in range(actions.shape[0]):
            action_cpu = actions[idx : idx + 1].detach().cpu()
            action_tokens = self.policy.action_tokenizer(action_cpu)
            if not torch.is_tensor(action_tokens):
                action_tokens = torch.tensor(action_tokens, dtype=torch.long)
            action_tokens = action_tokens.flatten().to(device=device, dtype=torch.long)
            paligemma_action = paligemma.vocab_size - 1 - self.policy.config.fast_skip_tokens - action_tokens
            rows.append(torch.cat([prefix, paligemma_action, end], dim=0))
        max_len = max(int(row.numel()) for row in rows)
        padded = []
        for row in rows:
            if row.numel() < max_len:
                row = torch.nn.functional.pad(row, (0, max_len - row.numel()), value=0)
            padded.append(row)
        return torch.stack(padded, dim=0)

    @torch.no_grad()
    def predict_action_chunk_ngram_speculative(
        self,
        batch: dict[str, torch.Tensor],
        drafter: Any,
        lookahead: int = 8,
        reuse_full_blocks: bool = False,
        min_verify_margin: float = 0.0,
        verify_from_scratch: bool = False,
        emit_bonus_token: bool = False,
        defer_correction_token: bool = False,
        dynamic_lookahead: bool = False,
        min_lookahead: int = 1,
        lookahead_growth: int = 1,
        lookahead_shrink: int = 4,
        tree_width: int = 1,
        tree_branch_width: int = 4,
        dynamic_tree_width: bool = False,
        min_tree_width: int = 1,
        tree_width_growth: int = 1,
        tree_width_shrink: int = 1,
        tree_anchor_target_token: bool = False,
        tree_anchor_target_continuation: bool = False,
        early_stop_action_end: bool = False,
        replay_accepted_cache: bool = False,
        resync_accepted_cache: bool = False,
        diagnose_verify_alignment: bool = False,
    ) -> PI0FastGenerationTrace:
        """Return a PI0-FAST action chunk decoded with exact n-gram speculation."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        token_ids, logits, stats = self.sample_actions_fast_ngram_speculative(
            images,
            img_masks,
            tokens,
            masks,
            drafter=drafter,
            max_decoding_steps=self.policy.config.max_decoding_steps,
            lookahead=lookahead,
            temperature=0.0,
            reuse_full_blocks=reuse_full_blocks,
            min_verify_margin=min_verify_margin,
            verify_from_scratch=verify_from_scratch,
            emit_bonus_token=emit_bonus_token,
            defer_correction_token=defer_correction_token,
            dynamic_lookahead=dynamic_lookahead,
            min_lookahead=min_lookahead,
            lookahead_growth=lookahead_growth,
            lookahead_shrink=lookahead_shrink,
            tree_width=tree_width,
            tree_branch_width=tree_branch_width,
            dynamic_tree_width=dynamic_tree_width,
            min_tree_width=min_tree_width,
            tree_width_growth=tree_width_growth,
            tree_width_shrink=tree_width_shrink,
            tree_anchor_target_token=tree_anchor_target_token,
            tree_anchor_target_continuation=tree_anchor_target_continuation,
            early_stop_action_end=early_stop_action_end,
            replay_accepted_cache=replay_accepted_cache,
            resync_accepted_cache=resync_accepted_cache,
            diagnose_verify_alignment=diagnose_verify_alignment,
        )
        actions = self._detokenize_generated_actions(token_ids)
        return PI0FastGenerationTrace(actions=actions, token_ids=token_ids, logits=logits, stats=stats)

    @torch.no_grad()
    def predict_action_chunk_medusa_speculative(
        self,
        batch: dict[str, torch.Tensor],
        medusa_head: Any,
        token_map: Any,
        lookahead: int = 4,
        min_draft_confidence: float = 0.0,
        min_verify_confidence: float = 0.0,
        min_spec_position: int = 0,
        accept_partial_blocks: bool = True,
        replay_accepted_cache: bool = False,
        resync_accepted_cache: bool = False,
        verify_from_scratch: bool = False,
        early_stop_action_end: bool = True,
    ) -> PI0FastGenerationTrace:
        """Return a PI0-FAST action chunk decoded with exact Medusa speculation."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        token_ids, logits, stats = self.sample_actions_fast_medusa_speculative(
            images,
            img_masks,
            tokens,
            masks,
            medusa_head=medusa_head,
            token_map=token_map,
            max_decoding_steps=self.policy.config.max_decoding_steps,
            lookahead=lookahead,
            min_draft_confidence=min_draft_confidence,
            min_verify_confidence=min_verify_confidence,
            min_spec_position=min_spec_position,
            accept_partial_blocks=accept_partial_blocks,
            replay_accepted_cache=replay_accepted_cache,
            resync_accepted_cache=resync_accepted_cache,
            verify_from_scratch=verify_from_scratch,
            early_stop_action_end=early_stop_action_end,
        )
        actions = self._detokenize_generated_actions(token_ids)
        return PI0FastGenerationTrace(actions=actions, token_ids=token_ids, logits=logits, stats=stats)

    @torch.no_grad()
    def predict_action_chunk_draft_transformer_speculative(
        self,
        batch: dict[str, torch.Tensor],
        draft_model: Any,
        token_map: Any,
        lookahead: int = 4,
        min_draft_confidence: float = 0.0,
        min_spec_position: int = 0,
        early_stop_action_end: bool = True,
        accept_partial_blocks: bool = False,
        debug_event_limit: int = 64,
    ) -> PI0FastGenerationTrace:
        """Return a PI0-FAST action chunk decoded with a small transformer drafter."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        token_ids, logits, stats = self.sample_actions_fast_draft_transformer_speculative(
            images,
            img_masks,
            tokens,
            masks,
            draft_model=draft_model,
            token_map=token_map,
            max_decoding_steps=self.policy.config.max_decoding_steps,
            lookahead=lookahead,
            min_draft_confidence=min_draft_confidence,
            min_spec_position=min_spec_position,
            early_stop_action_end=early_stop_action_end,
            accept_partial_blocks=accept_partial_blocks,
            debug_event_limit=debug_event_limit,
        )
        actions = self._detokenize_generated_actions(token_ids)
        return PI0FastGenerationTrace(actions=actions, token_ids=token_ids, logits=logits, stats=stats)

    @torch.no_grad()
    def predict_action_chunk_block_speculative(
        self,
        batch: dict[str, torch.Tensor],
        block_drafter: Any,
        token_map: Any,
        block_gate: Any | None = None,
        lookahead: int = 7,
        min_draft_confidence: float = 0.0,
        min_verify_confidence: float = 0.0,
        min_verify_margin: float = 0.0,
        block_gate_threshold: float = 0.0,
        max_future_accept: int | None = None,
        min_future_accept: int = 0,
        min_spec_position: int = 0,
        reject_cooldown_steps: int = 0,
        reject_cooldown_after: int = 1,
        spec_fallback_cooldown_steps: int = 0,
        spec_fallback_cooldown_after: int = 0,
        allow_unknown_context: bool = False,
        repeat_token_draft: bool = False,
        repeat_token_min_run: int = 2,
        repeat_pattern_draft: bool = False,
        repeat_pattern_max_period: int = 8,
        repeat_pattern_min_position: int = 0,
        pattern_only: bool = False,
        unverified_pattern_tail: bool = False,
        unverified_pattern_eos: bool = False,
        full_block_only: bool = False,
        early_stop_action_end: bool = True,
        accept_partial_blocks: bool = True,
        refine_steps: int = 1,
        verify_from_scratch: bool = False,
        resync_accepted_cache: bool = False,
        draft_after_known_token: bool = False,
        max_decoding_steps: int | None = None,
        force_action_end: bool = False,
        debug_event_limit: int = 64,
    ) -> PI0FastGenerationTrace:
        """Return a PI0-FAST action chunk decoded with masked-block speculation."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        token_ids, logits, stats = self.sample_actions_fast_block_speculative(
            images,
            img_masks,
            tokens,
            masks,
            block_drafter=block_drafter,
            token_map=token_map,
            block_gate=block_gate,
            max_decoding_steps=max_decoding_steps or self.policy.config.max_decoding_steps,
            lookahead=lookahead,
            min_draft_confidence=min_draft_confidence,
            min_verify_confidence=min_verify_confidence,
            min_verify_margin=min_verify_margin,
            block_gate_threshold=block_gate_threshold,
            max_future_accept=max_future_accept,
            min_future_accept=min_future_accept,
            min_spec_position=min_spec_position,
            reject_cooldown_steps=reject_cooldown_steps,
            reject_cooldown_after=reject_cooldown_after,
            spec_fallback_cooldown_steps=spec_fallback_cooldown_steps,
            spec_fallback_cooldown_after=spec_fallback_cooldown_after,
            allow_unknown_context=allow_unknown_context,
            repeat_token_draft=repeat_token_draft,
            repeat_token_min_run=repeat_token_min_run,
            repeat_pattern_draft=repeat_pattern_draft,
            repeat_pattern_max_period=repeat_pattern_max_period,
            repeat_pattern_min_position=repeat_pattern_min_position,
            pattern_only=pattern_only,
            unverified_pattern_tail=unverified_pattern_tail,
            unverified_pattern_eos=unverified_pattern_eos,
            full_block_only=full_block_only,
            early_stop_action_end=early_stop_action_end,
            accept_partial_blocks=accept_partial_blocks,
            refine_steps=refine_steps,
            verify_from_scratch=verify_from_scratch,
            resync_accepted_cache=resync_accepted_cache,
            draft_after_known_token=draft_after_known_token,
            debug_event_limit=debug_event_limit,
        )
        if force_action_end:
            action_end = self._action_end_token_id()
            if token_ids.shape[1] == 0 or int(token_ids[0, -1].item()) != action_end:
                eos = torch.tensor([[action_end]], dtype=token_ids.dtype, device=token_ids.device)
                token_ids = torch.cat([token_ids, eos], dim=1)
            stats = {
                **(stats or {}),
                "forced_action_end": 1.0,
                "forced_cutoff_tokens": int(max_decoding_steps or self.policy.config.max_decoding_steps),
                "emitted_tokens": int(token_ids.shape[1]),
            }
        actions = self._detokenize_generated_actions(token_ids)
        return PI0FastGenerationTrace(actions=actions, token_ids=token_ids, logits=logits, stats=stats)

    @torch.no_grad()
    def predict_action_chunk_prefix_cutoff(
        self,
        batch: dict[str, torch.Tensor],
        cutoff_tokens: int,
        early_stop_action_end: bool = True,
        collect_logits: bool = False,
        force_action_prefix: bool = False,
    ) -> PI0FastGenerationTrace:
        """Decode a bounded FAST-token prefix, then append action-end if needed."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        if collect_logits or not self.policy.config.use_kv_cache:
            if force_action_prefix:
                raise ValueError("Forced PI0-FAST action prefix is only supported on the no-logits cutoff path")
            if self.policy.config.use_kv_cache:
                token_ids, logits, hidden_states = self.sample_actions_fast_kv_cache_with_logits(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    max_decoding_steps=cutoff_tokens,
                    temperature=0.0,
                    return_hidden_states=False,
                    early_stop_action_end=early_stop_action_end,
                )
            else:
                token_ids, logits, hidden_states = self.sample_actions_fast_with_logits(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    max_decoding_steps=cutoff_tokens,
                    temperature=0.0,
                    return_hidden_states=False,
                    early_stop_action_end=early_stop_action_end,
                )
            mode = "prefix_cutoff_with_logits"
        else:
            token_ids = self.sample_actions_fast_kv_cache_action_end(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=cutoff_tokens,
                temperature=0.0,
                force_action_prefix=force_action_prefix,
            )
            logits = torch.empty((token_ids.shape[0], 0, 0), dtype=torch.float32, device=token_ids.device)
            hidden_states = None
            mode = "prefix_cutoff_prefix_no_logits" if force_action_prefix else "prefix_cutoff_no_logits"
        action_end = self._action_end_token_id()
        pre_append_tokens = int(token_ids.shape[1])
        ended = token_ids.shape[1] > 0 and bool(torch.all(token_ids[:, -1] == action_end))
        forced_end = int(not ended)
        if forced_end:
            eos = torch.full((token_ids.shape[0], 1), action_end, dtype=token_ids.dtype, device=token_ids.device)
            token_ids = torch.cat([token_ids, eos], dim=1)
        actions = self._detokenize_generated_actions(token_ids)
        stats = {
            "mode": mode,
            "cutoff_tokens": int(cutoff_tokens),
            "emitted_tokens": int(token_ids.shape[1]),
            "pre_append_tokens": pre_append_tokens,
            "forced_action_end": forced_end,
            "stopped_on_action_end": int(ended),
            "row_token_counts": getattr(self, "_last_action_end_row_token_counts", None),
            "forced_action_prefix": int(bool(force_action_prefix)),
            "forced_action_prefix_tokens": int(getattr(self, "_last_forced_action_prefix_token_count", 0)),
        }
        return PI0FastGenerationTrace(
            actions=actions,
            token_ids=token_ids,
            logits=logits,
            hidden_states=hidden_states,
            stats=stats,
        )

    @torch.no_grad()
    def predict_action_chunk_adaptive_prefix_cutoff(
        self,
        batch: dict[str, torch.Tensor],
        checkpoints: list[int],
        stable_tolerance: float = 0.0,
        stable_checks: int = 1,
        early_stable_checks: int | None = None,
        early_max_stable_checkpoint: int | None = None,
        checkpoint_stable_checks: dict[int, int] | None = None,
        max_stable_checkpoint: int | None = None,
        skip_unproductive_checks: bool = False,
        skip_unproductive_after_checkpoint: int = 0,
        continue_to_action_end_on_unstable: bool = False,
        stability_continue_to_action_end: bool = False,
        stability_risk_gate: dict[str, Any] | None = None,
        prefix_gate: Any | None = None,
        prefix_gate_threshold: float = 0.98,
        early_stop_action_end: bool = True,
    ) -> PI0FastGenerationTrace:
        """Decode until prefix+action-end detokenized actions stabilize."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
        if not checkpoints:
            raise ValueError("Adaptive prefix cutoff needs at least one checkpoint")
        checkpoints = sorted({int(c) for c in checkpoints if int(c) > 0})
        max_decoding_steps = max(checkpoints)
        bsize = tokens.shape[0]
        if bsize != 1:
            raise ValueError("Adaptive prefix cutoff currently supports batch size 1")

        device = tokens.device
        action_end = self._action_end_token_id()
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )
        prev_logits = lm_head(prefix_out[:, -1:, :])
        current_pad_mask = prefix_pad_masks
        generated_tokens: list[int] = []
        logits_by_step: list[torch.Tensor] = []
        checkpoint_idx = 0
        previous_actions: torch.Tensor | None = None
        stable_count = 0
        action_snapshots = 0
        checked: list[dict[str, Any]] = []
        continuing_after_stability_stop: dict[str, float] | None = None
        stability_risk_gate_rejections = 0

        def candidate_trace() -> tuple[torch.Tensor, torch.Tensor]:
            token_ids = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            if token_ids.shape[1] == 0 or int(token_ids[0, -1].item()) != action_end:
                eos = torch.tensor([[action_end]], dtype=token_ids.dtype, device=device)
                token_ids = torch.cat([token_ids, eos], dim=1)
            return token_ids, self._detokenize_generated_actions(token_ids)

        def candidate_feature_row(
            checkpoint: int,
            token_ids: torch.Tensor,
            actions: torch.Tensor,
            *,
            max_delta: float | None,
            stable_count: int,
            required_stable_checks: int,
        ) -> dict[str, float]:
            steps = min(token_ids.shape[1], len(logits_by_step))
            if steps:
                logits = torch.cat(logits_by_step[:steps], dim=1).float()
                ids = token_ids[:, :steps].to(logits.device)
                log_probs = F.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                token_stats = {
                    "logprob_mean": float(log_probs.mean().item()),
                    "logprob_min": float(log_probs.min().item()),
                    "entropy_mean": float(entropy.mean().item()),
                    "entropy_max": float(entropy.max().item()),
                }
            else:
                token_stats = {"logprob_mean": 0.0, "logprob_min": 0.0, "entropy_mean": 0.0, "entropy_max": 0.0}
            forced_eos = float(token_ids.shape[1] > len(generated_tokens))
            row = {
                "cutoff_norm": checkpoint / 256.0,
                "token_count_norm": int(token_ids.shape[1]) / 256.0,
                "forced_eos": forced_eos,
                **token_stats,
                **action_feature_values(actions.detach().float().cpu().numpy()),
            }
            row.update(
                {
                    "checkpoint": float(checkpoint),
                    "token_count": float(token_ids.shape[1]),
                    "max_delta": 0.0 if max_delta is None else float(max_delta),
                    "stable_count": float(stable_count),
                    "required_stable_checks": float(required_stable_checks),
                }
            )
            return row

        def prefixed_feature_stats(row: dict[str, float]) -> dict[str, float]:
            return {f"stability_stop_{key}": float(value) for key, value in row.items()}

        def gate_probability(row: dict[str, float]) -> float | None:
            if prefix_gate is None:
                return None
            features = torch.tensor(
                [[float(row.get(name, 0.0)) for name in PREFIX_GATE_FEATURES]],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                return float(prefix_gate.probability(features).item())

        def risk_gate_clauses() -> list[dict[str, float]]:
            if not stability_risk_gate:
                return []
            clauses = stability_risk_gate.get("clauses")
            if isinstance(clauses, list):
                return [clause for clause in clauses if isinstance(clause, dict)]
            return [stability_risk_gate]

        def risk_gate_rejects_clause(row: dict[str, float], clause: dict[str, float]) -> bool:
            checks = {
                "min_checkpoint": ("checkpoint", ">="),
                "max_checkpoint": ("checkpoint", "<="),
                "min_token_count": ("token_count", ">="),
                "max_token_count": ("token_count", "<="),
                "max_logprob_mean": ("logprob_mean", "<="),
                "min_entropy_mean": ("entropy_mean", ">="),
                "min_action_abs_max": ("action_abs_max", ">="),
                "max_action_abs_max": ("action_abs_max", "<="),
                "min_position_span": ("position_span", ">="),
                "max_position_span": ("position_span", "<="),
                "min_rotation_span": ("rotation_span", ">="),
                "max_rotation_span": ("rotation_span", "<="),
                "min_max_step_delta": ("max_step_delta", ">="),
                "max_max_step_delta": ("max_step_delta", "<="),
            }
            matched = False
            for threshold_name, (feature_name, op) in checks.items():
                if threshold_name not in clause:
                    continue
                matched = True
                value = float(row.get(feature_name, 0.0))
                threshold = float(clause[threshold_name])
                if op == ">=" and value < threshold:
                    return False
                if op == "<=" and value > threshold:
                    return False
            return matched

        def risk_gate_rejects(row: dict[str, float]) -> bool:
            return any(risk_gate_rejects_clause(row, clause) for clause in risk_gate_clauses())

        def risk_gate_stats() -> dict[str, float]:
            if stability_risk_gate_rejections <= 0:
                return {}
            return {
                "stability_risk_gate_rejections": float(stability_risk_gate_rejections),
            }

        while len(generated_tokens) < max_decoding_steps:
            next_token = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)
            logits_by_step.append(prev_logits)
            generated_tokens.append(int(next_token.item()))
            if early_stop_action_end and int(next_token.item()) == action_end:
                token_ids = torch.tensor([generated_tokens], dtype=torch.long, device=device)
                actions = self._detokenize_generated_actions(token_ids)
                stats = {
                    "mode": "adaptive_prefix_cutoff",
                    "emitted_tokens": int(token_ids.shape[1]),
                    "checked": checked,
                    "stopped_on_stability": False,
                    "stopped_on_action_end": True,
                    **risk_gate_stats(),
                }
                if continuing_after_stability_stop is not None:
                    stats.update(
                        {
                            "continued_after_stability_stop": True,
                            **continuing_after_stability_stop,
                        }
                    )
                return PI0FastGenerationTrace(actions=actions, token_ids=token_ids, logits=torch.cat(logits_by_step, dim=1), stats=stats)

            while checkpoint_idx < len(checkpoints) and len(generated_tokens) >= checkpoints[checkpoint_idx]:
                if (
                    skip_unproductive_checks
                    and previous_actions is not None
                    and stable_checks > 1
                    and checkpoints[checkpoint_idx] >= skip_unproductive_after_checkpoint
                    and action_snapshots >= stable_checks
                    and stable_count == 0
                ):
                    checked.append(
                        {
                            "checkpoint": checkpoints[checkpoint_idx],
                            "tokens": int(len(generated_tokens) + 1),
                            "max_delta": None,
                            "stable_count": stable_count,
                            "skipped": True,
                        }
                    )
                    checkpoint_idx += 1
                    continue

                token_ids, actions = candidate_trace()
                max_delta = None
                if previous_actions is not None:
                    max_delta = float(torch.max(torch.abs(actions.float() - previous_actions.float())).item())
                    if max_delta <= stable_tolerance:
                        stable_count += 1
                    else:
                        stable_count = 0
                previous_actions = actions
                action_snapshots += 1
                checked.append(
                    {
                        "checkpoint": checkpoints[checkpoint_idx],
                        "tokens": int(token_ids.shape[1]),
                        "max_delta": max_delta,
                        "stable_count": stable_count,
                    }
                )
                checkpoint_idx += 1
                checkpoint = checkpoints[checkpoint_idx - 1]
                required_stable_checks = stable_checks
                if (
                    early_stable_checks is not None
                    and early_max_stable_checkpoint is not None
                    and checkpoint <= early_max_stable_checkpoint
                ):
                    required_stable_checks = early_stable_checks
                if checkpoint_stable_checks and checkpoint in checkpoint_stable_checks:
                    required_stable_checks = checkpoint_stable_checks[checkpoint]
                can_stop_at_checkpoint = max_stable_checkpoint is None or checkpoint <= max_stable_checkpoint
                if (
                    stable_count >= required_stable_checks
                    and can_stop_at_checkpoint
                    and continuing_after_stability_stop is None
                ):
                    feature_row = candidate_feature_row(
                        checkpoint,
                        token_ids,
                        actions,
                        max_delta=max_delta,
                        stable_count=stable_count,
                        required_stable_checks=required_stable_checks,
                    )
                    feature_stats = prefixed_feature_stats(feature_row)
                    if stability_continue_to_action_end:
                        checked[-1]["continued_to_action_end"] = True
                        continuing_after_stability_stop = {
                            "stability_continue_checkpoint": float(checkpoint),
                            "stability_continue_required_checks": float(required_stable_checks),
                            "stability_continue_stable_count": float(stable_count),
                            **feature_stats,
                        }
                        continue
                    if risk_gate_rejects(feature_row):
                        stability_risk_gate_rejections += 1
                        checked[-1]["risk_gate_rejected"] = True
                        checked[-1]["risk_gate_rejection_index"] = stability_risk_gate_rejections
                        continue
                    gate_prob = gate_probability(feature_row)
                    if gate_prob is not None and gate_prob < prefix_gate_threshold:
                        checked[-1]["gate_probability"] = gate_prob
                        checked[-1]["gate_rejected"] = True
                        continue
                    if gate_prob is not None:
                        checked[-1]["gate_probability"] = gate_prob
                    stats = {
                        "mode": "adaptive_prefix_cutoff",
                        "emitted_tokens": int(token_ids.shape[1]),
                        "checked": checked,
                        "stopped_on_stability": True,
                        "stopped_on_action_end": False,
                        "stable_stop_checkpoint": float(checkpoint),
                        "stable_stop_required_checks": float(required_stable_checks),
                        "stable_stop_stable_count": float(stable_count),
                        **risk_gate_stats(),
                        **feature_stats,
                    }
                    return PI0FastGenerationTrace(
                        actions=actions,
                        token_ids=token_ids,
                        logits=torch.cat(logits_by_step, dim=1),
                        stats=stats,
                    )

            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(current_pad_mask.unsqueeze(1), dtype=next_token_emb.dtype)
            (step_out, _), past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[next_token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            prev_logits = lm_head(step_out[:, -1:, :])

        if continue_to_action_end_on_unstable or continuing_after_stability_stop is not None:
            max_action_tokens = int(self.model.config.max_action_tokens)
            while len(generated_tokens) < max_action_tokens:
                next_token = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)
                logits_by_step.append(prev_logits)
                generated_tokens.append(int(next_token.item()))
                if early_stop_action_end and int(next_token.item()) == action_end:
                    token_ids = torch.tensor([generated_tokens], dtype=torch.long, device=device)
                    actions = self._detokenize_generated_actions(token_ids)
                    stats = {
                        "mode": "adaptive_prefix_cutoff",
                        "emitted_tokens": int(token_ids.shape[1]),
                        "checked": checked,
                        "stopped_on_stability": False,
                        "stopped_on_action_end": True,
                        "continued_on_unstable": True,
                        **risk_gate_stats(),
                    }
                    if continuing_after_stability_stop is not None:
                        stats.update(
                            {
                                "continued_after_stability_stop": True,
                                **continuing_after_stability_stop,
                            }
                        )
                    return PI0FastGenerationTrace(
                        actions=actions,
                        token_ids=token_ids,
                        logits=torch.cat(logits_by_step, dim=1),
                        stats=stats,
                    )

                next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
                next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
                next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
                current_pad_mask = torch.cat(
                    [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                    dim=1,
                )
                current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
                step_att_mask = self.model._prepare_attention_masks_4d(
                    current_pad_mask.unsqueeze(1),
                    dtype=next_token_emb.dtype,
                )
                (step_out, _), past_key_values = self._forward_prefix_language_model(
                    attention_mask=step_att_mask,
                    position_ids=current_position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=next_token_emb,
                    use_cache=True,
                    cache_position=torch.tensor([current_pad_mask.shape[1] - 1], device=device, dtype=torch.long),
                )
                prev_logits = lm_head(step_out[:, -1:, :])

            token_ids = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            actions = self._detokenize_generated_actions(token_ids)
            stats = {
                "mode": "adaptive_prefix_cutoff",
                "emitted_tokens": int(token_ids.shape[1]),
                "checked": checked,
                "stopped_on_stability": False,
                "stopped_on_action_end": False,
                "continued_on_unstable": True,
                **risk_gate_stats(),
            }
            if continuing_after_stability_stop is not None:
                stats.update(
                    {
                        "continued_after_stability_stop": True,
                        **continuing_after_stability_stop,
                    }
                )
            return PI0FastGenerationTrace(
                actions=actions,
                token_ids=token_ids,
                logits=torch.cat(logits_by_step, dim=1),
                stats=stats,
            )

        token_ids, actions = candidate_trace()
        stats = {
            "mode": "adaptive_prefix_cutoff",
            "emitted_tokens": int(token_ids.shape[1]),
            "checked": checked,
            "stopped_on_stability": False,
            "stopped_on_action_end": False,
            **risk_gate_stats(),
        }
        return PI0FastGenerationTrace(actions=actions, token_ids=token_ids, logits=torch.cat(logits_by_step, dim=1), stats=stats)

    @torch.no_grad()
    def sample_actions_fast_with_logits(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps: int | None = None,
        temperature: float = 0.0,
        return_hidden_states: bool = False,
        early_stop_action_end: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Non-KV FAST decode that returns generated token IDs and logits."""

        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head

        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        generated = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)
        logits_by_step: list[torch.Tensor] = []
        hidden_by_step: list[torch.Tensor] = []
        action_end_token_id = self._action_end_token_id() if early_stop_action_end else None

        for t in range(max_decoding_steps):
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
            (prefix_out, _), _ = self.model.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
            last_logits = lm_head(prefix_out[:, -1:, :])
            logits_by_step.append(last_logits)
            if return_hidden_states:
                hidden_by_step.append(prefix_out[:, -1:, :])
            next_token = self._select_next_token(last_logits, temperature)
            generated[:, t] = next_token.squeeze(-1)
            if action_end_token_id is not None and bool(torch.all(next_token == action_end_token_id)):
                generated = generated[:, : t + 1]
                break

            if t < max_decoding_steps - 1:
                next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
                next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
                next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
                prefix_embs = torch.cat([prefix_embs, next_token_emb], dim=1)
                prefix_pad_masks = torch.cat(
                    [prefix_pad_masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                    dim=1,
                )
                old_len = prefix_att_masks.shape[1]
                new_len = old_len + 1
                new_att_masks = torch.zeros((bsize, new_len, new_len), dtype=torch.bool, device=device)
                new_att_masks[:, :old_len, :old_len] = prefix_att_masks
                new_att_masks[:, -1, :] = prefix_pad_masks
                prefix_att_masks = new_att_masks

        hidden = torch.cat(hidden_by_step, dim=1) if return_hidden_states else None
        return generated, torch.cat(logits_by_step, dim=1), hidden

    @torch.no_grad()
    def sample_actions_fast_kv_cache_with_logits(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps: int | None = None,
        temperature: float = 0.0,
        return_hidden_states: bool = False,
        early_stop_action_end: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """KV-cache FAST decode that returns generated token IDs and logits."""

        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head

        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens_in,
            masks_in,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )
        first_logits = lm_head(prefix_out[:, -1:, :])
        next_token = self._select_next_token(first_logits, temperature)
        generated = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)
        generated[:, 0] = next_token.squeeze(-1)
        logits_by_step = [first_logits]
        hidden_by_step = [prefix_out[:, -1:, :]] if return_hidden_states else []
        current_pad_mask = prefix_pad_masks
        action_end_token_id = self._action_end_token_id() if early_stop_action_end else None
        if action_end_token_id is not None and bool(torch.all(next_token == action_end_token_id)):
            hidden = torch.cat(hidden_by_step, dim=1) if return_hidden_states else None
            return generated[:, :1], torch.cat(logits_by_step, dim=1), hidden

        for t in range(1, max_decoding_steps):
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[next_token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            step_logits = lm_head(step_out[:, -1:, :])
            logits_by_step.append(step_logits)
            if return_hidden_states:
                hidden_by_step.append(step_out[:, -1:, :])
            next_token = self._select_next_token(step_logits, temperature)
            generated[:, t] = next_token.squeeze(-1)
            if action_end_token_id is not None and bool(torch.all(next_token == action_end_token_id)):
                generated = generated[:, : t + 1]
                break

        hidden = torch.cat(hidden_by_step, dim=1) if return_hidden_states else None
        return generated, torch.cat(logits_by_step, dim=1), hidden

    @torch.no_grad()
    def sample_actions_fast_kv_cache_action_end(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps: int | None = None,
        temperature: float = 0.0,
        force_action_prefix: bool = False,
        stop_on_action_chars: bool = False,
        action_char_min_chars: int | None = None,
        action_char_plateau_tokens: int = 0,
        action_char_stable_checks: int = 0,
        action_char_stable_tolerance: float = 0.0,
        action_char_stable_max_chars: int = 0,
        action_char_plateau_reject_eos_restart: bool = False,
        action_char_restart_continue: bool = False,
        action_char_restart_reset: bool = False,
        action_char_restart_reset_low_text_tokens: int = 0,
        action_char_plateau_low_text_tail_tokens: int = 0,
        action_char_strict_target_stop: bool = False,
        action_char_reset_requires_action_end: bool = False,
        action_char_target_confirm_tokens: int = 0,
    ) -> torch.Tensor:
        """KV-cache FAST decode with per-row action-end compaction.

        Rows that emit the FAST action-end token are removed from subsequent
        decode steps.  This keeps batched serving from paying for already
        completed robot requests when another row needs more tokens.
        """

        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        action_end_token_id = self._action_end_token_id()
        action_char_target = 0
        if stop_on_action_chars:
            if temperature != 0.0:
                raise ValueError("PI0-FAST action-char stop currently supports greedy temperature=0 only")
            action_dim = self.policy.config.output_features[self._action_key()].shape[0]
            action_char_target = int(self.policy.config.n_action_steps) * int(action_dim)
        action_char_min_chars = int(action_char_target if action_char_min_chars is None else action_char_min_chars)
        action_char_min_chars = max(0, int(action_char_min_chars))
        action_char_plateau_tokens = max(0, int(action_char_plateau_tokens))
        action_char_stable_checks = max(0, int(action_char_stable_checks))
        action_char_stable_tolerance = max(0.0, float(action_char_stable_tolerance))
        action_char_stable_max_chars = max(0, int(action_char_stable_max_chars))
        action_char_plateau_reject_eos_restart = bool(action_char_plateau_reject_eos_restart)
        action_char_restart_continue = bool(action_char_restart_continue)
        action_char_restart_reset = bool(action_char_restart_reset)
        action_char_restart_reset_low_text_tokens = max(0, int(action_char_restart_reset_low_text_tokens))
        action_char_plateau_low_text_tail_tokens = max(0, int(action_char_plateau_low_text_tail_tokens))
        action_char_strict_target_stop = bool(action_char_strict_target_stop)
        action_char_reset_requires_action_end = bool(action_char_reset_requires_action_end)
        action_char_target_confirm_tokens = max(0, int(action_char_target_confirm_tokens))
        self._last_action_char_target = int(action_char_target)
        self._last_action_char_min_chars = int(action_char_min_chars if stop_on_action_chars else 0)
        self._last_action_char_plateau_tokens = int(action_char_plateau_tokens if stop_on_action_chars else 0)
        self._last_action_char_stable_checks = int(action_char_stable_checks if stop_on_action_chars else 0)
        self._last_action_char_stable_tolerance = float(action_char_stable_tolerance if stop_on_action_chars else 0.0)
        self._last_action_char_stable_max_chars = int(action_char_stable_max_chars if stop_on_action_chars else 0)
        self._last_action_char_plateau_reject_eos_restart = bool(
            action_char_plateau_reject_eos_restart if stop_on_action_chars else False
        )
        self._last_action_char_restart_continue = bool(action_char_restart_continue if stop_on_action_chars else False)
        self._last_action_char_restart_reset = bool(action_char_restart_reset if stop_on_action_chars else False)
        self._last_action_char_restart_reset_low_text_tokens = int(
            action_char_restart_reset_low_text_tokens if stop_on_action_chars else 0
        )
        self._last_action_char_plateau_low_text_tail_tokens = int(
            action_char_plateau_low_text_tail_tokens if stop_on_action_chars else 0
        )
        self._last_action_char_strict_target_stop = bool(
            action_char_strict_target_stop if stop_on_action_chars else False
        )
        self._last_action_char_reset_requires_action_end = bool(
            action_char_reset_requires_action_end if stop_on_action_chars else False
        )
        self._last_action_char_target_confirm_tokens = int(
            action_char_target_confirm_tokens if stop_on_action_chars else 0
        )
        self._last_action_char_stop_count = 0
        self._last_action_char_plateau_stop_count = 0
        self._last_action_char_plateau_stability_block_count = 0
        self._last_action_char_plateau_eos_restart_block_count = 0
        self._last_action_char_restart_fallback_count = 0
        self._last_action_char_restart_continue_count = 0
        self._last_action_char_restart_reset_count = 0
        self._last_action_char_reset_requires_action_end_block_count = 0
        self._last_action_char_target_confirm_block_count = 0
        self._last_action_char_plateau_low_text_tail_block_count = 0
        self._last_action_char_count_mean = 0.0
        self._last_action_char_stable_snapshot_count_mean = 0.0
        self._last_action_char_stable_count_mean = 0.0
        action_char_lookup = self.prepare_action_char_length_lookup(device) if stop_on_action_chars else None
        action_char_prefix_tokens = torch.as_tensor(
            self.model._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ) if stop_on_action_chars else torch.empty((0,), dtype=torch.long, device=device)
        action_char_prefix_low_token_ids = tuple(
            int(token_id)
            for token_id in action_char_prefix_tokens.detach().cpu().tolist()
            if 0 <= int(token_id) < 8192
        )
        prefix_tokens = None
        if force_action_prefix:
            if temperature != 0.0:
                raise ValueError("Forced PI0-FAST action prefix currently supports greedy temperature=0 only")
            prefix_tokens = torch.as_tensor(
                self.model._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
                dtype=torch.long,
                device=device,
            )
            if prefix_tokens.numel() == 0:
                raise RuntimeError("Could not resolve PI0-FAST Action prefix tokens")
        self._last_forced_action_prefix_token_count = int(prefix_tokens.numel()) if prefix_tokens is not None else 0

        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens_in,
            masks_in,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )
        generated_width = int(max_decoding_steps) + (1 if stop_on_action_chars else 0)
        generated = torch.full((bsize, generated_width), action_end_token_id, dtype=torch.long, device=device)
        current_pad_mask = prefix_pad_masks
        active_indices = torch.arange(bsize, dtype=torch.long, device=device)
        emitted_lengths = torch.zeros((bsize,), dtype=torch.long, device=device)
        action_char_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        last_action_char_increment_lengths = torch.zeros((bsize,), dtype=torch.long, device=device)
        action_stable_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        action_stable_snapshot_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        target_reached_lengths = torch.zeros((bsize,), dtype=torch.long, device=device)
        previous_action_snapshots: list[torch.Tensor | None] = [None] * bsize
        stopped_by_chars = torch.zeros((bsize,), dtype=torch.bool, device=device)
        stopped_by_char_plateau = torch.zeros((bsize,), dtype=torch.bool, device=device)
        stopped_by_restart_fallback = torch.zeros((bsize,), dtype=torch.bool, device=device)
        restarted_with_count_reset = torch.zeros((bsize,), dtype=torch.bool, device=device)
        reset_requires_action_end = torch.zeros((bsize,), dtype=torch.bool, device=device)
        pending_restart_reset = torch.zeros((bsize,), dtype=torch.bool, device=device)
        post_restart_action_char_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        post_restart_last_action_char_increment_lengths = torch.zeros((bsize,), dtype=torch.long, device=device)
        restart_low_text_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        action_char_low_text_tail_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        blocked_by_char_plateau_stability = torch.zeros((bsize,), dtype=torch.bool, device=device)
        blocked_by_char_plateau_eos_restart = torch.zeros((bsize,), dtype=torch.bool, device=device)
        blocked_by_char_plateau_low_text_tail = torch.zeros((bsize,), dtype=torch.bool, device=device)
        blocked_by_reset_requires_action_end = torch.zeros((bsize,), dtype=torch.bool, device=device)
        blocked_by_target_confirmation = torch.zeros((bsize,), dtype=torch.bool, device=device)
        plateau_tail_seen_eos = torch.zeros((bsize,), dtype=torch.bool, device=device)
        plateau_tail_eos_restart = torch.zeros((bsize,), dtype=torch.bool, device=device)
        eos_token_id = int(getattr(self.model._paligemma_tokenizer, "eos_token_id", -1) or -1)
        bos_token_id = int(getattr(self.model._paligemma_tokenizer, "bos_token_id", -1) or -1)
        has_restart_guard_tokens = eos_token_id >= 0 or bos_token_id >= 0

        def token_ids_with_action_end(row_idx: int) -> torch.Tensor:
            length = max(int(emitted_lengths[row_idx].item()), 1)
            token_ids = generated[row_idx : row_idx + 1, :length]
            if int(token_ids[0, -1].item()) != action_end_token_id:
                eos = torch.full((1, 1), action_end_token_id, dtype=token_ids.dtype, device=token_ids.device)
                token_ids = torch.cat([token_ids, eos], dim=1)
            return token_ids

        def record_action_stability(row_indices: torch.Tensor) -> None:
            if action_char_stable_checks <= 0:
                return
            for row_idx in row_indices.detach().cpu().tolist():
                token_ids = token_ids_with_action_end(int(row_idx))
                actions = self._detokenize_generated_actions(token_ids).detach().float().cpu()
                previous = previous_action_snapshots[int(row_idx)]
                if previous is None:
                    action_stable_counts[int(row_idx)] = 0
                else:
                    max_delta = float(torch.max(torch.abs(actions - previous)).item())
                    if max_delta <= action_char_stable_tolerance:
                        action_stable_counts[int(row_idx)] += 1
                    else:
                        action_stable_counts[int(row_idx)] = 0
                previous_action_snapshots[int(row_idx)] = actions
                action_stable_snapshot_counts[int(row_idx)] += 1

        def record_action_chars(selected: torch.Tensor) -> torch.Tensor:
            if action_char_lookup is None:
                return torch.zeros_like(selected, dtype=torch.bool)
            increments = action_char_lookup.index_select(0, selected).to(dtype=action_char_counts.dtype)
            low_text = selected < 8192
            if eos_token_id >= 0:
                low_text = low_text & (selected != int(eos_token_id))
            if bos_token_id >= 0:
                low_text = low_text & (selected != int(bos_token_id))
            for token_id in action_char_prefix_low_token_ids:
                low_text = low_text & (selected != token_id)
            restarted = torch.zeros_like(selected, dtype=torch.bool)
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                selected_is_eos = selected == int(eos_token_id) if eos_token_id >= 0 else torch.zeros_like(selected, dtype=torch.bool)
                selected_is_bos = selected == int(bos_token_id) if bos_token_id >= 0 else torch.zeros_like(selected, dtype=torch.bool)
                restarted = selected_is_bos | (plateau_tail_seen_eos[active_indices] & ~selected_is_eos)
                if action_char_restart_reset and bool(torch.any(restarted)):
                    restart_rows = active_indices[restarted]
                    if action_char_restart_reset_low_text_tokens > 0:
                        pending_restart_reset[restart_rows] = True
                        post_restart_action_char_counts[restart_rows] = 0
                        post_restart_last_action_char_increment_lengths[restart_rows] = 0
                        restart_low_text_counts[restart_rows] = 0
                        plateau_tail_seen_eos[restart_rows] = False
                        plateau_tail_eos_restart[restart_rows] = False
                    else:
                        action_char_counts[restart_rows] = 0
                        last_action_char_increment_lengths[restart_rows] = 0
                        target_reached_lengths[restart_rows] = 0
                        action_stable_counts[restart_rows] = 0
                        action_stable_snapshot_counts[restart_rows] = 0
                        plateau_tail_seen_eos[restart_rows] = False
                        plateau_tail_eos_restart[restart_rows] = False
                        restarted_with_count_reset[restart_rows] = True
                        if action_char_reset_requires_action_end:
                            reset_requires_action_end[restart_rows] = True
                        for row_idx in restart_rows.detach().cpu().tolist():
                            previous_action_snapshots[int(row_idx)] = None
                else:
                    plateau_tail_eos_restart[active_indices] |= restarted
            action_char_counts[active_indices] += increments
            pending_rows = pending_restart_reset[active_indices]
            if bool(torch.any(pending_rows)):
                pending_global_rows = active_indices[pending_rows]
                post_restart_action_char_counts[pending_global_rows] += increments[pending_rows]
                restart_low_text_counts[pending_global_rows] += low_text[pending_rows].to(
                    dtype=restart_low_text_counts.dtype
                )
            incremented = increments > 0
            if bool(torch.any(incremented)):
                incremented_rows = active_indices[incremented]
                last_action_char_increment_lengths[incremented_rows] = emitted_lengths[incremented_rows]
                action_char_low_text_tail_counts[incremented_rows] = 0
                pending_incremented = incremented & pending_restart_reset[active_indices]
                if bool(torch.any(pending_incremented)):
                    pending_incremented_rows = active_indices[pending_incremented]
                    post_restart_last_action_char_increment_lengths[pending_incremented_rows] = emitted_lengths[
                        pending_incremented_rows
                    ]
                if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                    safe_rows = incremented_rows[~plateau_tail_eos_restart[incremented_rows]]
                    plateau_tail_seen_eos[safe_rows] = False
                else:
                    plateau_tail_seen_eos[incremented_rows] = False
                    plateau_tail_eos_restart[incremented_rows] = False
                record_action_stability(incremented_rows)
            if bool(torch.any(~incremented)):
                tail_rows = active_indices[~incremented]
                action_char_low_text_tail_counts[tail_rows] += low_text[~incremented].to(
                    dtype=action_char_low_text_tail_counts.dtype
                )
            if action_char_restart_reset and action_char_restart_reset_low_text_tokens > 0:
                should_reset = (
                    pending_restart_reset[active_indices]
                    & (restart_low_text_counts[active_indices] >= int(action_char_restart_reset_low_text_tokens))
                )
                if bool(torch.any(should_reset)):
                    reset_rows = active_indices[should_reset]
                    action_char_counts[reset_rows] = post_restart_action_char_counts[reset_rows]
                    last_action_char_increment_lengths[reset_rows] = post_restart_last_action_char_increment_lengths[
                        reset_rows
                    ]
                    target_reached_lengths[reset_rows] = 0
                    action_stable_counts[reset_rows] = 0
                    action_stable_snapshot_counts[reset_rows] = 0
                    plateau_tail_seen_eos[reset_rows] = False
                    plateau_tail_eos_restart[reset_rows] = False
                    pending_restart_reset[reset_rows] = False
                    restarted_with_count_reset[reset_rows] = True
                    if action_char_reset_requires_action_end:
                        reset_requires_action_end[reset_rows] = True
                    for row_idx in reset_rows.detach().cpu().tolist():
                        previous_action_snapshots[int(row_idx)] = None
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                selected_is_eos = selected == int(eos_token_id) if eos_token_id >= 0 else torch.zeros_like(selected, dtype=torch.bool)
                not_incremented = ~incremented
                if bool(torch.any(not_incremented)):
                    tail_rows = active_indices[not_incremented]
                    plateau_tail_seen_eos[tail_rows] |= selected_is_eos[not_incremented]
            if action_char_strict_target_stop:
                reached_target = action_char_counts[active_indices] > int(action_char_target)
            else:
                reached_target = action_char_counts[active_indices] >= int(action_char_target)
            if action_char_target <= 0:
                reached_target = torch.zeros_like(reached_target, dtype=torch.bool)
            if action_char_target_confirm_tokens > 0 and bool(torch.any(reached_target)):
                first_reached = reached_target & (target_reached_lengths[active_indices] <= 0)
                if bool(torch.any(first_reached)):
                    first_rows = active_indices[first_reached]
                    target_reached_lengths[first_rows] = emitted_lengths[first_rows]
                confirmed = (
                    emitted_lengths[active_indices] - target_reached_lengths[active_indices]
                    >= int(action_char_target_confirm_tokens)
                )
                blocked = reached_target & ~confirmed
                blocked_by_target_confirmation[active_indices] |= blocked
                reached_target = reached_target & confirmed
            restart_fallback = torch.zeros_like(reached_target, dtype=torch.bool)
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                restart_tainted = plateau_tail_eos_restart[active_indices]
                if action_char_restart_continue or action_char_restart_reset:
                    restart_fallback = torch.zeros_like(reached_target, dtype=torch.bool)
                else:
                    restart_fallback = restart_tainted
                reached_target = reached_target & ~restart_tainted
            plateau_candidate = torch.zeros_like(reached_target, dtype=torch.bool)
            if action_char_plateau_tokens > 0:
                plateau_candidate = (
                    (action_char_counts[active_indices] >= int(action_char_min_chars))
                    & (last_action_char_increment_lengths[active_indices] > 0)
                    & (
                        emitted_lengths[active_indices] - last_action_char_increment_lengths[active_indices]
                        >= int(action_char_plateau_tokens)
                    )
                )
            plateau = plateau_candidate
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                blocked = plateau_candidate & plateau_tail_eos_restart[active_indices]
                blocked_by_char_plateau_eos_restart[active_indices] |= blocked
                plateau = plateau & ~plateau_tail_eos_restart[active_indices]
            if action_char_plateau_low_text_tail_tokens > 0:
                blocked = plateau & (
                    action_char_low_text_tail_counts[active_indices]
                    >= int(action_char_plateau_low_text_tail_tokens)
                )
                blocked_by_char_plateau_low_text_tail[active_indices] |= blocked
                plateau = plateau & ~blocked
            if action_char_stable_checks > 0:
                stable_guard = plateau
                if action_char_stable_max_chars > 0:
                    stable_guard = stable_guard & (
                        action_char_counts[active_indices] < int(action_char_stable_max_chars)
                    )
                stable_enough = action_stable_counts[active_indices] >= int(action_char_stable_checks)
                blocked = stable_guard & ~stable_enough
                blocked_by_char_plateau_stability[active_indices] |= blocked
                plateau = plateau & (~stable_guard | stable_enough)
            if action_char_reset_requires_action_end:
                require_action_end = reset_requires_action_end[active_indices]
                blocked = (reached_target | plateau) & require_action_end
                blocked_by_reset_requires_action_end[active_indices] |= blocked
                reached_target = reached_target & ~require_action_end
                plateau = plateau & ~require_action_end
            reached = reached_target | plateau
            stopped_by_chars[active_indices] |= reached
            stopped_by_char_plateau[active_indices] |= plateau
            stopped_by_restart_fallback[active_indices] |= restart_fallback
            return reached | restart_fallback

        def finish_return(max_len: int, *, append_action_end: bool | None = None) -> torch.Tensor:
            self._last_action_end_row_token_counts = emitted_lengths.detach().cpu().tolist()
            if stop_on_action_chars:
                self._last_action_char_stop_count = int(stopped_by_chars.sum().item())
                self._last_action_char_plateau_stop_count = int(stopped_by_char_plateau.sum().item())
                self._last_action_char_plateau_stability_block_count = int(
                    blocked_by_char_plateau_stability.sum().item()
                )
                self._last_action_char_plateau_eos_restart_block_count = int(
                    blocked_by_char_plateau_eos_restart.sum().item()
                )
                self._last_action_char_plateau_low_text_tail_block_count = int(
                    blocked_by_char_plateau_low_text_tail.sum().item()
                )
                self._last_action_char_restart_fallback_count = int(stopped_by_restart_fallback.sum().item())
                self._last_action_char_restart_continue_count = (
                    int(plateau_tail_eos_restart.sum().item()) if action_char_restart_continue else 0
                )
                self._last_action_char_restart_reset_count = int(restarted_with_count_reset.sum().item())
                self._last_action_char_reset_requires_action_end_block_count = int(
                    blocked_by_reset_requires_action_end.sum().item()
                )
                self._last_action_char_target_confirm_block_count = int(
                    blocked_by_target_confirmation.sum().item()
                )
                self._last_action_char_count_mean = float(action_char_counts.float().mean().item())
                self._last_action_char_stable_snapshot_count_mean = float(
                    action_stable_snapshot_counts.float().mean().item()
                )
                self._last_action_char_stable_count_mean = float(action_stable_counts.float().mean().item())
                if append_action_end is None:
                    append_action_end = True
                width = int(max_len) + (1 if append_action_end else 0)
                return generated[:, : min(width, generated_width)]
            return generated[:, : int(max_len)]

        if prefix_tokens is not None:
            next_token = prefix_tokens[0].view(1, 1).expand(bsize, 1)
            generated[:, 0] = next_token.squeeze(-1)
            emitted_lengths[:] = 1
            loop_start = 1
        else:
            next_token = self._select_next_token(lm_head(prefix_out[:, -1:, :]), temperature)
            generated[:, 0] = next_token.squeeze(-1)
            emitted_lengths[:] = 1
            loop_start = 1

        selected = next_token.squeeze(-1)
        finished_by_chars = record_action_chars(selected)
        finished_by_action_end = selected == action_end_token_id
        if action_char_plateau_reject_eos_restart and has_restart_guard_tokens and not action_char_restart_continue:
            finished_by_action_end = finished_by_action_end & ~plateau_tail_eos_restart[active_indices]
        finished = finished_by_action_end | finished_by_chars
        if bool(torch.all(finished)):
            append_action_end = bool(torch.any(finished_by_chars & ~finished_by_action_end).item())
            return finish_return(1, append_action_end=append_action_end)
        keep = torch.nonzero(~finished, as_tuple=False).flatten()
        if keep.numel() != bsize:
            past_key_values = past_key_values.batch_select_indices(keep)
            current_pad_mask = current_pad_mask[keep]
            next_token = next_token[keep]
            active_indices = active_indices[keep]

        for t in range(loop_start, max_decoding_steps):
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            active_bsize = int(active_indices.numel())
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((active_bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[next_token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            if prefix_tokens is not None and t < int(prefix_tokens.numel()):
                next_token = prefix_tokens[t].view(1, 1).expand(active_bsize, 1)
            else:
                next_token = self._select_next_token(lm_head(step_out[:, -1:, :]), temperature)
            selected = next_token.squeeze(-1)
            generated[active_indices, t] = selected
            emitted_lengths[active_indices] = t + 1
            finished_by_chars = record_action_chars(selected)
            finished_by_action_end = selected == action_end_token_id
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens and not action_char_restart_continue:
                finished_by_action_end = finished_by_action_end & ~plateau_tail_eos_restart[active_indices]
            finished = finished_by_action_end | finished_by_chars
            if bool(torch.all(finished)):
                max_len = int(emitted_lengths.max().item())
                append_action_end = bool(torch.any(finished_by_chars & ~finished_by_action_end).item())
                return finish_return(max_len, append_action_end=append_action_end)
            if bool(torch.any(finished)):
                keep = torch.nonzero(~finished, as_tuple=False).flatten()
                past_key_values = past_key_values.batch_select_indices(keep)
                current_pad_mask = current_pad_mask[keep]
                next_token = next_token[keep]
                active_indices = active_indices[keep]

        self._last_action_end_row_token_counts = emitted_lengths.detach().cpu().tolist()
        if stop_on_action_chars:
            self._last_action_char_stop_count = int(stopped_by_chars.sum().item())
            self._last_action_char_plateau_stop_count = int(stopped_by_char_plateau.sum().item())
            self._last_action_char_plateau_stability_block_count = int(
                blocked_by_char_plateau_stability.sum().item()
            )
            self._last_action_char_plateau_eos_restart_block_count = int(
                blocked_by_char_plateau_eos_restart.sum().item()
            )
            self._last_action_char_plateau_low_text_tail_block_count = int(
                blocked_by_char_plateau_low_text_tail.sum().item()
            )
            self._last_action_char_restart_fallback_count = int(stopped_by_restart_fallback.sum().item())
            self._last_action_char_restart_continue_count = (
                int(plateau_tail_eos_restart.sum().item()) if action_char_restart_continue else 0
            )
            self._last_action_char_restart_reset_count = int(restarted_with_count_reset.sum().item())
            self._last_action_char_reset_requires_action_end_block_count = int(
                blocked_by_reset_requires_action_end.sum().item()
            )
            self._last_action_char_target_confirm_block_count = int(
                blocked_by_target_confirmation.sum().item()
            )
            self._last_action_char_count_mean = float(action_char_counts.float().mean().item())
            self._last_action_char_stable_snapshot_count_mean = float(
                action_stable_snapshot_counts.float().mean().item()
            )
            self._last_action_char_stable_count_mean = float(action_stable_counts.float().mean().item())
        return generated

    @torch.no_grad()
    def sample_actions_fast_kv_cache_action_end_constrained(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps: int | None = None,
        temperature: float = 0.0,
        action_vocab_size: int | None = None,
        text_vocab_size: int | None = None,
        full_head_margin: float | None = None,
        force_action_prefix: bool = True,
        structural_token_radius: int = 512,
        full_head_prefix_tokens: int = 0,
        extra_token_ids: torch.Tensor | list[int] | None = None,
        stop_on_action_chars: bool = False,
        action_char_min_chars: int | None = None,
        action_char_plateau_tokens: int = 0,
        action_char_stable_checks: int = 0,
        action_char_stable_tolerance: float = 0.0,
        action_char_stable_max_chars: int = 0,
        action_char_plateau_reject_eos_restart: bool = False,
        action_char_restart_continue: bool = False,
        action_char_restart_reset: bool = False,
        action_char_restart_reset_low_text_tokens: int = 0,
        action_char_plateau_low_text_tail_tokens: int = 0,
        action_char_strict_target_stop: bool = False,
        action_char_reset_requires_action_end: bool = False,
        action_char_target_confirm_tokens: int = 0,
    ) -> torch.Tensor:
        """KV-cache FAST decode using the known PI0-FAST action-token support.

        PI0-FAST action chunks are emitted as ``Action: `` followed by FAST BPE
        action tokens and the ``|`` action-end marker.  This path avoids the
        257k-way language-model head on every decode step: it forces the fixed
        text prefix and computes argmax only over the FAST action-token IDs plus
        the action-end token.  It is intended to be paired with validation
        against the full-vocabulary target decode before use as a proof mode.
        """

        if temperature != 0.0:
            raise ValueError("Constrained PI0-FAST decode currently supports greedy temperature=0 only")
        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        action_end_token_id = self._action_end_token_id()
        action_char_target = 0
        if stop_on_action_chars:
            action_dim = self.policy.config.output_features[self._action_key()].shape[0]
            action_char_target = int(self.policy.config.n_action_steps) * int(action_dim)
        action_char_min_chars = int(action_char_target if action_char_min_chars is None else action_char_min_chars)
        action_char_min_chars = max(0, int(action_char_min_chars))
        action_char_plateau_tokens = max(0, int(action_char_plateau_tokens))
        action_char_stable_checks = max(0, int(action_char_stable_checks))
        action_char_stable_tolerance = max(0.0, float(action_char_stable_tolerance))
        action_char_stable_max_chars = max(0, int(action_char_stable_max_chars))
        action_char_plateau_reject_eos_restart = bool(action_char_plateau_reject_eos_restart)
        action_char_restart_continue = bool(action_char_restart_continue)
        action_char_restart_reset = bool(action_char_restart_reset)
        action_char_restart_reset_low_text_tokens = max(0, int(action_char_restart_reset_low_text_tokens))
        action_char_plateau_low_text_tail_tokens = max(0, int(action_char_plateau_low_text_tail_tokens))
        action_char_strict_target_stop = bool(action_char_strict_target_stop)
        action_char_reset_requires_action_end = bool(action_char_reset_requires_action_end)
        action_char_target_confirm_tokens = max(0, int(action_char_target_confirm_tokens))
        self._last_action_char_target = int(action_char_target)
        self._last_action_char_min_chars = int(action_char_min_chars if stop_on_action_chars else 0)
        self._last_action_char_plateau_tokens = int(action_char_plateau_tokens if stop_on_action_chars else 0)
        self._last_action_char_stable_checks = int(action_char_stable_checks if stop_on_action_chars else 0)
        self._last_action_char_stable_tolerance = float(action_char_stable_tolerance if stop_on_action_chars else 0.0)
        self._last_action_char_stable_max_chars = int(action_char_stable_max_chars if stop_on_action_chars else 0)
        self._last_action_char_plateau_reject_eos_restart = bool(
            action_char_plateau_reject_eos_restart if stop_on_action_chars else False
        )
        self._last_action_char_restart_continue = bool(action_char_restart_continue if stop_on_action_chars else False)
        self._last_action_char_restart_reset = bool(action_char_restart_reset if stop_on_action_chars else False)
        self._last_action_char_restart_reset_low_text_tokens = int(
            action_char_restart_reset_low_text_tokens if stop_on_action_chars else 0
        )
        self._last_action_char_plateau_low_text_tail_tokens = int(
            action_char_plateau_low_text_tail_tokens if stop_on_action_chars else 0
        )
        self._last_action_char_strict_target_stop = bool(
            action_char_strict_target_stop if stop_on_action_chars else False
        )
        self._last_action_char_reset_requires_action_end = bool(
            action_char_reset_requires_action_end if stop_on_action_chars else False
        )
        self._last_action_char_target_confirm_tokens = int(
            action_char_target_confirm_tokens if stop_on_action_chars else 0
        )
        self._last_action_char_stop_count = 0
        self._last_action_char_plateau_stop_count = 0
        self._last_action_char_plateau_stability_block_count = 0
        self._last_action_char_plateau_eos_restart_block_count = 0
        self._last_action_char_restart_fallback_count = 0
        self._last_action_char_restart_continue_count = 0
        self._last_action_char_restart_reset_count = 0
        self._last_action_char_reset_requires_action_end_block_count = 0
        self._last_action_char_target_confirm_block_count = 0
        self._last_action_char_plateau_low_text_tail_block_count = 0
        self._last_action_char_count_mean = 0.0
        self._last_action_char_stable_snapshot_count_mean = 0.0
        self._last_action_char_stable_count_mean = 0.0
        action_char_lookup = self.prepare_action_char_length_lookup(device) if stop_on_action_chars else None
        prefix_tokens = torch.as_tensor(
            self.model._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
            dtype=torch.long,
            device=device,
        )
        if prefix_tokens.numel() == 0:
            raise RuntimeError("Could not resolve PI0-FAST Action prefix tokens")
        action_char_prefix_low_token_ids = tuple(
            int(token_id)
            for token_id in prefix_tokens.detach().cpu().tolist()
            if 0 <= int(token_id) < 8192
        )

        # The public FAST tokenizer reports a 1024-token BPE vocab, but the
        # PI0-FAST target occasionally emits nearby PaliGemma tokens that map
        # to larger raw action-token IDs.  LeRobot's relaxed FAST decoder still
        # consumes those IDs.  The target can also emit low text/special tokens
        # inside the action chunk, so exact constrained decoding needs both a
        # high FAST-action band and a small low-token band.
        paligemma_vocab_size = int(self.model._paligemma_tokenizer.vocab_size)
        if action_vocab_size is None:
            action_vocab_size = max(int(getattr(self.policy.action_tokenizer, "vocab_size", 1024)), 8192)
        if text_vocab_size is None:
            text_vocab_size = 8192
        action_vocab_size = max(int(getattr(self.policy.action_tokenizer, "vocab_size", 1024)), int(action_vocab_size))
        text_vocab_size = min(paligemma_vocab_size, max(0, int(text_vocab_size)))
        structural_token_radius = max(0, int(structural_token_radius))
        full_head_prefix_tokens = max(0, int(full_head_prefix_tokens))
        if extra_token_ids is None:
            extra_candidate_ids = torch.empty((0,), dtype=torch.long, device=device)
        elif torch.is_tensor(extra_token_ids):
            extra_candidate_ids = extra_token_ids.to(device=device, dtype=torch.long).flatten()
        else:
            extra_candidate_ids = torch.as_tensor(
                [int(token_id) for token_id in extra_token_ids],
                dtype=torch.long,
                device=device,
            ).flatten()
        extra_candidate_ids = extra_candidate_ids[
            (extra_candidate_ids >= 0) & (extra_candidate_ids < paligemma_vocab_size)
        ]
        extra_candidate_ids = torch.unique(extra_candidate_ids)
        extra_token_key = tuple(int(token_id) for token_id in extra_candidate_ids.detach().cpu().tolist())
        candidate_cache_key = (
            int(action_vocab_size),
            int(text_vocab_size),
            int(structural_token_radius),
            extra_token_key,
            str(device),
            str(lm_head.weight.dtype),
            int(lm_head.weight.data_ptr()),
        )
        cached = getattr(self, "_constrained_lm_head_cache", None)
        if cached is None or cached.get("key") != candidate_cache_key:
            action_token_ids = (
                paligemma_vocab_size
                - 1
                - int(self.policy.config.fast_skip_tokens)
                - torch.arange(action_vocab_size, dtype=torch.long, device=device)
            )
            action_token_ids = action_token_ids[(action_token_ids >= 0) & (action_token_ids < paligemma_vocab_size)]
            text_token_ids = torch.arange(text_vocab_size, dtype=torch.long, device=device)
            structural_start = max(0, action_end_token_id - structural_token_radius)
            structural_end = min(paligemma_vocab_size, action_end_token_id + structural_token_radius + 1)
            structural_token_ids = torch.arange(structural_start, structural_end, dtype=torch.long, device=device)
            candidate_ids = torch.cat(
                [
                    text_token_ids,
                    structural_token_ids,
                    action_token_ids,
                    extra_candidate_ids,
                    prefix_tokens,
                    torch.tensor([action_end_token_id], dtype=torch.long, device=device),
                ],
                dim=0,
            )
            candidate_ids = torch.unique(candidate_ids)
            candidate_weight = lm_head.weight.index_select(0, candidate_ids).contiguous()
            lm_head_bias = getattr(lm_head, "bias", None)
            candidate_bias = (
                None
                if lm_head_bias is None
                else lm_head_bias.index_select(0, candidate_ids).contiguous()
            )
            cached = {
                "key": candidate_cache_key,
                "candidate_ids": candidate_ids,
                "candidate_weight": candidate_weight,
                "candidate_bias": candidate_bias,
                "candidate_size": int(candidate_ids.numel()),
                "action_vocab_size": int(action_vocab_size),
                "text_vocab_size": int(text_vocab_size),
                "structural_token_radius": int(structural_token_radius),
                "extra_token_count": int(extra_candidate_ids.numel()),
            }
            self._constrained_lm_head_cache = cached
        candidate_ids = cached["candidate_ids"]
        candidate_weight = cached["candidate_weight"]
        candidate_bias = cached["candidate_bias"]
        self._last_constrained_candidate_size = int(candidate_ids.numel())
        self._last_constrained_action_vocab_size = int(cached["action_vocab_size"])
        self._last_constrained_text_vocab_size = int(cached["text_vocab_size"])
        self._last_constrained_structural_token_radius = int(cached["structural_token_radius"])
        self._last_constrained_extra_token_count = int(cached["extra_token_count"])
        self._last_constrained_force_action_prefix = int(bool(force_action_prefix))
        self._last_forced_action_prefix_token_count = int(prefix_tokens.numel()) if force_action_prefix else 0
        self._last_constrained_full_head_margin = float(full_head_margin) if full_head_margin is not None else -1.0
        self._last_constrained_full_head_prefix_tokens = int(full_head_prefix_tokens)
        self._last_constrained_full_head_prefix_calls = 0
        self._last_constrained_full_head_fallbacks = 0
        self._last_constrained_restricted_head_calls = 0
        self._last_constrained_full_head_fallback_rate = 0.0
        self._last_constrained_margin_min = 0.0
        self._last_constrained_margin_mean = 0.0
        restricted_head_calls = 0
        full_head_fallbacks = 0
        full_head_prefix_calls = 0
        margin_min = float("inf")
        margin_sum = 0.0

        def full_next(hidden: torch.Tensor) -> torch.Tensor:
            return torch.argmax(lm_head(hidden)[:, -1, :], dim=-1, keepdim=True)

        def constrained_next(hidden: torch.Tensor, token_position: int) -> torch.Tensor:
            nonlocal full_head_prefix_calls
            if token_position < full_head_prefix_tokens:
                full_head_prefix_calls += int(hidden.shape[0])
                return full_next(hidden)
            return restricted_next(hidden)

        def restricted_next(hidden: torch.Tensor) -> torch.Tensor:
            nonlocal restricted_head_calls, full_head_fallbacks, margin_min, margin_sum
            logits = F.linear(hidden, candidate_weight, candidate_bias)
            step_logits = logits[:, -1, :].float()
            restricted_head_calls += int(step_logits.shape[0])
            if step_logits.shape[-1] >= 2:
                local = torch.argmax(step_logits, dim=-1)
                top_values, top_indices = torch.topk(step_logits, k=2, dim=-1)
                margins = top_values[:, 0] - top_values[:, 1]
                margin_min = min(margin_min, float(torch.min(margins).item()))
                margin_sum += float(torch.sum(margins).item())
                if full_head_margin is not None and bool(torch.any(margins < float(full_head_margin))):
                    full_head_fallbacks += int(step_logits.shape[0])
                    return full_next(hidden)
            else:
                local = torch.argmax(step_logits, dim=-1)
            return candidate_ids.index_select(0, local).unsqueeze(-1)

        def finish_constrained_stats() -> None:
            self._last_constrained_full_head_prefix_calls = int(full_head_prefix_calls)
            self._last_constrained_full_head_fallbacks = int(full_head_fallbacks)
            self._last_constrained_restricted_head_calls = int(restricted_head_calls)
            self._last_constrained_full_head_fallback_rate = (
                float(full_head_fallbacks) / float(restricted_head_calls) if restricted_head_calls else 0.0
            )
            self._last_constrained_margin_min = float(margin_min if margin_min != float("inf") else 0.0)
            self._last_constrained_margin_mean = (
                float(margin_sum) / float(restricted_head_calls) if restricted_head_calls else 0.0
            )

        generated_width = int(max_decoding_steps) + (1 if stop_on_action_chars else 0)
        generated = torch.full((bsize, generated_width), action_end_token_id, dtype=torch.long, device=device)
        active_indices = torch.arange(bsize, dtype=torch.long, device=device)
        emitted_lengths = torch.ones((bsize,), dtype=torch.long, device=device)
        action_char_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        last_action_char_increment_lengths = torch.zeros((bsize,), dtype=torch.long, device=device)
        action_stable_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        action_stable_snapshot_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        target_reached_lengths = torch.zeros((bsize,), dtype=torch.long, device=device)
        previous_action_snapshots: list[torch.Tensor | None] = [None] * bsize
        stopped_by_chars = torch.zeros((bsize,), dtype=torch.bool, device=device)
        stopped_by_char_plateau = torch.zeros((bsize,), dtype=torch.bool, device=device)
        stopped_by_restart_fallback = torch.zeros((bsize,), dtype=torch.bool, device=device)
        restarted_with_count_reset = torch.zeros((bsize,), dtype=torch.bool, device=device)
        reset_requires_action_end = torch.zeros((bsize,), dtype=torch.bool, device=device)
        pending_restart_reset = torch.zeros((bsize,), dtype=torch.bool, device=device)
        post_restart_action_char_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        post_restart_last_action_char_increment_lengths = torch.zeros((bsize,), dtype=torch.long, device=device)
        restart_low_text_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        action_char_low_text_tail_counts = torch.zeros((bsize,), dtype=torch.long, device=device)
        blocked_by_char_plateau_stability = torch.zeros((bsize,), dtype=torch.bool, device=device)
        blocked_by_char_plateau_eos_restart = torch.zeros((bsize,), dtype=torch.bool, device=device)
        blocked_by_char_plateau_low_text_tail = torch.zeros((bsize,), dtype=torch.bool, device=device)
        blocked_by_reset_requires_action_end = torch.zeros((bsize,), dtype=torch.bool, device=device)
        blocked_by_target_confirmation = torch.zeros((bsize,), dtype=torch.bool, device=device)
        plateau_tail_seen_eos = torch.zeros((bsize,), dtype=torch.bool, device=device)
        plateau_tail_eos_restart = torch.zeros((bsize,), dtype=torch.bool, device=device)
        eos_token_id = int(getattr(self.model._paligemma_tokenizer, "eos_token_id", -1) or -1)
        bos_token_id = int(getattr(self.model._paligemma_tokenizer, "bos_token_id", -1) or -1)
        has_restart_guard_tokens = eos_token_id >= 0 or bos_token_id >= 0

        def token_ids_with_action_end(row_idx: int) -> torch.Tensor:
            length = max(int(emitted_lengths[row_idx].item()), 1)
            token_ids = generated[row_idx : row_idx + 1, :length]
            if int(token_ids[0, -1].item()) != action_end_token_id:
                eos = torch.full((1, 1), action_end_token_id, dtype=token_ids.dtype, device=token_ids.device)
                token_ids = torch.cat([token_ids, eos], dim=1)
            return token_ids

        def record_action_stability(row_indices: torch.Tensor) -> None:
            if action_char_stable_checks <= 0:
                return
            for row_idx in row_indices.detach().cpu().tolist():
                token_ids = token_ids_with_action_end(int(row_idx))
                actions = self._detokenize_generated_actions(token_ids).detach().float().cpu()
                previous = previous_action_snapshots[int(row_idx)]
                if previous is None:
                    action_stable_counts[int(row_idx)] = 0
                else:
                    max_delta = float(torch.max(torch.abs(actions - previous)).item())
                    if max_delta <= action_char_stable_tolerance:
                        action_stable_counts[int(row_idx)] += 1
                    else:
                        action_stable_counts[int(row_idx)] = 0
                previous_action_snapshots[int(row_idx)] = actions
                action_stable_snapshot_counts[int(row_idx)] += 1

        def record_action_chars(selected: torch.Tensor) -> torch.Tensor:
            if action_char_lookup is None:
                return torch.zeros_like(selected, dtype=torch.bool)
            increments = action_char_lookup.index_select(0, selected).to(dtype=action_char_counts.dtype)
            low_text = selected < 8192
            if eos_token_id >= 0:
                low_text = low_text & (selected != int(eos_token_id))
            if bos_token_id >= 0:
                low_text = low_text & (selected != int(bos_token_id))
            for token_id in action_char_prefix_low_token_ids:
                low_text = low_text & (selected != token_id)
            restarted = torch.zeros_like(selected, dtype=torch.bool)
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                selected_is_eos = selected == int(eos_token_id) if eos_token_id >= 0 else torch.zeros_like(selected, dtype=torch.bool)
                selected_is_bos = selected == int(bos_token_id) if bos_token_id >= 0 else torch.zeros_like(selected, dtype=torch.bool)
                restarted = selected_is_bos | (plateau_tail_seen_eos[active_indices] & ~selected_is_eos)
                if action_char_restart_reset and bool(torch.any(restarted)):
                    restart_rows = active_indices[restarted]
                    if action_char_restart_reset_low_text_tokens > 0:
                        pending_restart_reset[restart_rows] = True
                        post_restart_action_char_counts[restart_rows] = 0
                        post_restart_last_action_char_increment_lengths[restart_rows] = 0
                        restart_low_text_counts[restart_rows] = 0
                        plateau_tail_seen_eos[restart_rows] = False
                        plateau_tail_eos_restart[restart_rows] = False
                    else:
                        action_char_counts[restart_rows] = 0
                        last_action_char_increment_lengths[restart_rows] = 0
                        target_reached_lengths[restart_rows] = 0
                        action_stable_counts[restart_rows] = 0
                        action_stable_snapshot_counts[restart_rows] = 0
                        plateau_tail_seen_eos[restart_rows] = False
                        plateau_tail_eos_restart[restart_rows] = False
                        restarted_with_count_reset[restart_rows] = True
                        if action_char_reset_requires_action_end:
                            reset_requires_action_end[restart_rows] = True
                        for row_idx in restart_rows.detach().cpu().tolist():
                            previous_action_snapshots[int(row_idx)] = None
                else:
                    plateau_tail_eos_restart[active_indices] |= restarted
            action_char_counts[active_indices] += increments
            pending_rows = pending_restart_reset[active_indices]
            if bool(torch.any(pending_rows)):
                pending_global_rows = active_indices[pending_rows]
                post_restart_action_char_counts[pending_global_rows] += increments[pending_rows]
                restart_low_text_counts[pending_global_rows] += low_text[pending_rows].to(
                    dtype=restart_low_text_counts.dtype
                )
            incremented = increments > 0
            if bool(torch.any(incremented)):
                incremented_rows = active_indices[incremented]
                last_action_char_increment_lengths[incremented_rows] = emitted_lengths[incremented_rows]
                action_char_low_text_tail_counts[incremented_rows] = 0
                pending_incremented = incremented & pending_restart_reset[active_indices]
                if bool(torch.any(pending_incremented)):
                    pending_incremented_rows = active_indices[pending_incremented]
                    post_restart_last_action_char_increment_lengths[pending_incremented_rows] = emitted_lengths[
                        pending_incremented_rows
                    ]
                if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                    safe_rows = incremented_rows[~plateau_tail_eos_restart[incremented_rows]]
                    plateau_tail_seen_eos[safe_rows] = False
                else:
                    plateau_tail_seen_eos[incremented_rows] = False
                    plateau_tail_eos_restart[incremented_rows] = False
                record_action_stability(incremented_rows)
            if bool(torch.any(~incremented)):
                tail_rows = active_indices[~incremented]
                action_char_low_text_tail_counts[tail_rows] += low_text[~incremented].to(
                    dtype=action_char_low_text_tail_counts.dtype
                )
            if action_char_restart_reset and action_char_restart_reset_low_text_tokens > 0:
                should_reset = (
                    pending_restart_reset[active_indices]
                    & (restart_low_text_counts[active_indices] >= int(action_char_restart_reset_low_text_tokens))
                )
                if bool(torch.any(should_reset)):
                    reset_rows = active_indices[should_reset]
                    action_char_counts[reset_rows] = post_restart_action_char_counts[reset_rows]
                    last_action_char_increment_lengths[reset_rows] = post_restart_last_action_char_increment_lengths[
                        reset_rows
                    ]
                    target_reached_lengths[reset_rows] = 0
                    action_stable_counts[reset_rows] = 0
                    action_stable_snapshot_counts[reset_rows] = 0
                    plateau_tail_seen_eos[reset_rows] = False
                    plateau_tail_eos_restart[reset_rows] = False
                    pending_restart_reset[reset_rows] = False
                    restarted_with_count_reset[reset_rows] = True
                    if action_char_reset_requires_action_end:
                        reset_requires_action_end[reset_rows] = True
                    for row_idx in reset_rows.detach().cpu().tolist():
                        previous_action_snapshots[int(row_idx)] = None
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                selected_is_eos = selected == int(eos_token_id) if eos_token_id >= 0 else torch.zeros_like(selected, dtype=torch.bool)
                not_incremented = ~incremented
                if bool(torch.any(not_incremented)):
                    tail_rows = active_indices[not_incremented]
                    plateau_tail_seen_eos[tail_rows] |= selected_is_eos[not_incremented]
            if action_char_strict_target_stop:
                reached_target = action_char_counts[active_indices] > int(action_char_target)
            else:
                reached_target = action_char_counts[active_indices] >= int(action_char_target)
            if action_char_target <= 0:
                reached_target = torch.zeros_like(reached_target, dtype=torch.bool)
            if action_char_target_confirm_tokens > 0 and bool(torch.any(reached_target)):
                first_reached = reached_target & (target_reached_lengths[active_indices] <= 0)
                if bool(torch.any(first_reached)):
                    first_rows = active_indices[first_reached]
                    target_reached_lengths[first_rows] = emitted_lengths[first_rows]
                confirmed = (
                    emitted_lengths[active_indices] - target_reached_lengths[active_indices]
                    >= int(action_char_target_confirm_tokens)
                )
                blocked = reached_target & ~confirmed
                blocked_by_target_confirmation[active_indices] |= blocked
                reached_target = reached_target & confirmed
            restart_fallback = torch.zeros_like(reached_target, dtype=torch.bool)
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                restart_tainted = plateau_tail_eos_restart[active_indices]
                restart_fallback = (
                    torch.zeros_like(reached_target, dtype=torch.bool)
                    if action_char_restart_continue or action_char_restart_reset
                    else restart_tainted
                )
                reached_target = reached_target & ~restart_tainted
            plateau_candidate = torch.zeros_like(reached_target, dtype=torch.bool)
            if action_char_plateau_tokens > 0:
                plateau_candidate = (
                    (action_char_counts[active_indices] >= int(action_char_min_chars))
                    & (last_action_char_increment_lengths[active_indices] > 0)
                    & (
                        emitted_lengths[active_indices] - last_action_char_increment_lengths[active_indices]
                        >= int(action_char_plateau_tokens)
                    )
                )
            plateau = plateau_candidate
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens:
                blocked = plateau_candidate & plateau_tail_eos_restart[active_indices]
                blocked_by_char_plateau_eos_restart[active_indices] |= blocked
                plateau = plateau & ~plateau_tail_eos_restart[active_indices]
            if action_char_plateau_low_text_tail_tokens > 0:
                blocked = plateau & (
                    action_char_low_text_tail_counts[active_indices]
                    >= int(action_char_plateau_low_text_tail_tokens)
                )
                blocked_by_char_plateau_low_text_tail[active_indices] |= blocked
                plateau = plateau & ~blocked
            if action_char_stable_checks > 0:
                stable_guard = plateau
                if action_char_stable_max_chars > 0:
                    stable_guard = stable_guard & (
                        action_char_counts[active_indices] < int(action_char_stable_max_chars)
                    )
                stable_enough = action_stable_counts[active_indices] >= int(action_char_stable_checks)
                blocked = stable_guard & ~stable_enough
                blocked_by_char_plateau_stability[active_indices] |= blocked
                plateau = plateau & (~stable_guard | stable_enough)
            if action_char_reset_requires_action_end:
                require_action_end = reset_requires_action_end[active_indices]
                blocked = (reached_target | plateau) & require_action_end
                blocked_by_reset_requires_action_end[active_indices] |= blocked
                reached_target = reached_target & ~require_action_end
                plateau = plateau & ~require_action_end
            reached = reached_target | plateau
            stopped_by_chars[active_indices] |= reached
            stopped_by_char_plateau[active_indices] |= plateau
            stopped_by_restart_fallback[active_indices] |= restart_fallback
            return reached | restart_fallback

        def finish_return(max_len: int, *, append_action_end: bool | None = None) -> torch.Tensor:
            finish_constrained_stats()
            self._last_action_end_row_token_counts = emitted_lengths.detach().cpu().tolist()
            if stop_on_action_chars:
                self._last_action_char_stop_count = int(stopped_by_chars.sum().item())
                self._last_action_char_plateau_stop_count = int(stopped_by_char_plateau.sum().item())
                self._last_action_char_plateau_stability_block_count = int(
                    blocked_by_char_plateau_stability.sum().item()
                )
                self._last_action_char_plateau_eos_restart_block_count = int(
                    blocked_by_char_plateau_eos_restart.sum().item()
                )
                self._last_action_char_plateau_low_text_tail_block_count = int(
                    blocked_by_char_plateau_low_text_tail.sum().item()
                )
                self._last_action_char_restart_fallback_count = int(stopped_by_restart_fallback.sum().item())
                self._last_action_char_restart_continue_count = (
                    int(plateau_tail_eos_restart.sum().item()) if action_char_restart_continue else 0
                )
                self._last_action_char_restart_reset_count = int(restarted_with_count_reset.sum().item())
                self._last_action_char_reset_requires_action_end_block_count = int(
                    blocked_by_reset_requires_action_end.sum().item()
                )
                self._last_action_char_target_confirm_block_count = int(
                    blocked_by_target_confirmation.sum().item()
                )
                self._last_action_char_count_mean = float(action_char_counts.float().mean().item())
                self._last_action_char_stable_snapshot_count_mean = float(
                    action_stable_snapshot_counts.float().mean().item()
                )
                self._last_action_char_stable_count_mean = float(action_stable_counts.float().mean().item())
                if append_action_end is None:
                    append_action_end = True
                width = int(max_len) + (1 if append_action_end else 0)
                return generated[:, : min(width, generated_width)]
            return generated[:, : int(max_len)]

        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens_in,
            masks_in,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        generated = torch.full((bsize, max_decoding_steps), action_end_token_id, dtype=torch.long, device=device)
        if force_action_prefix:
            next_token = prefix_tokens[0].view(1, 1).expand(bsize, 1)
        else:
            next_token = constrained_next(prefix_out[:, -1:, :], 0)
        generated[:, 0] = next_token.squeeze(-1)
        current_pad_mask = prefix_pad_masks
        selected = next_token.squeeze(-1)
        finished_by_chars = record_action_chars(selected)
        finished_by_action_end = selected == action_end_token_id
        if action_char_plateau_reject_eos_restart and has_restart_guard_tokens and not action_char_restart_continue:
            finished_by_action_end = finished_by_action_end & ~plateau_tail_eos_restart[active_indices]
        finished = finished_by_action_end | finished_by_chars
        if bool(torch.all(finished)):
            append_action_end = bool(torch.any(finished_by_chars & ~finished_by_action_end).item())
            return finish_return(1, append_action_end=append_action_end)
        if bool(torch.any(finished)):
            keep = torch.nonzero(~finished, as_tuple=False).flatten()
            past_key_values = past_key_values.batch_select_indices(keep)
            current_pad_mask = current_pad_mask[keep]
            next_token = next_token[keep]
            active_indices = active_indices[keep]

        for t in range(1, max_decoding_steps):
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            active_bsize = int(active_indices.numel())
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((active_bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[next_token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            if force_action_prefix and t < int(prefix_tokens.numel()):
                next_token = prefix_tokens[t].view(1, 1).expand(active_bsize, 1)
            else:
                next_token = constrained_next(step_out[:, -1:, :], t)
            selected = next_token.squeeze(-1)
            generated[active_indices, t] = selected
            emitted_lengths[active_indices] = t + 1
            finished_by_chars = record_action_chars(selected)
            finished_by_action_end = selected == action_end_token_id
            if action_char_plateau_reject_eos_restart and has_restart_guard_tokens and not action_char_restart_continue:
                finished_by_action_end = finished_by_action_end & ~plateau_tail_eos_restart[active_indices]
            finished = finished_by_action_end | finished_by_chars
            if bool(torch.all(finished)):
                max_len = int(emitted_lengths.max().item())
                append_action_end = bool(torch.any(finished_by_chars & ~finished_by_action_end).item())
                return finish_return(max_len, append_action_end=append_action_end)
            if bool(torch.any(finished)):
                keep = torch.nonzero(~finished, as_tuple=False).flatten()
                past_key_values = past_key_values.batch_select_indices(keep)
                current_pad_mask = current_pad_mask[keep]
                next_token = next_token[keep]
                active_indices = active_indices[keep]

        return finish_return(int(max_decoding_steps), append_action_end=False)

    @torch.no_grad()
    def sample_actions_fast_ngram_speculative(
        self,
        images,
        img_masks,
        tokens,
        masks,
        drafter: Any,
        max_decoding_steps: int | None = None,
        lookahead: int = 8,
        temperature: float = 0.0,
        reuse_full_blocks: bool = False,
        min_verify_margin: float = 0.0,
        verify_from_scratch: bool = False,
        emit_bonus_token: bool = False,
        defer_correction_token: bool = False,
        dynamic_lookahead: bool = False,
        min_lookahead: int = 1,
        lookahead_growth: int = 1,
        lookahead_shrink: int = 4,
        tree_width: int = 1,
        tree_branch_width: int = 4,
        dynamic_tree_width: bool = False,
        min_tree_width: int = 1,
        tree_width_growth: int = 1,
        tree_width_shrink: int = 1,
        tree_anchor_target_token: bool = False,
        tree_anchor_target_continuation: bool = False,
        early_stop_action_end: bool = False,
        replay_accepted_cache: bool = False,
        resync_accepted_cache: bool = False,
        diagnose_verify_alignment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Greedy exact FAST-token decode with n-gram speculative verification.

        This is a conservative exact-SD path: accepted drafted tokens are added
        to the target KV cache, while mismatches fall back to the target token.
        With ``emit_bonus_token`` enabled, a fully accepted block also emits the
        verifier's next greedy token. That token is processed by the next
        verify pass, matching standard greedy speculative decoding.
        With ``defer_correction_token`` enabled, rejected-block correction
        tokens use the same pending-token path instead of an immediate fallback
        forward, so the next verifier pass can process the correction and draft
        together.
        """

        if temperature != 0.0:
            raise ValueError("ngram speculative PI0-FAST decode currently supports greedy temperature=0 only")
        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        if bsize != 1:
            raise ValueError("ngram speculative PI0-FAST decode currently supports batch size 1")
        if int(tree_width) > 1 and verify_from_scratch:
            raise ValueError("PI0-FAST tree speculation requires cached verification; disable verify_from_scratch")
        defer_correction_token = bool(defer_correction_token)

        device = tokens.device
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        action_end_token_id = self._action_end_token_id() if early_stop_action_end else None
        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), past_key_values = self._forward_prefix_language_model(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=prefix_embs,
            use_cache=True,
            cache_position=torch.arange(prefix_embs.shape[1], device=device, dtype=torch.long),
        )
        prev_logits = lm_head(prefix_out[:, -1:, :])
        current_pad_mask = prefix_pad_masks
        generated_tokens: list[int] = []
        logits_by_step: list[torch.Tensor] = []
        target_forwards = 1
        verify_forwards = 0
        fallback_forwards = 0
        replay_forwards = 0
        resync_forwards = 0
        full_block_reuses = 0
        margin_block_rejects = 0
        verify_margin_checks = 0
        verify_margin_min = float("inf")
        verify_margin_sum = 0.0
        bonus_tokens = 0
        pending_verifies = 0
        pending_corrections = 0
        pending_processes = 0
        pending_unprocessed_token: torch.Tensor | None = None
        drafted_tokens = 0
        accepted_tokens = 0
        debug_events: list[dict[str, Any]] = []
        max_lookahead = max(1, int(lookahead))
        current_lookahead = max_lookahead
        min_dynamic_lookahead = max(1, min(int(min_lookahead), max_lookahead))
        lookahead_values: list[int] = []
        max_tree_width = max(1, int(tree_width))
        current_tree_width = max(
            1,
            min(max_tree_width, int(min_tree_width) if dynamic_tree_width else max_tree_width),
        )
        tree_width_values: list[int] = []
        max_tree_branch_width = max(1, int(tree_branch_width))
        tree_verifies = 0
        tree_candidates = 0
        tree_accepted_tokens = 0
        tree_anchor_verifies = 0
        tree_anchor_candidates = 0
        tree_anchor_accepted_tokens = 0
        tree_first_token_checks = 0
        tree_first_token_misses = 0
        verify_alignment_blocks = 0
        verify_alignment_checks = 0
        verify_alignment_argmax_mismatches = 0
        verify_alignment_blocks_with_mismatch = 0
        verify_alignment_false_accepts = 0
        verify_alignment_accept_disagreements = 0
        verify_alignment_first_mismatch_min = float("inf")
        verify_alignment_first_mismatch_sum = 0.0
        verify_alignment_forwards = 0
        verify_alignment_batched_accept_sum = 0
        verify_alignment_step_accept_sum = 0
        verify_alignment_correction_checks = 0
        verify_alignment_correction_mismatches = 0
        second_order_action_extrapolation_drafted_tokens = 0
        second_order_action_extrapolation_accepted_tokens = 0
        previous_chunk_position_drafted_tokens = 0
        previous_chunk_position_accepted_tokens = 0
        chunk_prefix_retrieval_drafted_tokens = 0
        chunk_prefix_retrieval_accepted_tokens = 0
        chunk_length_stop_drafted_tokens = 0
        chunk_length_stop_accepted_tokens = 0
        position_mode_histogram_drafted_tokens = 0
        position_mode_histogram_accepted_tokens = 0
        global_position_mode_drafted_tokens = 0
        global_position_mode_accepted_tokens = 0
        action_dimension_mode_drafted_tokens = 0
        action_dimension_mode_accepted_tokens = 0
        hold_action_token_drafted_tokens = 0
        hold_action_token_accepted_tokens = 0
        ngram_continuation_drafted_tokens = 0
        ngram_continuation_accepted_tokens = 0
        source_agreement_drafted_tokens = 0
        source_agreement_accepted_tokens = 0
        action_trend_regression_drafted_tokens = 0
        action_trend_regression_accepted_tokens = 0
        action_prefix_lookup_drafted_tokens = 0
        action_prefix_lookup_accepted_tokens = 0
        action_vector_transition_drafted_tokens = 0
        action_vector_transition_accepted_tokens = 0
        action_repeat_vector_drafted_tokens = 0
        action_repeat_vector_accepted_tokens = 0
        action_token_neighborhood_drafted_tokens = 0
        action_token_neighborhood_accepted_tokens = 0
        action_context_tree_drafted_tokens = 0
        action_context_tree_accepted_tokens = 0
        action_transition_histogram_drafted_tokens = 0
        action_transition_histogram_accepted_tokens = 0
        action_delta_histogram_drafted_tokens = 0
        action_delta_histogram_accepted_tokens = 0
        action_delta_ngram_drafted_tokens = 0
        action_delta_ngram_accepted_tokens = 0
        chunk_delta_template_drafted_tokens = 0
        chunk_delta_template_accepted_tokens = 0
        chunk_position_delta_drafted_tokens = 0
        chunk_position_delta_accepted_tokens = 0

        def source_count(sources: list[str], source: str, limit: int | None = None) -> int:
            values = sources if limit is None else sources[: max(int(limit), 0)]
            return sum(1 for value in values if value == source)

        def last_draft_sources() -> list[str]:
            if hasattr(drafter, "last_draft_sources"):
                return [str(value) for value in drafter.last_draft_sources()]
            return []

        def last_many_sources() -> list[list[str]]:
            if hasattr(drafter, "last_many_sources"):
                return [[str(value) for value in row] for row in drafter.last_many_sources()]
            return []

        def record_source_feedback(sources: list[str], accepted: int) -> None:
            if hasattr(drafter, "record_source_feedback"):
                drafter.record_source_feedback(sources, accepted)

        def record_source_miss() -> None:
            if hasattr(drafter, "record_source_miss"):
                drafter.record_source_miss()

        def source_cooldown_stats() -> dict[str, Any]:
            if hasattr(drafter, "source_cooldown_stats"):
                return dict(drafter.source_cooldown_stats())
            return {}

        def update_dynamic_lookahead(*, accepted: int, drafted: int, miss: bool = False) -> None:
            nonlocal current_lookahead
            if not dynamic_lookahead:
                return
            if miss or accepted < drafted:
                current_lookahead = max(
                    min_dynamic_lookahead,
                    current_lookahead - max(int(lookahead_shrink), 1),
                )
            else:
                current_lookahead = min(
                    max_lookahead,
                    current_lookahead + max(int(lookahead_growth), 1),
                )

        def advance_one(
            next_token: torch.Tensor,
            logits_for_token: torch.Tensor,
            *,
            fallback: bool,
        ) -> torch.Tensor:
            nonlocal current_pad_mask, past_key_values, target_forwards, fallback_forwards, replay_forwards
            logits_by_step.append(logits_for_token)
            generated_tokens.append(int(next_token.item()))
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_token_emb,
                use_cache=True,
                cache_position=torch.tensor([current_pad_mask.shape[1] - 1], device=device, dtype=torch.long),
            )
            target_forwards += 1
            if fallback:
                fallback_forwards += 1
            else:
                replay_forwards += 1
            return lm_head(step_out[:, -1:, :])

        def replay_one(next_token: torch.Tensor) -> torch.Tensor:
            nonlocal current_pad_mask, past_key_values, target_forwards, replay_forwards
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_token_emb,
                use_cache=True,
                cache_position=torch.tensor([current_pad_mask.shape[1] - 1], device=device, dtype=torch.long),
            )
            target_forwards += 1
            replay_forwards += 1
            return lm_head(step_out[:, -1:, :])

        def diagnostic_stepwise_logits(candidate_tokens: list[int], steps: int) -> torch.Tensor:
            nonlocal verify_alignment_forwards
            steps = max(int(steps), 1)
            diag_past = clone_kv(past_key_values)
            diag_pad = current_pad_mask.clone()
            step_logits = [prev_logits.detach()]
            for token in candidate_tokens[: steps - 1]:
                token_tensor = torch.tensor([[int(token)]], dtype=torch.long, device=device)
                token_emb = self.model.paligemma_with_expert.embed_language_tokens(token_tensor)
                token_emb = token_emb * math.sqrt(token_emb.shape[-1])
                token_emb = token_emb.to(dtype=prefix_embs.dtype)
                diag_pad = torch.cat(
                    [diag_pad, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                    dim=1,
                )
                position_ids = (torch.sum(diag_pad, dim=1, keepdim=True) - 1).long()
                att_mask = self.model._prepare_attention_masks_4d(
                    diag_pad.unsqueeze(1),
                    dtype=token_emb.dtype,
                )
                (step_out, _), diag_past = self._forward_prefix_language_model(
                    attention_mask=att_mask,
                    position_ids=position_ids,
                    past_key_values=diag_past,
                    inputs_embeds=token_emb,
                    use_cache=True,
                    cache_position=torch.tensor([diag_pad.shape[1] - 1], device=device, dtype=torch.long),
                )
                verify_alignment_forwards += 1
                step_logits.append(lm_head(step_out[:, -1:, :]).detach())
            return torch.cat(step_logits, dim=1)

        def process_pending_one() -> torch.Tensor:
            nonlocal pending_unprocessed_token, current_pad_mask, past_key_values
            nonlocal target_forwards, fallback_forwards, pending_processes
            if pending_unprocessed_token is None:
                raise RuntimeError("No pending token to process")
            token = pending_unprocessed_token
            token_emb = self.model.paligemma_with_expert.embed_language_tokens(token)
            token_emb = token_emb * math.sqrt(token_emb.shape[-1])
            token_emb = token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=token_emb.dtype,
            )
            (step_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=token_emb,
                use_cache=True,
                cache_position=torch.tensor([current_pad_mask.shape[1] - 1], device=device, dtype=torch.long),
            )
            pending_unprocessed_token = None
            pending_processes += 1
            target_forwards += 1
            fallback_forwards += 1
            return lm_head(step_out[:, -1:, :])

        def resync_from_generated() -> torch.Tensor:
            nonlocal current_pad_mask, past_key_values, target_forwards, resync_forwards
            if not generated_tokens:
                raise RuntimeError("Cannot resync pattern_sd cache before any generated token")
            full_fast_tokens = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            full_fast_masks = torch.ones_like(full_fast_tokens, dtype=torch.bool)
            full_embs, full_pad_masks, full_att_masks, _total_t_images, _num_fast_embs = self.model.embed_prefix_fast(
                images,
                img_masks,
                tokens,
                masks,
                fast_action_tokens=full_fast_tokens,
                fast_action_masks=full_fast_masks,
            )
            full_embs = self._match_model_precision(full_embs)
            full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
            full_att_4d = self.model._prepare_attention_masks_4d(full_att_masks, dtype=full_embs.dtype)
            (full_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=full_att_4d,
                position_ids=full_position_ids,
                past_key_values=None,
                inputs_embeds=full_embs,
                use_cache=True,
                cache_position=torch.arange(full_embs.shape[1], device=device, dtype=torch.long),
            )
            current_pad_mask = full_pad_masks
            target_forwards += 1
            resync_forwards += 1
            return lm_head(full_out[:, -1:, :])

        def verify_candidate_batch(
            candidate_rows: list[list[int]],
            *,
            prepend_prev_logits: bool,
        ) -> tuple[torch.Tensor, Any, torch.Tensor, list[int]]:
            nonlocal target_forwards, verify_forwards, tree_verifies, tree_candidates
            if not candidate_rows:
                raise ValueError("verify_candidate_batch requires at least one candidate")
            lengths = [len(row) for row in candidate_rows]
            max_len = max(lengths)
            batch_candidates = len(candidate_rows)
            padded_rows = [row + [row[-1]] * (max_len - len(row)) for row in candidate_rows]
            candidate_tensor = torch.tensor(padded_rows, dtype=torch.long, device=device)
            old_mask_len = current_pad_mask.shape[1]
            verify_past_key_values = repeat_kv_batch(past_key_values, batch_candidates)
            candidate_embs = self.model.paligemma_with_expert.embed_language_tokens(candidate_tensor)
            candidate_embs = candidate_embs * math.sqrt(candidate_embs.shape[-1])
            candidate_embs = candidate_embs.to(dtype=prefix_embs.dtype)
            verify_pad_mask = torch.cat(
                [
                    current_pad_mask.repeat(batch_candidates, 1),
                    torch.ones((batch_candidates, max_len), dtype=torch.bool, device=device),
                ],
                dim=1,
            )
            verify_mask = torch.zeros(
                (batch_candidates, max_len, old_mask_len + max_len),
                dtype=torch.bool,
                device=device,
            )
            for row in range(max_len):
                verify_mask[:, row, : old_mask_len + row + 1] = verify_pad_mask[:, : old_mask_len + row + 1]
            verify_att_mask = self.model._prepare_attention_masks_4d(verify_mask, dtype=candidate_embs.dtype)
            position_start = int(torch.sum(current_pad_mask, dim=1).item())
            verify_position_ids = torch.arange(
                position_start,
                position_start + max_len,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).repeat(batch_candidates, 1)
            (verify_out, _), verify_kv = self._forward_prefix_language_model(
                attention_mask=verify_att_mask,
                position_ids=verify_position_ids,
                past_key_values=verify_past_key_values,
                inputs_embeds=candidate_embs,
                use_cache=True,
                cache_position=verify_position_ids[0],
            )
            verify_logits = lm_head(verify_out)
            target_forwards += 1
            verify_forwards += 1
            tree_verifies += 1
            tree_candidates += batch_candidates
            if prepend_prev_logits:
                return torch.cat([prev_logits.repeat(batch_candidates, 1, 1), verify_logits], dim=1), verify_kv, verify_pad_mask, lengths
            return verify_logits, verify_kv, verify_pad_mask, lengths

        if hasattr(drafter, "reset_source_feedback"):
            drafter.reset_source_feedback()

        while len(generated_tokens) < max_decoding_steps:
            remaining = max_decoding_steps - len(generated_tokens)
            block_lookahead = current_lookahead if dynamic_lookahead else max_lookahead
            block_lookahead = max(1, min(int(block_lookahead), remaining))
            lookahead_values.append(block_lookahead)
            block_tree_width = current_tree_width if dynamic_tree_width else max_tree_width
            block_tree_width = max(1, min(max_tree_width, int(block_tree_width)))
            tree_width_values.append(block_tree_width)
            if pending_unprocessed_token is not None:
                if action_end_token_id is not None and int(pending_unprocessed_token.item()) == action_end_token_id:
                    break
                if block_tree_width > 1 and hasattr(drafter, "draft_many"):
                    raw_future_rows = drafter.draft_many(
                        generated_tokens,
                        lookahead=block_lookahead,
                        max_candidates=block_tree_width,
                        branch_width=max_tree_branch_width,
                    )
                    raw_future_sources = last_many_sources()
                    future_rows: list[list[int]] = []
                    future_sources: list[list[str]] = []
                    for row_idx, row in enumerate(raw_future_rows):
                        if not row:
                            continue
                        future_rows.append([int(token) for token in row[:remaining]])
                        sources = raw_future_sources[row_idx] if row_idx < len(raw_future_sources) else []
                        future_sources.append(sources[: len(future_rows[-1])])
                    if not future_rows:
                        update_dynamic_lookahead(accepted=0, drafted=0, miss=True)
                        if dynamic_tree_width:
                            current_tree_width = max(
                                1,
                                current_tree_width - max(int(tree_width_shrink), 1),
                            )
                        record_source_miss()
                        prev_logits = process_pending_one()
                        continue

                    pending_value = int(pending_unprocessed_token.item())
                    candidate_rows = [[pending_value, *future] for future in future_rows]
                    drafted_tokens += sum(len(future) for future in future_rows)
                    second_order_action_extrapolation_drafted_tokens += sum(
                        source_count(sources, "second_order_action_extrapolation")
                        for sources in future_sources
                    )
                    previous_chunk_position_drafted_tokens += sum(
                        source_count(sources, "previous_chunk_position") for sources in future_sources
                    )
                    chunk_prefix_retrieval_drafted_tokens += sum(
                        source_count(sources, "chunk_prefix_retrieval") for sources in future_sources
                    )
                    chunk_length_stop_drafted_tokens += sum(
                        source_count(sources, "chunk_length_stop") for sources in future_sources
                    )
                    position_mode_histogram_drafted_tokens += sum(
                        source_count(sources, "position_mode_histogram") for sources in future_sources
                    )
                    global_position_mode_drafted_tokens += sum(
                        source_count(sources, "global_position_mode") for sources in future_sources
                    )
                    action_dimension_mode_drafted_tokens += sum(
                        source_count(sources, "action_dimension_mode") for sources in future_sources
                    )
                    hold_action_token_drafted_tokens += sum(
                        source_count(sources, "hold_action_token") for sources in future_sources
                    )
                    ngram_continuation_drafted_tokens += sum(
                        source_count(sources, "ngram_continuation") for sources in future_sources
                    )
                    source_agreement_drafted_tokens += sum(
                        source_count(sources, "source_agreement") for sources in future_sources
                    )
                    action_trend_regression_drafted_tokens += sum(
                        source_count(sources, "action_trend_regression") for sources in future_sources
                    )
                    action_prefix_lookup_drafted_tokens += sum(
                        source_count(sources, "action_prefix_lookup") for sources in future_sources
                    )
                    action_vector_transition_drafted_tokens += sum(
                        source_count(sources, "action_vector_transition") for sources in future_sources
                    )
                    action_repeat_vector_drafted_tokens += sum(
                        source_count(sources, "action_repeat_vector") for sources in future_sources
                    )
                    action_token_neighborhood_drafted_tokens += sum(
                        source_count(sources, "action_token_neighborhood") for sources in future_sources
                    )
                    action_context_tree_drafted_tokens += sum(
                        source_count(sources, "action_context_tree") for sources in future_sources
                    )
                    action_transition_histogram_drafted_tokens += sum(
                        source_count(sources, "action_transition_histogram") for sources in future_sources
                    )
                    action_delta_histogram_drafted_tokens += sum(
                        source_count(sources, "action_delta_histogram") for sources in future_sources
                    )
                    action_delta_ngram_drafted_tokens += sum(
                        source_count(sources, "action_delta_ngram") for sources in future_sources
                    )
                    chunk_position_delta_drafted_tokens += sum(
                        source_count(sources, "chunk_position_delta") for sources in future_sources
                    )
                    chunk_delta_template_drafted_tokens += sum(
                        source_count(sources, "chunk_delta_template") for sources in future_sources
                    )
                    old_mask_len = current_pad_mask.shape[1]
                    future_logits_all, verify_kv, verify_pad_mask, _candidate_lengths = verify_candidate_batch(
                        candidate_rows,
                        prepend_prev_logits=False,
                    )
                    pending_verifies += 1

                    best_idx = 0
                    future_accepted = -1
                    for row_idx, future_candidate in enumerate(future_rows):
                        accepted = 0
                        for idx, token in enumerate(future_candidate):
                            predicted_future = int(
                                torch.argmax(
                                    future_logits_all[row_idx : row_idx + 1, idx : idx + 1, :][:, -1],
                                    dim=-1,
                                ).item()
                            )
                            if predicted_future != int(token):
                                break
                            accepted += 1
                        if accepted > future_accepted:
                            best_idx = row_idx
                            future_accepted = accepted
                            if accepted == len(future_candidate):
                                break

                    future = future_rows[best_idx]
                    future_logits_all = future_logits_all[best_idx : best_idx + 1]
                    verify_kv = select_kv_batch(verify_kv, best_idx)
                    verify_pad_mask = verify_pad_mask[best_idx : best_idx + 1]
                    future_accepted = max(future_accepted, 0)
                    if len(debug_events) < 64:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "pending_tree_verify",
                                "candidates": len(future_rows),
                                "future_len": len(future),
                                "accepted": future_accepted,
                            }
                        )
                    accepted_emit = future_accepted
                    if action_end_token_id is not None:
                        for idx in range(future_accepted):
                            if int(future[idx]) == action_end_token_id:
                                accepted_emit = idx + 1
                                break
                    for idx in range(accepted_emit):
                        logits_for_token = future_logits_all[:, idx : idx + 1, :]
                        logits_by_step.append(logits_for_token)
                        generated_tokens.append(int(future[idx]))
                    accepted_tokens += accepted_emit
                    tree_accepted_tokens += accepted_emit
                    if best_idx < len(future_sources):
                        record_source_feedback(future_sources[best_idx], future_accepted)
                        second_order_action_extrapolation_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "second_order_action_extrapolation",
                            accepted_emit,
                        )
                        previous_chunk_position_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "previous_chunk_position",
                            accepted_emit,
                        )
                        chunk_prefix_retrieval_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "chunk_prefix_retrieval",
                            accepted_emit,
                        )
                        chunk_length_stop_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "chunk_length_stop",
                            accepted_emit,
                        )
                        position_mode_histogram_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "position_mode_histogram",
                            accepted_emit,
                        )
                        global_position_mode_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "global_position_mode",
                            accepted_emit,
                        )
                        action_dimension_mode_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_dimension_mode",
                            accepted_emit,
                        )
                        hold_action_token_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "hold_action_token",
                            accepted_emit,
                        )
                        ngram_continuation_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "ngram_continuation",
                            accepted_emit,
                        )
                        source_agreement_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "source_agreement",
                            accepted_emit,
                        )
                        action_trend_regression_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_trend_regression",
                            accepted_emit,
                        )
                        action_prefix_lookup_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_prefix_lookup",
                            accepted_emit,
                        )
                        action_vector_transition_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_vector_transition",
                            accepted_emit,
                        )
                        action_repeat_vector_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_repeat_vector",
                            accepted_emit,
                        )
                        action_token_neighborhood_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_token_neighborhood",
                            accepted_emit,
                        )
                        action_context_tree_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_context_tree",
                            accepted_emit,
                        )
                        action_transition_histogram_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_transition_histogram",
                            accepted_emit,
                        )
                        action_delta_histogram_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_delta_histogram",
                            accepted_emit,
                        )
                        action_delta_ngram_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "action_delta_ngram",
                            accepted_emit,
                        )
                        chunk_position_delta_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "chunk_position_delta",
                            accepted_emit,
                        )
                        chunk_delta_template_accepted_tokens += source_count(
                            future_sources[best_idx],
                            "chunk_delta_template",
                            accepted_emit,
                        )
                    else:
                        record_source_miss()
                    update_dynamic_lookahead(accepted=future_accepted, drafted=len(future))
                    if dynamic_tree_width:
                        if future_accepted == len(future):
                            current_tree_width = min(
                                max_tree_width,
                                current_tree_width + max(int(tree_width_growth), 1),
                            )
                        else:
                            current_tree_width = max(
                                1,
                                current_tree_width - max(int(tree_width_shrink), 1),
                            )

                    keep_len = old_mask_len + 1 + accepted_emit
                    past_key_values = trim_kv(verify_kv, keep_len)
                    current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()
                    pending_unprocessed_token = None

                    if len(generated_tokens) >= max_decoding_steps:
                        break
                    if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                        break
                    if future_accepted == len(future):
                        full_block_reuses += 1
                        bonus_logits = future_logits_all[:, len(future) : len(future) + 1, :]
                        if emit_bonus_token and bonus_logits.shape[1] == 1 and len(generated_tokens) < max_decoding_steps:
                            bonus_token = torch.argmax(bonus_logits[:, -1], dim=-1, keepdim=True)
                            logits_by_step.append(bonus_logits)
                            generated_tokens.append(int(bonus_token.item()))
                            bonus_tokens += 1
                            pending_unprocessed_token = bonus_token
                            prev_logits = bonus_logits
                            if action_end_token_id is not None and int(bonus_token.item()) == action_end_token_id:
                                break
                        else:
                            prev_logits = bonus_logits
                    else:
                        correction_logits = future_logits_all[:, future_accepted : future_accepted + 1, :]
                        correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
                        logits_by_step.append(correction_logits)
                        generated_tokens.append(int(correction_token.item()))
                        pending_corrections += 1
                        pending_unprocessed_token = correction_token
                        prev_logits = correction_logits
                        if action_end_token_id is not None and int(correction_token.item()) == action_end_token_id:
                            break
                    continue
                future = drafter.draft(generated_tokens, lookahead=block_lookahead)
                future_sources = last_draft_sources()
                future = future[:remaining]
                future_sources = future_sources[: len(future)]
                if not future:
                    update_dynamic_lookahead(accepted=0, drafted=0, miss=True)
                    record_source_miss()
                    prev_logits = process_pending_one()
                    continue

                candidate = [int(pending_unprocessed_token.item()), *[int(token) for token in future]]
                candidate_tensor = torch.tensor([candidate], dtype=torch.long, device=device)
                drafted_tokens += len(future)
                second_order_action_extrapolation_drafted_tokens += source_count(
                    future_sources,
                    "second_order_action_extrapolation",
                )
                previous_chunk_position_drafted_tokens += source_count(
                    future_sources,
                    "previous_chunk_position",
                )
                chunk_prefix_retrieval_drafted_tokens += source_count(
                    future_sources,
                    "chunk_prefix_retrieval",
                )
                chunk_length_stop_drafted_tokens += source_count(
                    future_sources,
                    "chunk_length_stop",
                )
                position_mode_histogram_drafted_tokens += source_count(
                    future_sources,
                    "position_mode_histogram",
                )
                global_position_mode_drafted_tokens += source_count(
                    future_sources,
                    "global_position_mode",
                )
                action_dimension_mode_drafted_tokens += source_count(
                    future_sources,
                    "action_dimension_mode",
                )
                hold_action_token_drafted_tokens += source_count(
                    future_sources,
                    "hold_action_token",
                )
                ngram_continuation_drafted_tokens += source_count(
                    future_sources,
                    "ngram_continuation",
                )
                source_agreement_drafted_tokens += source_count(
                    future_sources,
                    "source_agreement",
                )
                action_trend_regression_drafted_tokens += source_count(
                    future_sources,
                    "action_trend_regression",
                )
                action_prefix_lookup_drafted_tokens += source_count(
                    future_sources,
                    "action_prefix_lookup",
                )
                action_vector_transition_drafted_tokens += source_count(
                    future_sources,
                    "action_vector_transition",
                )
                action_repeat_vector_drafted_tokens += source_count(
                    future_sources,
                    "action_repeat_vector",
                )
                action_token_neighborhood_drafted_tokens += source_count(
                    future_sources,
                    "action_token_neighborhood",
                )
                action_context_tree_drafted_tokens += source_count(
                    future_sources,
                    "action_context_tree",
                )
                action_transition_histogram_drafted_tokens += source_count(
                    future_sources,
                    "action_transition_histogram",
                )
                action_delta_histogram_drafted_tokens += source_count(
                    future_sources,
                    "action_delta_histogram",
                )
                action_delta_ngram_drafted_tokens += source_count(
                    future_sources,
                    "action_delta_ngram",
                )
                chunk_position_delta_drafted_tokens += source_count(
                    future_sources,
                    "chunk_position_delta",
                )
                chunk_delta_template_drafted_tokens += source_count(
                    future_sources,
                    "chunk_delta_template",
                )
                old_mask_len = current_pad_mask.shape[1]
                if verify_from_scratch:
                    full_fast_tokens = torch.tensor(
                        [generated_tokens + [int(token) for token in future]],
                        dtype=torch.long,
                        device=device,
                    )
                    full_fast_masks = torch.ones_like(full_fast_tokens, dtype=torch.bool)
                    full_embs, full_pad_masks, full_att_masks, _total_t_images, num_fast_embs = self.model.embed_prefix_fast(
                        images,
                        img_masks,
                        tokens,
                        masks,
                        fast_action_tokens=full_fast_tokens,
                        fast_action_masks=full_fast_masks,
                    )
                    full_embs = self._match_model_precision(full_embs)
                    full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
                    full_att_4d = self.model._prepare_attention_masks_4d(full_att_masks, dtype=full_embs.dtype)
                    (verify_out, _), verify_kv = self._forward_prefix_language_model(
                        attention_mask=full_att_4d,
                        position_ids=full_position_ids,
                        past_key_values=None,
                        inputs_embeds=full_embs,
                        use_cache=True,
                        cache_position=torch.arange(full_embs.shape[1], device=device, dtype=torch.long),
                    )
                    bos_hidden_idx = verify_out.shape[1] - num_fast_embs - 1
                    start = bos_hidden_idx + len(generated_tokens)
                    future_logits_all = lm_head(verify_out[:, start : start + len(future) + 1, :])
                    verify_pad_mask = full_pad_masks
                else:
                    verify_past_key_values = clone_kv(past_key_values)
                    candidate_embs = self.model.paligemma_with_expert.embed_language_tokens(candidate_tensor)
                    candidate_embs = candidate_embs * math.sqrt(candidate_embs.shape[-1])
                    candidate_embs = candidate_embs.to(dtype=prefix_embs.dtype)
                    verify_pad_mask = torch.cat(
                        [current_pad_mask, torch.ones((bsize, len(candidate)), dtype=torch.bool, device=device)],
                        dim=1,
                    )
                    verify_mask = torch.zeros(
                        (bsize, len(candidate), old_mask_len + len(candidate)),
                        dtype=torch.bool,
                        device=device,
                    )
                    for row in range(len(candidate)):
                        verify_mask[:, row, : old_mask_len + row + 1] = verify_pad_mask[:, : old_mask_len + row + 1]
                    verify_att_mask = self.model._prepare_attention_masks_4d(verify_mask, dtype=candidate_embs.dtype)
                    position_start = int(torch.sum(current_pad_mask, dim=1).item())
                    verify_position_ids = torch.arange(
                        position_start,
                        position_start + len(candidate),
                        device=device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                    (verify_out, _), verify_kv = self._forward_prefix_language_model(
                        attention_mask=verify_att_mask,
                        position_ids=verify_position_ids,
                        past_key_values=verify_past_key_values,
                        inputs_embeds=candidate_embs,
                        use_cache=True,
                        cache_position=verify_position_ids.squeeze(0),
                    )
                    future_logits_all = lm_head(verify_out)
                target_forwards += 1
                verify_forwards += 1
                pending_verifies += 1

                future_accepted = 0
                for idx, token in enumerate(future):
                    predicted_future = int(torch.argmax(future_logits_all[:, idx : idx + 1, :][:, -1], dim=-1).item())
                    if predicted_future != int(token):
                        break
                    future_accepted += 1
                if len(debug_events) < 64:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "pending_verify",
                            "future_len": len(future),
                            "accepted": future_accepted,
                        }
                    )
                accepted_emit = future_accepted
                if action_end_token_id is not None:
                    for idx in range(future_accepted):
                        if int(future[idx]) == action_end_token_id:
                            accepted_emit = idx + 1
                            break
                for idx in range(accepted_emit):
                    logits_for_token = future_logits_all[:, idx : idx + 1, :]
                    logits_by_step.append(logits_for_token)
                    generated_tokens.append(int(future[idx]))
                accepted_tokens += accepted_emit
                record_source_feedback(future_sources, future_accepted)
                second_order_action_extrapolation_accepted_tokens += source_count(
                    future_sources,
                    "second_order_action_extrapolation",
                    accepted_emit,
                )
                previous_chunk_position_accepted_tokens += source_count(
                    future_sources,
                    "previous_chunk_position",
                    accepted_emit,
                )
                chunk_prefix_retrieval_accepted_tokens += source_count(
                    future_sources,
                    "chunk_prefix_retrieval",
                    accepted_emit,
                )
                chunk_length_stop_accepted_tokens += source_count(
                    future_sources,
                    "chunk_length_stop",
                    accepted_emit,
                )
                position_mode_histogram_accepted_tokens += source_count(
                    future_sources,
                    "position_mode_histogram",
                    accepted_emit,
                )
                global_position_mode_accepted_tokens += source_count(
                    future_sources,
                    "global_position_mode",
                    accepted_emit,
                )
                action_dimension_mode_accepted_tokens += source_count(
                    future_sources,
                    "action_dimension_mode",
                    accepted_emit,
                )
                hold_action_token_accepted_tokens += source_count(
                    future_sources,
                    "hold_action_token",
                    accepted_emit,
                )
                ngram_continuation_accepted_tokens += source_count(
                    future_sources,
                    "ngram_continuation",
                    accepted_emit,
                )
                source_agreement_accepted_tokens += source_count(
                    future_sources,
                    "source_agreement",
                    accepted_emit,
                )
                action_trend_regression_accepted_tokens += source_count(
                    future_sources,
                    "action_trend_regression",
                    accepted_emit,
                )
                action_prefix_lookup_accepted_tokens += source_count(
                    future_sources,
                    "action_prefix_lookup",
                    accepted_emit,
                )
                action_vector_transition_accepted_tokens += source_count(
                    future_sources,
                    "action_vector_transition",
                    accepted_emit,
                )
                action_repeat_vector_accepted_tokens += source_count(
                    future_sources,
                    "action_repeat_vector",
                    accepted_emit,
                )
                action_token_neighborhood_accepted_tokens += source_count(
                    future_sources,
                    "action_token_neighborhood",
                    accepted_emit,
                )
                action_context_tree_accepted_tokens += source_count(
                    future_sources,
                    "action_context_tree",
                    accepted_emit,
                )
                action_transition_histogram_accepted_tokens += source_count(
                    future_sources,
                    "action_transition_histogram",
                    accepted_emit,
                )
                action_delta_histogram_accepted_tokens += source_count(
                    future_sources,
                    "action_delta_histogram",
                    accepted_emit,
                )
                action_delta_ngram_accepted_tokens += source_count(
                    future_sources,
                    "action_delta_ngram",
                    accepted_emit,
                )
                chunk_position_delta_accepted_tokens += source_count(
                    future_sources,
                    "chunk_position_delta",
                    accepted_emit,
                )
                chunk_delta_template_accepted_tokens += source_count(
                    future_sources,
                    "chunk_delta_template",
                    accepted_emit,
                )
                update_dynamic_lookahead(accepted=future_accepted, drafted=len(future))

                keep_len = old_mask_len + 1 + accepted_emit
                past_key_values = trim_kv(verify_kv, keep_len)
                current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()
                pending_unprocessed_token = None

                if len(generated_tokens) >= max_decoding_steps:
                    break
                if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                    break
                if future_accepted == len(future):
                    full_block_reuses += 1
                    bonus_logits = future_logits_all[:, len(future) : len(future) + 1, :]
                    if emit_bonus_token and bonus_logits.shape[1] == 1 and len(generated_tokens) < max_decoding_steps:
                        bonus_token = torch.argmax(bonus_logits[:, -1], dim=-1, keepdim=True)
                        logits_by_step.append(bonus_logits)
                        generated_tokens.append(int(bonus_token.item()))
                        bonus_tokens += 1
                        pending_unprocessed_token = bonus_token
                        prev_logits = bonus_logits
                        if action_end_token_id is not None and int(bonus_token.item()) == action_end_token_id:
                            break
                    else:
                        prev_logits = bonus_logits
                else:
                    correction_logits = future_logits_all[:, future_accepted : future_accepted + 1, :]
                    correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
                    logits_by_step.append(correction_logits)
                    generated_tokens.append(int(correction_token.item()))
                    pending_corrections += 1
                    pending_unprocessed_token = correction_token
                    prev_logits = correction_logits
                    if action_end_token_id is not None and int(correction_token.item()) == action_end_token_id:
                        break
                continue

            target_next = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)
            if action_end_token_id is not None and int(target_next.item()) == action_end_token_id:
                logits_by_step.append(prev_logits)
                generated_tokens.append(int(target_next.item()))
                pending_unprocessed_token = target_next
                if len(debug_events) < 64:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens) - 1,
                            "kind": "action_end",
                            "token": int(target_next.item()),
                        }
                )
                break
            if block_tree_width > 1 and hasattr(drafter, "draft_many"):
                target_token_value = int(target_next.item())
                force_anchor = bool(
                    tree_anchor_target_continuation and remaining > 1 and block_lookahead > 1
                )
                if force_anchor:
                    raw_draft_rows = [[target_token_value + 1]]
                    raw_draft_sources = [[]]
                else:
                    raw_draft_rows = drafter.draft_many(
                        generated_tokens,
                        lookahead=block_lookahead,
                        max_candidates=block_tree_width,
                        branch_width=max_tree_branch_width,
                    )
                    raw_draft_sources = last_many_sources()
                if raw_draft_rows and not force_anchor:
                    tree_first_token_checks += 1
                draft_rows: list[list[int]] = []
                draft_sources: list[list[str]] = []
                for row_idx, row in enumerate(raw_draft_rows):
                    if not row or int(row[0]) != target_token_value:
                        continue
                    draft_rows.append([int(token) for token in row[:remaining]])
                    sources = raw_draft_sources[row_idx] if row_idx < len(raw_draft_sources) else []
                    draft_sources.append(sources[: len(draft_rows[-1])])
                if not draft_rows:
                    if raw_draft_rows and not force_anchor:
                        tree_first_token_misses += 1
                    if (
                        (tree_anchor_target_token or force_anchor)
                        and raw_draft_rows
                        and remaining > 1
                        and block_lookahead > 1
                    ):
                        anchor_lookahead = max(1, min(block_lookahead - 1, remaining - 1))
                        anchor_prefix = [*generated_tokens, target_token_value]
                        raw_anchor_rows = drafter.draft_many(
                            anchor_prefix,
                            lookahead=anchor_lookahead,
                            max_candidates=block_tree_width,
                            branch_width=max_tree_branch_width,
                        )
                        raw_anchor_sources = last_many_sources()
                        anchor_future_rows: list[list[int]] = []
                        anchor_future_sources: list[list[str]] = []
                        for row_idx, row in enumerate(raw_anchor_rows):
                            future = [int(token) for token in row[: remaining - 1]]
                            if not future:
                                continue
                            anchor_future_rows.append(future)
                            sources = raw_anchor_sources[row_idx] if row_idx < len(raw_anchor_sources) else []
                            anchor_future_sources.append(sources[: len(future)])
                        if anchor_future_rows:
                            if raw_draft_sources:
                                record_source_feedback(raw_draft_sources[0], 0)
                            drafted_tokens += sum(len(row) for row in anchor_future_rows)
                            second_order_action_extrapolation_drafted_tokens += sum(
                                source_count(sources, "second_order_action_extrapolation")
                                for sources in anchor_future_sources
                            )
                            previous_chunk_position_drafted_tokens += sum(
                                source_count(sources, "previous_chunk_position")
                                for sources in anchor_future_sources
                            )
                            chunk_prefix_retrieval_drafted_tokens += sum(
                                source_count(sources, "chunk_prefix_retrieval")
                                for sources in anchor_future_sources
                            )
                            chunk_length_stop_drafted_tokens += sum(
                                source_count(sources, "chunk_length_stop")
                                for sources in anchor_future_sources
                            )
                            position_mode_histogram_drafted_tokens += sum(
                                source_count(sources, "position_mode_histogram")
                                for sources in anchor_future_sources
                            )
                            global_position_mode_drafted_tokens += sum(
                                source_count(sources, "global_position_mode")
                                for sources in anchor_future_sources
                            )
                            action_dimension_mode_drafted_tokens += sum(
                                source_count(sources, "action_dimension_mode")
                                for sources in anchor_future_sources
                            )
                            hold_action_token_drafted_tokens += sum(
                                source_count(sources, "hold_action_token") for sources in anchor_future_sources
                            )
                            ngram_continuation_drafted_tokens += sum(
                                source_count(sources, "ngram_continuation") for sources in anchor_future_sources
                            )
                            source_agreement_drafted_tokens += sum(
                                source_count(sources, "source_agreement") for sources in anchor_future_sources
                            )
                            action_trend_regression_drafted_tokens += sum(
                                source_count(sources, "action_trend_regression")
                                for sources in anchor_future_sources
                            )
                            action_prefix_lookup_drafted_tokens += sum(
                                source_count(sources, "action_prefix_lookup")
                                for sources in anchor_future_sources
                            )
                            action_vector_transition_drafted_tokens += sum(
                                source_count(sources, "action_vector_transition")
                                for sources in anchor_future_sources
                            )
                            action_repeat_vector_drafted_tokens += sum(
                                source_count(sources, "action_repeat_vector")
                                for sources in anchor_future_sources
                            )
                            action_token_neighborhood_drafted_tokens += sum(
                                source_count(sources, "action_token_neighborhood")
                                for sources in anchor_future_sources
                            )
                            action_context_tree_drafted_tokens += sum(
                                source_count(sources, "action_context_tree") for sources in anchor_future_sources
                            )
                            action_transition_histogram_drafted_tokens += sum(
                                source_count(sources, "action_transition_histogram")
                                for sources in anchor_future_sources
                            )
                            action_delta_histogram_drafted_tokens += sum(
                                source_count(sources, "action_delta_histogram")
                                for sources in anchor_future_sources
                            )
                            action_delta_ngram_drafted_tokens += sum(
                                source_count(sources, "action_delta_ngram")
                                for sources in anchor_future_sources
                            )
                            chunk_position_delta_drafted_tokens += sum(
                                source_count(sources, "chunk_position_delta")
                                for sources in anchor_future_sources
                            )
                            chunk_delta_template_drafted_tokens += sum(
                                source_count(sources, "chunk_delta_template")
                                for sources in anchor_future_sources
                            )
                            old_mask_len = current_pad_mask.shape[1]
                            anchor_rows = [[target_token_value, *future] for future in anchor_future_rows]
                            verify_logits_all, verify_kv, verify_pad_mask, _candidate_lengths = verify_candidate_batch(
                                anchor_rows,
                                prepend_prev_logits=True,
                            )
                            tree_anchor_verifies += 1
                            tree_anchor_candidates += len(anchor_future_rows)
                            best_idx = 0
                            future_accepted = -1
                            for row_idx, future in enumerate(anchor_future_rows):
                                accepted_prefix = 0
                                for idx, token in enumerate(future):
                                    logits_for_candidate = verify_logits_all[
                                        row_idx : row_idx + 1,
                                        idx + 1 : idx + 2,
                                        :,
                                    ]
                                    predicted_candidate = int(
                                        torch.argmax(logits_for_candidate[:, -1], dim=-1).item()
                                    )
                                    if predicted_candidate != int(token):
                                        break
                                    accepted_prefix += 1
                                if accepted_prefix > future_accepted:
                                    best_idx = row_idx
                                    future_accepted = accepted_prefix
                                    if accepted_prefix == len(future):
                                        break

                            future = anchor_future_rows[best_idx]
                            future_source_row = (
                                anchor_future_sources[best_idx] if best_idx < len(anchor_future_sources) else []
                            )
                            verify_logits_all = verify_logits_all[best_idx : best_idx + 1]
                            verify_kv = select_kv_batch(verify_kv, best_idx)
                            verify_pad_mask = verify_pad_mask[best_idx : best_idx + 1]
                            future_accepted = max(future_accepted, 0)
                            if len(debug_events) < 64:
                                debug_events.append(
                                    {
                                        "pos": len(generated_tokens),
                                        "kind": "tree_anchor_verify",
                                        "candidates": len(anchor_future_rows),
                                        "future_len": len(future),
                                        "accepted": future_accepted,
                                    }
                                )

                            logits_by_step.append(verify_logits_all[:, 0:1, :])
                            generated_tokens.append(target_token_value)
                            accepted_emit = future_accepted
                            if action_end_token_id is not None:
                                for idx in range(future_accepted):
                                    if int(future[idx]) == action_end_token_id:
                                        accepted_emit = idx + 1
                                        break
                            for idx in range(accepted_emit):
                                logits_for_token = verify_logits_all[:, idx + 1 : idx + 2, :]
                                logits_by_step.append(logits_for_token)
                                generated_tokens.append(int(future[idx]))

                            keep_len = old_mask_len + 1 + accepted_emit
                            past_key_values = trim_kv(verify_kv, keep_len)
                            current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()
                            accepted_tokens += accepted_emit
                            tree_accepted_tokens += accepted_emit
                            tree_anchor_accepted_tokens += accepted_emit
                            record_source_feedback(future_source_row, future_accepted)
                            second_order_action_extrapolation_accepted_tokens += source_count(
                                future_source_row,
                                "second_order_action_extrapolation",
                                accepted_emit,
                            )
                            previous_chunk_position_accepted_tokens += source_count(
                                future_source_row,
                                "previous_chunk_position",
                                accepted_emit,
                            )
                            chunk_prefix_retrieval_accepted_tokens += source_count(
                                future_source_row,
                                "chunk_prefix_retrieval",
                                accepted_emit,
                            )
                            chunk_length_stop_accepted_tokens += source_count(
                                future_source_row,
                                "chunk_length_stop",
                                accepted_emit,
                            )
                            position_mode_histogram_accepted_tokens += source_count(
                                future_source_row,
                                "position_mode_histogram",
                                accepted_emit,
                            )
                            global_position_mode_accepted_tokens += source_count(
                                future_source_row,
                                "global_position_mode",
                                accepted_emit,
                            )
                            action_dimension_mode_accepted_tokens += source_count(
                                future_source_row,
                                "action_dimension_mode",
                                accepted_emit,
                            )
                            hold_action_token_accepted_tokens += source_count(
                                future_source_row,
                                "hold_action_token",
                                accepted_emit,
                            )
                            ngram_continuation_accepted_tokens += source_count(
                                future_source_row,
                                "ngram_continuation",
                                accepted_emit,
                            )
                            source_agreement_accepted_tokens += source_count(
                                future_source_row,
                                "source_agreement",
                                accepted_emit,
                            )
                            action_trend_regression_accepted_tokens += source_count(
                                future_source_row,
                                "action_trend_regression",
                                accepted_emit,
                            )
                            action_prefix_lookup_accepted_tokens += source_count(
                                future_source_row,
                                "action_prefix_lookup",
                                accepted_emit,
                            )
                            action_vector_transition_accepted_tokens += source_count(
                                future_source_row,
                                "action_vector_transition",
                                accepted_emit,
                            )
                            action_repeat_vector_accepted_tokens += source_count(
                                future_source_row,
                                "action_repeat_vector",
                                accepted_emit,
                            )
                            action_token_neighborhood_accepted_tokens += source_count(
                                future_source_row,
                                "action_token_neighborhood",
                                accepted_emit,
                            )
                            action_context_tree_accepted_tokens += source_count(
                                future_source_row,
                                "action_context_tree",
                                accepted_emit,
                            )
                            action_transition_histogram_accepted_tokens += source_count(
                                future_source_row,
                                "action_transition_histogram",
                                accepted_emit,
                            )
                            action_delta_histogram_accepted_tokens += source_count(
                                future_source_row,
                                "action_delta_histogram",
                                accepted_emit,
                            )
                            action_delta_ngram_accepted_tokens += source_count(
                                future_source_row,
                                "action_delta_ngram",
                                accepted_emit,
                            )
                            chunk_position_delta_accepted_tokens += source_count(
                                future_source_row,
                                "chunk_position_delta",
                                accepted_emit,
                            )
                            chunk_delta_template_accepted_tokens += source_count(
                                future_source_row,
                                "chunk_delta_template",
                                accepted_emit,
                            )
                            update_dynamic_lookahead(accepted=future_accepted, drafted=len(future))
                            if dynamic_tree_width:
                                if future_accepted == len(future):
                                    current_tree_width = min(
                                        max_tree_width,
                                        current_tree_width + max(int(tree_width_growth), 1),
                                    )
                                else:
                                    current_tree_width = max(
                                        1,
                                        current_tree_width - max(int(tree_width_shrink), 1),
                                    )
                            pending_unprocessed_token = None
                            if len(generated_tokens) >= max_decoding_steps:
                                break
                            if (
                                action_end_token_id is not None
                                and generated_tokens
                                and generated_tokens[-1] == action_end_token_id
                            ):
                                break
                            if future_accepted == len(future):
                                full_block_reuses += 1
                                bonus_logits = verify_logits_all[
                                    :,
                                    1 + future_accepted : 2 + future_accepted,
                                    :,
                                ]
                                if (
                                    emit_bonus_token
                                    and bonus_logits.shape[1] == 1
                                    and len(generated_tokens) < max_decoding_steps
                                ):
                                    bonus_token = torch.argmax(bonus_logits[:, -1], dim=-1, keepdim=True)
                                    logits_by_step.append(bonus_logits)
                                    generated_tokens.append(int(bonus_token.item()))
                                    bonus_tokens += 1
                                    pending_unprocessed_token = bonus_token
                                    prev_logits = bonus_logits
                                    if action_end_token_id is not None and int(bonus_token.item()) == action_end_token_id:
                                        break
                                else:
                                    prev_logits = bonus_logits
                            else:
                                correction_logits = verify_logits_all[
                                    :,
                                    1 + future_accepted : 2 + future_accepted,
                                    :,
                                ]
                                correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
                                if (
                                    future_accepted < len(future)
                                    and int(correction_token.item()) == int(future[future_accepted])
                                ):
                                    raise RuntimeError(
                                        "Internal SD invariant failed: anchored prefix stopped before matching token"
                                    )
                                logits_by_step.append(correction_logits)
                                generated_tokens.append(int(correction_token.item()))
                                pending_corrections += 1
                                pending_unprocessed_token = correction_token
                                prev_logits = correction_logits
                                if action_end_token_id is not None and int(correction_token.item()) == action_end_token_id:
                                    break
                            continue
                    if len(debug_events) < 64:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "tree_fallback",
                                "token": target_token_value,
                            }
                    )
                    update_dynamic_lookahead(accepted=0, drafted=0, miss=True)
                    if dynamic_tree_width:
                        current_tree_width = max(1, current_tree_width - max(int(tree_width_shrink), 1))
                    if raw_draft_sources:
                        record_source_feedback(raw_draft_sources[0], 0)
                    else:
                        record_source_miss()
                    prev_logits = advance_one(target_next, prev_logits, fallback=True)
                    continue

                drafted_tokens += sum(len(row) for row in draft_rows)
                second_order_action_extrapolation_drafted_tokens += sum(
                    source_count(sources, "second_order_action_extrapolation") for sources in draft_sources
                )
                previous_chunk_position_drafted_tokens += sum(
                    source_count(sources, "previous_chunk_position") for sources in draft_sources
                )
                chunk_prefix_retrieval_drafted_tokens += sum(
                    source_count(sources, "chunk_prefix_retrieval") for sources in draft_sources
                )
                chunk_length_stop_drafted_tokens += sum(
                    source_count(sources, "chunk_length_stop") for sources in draft_sources
                )
                position_mode_histogram_drafted_tokens += sum(
                    source_count(sources, "position_mode_histogram") for sources in draft_sources
                )
                global_position_mode_drafted_tokens += sum(
                    source_count(sources, "global_position_mode") for sources in draft_sources
                )
                action_dimension_mode_drafted_tokens += sum(
                    source_count(sources, "action_dimension_mode") for sources in draft_sources
                )
                hold_action_token_drafted_tokens += sum(
                    source_count(sources, "hold_action_token") for sources in draft_sources
                )
                ngram_continuation_drafted_tokens += sum(
                    source_count(sources, "ngram_continuation") for sources in draft_sources
                )
                source_agreement_drafted_tokens += sum(
                    source_count(sources, "source_agreement") for sources in draft_sources
                )
                action_trend_regression_drafted_tokens += sum(
                    source_count(sources, "action_trend_regression") for sources in draft_sources
                )
                action_prefix_lookup_drafted_tokens += sum(
                    source_count(sources, "action_prefix_lookup") for sources in draft_sources
                )
                action_vector_transition_drafted_tokens += sum(
                    source_count(sources, "action_vector_transition") for sources in draft_sources
                )
                action_repeat_vector_drafted_tokens += sum(
                    source_count(sources, "action_repeat_vector") for sources in draft_sources
                )
                action_token_neighborhood_drafted_tokens += sum(
                    source_count(sources, "action_token_neighborhood") for sources in draft_sources
                )
                action_context_tree_drafted_tokens += sum(
                    source_count(sources, "action_context_tree") for sources in draft_sources
                )
                action_transition_histogram_drafted_tokens += sum(
                    source_count(sources, "action_transition_histogram") for sources in draft_sources
                )
                action_delta_histogram_drafted_tokens += sum(
                    source_count(sources, "action_delta_histogram") for sources in draft_sources
                )
                action_delta_ngram_drafted_tokens += sum(
                    source_count(sources, "action_delta_ngram") for sources in draft_sources
                )
                chunk_position_delta_drafted_tokens += sum(
                    source_count(sources, "chunk_position_delta") for sources in draft_sources
                )
                chunk_delta_template_drafted_tokens += sum(
                    source_count(sources, "chunk_delta_template") for sources in draft_sources
                )
                old_mask_len = current_pad_mask.shape[1]
                verify_logits_all, verify_kv, verify_pad_mask, _candidate_lengths = verify_candidate_batch(
                    draft_rows,
                    prepend_prev_logits=True,
                )
                best_idx = 0
                block_accepted = -1
                for row_idx, candidate in enumerate(draft_rows):
                    accepted_prefix = 0
                    for idx, token in enumerate(candidate):
                        logits_for_candidate = verify_logits_all[row_idx : row_idx + 1, idx : idx + 1, :]
                        predicted_candidate = int(torch.argmax(logits_for_candidate[:, -1], dim=-1).item())
                        if predicted_candidate != int(token):
                            break
                        accepted_prefix += 1
                    if accepted_prefix > block_accepted:
                        best_idx = row_idx
                        block_accepted = accepted_prefix
                        if accepted_prefix == len(candidate):
                            break

                draft = draft_rows[best_idx]
                draft_source_row = draft_sources[best_idx] if best_idx < len(draft_sources) else []
                verify_logits_all = verify_logits_all[best_idx : best_idx + 1]
                verify_kv = select_kv_batch(verify_kv, best_idx)
                verify_pad_mask = verify_pad_mask[best_idx : best_idx + 1]
                block_accepted = min(max(block_accepted, 0), remaining)
                if len(debug_events) < 64:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "tree_verify",
                            "candidates": len(draft_rows),
                            "draft_len": len(draft),
                            "accepted": block_accepted,
                            "draft": [int(token) for token in draft[: min(len(draft), 12)]],
                        }
                    )
                accepted = 0
                accepted_emit = block_accepted
                if action_end_token_id is not None:
                    for idx in range(block_accepted):
                        if int(draft[idx]) == action_end_token_id:
                            accepted_emit = idx + 1
                            break
                if reuse_full_blocks and block_accepted == len(draft):
                    full_block_reuses += 1
                    for idx in range(accepted_emit):
                        logits_for_token = verify_logits_all[:, idx : idx + 1, :]
                        logits_by_step.append(logits_for_token)
                        generated_tokens.append(int(draft[idx]))
                    accepted = accepted_emit
                    keep_len = old_mask_len + accepted
                    past_key_values = trim_kv(verify_kv, keep_len)
                    current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()
                    if (
                        emit_bonus_token
                        and accepted == len(draft)
                        and len(generated_tokens) < max_decoding_steps
                        and (
                            action_end_token_id is None
                            or not generated_tokens
                            or generated_tokens[-1] != action_end_token_id
                        )
                    ):
                        bonus_logits = verify_logits_all[:, accepted : accepted + 1, :]
                        bonus_token = torch.argmax(bonus_logits[:, -1], dim=-1, keepdim=True)
                        logits_by_step.append(bonus_logits)
                        generated_tokens.append(int(bonus_token.item()))
                        bonus_tokens += 1
                        pending_unprocessed_token = bonus_token
                        prev_logits = bonus_logits
                    else:
                        prev_logits = verify_logits_all[:, accepted : accepted + 1, :]
                else:
                    accepted = accepted_emit
                    for idx in range(accepted):
                        logits_for_token = verify_logits_all[:, idx : idx + 1, :]
                        logits_by_step.append(logits_for_token)
                        generated_tokens.append(int(draft[idx]))

                    keep_len = old_mask_len + accepted
                    past_key_values = trim_kv(verify_kv, keep_len)
                    current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()

                    if (
                        len(generated_tokens) < max_decoding_steps
                        and (
                            action_end_token_id is None
                            or not generated_tokens
                            or generated_tokens[-1] != action_end_token_id
                        )
                    ):
                        correction_logits = verify_logits_all[:, accepted : accepted + 1, :]
                        correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
                        if accepted < len(draft) and int(correction_token.item()) == int(draft[accepted]):
                            raise RuntimeError("Internal SD invariant failed: tree prefix stopped before matching token")
                        if len(debug_events) < 64:
                            debug_events.append(
                                {
                                    "pos": len(generated_tokens),
                                    "kind": "tree_correction",
                                    "accepted": accepted,
                                    "token": int(correction_token.item()),
                                    "rejected_draft": int(draft[accepted]) if accepted < len(draft) else None,
                                }
                            )
                        if defer_correction_token:
                            logits_by_step.append(correction_logits)
                            generated_tokens.append(int(correction_token.item()))
                            pending_corrections += 1
                            pending_unprocessed_token = correction_token
                            prev_logits = correction_logits
                            if action_end_token_id is not None and int(correction_token.item()) == action_end_token_id:
                                break
                        else:
                            prev_logits = advance_one(correction_token, correction_logits, fallback=True)
                accepted_tokens += accepted
                tree_accepted_tokens += accepted
                record_source_feedback(draft_source_row, block_accepted)
                second_order_action_extrapolation_accepted_tokens += source_count(
                    draft_source_row,
                    "second_order_action_extrapolation",
                    accepted,
                )
                previous_chunk_position_accepted_tokens += source_count(
                    draft_source_row,
                    "previous_chunk_position",
                    accepted,
                )
                chunk_prefix_retrieval_accepted_tokens += source_count(
                    draft_source_row,
                    "chunk_prefix_retrieval",
                    accepted,
                )
                chunk_length_stop_accepted_tokens += source_count(
                    draft_source_row,
                    "chunk_length_stop",
                    accepted,
                )
                position_mode_histogram_accepted_tokens += source_count(
                    draft_source_row,
                    "position_mode_histogram",
                    accepted,
                )
                global_position_mode_accepted_tokens += source_count(
                    draft_source_row,
                    "global_position_mode",
                    accepted,
                )
                action_dimension_mode_accepted_tokens += source_count(
                    draft_source_row,
                    "action_dimension_mode",
                    accepted,
                )
                hold_action_token_accepted_tokens += source_count(
                    draft_source_row,
                    "hold_action_token",
                    accepted,
                )
                ngram_continuation_accepted_tokens += source_count(
                    draft_source_row,
                    "ngram_continuation",
                    accepted,
                )
                source_agreement_accepted_tokens += source_count(
                    draft_source_row,
                    "source_agreement",
                    accepted,
                )
                action_trend_regression_accepted_tokens += source_count(
                    draft_source_row,
                    "action_trend_regression",
                    accepted,
                )
                action_prefix_lookup_accepted_tokens += source_count(
                    draft_source_row,
                    "action_prefix_lookup",
                    accepted,
                )
                action_vector_transition_accepted_tokens += source_count(
                    draft_source_row,
                    "action_vector_transition",
                    accepted,
                )
                action_repeat_vector_accepted_tokens += source_count(
                    draft_source_row,
                    "action_repeat_vector",
                    accepted,
                )
                action_token_neighborhood_accepted_tokens += source_count(
                    draft_source_row,
                    "action_token_neighborhood",
                    accepted,
                )
                action_context_tree_accepted_tokens += source_count(
                    draft_source_row,
                    "action_context_tree",
                    accepted,
                )
                action_transition_histogram_accepted_tokens += source_count(
                    draft_source_row,
                    "action_transition_histogram",
                    accepted,
                )
                action_delta_histogram_accepted_tokens += source_count(
                    draft_source_row,
                    "action_delta_histogram",
                    accepted,
                )
                action_delta_ngram_accepted_tokens += source_count(
                    draft_source_row,
                    "action_delta_ngram",
                    accepted,
                )
                chunk_position_delta_accepted_tokens += source_count(
                    draft_source_row,
                    "chunk_position_delta",
                    accepted,
                )
                chunk_delta_template_accepted_tokens += source_count(
                    draft_source_row,
                    "chunk_delta_template",
                    accepted,
                )
                update_dynamic_lookahead(accepted=block_accepted, drafted=len(draft))
                if dynamic_tree_width:
                    if block_accepted == len(draft):
                        current_tree_width = min(
                            max_tree_width,
                            current_tree_width + max(int(tree_width_growth), 1),
                        )
                    else:
                        current_tree_width = max(1, current_tree_width - max(int(tree_width_shrink), 1))
                if len(generated_tokens) >= max_decoding_steps:
                    break
                if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                    break
                continue
            draft = drafter.draft(generated_tokens, lookahead=block_lookahead)
            draft_sources = last_draft_sources()
            if not draft or int(draft[0]) != int(target_next.item()):
                if len(debug_events) < 64:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "fallback",
                            "token": int(target_next.item()),
                            "draft0": int(draft[0]) if draft else None,
                        }
                    )
                update_dynamic_lookahead(accepted=0, drafted=len(draft), miss=True)
                if dynamic_tree_width:
                    current_tree_width = max(1, current_tree_width - max(int(tree_width_shrink), 1))
                if draft_sources:
                    record_source_feedback(draft_sources, 0)
                else:
                    record_source_miss()
                prev_logits = advance_one(target_next, prev_logits, fallback=True)
                continue

            draft = draft[:remaining]
            draft_sources = draft_sources[: len(draft)]
            draft_tensor = torch.tensor([draft], dtype=torch.long, device=device)
            drafted_tokens += len(draft)
            second_order_action_extrapolation_drafted_tokens += source_count(
                draft_sources,
                "second_order_action_extrapolation",
            )
            previous_chunk_position_drafted_tokens += source_count(
                draft_sources,
                "previous_chunk_position",
            )
            chunk_prefix_retrieval_drafted_tokens += source_count(
                draft_sources,
                "chunk_prefix_retrieval",
            )
            chunk_length_stop_drafted_tokens += source_count(
                draft_sources,
                "chunk_length_stop",
            )
            position_mode_histogram_drafted_tokens += source_count(
                draft_sources,
                "position_mode_histogram",
            )
            global_position_mode_drafted_tokens += source_count(
                draft_sources,
                "global_position_mode",
            )
            action_dimension_mode_drafted_tokens += source_count(
                draft_sources,
                "action_dimension_mode",
            )
            hold_action_token_drafted_tokens += source_count(
                draft_sources,
                "hold_action_token",
            )
            ngram_continuation_drafted_tokens += source_count(
                draft_sources,
                "ngram_continuation",
            )
            source_agreement_drafted_tokens += source_count(
                draft_sources,
                "source_agreement",
            )
            action_trend_regression_drafted_tokens += source_count(
                draft_sources,
                "action_trend_regression",
            )
            action_prefix_lookup_drafted_tokens += source_count(
                draft_sources,
                "action_prefix_lookup",
            )
            action_vector_transition_drafted_tokens += source_count(
                draft_sources,
                "action_vector_transition",
            )
            action_repeat_vector_drafted_tokens += source_count(
                draft_sources,
                "action_repeat_vector",
            )
            action_token_neighborhood_drafted_tokens += source_count(
                draft_sources,
                "action_token_neighborhood",
            )
            action_context_tree_drafted_tokens += source_count(
                draft_sources,
                "action_context_tree",
            )
            action_transition_histogram_drafted_tokens += source_count(
                draft_sources,
                "action_transition_histogram",
            )
            action_delta_histogram_drafted_tokens += source_count(
                draft_sources,
                "action_delta_histogram",
            )
            action_delta_ngram_drafted_tokens += source_count(
                draft_sources,
                "action_delta_ngram",
            )
            chunk_position_delta_drafted_tokens += source_count(
                draft_sources,
                "chunk_position_delta",
            )
            chunk_delta_template_drafted_tokens += source_count(
                draft_sources,
                "chunk_delta_template",
            )
            old_mask_len = current_pad_mask.shape[1]
            generated_before_verify = list(generated_tokens)
            if verify_from_scratch:
                full_fast_tokens = torch.tensor(
                    [generated_before_verify + [int(token) for token in draft]],
                    dtype=torch.long,
                    device=device,
                )
                full_fast_masks = torch.ones_like(full_fast_tokens, dtype=torch.bool)
                full_embs, full_pad_masks, full_att_masks, _total_t_images, num_fast_embs = self.model.embed_prefix_fast(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    fast_action_tokens=full_fast_tokens,
                    fast_action_masks=full_fast_masks,
                )
                full_embs = self._match_model_precision(full_embs)
                full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
                full_att_4d = self.model._prepare_attention_masks_4d(full_att_masks, dtype=full_embs.dtype)
                (verify_out, _), verify_kv = self._forward_prefix_language_model(
                    attention_mask=full_att_4d,
                    position_ids=full_position_ids,
                    past_key_values=None,
                    inputs_embeds=full_embs,
                    use_cache=True,
                    cache_position=torch.arange(full_embs.shape[1], device=device, dtype=torch.long),
                )
                bos_hidden_idx = verify_out.shape[1] - num_fast_embs - 1
                start = bos_hidden_idx + len(generated_before_verify)
                verify_logits_all = lm_head(verify_out[:, start : start + len(draft) + 1, :])
                verify_pad_mask = full_pad_masks
            else:
                verify_past_key_values = clone_kv(past_key_values)

                draft_embs = self.model.paligemma_with_expert.embed_language_tokens(draft_tensor)
                draft_embs = draft_embs * math.sqrt(draft_embs.shape[-1])
                draft_embs = draft_embs.to(dtype=prefix_embs.dtype)
                verify_pad_mask = torch.cat(
                    [current_pad_mask, torch.ones((bsize, len(draft)), dtype=torch.bool, device=device)],
                    dim=1,
                )
                verify_mask = torch.zeros(
                    (bsize, len(draft), old_mask_len + len(draft)),
                    dtype=torch.bool,
                    device=device,
                )
                for row in range(len(draft)):
                    verify_mask[:, row, : old_mask_len + row + 1] = verify_pad_mask[:, : old_mask_len + row + 1]
                verify_att_mask = self.model._prepare_attention_masks_4d(verify_mask, dtype=draft_embs.dtype)
                position_start = int(torch.sum(current_pad_mask, dim=1).item())
                verify_position_ids = torch.arange(
                    position_start,
                    position_start + len(draft),
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                (verify_out, _), verify_kv = self._forward_prefix_language_model(
                    attention_mask=verify_att_mask,
                    position_ids=verify_position_ids,
                    past_key_values=verify_past_key_values,
                    inputs_embeds=draft_embs,
                    use_cache=True,
                    cache_position=verify_position_ids.squeeze(0),
                )
                verify_logits = lm_head(verify_out)
                verify_logits_all = torch.cat([prev_logits, verify_logits], dim=1)
            target_forwards += 1
            verify_forwards += 1

            diagnostic_step_accept: int | None = None
            diagnostic_batched_pred: torch.Tensor | None = None
            diagnostic_stepwise_pred: torch.Tensor | None = None
            if diagnose_verify_alignment:
                align_len = min(int(verify_logits_all.shape[1]), len(draft) + 1)
                if align_len > 0:
                    verify_alignment_blocks += 1
                    verify_alignment_checks += align_len
                    stepwise_logits_all = diagnostic_stepwise_logits(
                        [int(token) for token in draft],
                        align_len,
                    )
                    batched_pred = torch.argmax(verify_logits_all[:, :align_len, :].float(), dim=-1)
                    stepwise_pred = torch.argmax(stepwise_logits_all[:, :align_len, :].float(), dim=-1)
                    diagnostic_batched_pred = batched_pred
                    diagnostic_stepwise_pred = stepwise_pred
                    mismatch_mask = batched_pred != stepwise_pred
                    mismatches = int(torch.sum(mismatch_mask).item())
                    verify_alignment_argmax_mismatches += mismatches
                    if mismatches > 0:
                        verify_alignment_blocks_with_mismatch += 1
                        mismatch_positions = torch.nonzero(mismatch_mask[0], as_tuple=False).flatten()
                        first_mismatch = int(mismatch_positions[0].item())
                        verify_alignment_first_mismatch_min = min(
                            verify_alignment_first_mismatch_min,
                            first_mismatch,
                        )
                        verify_alignment_first_mismatch_sum += first_mismatch
                    diagnostic_step_accept = 0
                    for idx in range(min(len(draft), align_len)):
                        if int(stepwise_pred[0, idx].item()) != int(draft[idx]):
                            break
                        diagnostic_step_accept += 1

            block_accepted = 0
            matched_prefix = 0
            block_min_margin = float("inf")
            block_margin_ok = True
            margin_rejected_block = False
            for idx in range(len(draft)):
                logits_for_candidate = verify_logits_all[:, idx : idx + 1, :]
                candidate_logits = logits_for_candidate[:, -1, :].float()
                top_values, _top_indices = torch.topk(candidate_logits, k=2, dim=-1)
                predicted_candidate = int(torch.argmax(candidate_logits, dim=-1).item())
                margin = float((top_values[:, 0] - top_values[:, 1]).item())
                block_min_margin = min(block_min_margin, margin)
                verify_margin_checks += 1
                verify_margin_min = min(verify_margin_min, margin)
                verify_margin_sum += margin
                if predicted_candidate != int(draft[idx]):
                    break
                matched_prefix += 1
                if float(min_verify_margin) > 0.0 and margin < float(min_verify_margin):
                    block_margin_ok = False
                    break
                block_accepted += 1
            if float(min_verify_margin) > 0.0 and (
                block_accepted != len(draft) or matched_prefix != len(draft) or not block_margin_ok
            ):
                margin_block_rejects += 1
                margin_rejected_block = True
                block_accepted = 0
            block_accepted = min(block_accepted, remaining)
            if diagnostic_step_accept is not None:
                diagnostic_step_accept = min(diagnostic_step_accept, remaining)
                verify_alignment_batched_accept_sum += block_accepted
                verify_alignment_step_accept_sum += diagnostic_step_accept
                if block_accepted != diagnostic_step_accept:
                    verify_alignment_accept_disagreements += 1
                if block_accepted > diagnostic_step_accept:
                    verify_alignment_false_accepts += 1
                batched_correction_for_debug = None
                stepwise_correction_for_debug = None
                if diagnostic_batched_pred is not None and diagnostic_stepwise_pred is not None:
                    correction_idx = min(block_accepted, int(diagnostic_batched_pred.shape[1]) - 1)
                    verify_alignment_correction_checks += 1
                    batched_correction = int(diagnostic_batched_pred[0, correction_idx].item())
                    stepwise_correction = int(diagnostic_stepwise_pred[0, correction_idx].item())
                    batched_correction_for_debug = batched_correction
                    stepwise_correction_for_debug = stepwise_correction
                    if batched_correction != stepwise_correction:
                        verify_alignment_correction_mismatches += 1
                if len(debug_events) < 64:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "verify_alignment",
                            "draft_len": len(draft),
                            "batched_accept": block_accepted,
                            "step_accept": diagnostic_step_accept,
                            "batched_correction": batched_correction_for_debug,
                            "step_correction": stepwise_correction_for_debug,
                        }
                    )
            if len(debug_events) < 64:
                debug_events.append(
                    {
                        "pos": len(generated_tokens),
                        "kind": "verify",
                        "draft_len": len(draft),
                        "accepted": block_accepted,
                        "matched_prefix": matched_prefix,
                        "min_margin": block_min_margin if block_min_margin != float("inf") else None,
                        "draft": [int(token) for token in draft[: min(len(draft), 12)]],
                    }
                )
            accepted = 0
            accepted_emit = block_accepted
            if action_end_token_id is not None:
                for idx in range(block_accepted):
                    if int(draft[idx]) == action_end_token_id:
                        accepted_emit = idx + 1
                        break
            if reuse_full_blocks and block_accepted == len(draft) and not replay_accepted_cache:
                full_block_reuses += 1
                for idx in range(accepted_emit):
                    logits_for_token = verify_logits_all[:, idx : idx + 1, :]
                    logits_by_step.append(logits_for_token)
                    generated_tokens.append(int(draft[idx]))
                accepted = accepted_emit
                if replay_accepted_cache and accepted > 0:
                    for idx in range(accepted):
                        token = torch.tensor([[int(draft[idx])]], dtype=torch.long, device=device)
                        prev_logits = replay_one(token)
                elif resync_accepted_cache and accepted > 0:
                    prev_logits = resync_from_generated()
                else:
                    past_key_values = verify_kv
                    if accepted == len(draft):
                        current_pad_mask = verify_pad_mask
                    else:
                        keep_len = old_mask_len + accepted
                        past_key_values = trim_kv(verify_kv, keep_len)
                        current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()
                    prev_logits = verify_logits_all[:, accepted : accepted + 1, :]
                if (
                    emit_bonus_token
                    and accepted == len(draft)
                    and len(generated_tokens) < max_decoding_steps
                    and (
                        action_end_token_id is None
                        or not generated_tokens
                        or generated_tokens[-1] != action_end_token_id
                    )
                ):
                    bonus_logits = prev_logits
                    bonus_token = torch.argmax(bonus_logits[:, -1], dim=-1, keepdim=True)
                    logits_by_step.append(bonus_logits)
                    generated_tokens.append(int(bonus_token.item()))
                    bonus_tokens += 1
                    pending_unprocessed_token = bonus_token
                    prev_logits = bonus_logits
            else:
                if replay_accepted_cache:
                    accepted = 0
                    for idx in range(accepted_emit):
                        live_token = int(torch.argmax(prev_logits[:, -1], dim=-1).item())
                        if live_token != int(draft[idx]):
                            break
                        logits_by_step.append(prev_logits)
                        generated_tokens.append(int(draft[idx]))
                        token = torch.tensor([[int(draft[idx])]], dtype=torch.long, device=device)
                        prev_logits = replay_one(token)
                        accepted += 1
                    block_accepted = accepted
                    accepted_emit = accepted
                else:
                    accepted = accepted_emit
                    for idx in range(accepted):
                        logits_for_token = verify_logits_all[:, idx : idx + 1, :]
                        logits_by_step.append(logits_for_token)
                        generated_tokens.append(int(draft[idx]))
                    if resync_accepted_cache and accepted > 0:
                        prev_logits = resync_from_generated()
                    elif accepted > 0:
                        keep_len = old_mask_len + accepted
                        past_key_values = trim_kv(verify_kv, keep_len)
                        current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()
                        prev_logits = verify_logits_all[:, accepted : accepted + 1, :]
                    else:
                        prev_logits = verify_logits_all[:, :1, :]

                if (
                    len(generated_tokens) < max_decoding_steps
                    and (
                        action_end_token_id is None
                        or not generated_tokens
                        or generated_tokens[-1] != action_end_token_id
                    )
                ):
                    correction_logits = prev_logits
                    correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
                    if (
                        not margin_rejected_block
                        and accepted < len(draft)
                        and int(correction_token.item()) == int(draft[accepted])
                    ):
                        debug_predictions = [
                            int(torch.argmax(verify_logits_all[:, i : i + 1, :][:, -1], dim=-1).item())
                            for i in range(min(len(draft), 12))
                        ]
                        raise RuntimeError(
                            "Internal SD invariant failed: accepted prefix stopped before matching token "
                            f"accepted={accepted} accepted_emit={accepted_emit} "
                            f"block_accepted={block_accepted} matched_prefix={matched_prefix} "
                            f"draft={ [int(token) for token in draft[: min(len(draft), 12)]] } "
                            f"pred={debug_predictions} action_end={action_end_token_id}"
                        )
                    if len(debug_events) < 64:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "correction",
                                "accepted": accepted,
                                "token": int(correction_token.item()),
                                "rejected_draft": int(draft[accepted]) if accepted < len(draft) else None,
                            }
                        )
                    if defer_correction_token:
                        logits_by_step.append(correction_logits)
                        generated_tokens.append(int(correction_token.item()))
                        pending_corrections += 1
                        pending_unprocessed_token = correction_token
                        prev_logits = correction_logits
                        if action_end_token_id is not None and int(correction_token.item()) == action_end_token_id:
                            break
                    else:
                        prev_logits = advance_one(correction_token, correction_logits, fallback=True)
            accepted_tokens += accepted
            record_source_feedback(draft_sources, block_accepted)
            second_order_action_extrapolation_accepted_tokens += source_count(
                draft_sources,
                "second_order_action_extrapolation",
                accepted,
            )
            previous_chunk_position_accepted_tokens += source_count(
                draft_sources,
                "previous_chunk_position",
                accepted,
            )
            chunk_prefix_retrieval_accepted_tokens += source_count(
                draft_sources,
                "chunk_prefix_retrieval",
                accepted,
            )
            chunk_length_stop_accepted_tokens += source_count(
                draft_sources,
                "chunk_length_stop",
                accepted,
            )
            position_mode_histogram_accepted_tokens += source_count(
                draft_sources,
                "position_mode_histogram",
                accepted,
            )
            global_position_mode_accepted_tokens += source_count(
                draft_sources,
                "global_position_mode",
                accepted,
            )
            action_dimension_mode_accepted_tokens += source_count(
                draft_sources,
                "action_dimension_mode",
                accepted,
            )
            hold_action_token_accepted_tokens += source_count(
                draft_sources,
                "hold_action_token",
                accepted,
            )
            ngram_continuation_accepted_tokens += source_count(
                draft_sources,
                "ngram_continuation",
                accepted,
            )
            source_agreement_accepted_tokens += source_count(
                draft_sources,
                "source_agreement",
                accepted,
            )
            action_trend_regression_accepted_tokens += source_count(
                draft_sources,
                "action_trend_regression",
                accepted,
            )
            action_prefix_lookup_accepted_tokens += source_count(
                draft_sources,
                "action_prefix_lookup",
                accepted,
            )
            action_vector_transition_accepted_tokens += source_count(
                draft_sources,
                "action_vector_transition",
                accepted,
            )
            action_repeat_vector_accepted_tokens += source_count(
                draft_sources,
                "action_repeat_vector",
                accepted,
            )
            action_token_neighborhood_accepted_tokens += source_count(
                draft_sources,
                "action_token_neighborhood",
                accepted,
            )
            action_context_tree_accepted_tokens += source_count(
                draft_sources,
                "action_context_tree",
                accepted,
            )
            action_transition_histogram_accepted_tokens += source_count(
                draft_sources,
                "action_transition_histogram",
                accepted,
            )
            action_delta_histogram_accepted_tokens += source_count(
                draft_sources,
                "action_delta_histogram",
                accepted,
            )
            action_delta_ngram_accepted_tokens += source_count(
                draft_sources,
                "action_delta_ngram",
                accepted,
            )
            chunk_position_delta_accepted_tokens += source_count(
                draft_sources,
                "chunk_position_delta",
                accepted,
            )
            chunk_delta_template_accepted_tokens += source_count(
                draft_sources,
                "chunk_delta_template",
                accepted,
            )
            update_dynamic_lookahead(accepted=block_accepted, drafted=len(draft))
            if dynamic_tree_width:
                if block_accepted == len(draft):
                    current_tree_width = min(max_tree_width, current_tree_width + max(int(tree_width_growth), 1))
                else:
                    current_tree_width = max(1, current_tree_width - max(int(tree_width_shrink), 1))
            if len(generated_tokens) >= max_decoding_steps:
                break
            if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                break

        if len(logits_by_step) != len(generated_tokens):
            raise RuntimeError(
                "Internal SD invariant failed: logits/token length mismatch "
                f"({len(logits_by_step)} logits for {len(generated_tokens)} tokens)"
            )
        cached_generated_tokens = len(generated_tokens) - (1 if pending_unprocessed_token is not None else 0)
        expected_mask_len = prefix_pad_masks.shape[1] + cached_generated_tokens
        if current_pad_mask.shape[1] != expected_mask_len:
            raise RuntimeError(
                "Internal SD invariant failed: cache mask/token length mismatch "
                f"({current_pad_mask.shape[1]} mask positions, expected {expected_mask_len})"
            )

        generated = torch.tensor([generated_tokens[:max_decoding_steps]], dtype=torch.long, device=device)
        logits = torch.cat(logits_by_step, dim=1)
        if logits.shape[1] != generated.shape[1]:
            raise RuntimeError(
                "Internal SD invariant failed: returned logits/generated length mismatch "
                f"({logits.shape[1]} logits for {generated.shape[1]} tokens)"
            )
        stats = {
            "prefix_hidden": prefix_out[:, -1, :].detach().float(),
            "target_forwards": target_forwards,
            "verify_forwards": verify_forwards,
            "fallback_forwards": fallback_forwards,
            "replay_forwards": replay_forwards,
            "resync_forwards": resync_forwards,
            "full_block_reuses": full_block_reuses,
            "bonus_tokens": bonus_tokens,
            "pending_verifies": pending_verifies,
            "pending_corrections": pending_corrections,
            "pending_processes": pending_processes,
            "pending_token_at_return": pending_unprocessed_token is not None,
            "reuse_full_blocks": reuse_full_blocks,
            "replay_accepted_cache": replay_accepted_cache,
            "resync_accepted_cache": resync_accepted_cache,
            "min_verify_margin": float(min_verify_margin),
            "margin_block_rejects": margin_block_rejects,
            "verify_margin_checks": verify_margin_checks,
            "verify_margin_min": 0.0 if verify_margin_min == float("inf") else verify_margin_min,
            "verify_margin_mean": verify_margin_sum / max(verify_margin_checks, 1),
            "verify_from_scratch": verify_from_scratch,
            "emit_bonus_token": emit_bonus_token,
            "defer_correction_token": defer_correction_token,
            "dynamic_lookahead": dynamic_lookahead,
            "min_lookahead": min_dynamic_lookahead,
            "max_lookahead": max_lookahead,
            "final_lookahead": current_lookahead,
            "mean_lookahead": sum(lookahead_values) / max(len(lookahead_values), 1),
            "tree_width": max_tree_width,
            "tree_branch_width": max_tree_branch_width,
            "dynamic_tree_width": dynamic_tree_width,
            "min_tree_width": max(1, min(max_tree_width, int(min_tree_width))),
            "final_tree_width": current_tree_width,
            "tree_width_growth": max(int(tree_width_growth), 1),
            "tree_width_shrink": max(int(tree_width_shrink), 1),
            "tree_anchor_target_token": tree_anchor_target_token,
            "tree_anchor_target_continuation": tree_anchor_target_continuation,
            "mean_tree_width": sum(tree_width_values) / max(len(tree_width_values), 1),
            "tree_verifies": tree_verifies,
            "tree_candidates": tree_candidates,
            "tree_accepted_tokens": tree_accepted_tokens,
            "tree_anchor_verifies": tree_anchor_verifies,
            "tree_anchor_candidates": tree_anchor_candidates,
            "tree_anchor_accepted_tokens": tree_anchor_accepted_tokens,
            "tree_anchor_acceptance_rate": tree_anchor_accepted_tokens / max(tree_anchor_candidates, 1),
            "tree_first_token_checks": tree_first_token_checks,
            "tree_first_token_misses": tree_first_token_misses,
            "tree_first_token_miss_rate": tree_first_token_misses / max(tree_first_token_checks, 1),
            "mean_tree_candidates": tree_candidates / max(tree_verifies, 1),
            "verify_alignment_diagnostics": diagnose_verify_alignment,
            "verify_alignment_blocks": verify_alignment_blocks,
            "verify_alignment_checks": verify_alignment_checks,
            "verify_alignment_argmax_mismatches": verify_alignment_argmax_mismatches,
            "verify_alignment_blocks_with_mismatch": verify_alignment_blocks_with_mismatch,
            "verify_alignment_false_accepts": verify_alignment_false_accepts,
            "verify_alignment_accept_disagreements": verify_alignment_accept_disagreements,
            "verify_alignment_first_mismatch_min": (
                0
                if verify_alignment_first_mismatch_min == float("inf")
                else int(verify_alignment_first_mismatch_min)
            ),
            "verify_alignment_first_mismatch_mean": (
                verify_alignment_first_mismatch_sum / max(verify_alignment_blocks_with_mismatch, 1)
            ),
            "verify_alignment_forwards": verify_alignment_forwards,
            "verify_alignment_batched_accept_mean": (
                verify_alignment_batched_accept_sum / max(verify_alignment_blocks, 1)
            ),
            "verify_alignment_step_accept_mean": verify_alignment_step_accept_sum / max(verify_alignment_blocks, 1),
            "verify_alignment_correction_checks": verify_alignment_correction_checks,
            "verify_alignment_correction_mismatches": verify_alignment_correction_mismatches,
            "verify_alignment_correction_mismatch_rate": (
                verify_alignment_correction_mismatches / max(verify_alignment_correction_checks, 1)
            ),
            "second_order_action_extrapolation_drafted_tokens": second_order_action_extrapolation_drafted_tokens,
            "second_order_action_extrapolation_accepted_tokens": second_order_action_extrapolation_accepted_tokens,
            "second_order_action_extrapolation_acceptance_rate": (
                second_order_action_extrapolation_accepted_tokens
                / max(second_order_action_extrapolation_drafted_tokens, 1)
            ),
            "previous_chunk_position_drafted_tokens": previous_chunk_position_drafted_tokens,
            "previous_chunk_position_accepted_tokens": previous_chunk_position_accepted_tokens,
            "previous_chunk_position_acceptance_rate": previous_chunk_position_accepted_tokens
            / max(previous_chunk_position_drafted_tokens, 1),
            "chunk_prefix_retrieval_drafted_tokens": chunk_prefix_retrieval_drafted_tokens,
            "chunk_prefix_retrieval_accepted_tokens": chunk_prefix_retrieval_accepted_tokens,
            "chunk_prefix_retrieval_acceptance_rate": chunk_prefix_retrieval_accepted_tokens
            / max(chunk_prefix_retrieval_drafted_tokens, 1),
            "chunk_length_stop_drafted_tokens": chunk_length_stop_drafted_tokens,
            "chunk_length_stop_accepted_tokens": chunk_length_stop_accepted_tokens,
            "chunk_length_stop_acceptance_rate": chunk_length_stop_accepted_tokens
            / max(chunk_length_stop_drafted_tokens, 1),
            "action_token_neighborhood_drafted_tokens": action_token_neighborhood_drafted_tokens,
            "action_token_neighborhood_accepted_tokens": action_token_neighborhood_accepted_tokens,
            "action_token_neighborhood_acceptance_rate": action_token_neighborhood_accepted_tokens
            / max(action_token_neighborhood_drafted_tokens, 1),
            "position_mode_histogram_drafted_tokens": position_mode_histogram_drafted_tokens,
            "position_mode_histogram_accepted_tokens": position_mode_histogram_accepted_tokens,
            "position_mode_histogram_acceptance_rate": position_mode_histogram_accepted_tokens
            / max(position_mode_histogram_drafted_tokens, 1),
            "global_position_mode_drafted_tokens": global_position_mode_drafted_tokens,
            "global_position_mode_accepted_tokens": global_position_mode_accepted_tokens,
            "global_position_mode_acceptance_rate": global_position_mode_accepted_tokens
            / max(global_position_mode_drafted_tokens, 1),
            "action_dimension_mode_drafted_tokens": action_dimension_mode_drafted_tokens,
            "action_dimension_mode_accepted_tokens": action_dimension_mode_accepted_tokens,
            "action_dimension_mode_acceptance_rate": action_dimension_mode_accepted_tokens
            / max(action_dimension_mode_drafted_tokens, 1),
            "hold_action_token_drafted_tokens": hold_action_token_drafted_tokens,
            "hold_action_token_accepted_tokens": hold_action_token_accepted_tokens,
            "hold_action_token_acceptance_rate": hold_action_token_accepted_tokens
            / max(hold_action_token_drafted_tokens, 1),
            "ngram_continuation_drafted_tokens": ngram_continuation_drafted_tokens,
            "ngram_continuation_accepted_tokens": ngram_continuation_accepted_tokens,
            "ngram_continuation_acceptance_rate": ngram_continuation_accepted_tokens
            / max(ngram_continuation_drafted_tokens, 1),
            "source_agreement_drafted_tokens": source_agreement_drafted_tokens,
            "source_agreement_accepted_tokens": source_agreement_accepted_tokens,
            "source_agreement_acceptance_rate": source_agreement_accepted_tokens
            / max(source_agreement_drafted_tokens, 1),
            "action_trend_regression_drafted_tokens": action_trend_regression_drafted_tokens,
            "action_trend_regression_accepted_tokens": action_trend_regression_accepted_tokens,
            "action_trend_regression_acceptance_rate": action_trend_regression_accepted_tokens
            / max(action_trend_regression_drafted_tokens, 1),
            "action_prefix_lookup_drafted_tokens": action_prefix_lookup_drafted_tokens,
            "action_prefix_lookup_accepted_tokens": action_prefix_lookup_accepted_tokens,
            "action_prefix_lookup_acceptance_rate": action_prefix_lookup_accepted_tokens
            / max(action_prefix_lookup_drafted_tokens, 1),
            "action_vector_transition_drafted_tokens": action_vector_transition_drafted_tokens,
            "action_vector_transition_accepted_tokens": action_vector_transition_accepted_tokens,
            "action_vector_transition_acceptance_rate": action_vector_transition_accepted_tokens
            / max(action_vector_transition_drafted_tokens, 1),
            "action_repeat_vector_drafted_tokens": action_repeat_vector_drafted_tokens,
            "action_repeat_vector_accepted_tokens": action_repeat_vector_accepted_tokens,
            "action_repeat_vector_acceptance_rate": action_repeat_vector_accepted_tokens
            / max(action_repeat_vector_drafted_tokens, 1),
            "action_context_tree_drafted_tokens": action_context_tree_drafted_tokens,
            "action_context_tree_accepted_tokens": action_context_tree_accepted_tokens,
            "action_context_tree_acceptance_rate": action_context_tree_accepted_tokens
            / max(action_context_tree_drafted_tokens, 1),
            "action_transition_histogram_drafted_tokens": action_transition_histogram_drafted_tokens,
            "action_transition_histogram_accepted_tokens": action_transition_histogram_accepted_tokens,
            "action_transition_histogram_acceptance_rate": action_transition_histogram_accepted_tokens
            / max(action_transition_histogram_drafted_tokens, 1),
            "action_delta_histogram_drafted_tokens": action_delta_histogram_drafted_tokens,
            "action_delta_histogram_accepted_tokens": action_delta_histogram_accepted_tokens,
            "action_delta_histogram_acceptance_rate": action_delta_histogram_accepted_tokens
            / max(action_delta_histogram_drafted_tokens, 1),
            "action_delta_ngram_drafted_tokens": action_delta_ngram_drafted_tokens,
            "action_delta_ngram_accepted_tokens": action_delta_ngram_accepted_tokens,
            "action_delta_ngram_acceptance_rate": action_delta_ngram_accepted_tokens
            / max(action_delta_ngram_drafted_tokens, 1),
            "chunk_position_delta_drafted_tokens": chunk_position_delta_drafted_tokens,
            "chunk_position_delta_accepted_tokens": chunk_position_delta_accepted_tokens,
            "chunk_position_delta_acceptance_rate": chunk_position_delta_accepted_tokens
            / max(chunk_position_delta_drafted_tokens, 1),
            "chunk_delta_template_drafted_tokens": chunk_delta_template_drafted_tokens,
            "chunk_delta_template_accepted_tokens": chunk_delta_template_accepted_tokens,
            "chunk_delta_template_acceptance_rate": chunk_delta_template_accepted_tokens
            / max(chunk_delta_template_drafted_tokens, 1),
            "drafted_tokens": drafted_tokens,
            "accepted_tokens": accepted_tokens,
            "acceptance_rate": accepted_tokens / max(drafted_tokens, 1),
            "tokens_per_target_forward": len(generated_tokens) / max(target_forwards, 1),
            **source_cooldown_stats(),
            "debug_events": debug_events,
        }
        return generated, logits, stats

    @torch.no_grad()
    def sample_actions_fast_medusa_speculative(
        self,
        images,
        img_masks,
        tokens,
        masks,
        medusa_head: Any,
        token_map: Any,
        max_decoding_steps: int | None = None,
        lookahead: int = 4,
        min_draft_confidence: float = 0.0,
        min_verify_confidence: float = 0.0,
        min_spec_position: int = 0,
        accept_partial_blocks: bool = True,
        replay_accepted_cache: bool = False,
        resync_accepted_cache: bool = False,
        verify_from_scratch: bool = False,
        early_stop_action_end: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Greedy exact FAST-token decode with Medusa future-token verification."""

        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        if bsize != 1:
            raise ValueError("Medusa speculative PI0-FAST decode currently supports batch size 1")

        device = tokens.device
        medusa_dtype = self._match_model_precision(torch.empty((), device=device)).dtype
        first_medusa_param = next(medusa_head.parameters())
        if first_medusa_param.device != device or first_medusa_param.dtype != medusa_dtype:
            medusa_head = medusa_head.to(device=device, dtype=medusa_dtype)
        medusa_head.eval()
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        action_end_token_id = self._action_end_token_id() if early_stop_action_end else None
        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), past_key_values = self._forward_prefix_language_model(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=prefix_embs,
            use_cache=True,
            cache_position=torch.arange(prefix_embs.shape[1], device=device, dtype=torch.long),
        )
        prev_hidden = prefix_out[:, -1:, :]
        prev_logits = lm_head(prev_hidden)
        current_pad_mask = prefix_pad_masks
        generated_tokens: list[int] = []
        logits_by_step: list[torch.Tensor] = []
        target_forwards = 1
        verify_forwards = 0
        fallback_forwards = 0
        drafted_tokens = 0
        accepted_tokens = 0
        confidence_rejects = 0
        replay_forwards = 0
        debug_events: list[dict[str, Any]] = []

        def advance_one(
            next_token: torch.Tensor,
            logits_for_token: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal current_pad_mask, past_key_values, target_forwards, fallback_forwards
            logits_by_step.append(logits_for_token)
            generated_tokens.append(int(next_token.item()))
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_token_emb,
                use_cache=True,
                cache_position=torch.tensor([current_pad_mask.shape[1] - 1], device=device, dtype=torch.long),
            )
            target_forwards += 1
            fallback_forwards += 1
            return lm_head(step_out[:, -1:, :]), step_out[:, -1:, :]

        def replay_one(next_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal current_pad_mask, past_key_values, target_forwards, replay_forwards
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_token_emb,
                use_cache=True,
                cache_position=torch.tensor([current_pad_mask.shape[1] - 1], device=device, dtype=torch.long),
            )
            target_forwards += 1
            replay_forwards += 1
            return lm_head(step_out[:, -1:, :]), step_out[:, -1:, :]

        def resync_from_generated() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal current_pad_mask, past_key_values, target_forwards, replay_forwards
            if not generated_tokens:
                raise RuntimeError("Cannot resync Medusa cache before any generated token")
            full_fast_tokens = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            full_fast_masks = torch.ones_like(full_fast_tokens, dtype=torch.bool)
            full_embs, full_pad_masks, full_att_masks, _total_t_images, _num_fast_embs = self.model.embed_prefix_fast(
                images,
                img_masks,
                tokens,
                masks,
                fast_action_tokens=full_fast_tokens,
                fast_action_masks=full_fast_masks,
            )
            full_embs = self._match_model_precision(full_embs)
            full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
            full_att_4d = self.model._prepare_attention_masks_4d(full_att_masks, dtype=full_embs.dtype)
            (full_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=full_att_4d,
                position_ids=full_position_ids,
                past_key_values=None,
                inputs_embeds=full_embs,
                use_cache=True,
                cache_position=torch.arange(full_embs.shape[1], device=device, dtype=torch.long),
            )
            current_pad_mask = full_pad_masks
            target_forwards += 1
            replay_forwards += 1
            return lm_head(full_out[:, -1:, :]), full_out[:, -1:, :]

        def draft_from_hidden(hidden: torch.Tensor, steps: int) -> tuple[list[int], list[float]]:
            head_logits = medusa_head(hidden[:, -1, :].to(dtype=next(medusa_head.parameters()).dtype))
            draft_classes: list[int] = []
            confidences: list[float] = []
            for logit in head_logits[:steps]:
                probs = F.softmax(logit.float(), dim=-1)
                max_prob, max_idx = torch.max(probs, dim=-1)
                draft_classes.append(int(max_idx.item()))
                confidences.append(float(max_prob.item()))
            if not draft_classes:
                return [], []
            class_tensor = torch.tensor(draft_classes, dtype=torch.long, device=device)
            return [int(token) for token in token_map.decode_tensor(class_tensor).tolist()], confidences

        while len(generated_tokens) < max_decoding_steps:
            remaining = max_decoding_steps - len(generated_tokens)
            target_next = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)
            if action_end_token_id is not None and int(target_next.item()) == action_end_token_id:
                logits_by_step.append(prev_logits)
                generated_tokens.append(int(target_next.item()))
                break

            if len(generated_tokens) < min_spec_position:
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue

            draft, draft_confidences = draft_from_hidden(prev_hidden, min(lookahead, remaining))
            if draft_confidences and min(draft_confidences) < min_draft_confidence:
                confidence_rejects += 1
                if len(debug_events) < 64:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "confidence_fallback",
                            "token": int(target_next.item()),
                            "draft0": int(draft[0]) if draft else None,
                            "min_confidence": min(draft_confidences),
                        }
                    )
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue
            if not draft or int(draft[0]) != int(target_next.item()):
                if len(debug_events) < 64:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "fallback",
                            "token": int(target_next.item()),
                            "draft0": int(draft[0]) if draft else None,
                        }
                    )
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue

            draft_tensor = torch.tensor([draft], dtype=torch.long, device=device)
            drafted_tokens += len(draft)
            old_mask_len = current_pad_mask.shape[1]
            if verify_from_scratch:
                full_fast_tokens = torch.tensor(
                    [generated_tokens + [int(token) for token in draft]],
                    dtype=torch.long,
                    device=device,
                )
                full_fast_masks = torch.ones_like(full_fast_tokens, dtype=torch.bool)
                full_embs, full_pad_masks, full_att_masks, _total_t_images, num_fast_embs = self.model.embed_prefix_fast(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    fast_action_tokens=full_fast_tokens,
                    fast_action_masks=full_fast_masks,
                )
                full_embs = self._match_model_precision(full_embs)
                full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
                full_att_4d = self.model._prepare_attention_masks_4d(full_att_masks, dtype=full_embs.dtype)
                (full_out, _), verify_kv = self._forward_prefix_language_model(
                    attention_mask=full_att_4d,
                    position_ids=full_position_ids,
                    past_key_values=None,
                    inputs_embeds=full_embs,
                    use_cache=True,
                    cache_position=torch.arange(full_embs.shape[1], device=device, dtype=torch.long),
                )
                bos_hidden_idx = full_out.shape[1] - num_fast_embs - 1
                start = bos_hidden_idx + len(generated_tokens)
                verify_logits_all = lm_head(full_out[:, start : start + len(draft) + 1, :])
                verify_out = full_out[:, start + 1 : start + len(draft) + 1, :]
                verify_pad_mask = full_pad_masks
            else:
                verify_past_key_values = clone_kv(past_key_values)
                draft_embs = self.model.paligemma_with_expert.embed_language_tokens(draft_tensor)
                draft_embs = draft_embs * math.sqrt(draft_embs.shape[-1])
                draft_embs = draft_embs.to(dtype=prefix_embs.dtype)
                verify_pad_mask = torch.cat(
                    [current_pad_mask, torch.ones((bsize, len(draft)), dtype=torch.bool, device=device)],
                    dim=1,
                )
                verify_mask = torch.zeros(
                    (bsize, len(draft), old_mask_len + len(draft)),
                    dtype=torch.bool,
                    device=device,
                )
                for row in range(len(draft)):
                    verify_mask[:, row, : old_mask_len + row + 1] = verify_pad_mask[:, : old_mask_len + row + 1]
                verify_att_mask = self.model._prepare_attention_masks_4d(verify_mask, dtype=draft_embs.dtype)
                position_start = int(torch.sum(current_pad_mask, dim=1).item())
                verify_position_ids = torch.arange(
                    position_start,
                    position_start + len(draft),
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                (verify_out, _), verify_kv = self._forward_prefix_language_model(
                    attention_mask=verify_att_mask,
                    position_ids=verify_position_ids,
                    past_key_values=verify_past_key_values,
                    inputs_embeds=draft_embs,
                    use_cache=True,
                    cache_position=verify_position_ids.squeeze(0),
                )
                verify_logits = lm_head(verify_out)
                verify_logits_all = torch.cat([prev_logits, verify_logits], dim=1)
            target_forwards += 1
            verify_forwards += 1

            block_accepted = 0
            for idx in range(len(draft)):
                step_probs = F.softmax(verify_logits_all[:, idx].float(), dim=-1)
                step_confidence, step_prediction = torch.max(step_probs, dim=-1)
                predicted_candidate = int(step_prediction.item())
                if predicted_candidate != int(draft[idx]):
                    break
                if float(step_confidence.item()) < min_verify_confidence:
                    break
                block_accepted += 1
            accepted_emit = block_accepted if accept_partial_blocks else (block_accepted if block_accepted == len(draft) else 0)
            if action_end_token_id is not None:
                for idx in range(accepted_emit):
                    if int(draft[idx]) == action_end_token_id:
                        accepted_emit = idx + 1
                        break
            if len(debug_events) < 64:
                debug_events.append(
                    {
                        "pos": len(generated_tokens),
                        "kind": "verify",
                        "draft_len": len(draft),
                        "accepted": accepted_emit,
                        "matched_prefix": block_accepted,
                    }
                )
            for idx in range(accepted_emit):
                logits_by_step.append(verify_logits_all[:, idx : idx + 1, :])
                generated_tokens.append(int(draft[idx]))

            accepted_tokens += accepted_emit
            if replay_accepted_cache and accepted_emit > 0:
                for idx in range(accepted_emit):
                    token = torch.tensor([[int(draft[idx])]], dtype=torch.long, device=device)
                    prev_logits, prev_hidden = replay_one(token)
            elif resync_accepted_cache and accepted_emit > 0:
                prev_logits, prev_hidden = resync_from_generated()
            elif accepted_emit == len(draft):
                past_key_values = verify_kv
                current_pad_mask = verify_pad_mask
                prev_hidden = verify_out[:, accepted_emit - 1 : accepted_emit, :]
                prev_logits = verify_logits_all[:, accepted_emit : accepted_emit + 1, :]
            elif accepted_emit > 0:
                keep_len = old_mask_len + accepted_emit
                past_key_values = trim_kv(verify_kv, keep_len)
                current_pad_mask = verify_pad_mask[:, :keep_len]
                prev_hidden = verify_out[:, accepted_emit - 1 : accepted_emit, :]
                prev_logits = verify_logits_all[:, accepted_emit : accepted_emit + 1, :]
            if len(generated_tokens) >= max_decoding_steps:
                break
            if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                break
            if accepted_emit == len(draft):
                continue

            correction_logits = prev_logits
            correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
            prev_logits, prev_hidden = advance_one(correction_token, correction_logits)

        generated = torch.tensor([generated_tokens[:max_decoding_steps]], dtype=torch.long, device=device)
        stats = {
            "prefix_hidden": prefix_out[:, -1, :].detach().float(),
            "target_forwards": target_forwards,
            "verify_forwards": verify_forwards,
            "fallback_forwards": fallback_forwards,
            "replay_forwards": replay_forwards,
            "resync_accepted_cache": resync_accepted_cache,
            "drafted_tokens": drafted_tokens,
            "accepted_tokens": accepted_tokens,
            "confidence_rejects": confidence_rejects,
            "acceptance_rate": accepted_tokens / max(drafted_tokens, 1),
            "tokens_per_target_forward": len(generated_tokens) / max(target_forwards, 1),
            "debug_events": debug_events,
        }
        return generated, torch.cat(logits_by_step, dim=1), stats

    @torch.no_grad()
    def sample_actions_fast_draft_transformer_speculative(
        self,
        images,
        img_masks,
        tokens,
        masks,
        draft_model: Any,
        token_map: Any,
        max_decoding_steps: int | None = None,
        lookahead: int = 4,
        min_draft_confidence: float = 0.0,
        min_spec_position: int = 0,
        early_stop_action_end: bool = True,
        accept_partial_blocks: bool = False,
        debug_event_limit: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Greedy FAST-token decode with a small transformer draft model."""

        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        if bsize != 1:
            raise ValueError("Draft-transformer PI0-FAST decode currently supports batch size 1")

        device = tokens.device
        draft_dtype = next(draft_model.parameters()).dtype
        draft_model = draft_model.to(device=device, dtype=draft_dtype).eval()
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        action_end_token_id = self._action_end_token_id() if early_stop_action_end else None
        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), past_key_values = self._forward_prefix_language_model(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=prefix_embs,
            use_cache=True,
            cache_position=torch.arange(prefix_embs.shape[1], device=device, dtype=torch.long),
        )
        prev_hidden = prefix_out[:, -1:, :]
        prev_logits = lm_head(prev_hidden)
        current_pad_mask = prefix_pad_masks
        generated_tokens: list[int] = []
        logits_by_step: list[torch.Tensor] = []
        target_forwards = 1
        verify_forwards = 0
        fallback_forwards = 0
        drafted_tokens = 0
        accepted_tokens = 0
        confidence_rejects = 0
        debug_events: list[dict[str, Any]] = []

        def advance_one(
            next_token: torch.Tensor,
            logits_for_token: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal current_pad_mask, past_key_values, target_forwards, fallback_forwards
            logits_by_step.append(logits_for_token)
            generated_tokens.append(int(next_token.item()))
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_token_emb,
                use_cache=True,
                cache_position=torch.tensor([current_pad_mask.shape[1] - 1], device=device, dtype=torch.long),
            )
            target_forwards += 1
            fallback_forwards += 1
            return lm_head(step_out[:, -1:, :]), step_out[:, -1:, :]

        def draft_from_state(hidden: torch.Tensor, steps: int) -> tuple[list[int], list[float]]:
            context_len = int(draft_model.config.context_len)
            pad_id = len(token_map)
            encoded_context = [token_map.token_to_class.get(int(token), -1) for token in generated_tokens[-context_len:]]
            if any(cls < 0 for cls in encoded_context):
                return [], []
            context = torch.full((1, context_len), pad_id, dtype=torch.long, device=device)
            if encoded_context:
                context[0, -len(encoded_context) :] = torch.tensor(encoded_context, dtype=torch.long, device=device)
            draft_classes, confidences = draft_model.draft(
                hidden[:, -1, :].to(dtype=next(draft_model.parameters()).dtype),
                context,
                steps=steps,
            )
            draft_tokens = token_map.decode_tensor(draft_classes[0]).tolist()
            return [int(token) for token in draft_tokens], [float(conf) for conf in confidences[0].tolist()]

        while len(generated_tokens) < max_decoding_steps:
            remaining = max_decoding_steps - len(generated_tokens)
            target_next = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)
            if action_end_token_id is not None and int(target_next.item()) == action_end_token_id:
                logits_by_step.append(prev_logits)
                generated_tokens.append(int(target_next.item()))
                break

            if len(generated_tokens) < min_spec_position:
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue

            draft, draft_confidences = draft_from_state(prev_hidden, min(lookahead, remaining))
            if draft_confidences and min(draft_confidences) < min_draft_confidence:
                confidence_rejects += 1
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue
            if not draft or int(draft[0]) != int(target_next.item()):
                if len(debug_events) < debug_event_limit:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "fallback",
                            "token": int(target_next.item()),
                            "draft0": int(draft[0]) if draft else None,
                        }
                    )
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue

            draft_tensor = torch.tensor([draft], dtype=torch.long, device=device)
            drafted_tokens += len(draft)
            old_mask_len = current_pad_mask.shape[1]
            verify_past_key_values = clone_kv(past_key_values)
            draft_embs = self.model.paligemma_with_expert.embed_language_tokens(draft_tensor)
            draft_embs = draft_embs * math.sqrt(draft_embs.shape[-1])
            draft_embs = draft_embs.to(dtype=prefix_embs.dtype)
            verify_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, len(draft)), dtype=torch.bool, device=device)],
                dim=1,
            )
            verify_mask = torch.zeros(
                (bsize, len(draft), old_mask_len + len(draft)),
                dtype=torch.bool,
                device=device,
            )
            for row in range(len(draft)):
                verify_mask[:, row, : old_mask_len + row + 1] = verify_pad_mask[:, : old_mask_len + row + 1]
            verify_att_mask = self.model._prepare_attention_masks_4d(verify_mask, dtype=draft_embs.dtype)
            position_start = int(torch.sum(current_pad_mask, dim=1).item())
            verify_position_ids = torch.arange(
                position_start,
                position_start + len(draft),
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            (verify_out, _), verify_kv = self._forward_prefix_language_model(
                attention_mask=verify_att_mask,
                position_ids=verify_position_ids,
                past_key_values=verify_past_key_values,
                inputs_embeds=draft_embs,
                use_cache=True,
                cache_position=verify_position_ids.squeeze(0),
            )
            verify_logits = lm_head(verify_out)
            verify_logits_all = torch.cat([prev_logits, verify_logits], dim=1)
            target_forwards += 1
            verify_forwards += 1

            block_accepted = 0
            for idx in range(len(draft)):
                predicted_candidate = int(torch.argmax(verify_logits_all[:, idx], dim=-1).item())
                if predicted_candidate != int(draft[idx]):
                    break
                block_accepted += 1
            accepted_emit = block_accepted if accept_partial_blocks else 0
            if block_accepted == len(draft):
                accepted_emit = block_accepted
            if action_end_token_id is not None:
                for idx in range(accepted_emit):
                    if int(draft[idx]) == action_end_token_id:
                        accepted_emit = idx + 1
                        break
            if len(debug_events) < debug_event_limit:
                debug_events.append(
                    {
                        "pos": len(generated_tokens),
                        "kind": "verify",
                        "draft_len": len(draft),
                        "accepted": accepted_emit,
                        "matched_prefix": block_accepted,
                    }
                )
            for idx in range(accepted_emit):
                logits_by_step.append(verify_logits_all[:, idx : idx + 1, :])
                generated_tokens.append(int(draft[idx]))

            accepted_tokens += accepted_emit
            if accepted_emit > 0:
                if accepted_emit == len(draft):
                    past_key_values = verify_kv
                    current_pad_mask = verify_pad_mask
                else:
                    past_key_values = trim_kv(verify_kv, old_mask_len + accepted_emit)
                    current_pad_mask = verify_pad_mask[:, : old_mask_len + accepted_emit]
                prev_hidden = verify_out[:, accepted_emit - 1 : accepted_emit, :]
                prev_logits = verify_logits_all[:, accepted_emit : accepted_emit + 1, :]
            if len(generated_tokens) >= max_decoding_steps:
                break
            if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                break
            if accepted_emit == len(draft):
                continue

            correction_logits = prev_logits
            correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
            prev_logits, prev_hidden = advance_one(correction_token, correction_logits)

        generated = torch.tensor([generated_tokens[:max_decoding_steps]], dtype=torch.long, device=device)
        stats = {
            "prefix_hidden": prefix_out[:, -1, :].detach().float(),
            "target_forwards": target_forwards,
            "verify_forwards": verify_forwards,
            "fallback_forwards": fallback_forwards,
            "drafted_tokens": drafted_tokens,
            "accepted_tokens": accepted_tokens,
            "confidence_rejects": confidence_rejects,
            "acceptance_rate": accepted_tokens / max(drafted_tokens, 1),
            "tokens_per_target_forward": len(generated_tokens) / max(target_forwards, 1),
            "debug_events": debug_events,
        }
        return generated, torch.cat(logits_by_step, dim=1), stats

    @torch.no_grad()
    def sample_actions_fast_block_speculative(
        self,
        images,
        img_masks,
        tokens,
        masks,
        block_drafter: Any,
        token_map: Any,
        block_gate: Any | None = None,
        max_decoding_steps: int | None = None,
        lookahead: int = 7,
        min_draft_confidence: float = 0.0,
        min_verify_confidence: float = 0.0,
        min_verify_margin: float = 0.0,
        block_gate_threshold: float = 0.0,
        max_future_accept: int | None = None,
        min_future_accept: int = 0,
        min_spec_position: int = 0,
        reject_cooldown_steps: int = 0,
        reject_cooldown_after: int = 1,
        spec_fallback_cooldown_steps: int = 0,
        spec_fallback_cooldown_after: int = 0,
        allow_unknown_context: bool = False,
        repeat_token_draft: bool = False,
        repeat_token_min_run: int = 2,
        repeat_pattern_draft: bool = False,
        repeat_pattern_max_period: int = 8,
        repeat_pattern_min_position: int = 0,
        pattern_only: bool = False,
        unverified_pattern_tail: bool = False,
        unverified_pattern_eos: bool = False,
        full_block_only: bool = False,
        early_stop_action_end: bool = True,
        accept_partial_blocks: bool = True,
        refine_steps: int = 1,
        verify_from_scratch: bool = False,
        resync_accepted_cache: bool = False,
        draft_after_known_token: bool = False,
        debug_event_limit: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Greedy FAST-token decode with known-first masked-block speculation.

        The target model already gives the next greedy token in ``prev_logits``.
        We seed every speculative block with that known target token, then let
        the small block drafter propose only the following FAST tokens.
        """

        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        if bsize != 1:
            raise ValueError("Block-drafter PI0-FAST decode currently supports batch size 1")

        device = tokens.device
        first_draft_param = next(block_drafter.parameters())
        draft_dtype = first_draft_param.dtype
        if first_draft_param.device != device:
            block_drafter = block_drafter.to(device=device, dtype=draft_dtype)
        block_drafter.eval()
        if block_gate is not None:
            first_gate_param = next(block_gate.parameters())
            if first_gate_param.device != device or first_gate_param.dtype != draft_dtype:
                block_gate = block_gate.to(device=device, dtype=draft_dtype)
            block_gate.eval()
        lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        action_end_token_id = self._action_end_token_id() if early_stop_action_end else None
        bos_token = torch.full(
            (bsize, 1),
            self.model._paligemma_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _total_t_images, _ = self.model.embed_prefix_fast(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
        prefix_embs = self._match_model_precision(prefix_embs)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
        (prefix_out, _), past_key_values = self._forward_prefix_language_model(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=prefix_embs,
            use_cache=True,
            cache_position=torch.arange(prefix_embs.shape[1], device=device, dtype=torch.long),
        )
        prev_hidden = prefix_out[:, -1:, :]
        prev_logits = lm_head(prev_hidden)
        current_pad_mask = prefix_pad_masks
        generated_tokens: list[int] = []
        logits_by_step: list[torch.Tensor] = []
        target_forwards = 1
        verify_forwards = 0
        fallback_forwards = 0
        drafted_tokens = 0
        accepted_tokens = 0
        accepted_future_tokens = 0
        full_block_rejects = 0
        confidence_rejects = 0
        gate_rejects = 0
        verify_confidence_rejects = 0
        verify_margin_rejects = 0
        short_accept_rejects = 0
        resync_forwards = 0
        known_token_advances = 0
        cooldown_fallbacks = 0
        unknown_context_tokens = 0
        repeat_draft_attempts = 0
        repeat_drafted_tokens = 0
        repeat_accepted_future_tokens = 0
        pattern_draft_attempts = 0
        pattern_drafted_tokens = 0
        pattern_accepted_future_tokens = 0
        unverified_pattern_tokens = 0
        unverified_pattern_eos_tokens = 0
        reject_cooldown_remaining = 0
        consecutive_full_block_rejects = 0
        consecutive_spec_fallbacks = 0
        debug_events: list[dict[str, Any]] = []

        def advance_one(
            next_token: torch.Tensor,
            logits_for_token: torch.Tensor,
            *,
            count_fallback: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal current_pad_mask, past_key_values, target_forwards, fallback_forwards
            logits_by_step.append(logits_for_token)
            generated_tokens.append(int(next_token.item()))
            next_token_emb = self.model.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            next_token_emb = next_token_emb.to(dtype=prefix_embs.dtype)
            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self.model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1),
                dtype=next_token_emb.dtype,
            )
            (step_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_token_emb,
                use_cache=True,
                cache_position=torch.tensor([current_pad_mask.shape[1] - 1], device=device, dtype=torch.long),
            )
            target_forwards += 1
            if count_fallback:
                fallback_forwards += 1
            return lm_head(step_out[:, -1:, :]), step_out[:, -1:, :]

        def draft_future_from_state(current_token: torch.Tensor | None, steps: int) -> tuple[list[int], list[float]]:
            nonlocal unknown_context_tokens
            context_len = int(block_drafter.config.context_len)
            pad_id = len(token_map)
            raw_context = [*generated_tokens]
            if current_token is not None:
                raw_context.append(int(current_token.item()))
            encoded_context = [token_map.token_to_class.get(int(token), -1) for token in raw_context[-context_len:]]
            if any(cls < 0 for cls in encoded_context):
                unknown_context_tokens += sum(1 for cls in encoded_context if cls < 0)
                if not allow_unknown_context:
                    return [], []
                encoded_context = [pad_id if cls < 0 else cls for cls in encoded_context]
            context = torch.full((1, context_len), pad_id, dtype=torch.long, device=device)
            if encoded_context:
                context[0, -len(encoded_context) :] = torch.tensor(encoded_context, dtype=torch.long, device=device)
            draft_classes, confidences = block_drafter.draft(
                prev_hidden[:, -1, :].to(dtype=draft_dtype),
                context,
                steps=steps,
                refine_steps=refine_steps,
            )
            draft_tokens = token_map.decode_tensor(draft_classes[0]).tolist()
            return [int(token) for token in draft_tokens], [float(conf) for conf in confidences[0].tolist()]

        def mark_spec_fallback() -> None:
            nonlocal consecutive_spec_fallbacks, reject_cooldown_remaining
            consecutive_spec_fallbacks += 1
            if (
                spec_fallback_cooldown_steps > 0
                and spec_fallback_cooldown_after > 0
                and consecutive_spec_fallbacks >= int(spec_fallback_cooldown_after)
            ):
                reject_cooldown_remaining = max(reject_cooldown_remaining, int(spec_fallback_cooldown_steps))
                consecutive_spec_fallbacks = 0

        def causal_mask_from_pad(pad_mask: torch.Tensor) -> torch.Tensor:
            seq_len = int(pad_mask.shape[1])
            causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=pad_mask.device).tril()
            return pad_mask[:, None, :] & causal[None, :, :]

        def resync_from_generated() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal current_pad_mask, past_key_values, target_forwards, resync_forwards
            if not generated_tokens:
                raise RuntimeError("Cannot resync block drafter cache before any generated token")
            full_fast_tokens = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            full_fast_masks = torch.ones_like(full_fast_tokens, dtype=torch.bool)
            full_embs, full_pad_masks, _full_att_masks, _total_t_images, _num_fast_embs = self.model.embed_prefix_fast(
                images,
                img_masks,
                tokens,
                masks,
                fast_action_tokens=full_fast_tokens,
                fast_action_masks=full_fast_masks,
            )
            full_embs = self._match_model_precision(full_embs)
            full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
            full_att_4d = self.model._prepare_attention_masks_4d(
                causal_mask_from_pad(full_pad_masks),
                dtype=full_embs.dtype,
            )
            (full_out, _), past_key_values = self._forward_prefix_language_model(
                attention_mask=full_att_4d,
                position_ids=full_position_ids,
                past_key_values=None,
                inputs_embeds=full_embs,
                use_cache=True,
                cache_position=torch.arange(full_embs.shape[1], device=device, dtype=torch.long),
            )
            current_pad_mask = full_pad_masks
            target_forwards += 1
            resync_forwards += 1
            return lm_head(full_out[:, -1:, :]), full_out[:, -1:, :]

        while len(generated_tokens) < max_decoding_steps:
            remaining = max_decoding_steps - len(generated_tokens)
            target_next = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)
            if action_end_token_id is not None and int(target_next.item()) == action_end_token_id:
                logits_by_step.append(prev_logits)
                generated_tokens.append(int(target_next.item()))
                break

            future_steps = min(int(lookahead), remaining - 1)
            if len(generated_tokens) < min_spec_position or future_steps <= 0:
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue
            if draft_after_known_token:
                prev_logits, prev_hidden = advance_one(target_next, prev_logits, count_fallback=False)
                known_token_advances += 1
                if len(generated_tokens) >= max_decoding_steps:
                    break
                if action_end_token_id is not None and generated_tokens[-1] == action_end_token_id:
                    break

                future_steps = min(int(lookahead), max_decoding_steps - len(generated_tokens))
                if future_steps <= 0:
                    continue
                if reject_cooldown_remaining > 0:
                    reject_cooldown_remaining -= 1
                    cooldown_fallbacks += 1
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "post_known_reject_cooldown",
                                "remaining": reject_cooldown_remaining,
                            }
                        )
                    continue

                draft_source = "block"
                draft_future, draft_confidences = draft_future_from_state(None, future_steps)
                if draft_confidences and min(draft_confidences) < min_draft_confidence:
                    confidence_rejects += 1
                    mark_spec_fallback()
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "post_known_confidence_fallback",
                                "min_confidence": min(draft_confidences),
                            }
                        )
                    continue
                if not draft_future:
                    mark_spec_fallback()
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "post_known_fallback_no_draft",
                            }
                        )
                    continue
                if block_gate is not None and block_gate_threshold > 0.0:
                    gate_lookahead = int(getattr(block_gate.config, "lookahead", len(draft_future)))
                    features = block_gate_features(
                        position=len(generated_tokens),
                        max_decoding_steps=max_decoding_steps,
                        confidences=draft_confidences,
                        lookahead=gate_lookahead,
                    ).to(device=device, dtype=draft_dtype)[None, :]
                    gate_prob = float(
                        block_gate.probability(prev_hidden[:, -1, :].to(dtype=draft_dtype), features).item()
                    )
                    if gate_prob < block_gate_threshold:
                        gate_rejects += 1
                        mark_spec_fallback()
                        if len(debug_events) < debug_event_limit:
                            debug_events.append(
                                {
                                    "pos": len(generated_tokens),
                                    "kind": "post_known_gate_fallback",
                                    "gate_probability": gate_prob,
                                }
                            )
                        continue

                candidate = [int(token) for token in draft_future]
                draft_tensor = torch.tensor([candidate], dtype=torch.long, device=device)
                drafted_tokens += len(candidate)
                old_mask_len = current_pad_mask.shape[1]
                if verify_from_scratch:
                    full_fast_tokens = torch.tensor(
                        [generated_tokens + candidate],
                        dtype=torch.long,
                        device=device,
                    )
                    full_fast_masks = torch.ones_like(full_fast_tokens, dtype=torch.bool)
                    (
                        full_embs,
                        full_pad_masks,
                        _full_att_masks,
                        _total_t_images,
                        num_fast_embs,
                    ) = self.model.embed_prefix_fast(
                        images,
                        img_masks,
                        tokens,
                        masks,
                        fast_action_tokens=full_fast_tokens,
                        fast_action_masks=full_fast_masks,
                    )
                    full_embs = self._match_model_precision(full_embs)
                    full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
                    full_att_4d = self.model._prepare_attention_masks_4d(
                        causal_mask_from_pad(full_pad_masks),
                        dtype=full_embs.dtype,
                    )
                    (full_out, _), verify_kv = self._forward_prefix_language_model(
                        attention_mask=full_att_4d,
                        position_ids=full_position_ids,
                        past_key_values=None,
                        inputs_embeds=full_embs,
                        use_cache=True,
                        cache_position=torch.arange(full_embs.shape[1], device=device, dtype=torch.long),
                    )
                    bos_hidden_idx = full_out.shape[1] - num_fast_embs - 1
                    start = bos_hidden_idx + len(generated_tokens)
                    verify_logits_all = lm_head(full_out[:, start : start + len(candidate) + 1, :])
                    verify_out = full_out[:, start + 1 : start + len(candidate) + 1, :]
                    verify_pad_mask = full_pad_masks
                else:
                    verify_past_key_values = clone_kv(past_key_values)
                    draft_embs = self.model.paligemma_with_expert.embed_language_tokens(draft_tensor)
                    draft_embs = draft_embs * math.sqrt(draft_embs.shape[-1])
                    draft_embs = draft_embs.to(dtype=prefix_embs.dtype)
                    verify_pad_mask = torch.cat(
                        [current_pad_mask, torch.ones((bsize, len(candidate)), dtype=torch.bool, device=device)],
                        dim=1,
                    )
                    verify_mask = torch.zeros(
                        (bsize, len(candidate), old_mask_len + len(candidate)),
                        dtype=torch.bool,
                        device=device,
                    )
                    for row in range(len(candidate)):
                        verify_mask[:, row, : old_mask_len + row + 1] = verify_pad_mask[:, : old_mask_len + row + 1]
                    verify_att_mask = self.model._prepare_attention_masks_4d(verify_mask, dtype=draft_embs.dtype)
                    position_start = int(torch.sum(current_pad_mask, dim=1).item())
                    verify_position_ids = torch.arange(
                        position_start,
                        position_start + len(candidate),
                        device=device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                    (verify_out, _), verify_kv = self._forward_prefix_language_model(
                        attention_mask=verify_att_mask,
                        position_ids=verify_position_ids,
                        past_key_values=verify_past_key_values,
                        inputs_embeds=draft_embs,
                        use_cache=True,
                        cache_position=verify_position_ids.squeeze(0),
                    )
                    verify_logits = lm_head(verify_out)
                    verify_logits_all = torch.cat([prev_logits, verify_logits], dim=1)
                target_forwards += 1
                verify_forwards += 1

                verify_needs_scores = min_verify_confidence > 0.0 or min_verify_margin > 0.0
                if not verify_needs_scores:
                    step_predictions = torch.argmax(verify_logits_all[:, : len(candidate), :], dim=-1).squeeze(0)
                    matches = step_predictions.eq(draft_tensor[0])
                    mismatches = torch.nonzero(~matches, as_tuple=False)
                    block_accepted = len(candidate) if mismatches.numel() == 0 else int(mismatches[0].item())
                else:
                    block_accepted = 0
                    for idx, token in enumerate(candidate):
                        step_logits = verify_logits_all[:, idx].float()
                        step_top2 = torch.topk(step_logits, k=2, dim=-1)
                        step_prediction = step_top2.indices[:, 0]
                        step_logit_margin = step_top2.values[:, 0] - step_top2.values[:, 1]
                        step_probs = F.softmax(step_logits, dim=-1)
                        step_confidence = step_probs.gather(-1, step_prediction.unsqueeze(-1)).squeeze(-1)
                        if int(step_prediction.item()) != int(token):
                            break
                        if float(step_confidence.item()) < min_verify_confidence:
                            verify_confidence_rejects += 1
                            break
                        if float(step_logit_margin.item()) < min_verify_margin:
                            verify_margin_rejects += 1
                            break
                        block_accepted += 1
                if full_block_only and block_accepted < len(candidate):
                    full_block_rejects += 1
                    consecutive_spec_fallbacks = 0
                    consecutive_full_block_rejects += 1
                    if (
                        reject_cooldown_steps > 0
                        and consecutive_full_block_rejects >= max(int(reject_cooldown_after), 1)
                    ):
                        reject_cooldown_remaining = int(reject_cooldown_steps)
                        consecutive_full_block_rejects = 0
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "post_known_full_block_reject",
                                "candidate_len": len(candidate),
                                "matched_prefix": block_accepted,
                                "cooldown_remaining": reject_cooldown_remaining,
                            }
                        )
                    continue
                if min_future_accept > 0 and block_accepted < int(min_future_accept):
                    short_accept_rejects += 1
                    mark_spec_fallback()
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "post_known_short_accept_reject",
                                "candidate_len": len(candidate),
                                "matched_prefix": block_accepted,
                                "min_future_accept": int(min_future_accept),
                            }
                        )
                    continue
                consecutive_full_block_rejects = 0
                consecutive_spec_fallbacks = 0
                if max_future_accept is not None and max_future_accept >= 0:
                    block_accepted = min(block_accepted, int(max_future_accept))
                accepted_emit = block_accepted if accept_partial_blocks else (
                    block_accepted if block_accepted == len(candidate) else 0
                )
                if action_end_token_id is not None:
                    for idx in range(accepted_emit):
                        if int(candidate[idx]) == action_end_token_id:
                            accepted_emit = idx + 1
                            break
                if len(debug_events) < debug_event_limit:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "post_known_verify",
                            "source": draft_source,
                            "candidate_len": len(candidate),
                            "future_len": len(candidate),
                            "accepted": accepted_emit,
                            "accepted_future": accepted_emit,
                            "matched_prefix": block_accepted,
                            "max_future_accept": max_future_accept,
                        }
                    )
                for idx in range(accepted_emit):
                    logits_by_step.append(verify_logits_all[:, idx : idx + 1, :])
                    generated_tokens.append(int(candidate[idx]))

                accepted_tokens += accepted_emit
                accepted_future_tokens += accepted_emit
                if resync_accepted_cache and accepted_emit > 0:
                    prev_logits, prev_hidden = resync_from_generated()
                elif accepted_emit > 0:
                    if accepted_emit == len(candidate):
                        past_key_values = verify_kv
                        current_pad_mask = verify_pad_mask
                    else:
                        past_key_values = trim_kv(verify_kv, old_mask_len + accepted_emit)
                        current_pad_mask = verify_pad_mask[:, : old_mask_len + accepted_emit]
                    prev_hidden = verify_out[:, accepted_emit - 1 : accepted_emit, :]
                    prev_logits = verify_logits_all[:, accepted_emit : accepted_emit + 1, :]
                if len(generated_tokens) >= max_decoding_steps:
                    break
                if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                    break
                if accepted_emit == len(candidate):
                    continue

                correction_logits = prev_logits
                correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
                prev_logits, prev_hidden = advance_one(correction_token, correction_logits)
                if resync_accepted_cache:
                    prev_logits, prev_hidden = resync_from_generated()
                continue
            target_token_value = int(target_next.item())
            recent_run = 0
            for token in reversed(generated_tokens):
                if int(token) != target_token_value:
                    break
                recent_run += 1
            repeat_ready = repeat_token_draft and recent_run >= max(int(repeat_token_min_run), 1)
            pattern_future: list[int] = []
            pattern_tokens: list[int] = []
            if repeat_pattern_draft and len(generated_tokens) >= int(repeat_pattern_min_position):
                seq_with_target = [*generated_tokens, target_token_value]
                max_period = min(int(repeat_pattern_max_period), len(seq_with_target) // 2)
                for period in range(2, max_period + 1):
                    if seq_with_target[-period:] == seq_with_target[-2 * period : -period]:
                        pattern = seq_with_target[-period:]
                        pattern_tokens = [int(token) for token in pattern]
                        pattern_future = [int(pattern[idx % period]) for idx in range(future_steps)]
                        break
            pattern_ready = bool(pattern_future)
            if reject_cooldown_remaining > 0:
                if not repeat_ready and not pattern_ready:
                    reject_cooldown_remaining -= 1
                    cooldown_fallbacks += 1
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "reject_cooldown_fallback",
                                "known_token": int(target_next.item()),
                                "remaining": reject_cooldown_remaining,
                            }
                        )
                    prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                    continue
                reject_cooldown_remaining = 0

            draft_source = "block"
            if repeat_ready:
                repeat_draft_attempts += 1
                draft_source = "repeat"
                draft_future = [target_token_value] * future_steps
                draft_confidences = [1.0] * future_steps
                repeat_drafted_tokens += len(draft_future)
            elif pattern_ready:
                pattern_draft_attempts += 1
                draft_source = "pattern"
                draft_future = pattern_future
                draft_confidences = [1.0] * len(draft_future)
                pattern_drafted_tokens += len(draft_future)
                if unverified_pattern_eos and pattern_tokens and action_end_token_id is not None:
                    candidate = [target_token_value]
                    if target_token_value != action_end_token_id:
                        candidate.append(int(action_end_token_id))
                    for token in candidate:
                        logits_by_step.append(prev_logits)
                        generated_tokens.append(int(token))
                    drafted_tokens += len(candidate)
                    accepted_tokens += len(candidate)
                    accepted_future_tokens += max(len(candidate) - 1, 0)
                    pattern_accepted_future_tokens += max(len(candidate) - 1, 0)
                    unverified_pattern_tokens += len(candidate)
                    unverified_pattern_eos_tokens += len(candidate)
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens) - len(candidate),
                                "kind": "unverified_pattern_eos",
                                "candidate_len": len(candidate),
                                "period": len(pattern_tokens),
                            }
                        )
                    break
                if unverified_pattern_tail and pattern_tokens:
                    candidate = [
                        target_token_value,
                        *[
                            int(pattern_tokens[idx % len(pattern_tokens)])
                            for idx in range(max(remaining - 1, 0))
                        ],
                    ][:remaining]
                    if action_end_token_id is not None:
                        for idx, token in enumerate(candidate):
                            if int(token) == action_end_token_id:
                                candidate = candidate[: idx + 1]
                                break
                    for token in candidate:
                        logits_by_step.append(prev_logits)
                        generated_tokens.append(int(token))
                    drafted_tokens += len(candidate)
                    accepted_tokens += len(candidate)
                    accepted_future_tokens += max(len(candidate) - 1, 0)
                    pattern_accepted_future_tokens += max(len(candidate) - 1, 0)
                    unverified_pattern_tokens += len(candidate)
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens) - len(candidate),
                                "kind": "unverified_pattern_tail",
                                "candidate_len": len(candidate),
                                "period": len(pattern_tokens),
                            }
                        )
                    break
            elif pattern_only:
                mark_spec_fallback()
                if len(debug_events) < debug_event_limit:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "fallback_no_pattern",
                            "known_token": int(target_next.item()),
                        }
                    )
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue
            else:
                draft_future, draft_confidences = draft_future_from_state(target_next, future_steps)
            if draft_confidences and min(draft_confidences) < min_draft_confidence:
                confidence_rejects += 1
                mark_spec_fallback()
                if len(debug_events) < debug_event_limit:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "confidence_fallback",
                            "known_token": int(target_next.item()),
                            "min_confidence": min(draft_confidences),
                        }
                    )
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue
            if not draft_future:
                mark_spec_fallback()
                if len(debug_events) < debug_event_limit:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "fallback_no_draft",
                            "known_token": int(target_next.item()),
                        }
                    )
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue
            if block_gate is not None and block_gate_threshold > 0.0 and draft_source not in {"repeat", "pattern"}:
                gate_lookahead = int(getattr(block_gate.config, "lookahead", len(draft_future)))
                features = block_gate_features(
                    position=len(generated_tokens),
                    max_decoding_steps=max_decoding_steps,
                    confidences=draft_confidences,
                    lookahead=gate_lookahead,
                ).to(device=device, dtype=draft_dtype)[None, :]
                gate_prob = float(
                    block_gate.probability(prev_hidden[:, -1, :].to(dtype=draft_dtype), features).item()
                )
                if gate_prob < block_gate_threshold:
                    gate_rejects += 1
                    mark_spec_fallback()
                    if len(debug_events) < debug_event_limit:
                        debug_events.append(
                            {
                                "pos": len(generated_tokens),
                                "kind": "gate_fallback",
                                "known_token": int(target_next.item()),
                                "gate_probability": gate_prob,
                            }
                        )
                    prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                    continue

            candidate = [int(target_next.item()), *draft_future]
            draft_tensor = torch.tensor([candidate], dtype=torch.long, device=device)
            drafted_tokens += len(candidate)
            old_mask_len = current_pad_mask.shape[1]
            if verify_from_scratch:
                full_fast_tokens = torch.tensor(
                    [generated_tokens + candidate],
                    dtype=torch.long,
                    device=device,
                )
                full_fast_masks = torch.ones_like(full_fast_tokens, dtype=torch.bool)
                full_embs, full_pad_masks, _full_att_masks, _total_t_images, num_fast_embs = self.model.embed_prefix_fast(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    fast_action_tokens=full_fast_tokens,
                    fast_action_masks=full_fast_masks,
                )
                full_embs = self._match_model_precision(full_embs)
                full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
                full_att_4d = self.model._prepare_attention_masks_4d(
                    causal_mask_from_pad(full_pad_masks),
                    dtype=full_embs.dtype,
                )
                (full_out, _), verify_kv = self._forward_prefix_language_model(
                    attention_mask=full_att_4d,
                    position_ids=full_position_ids,
                    past_key_values=None,
                    inputs_embeds=full_embs,
                    use_cache=True,
                    cache_position=torch.arange(full_embs.shape[1], device=device, dtype=torch.long),
                )
                bos_hidden_idx = full_out.shape[1] - num_fast_embs - 1
                start = bos_hidden_idx + len(generated_tokens)
                verify_logits_all = lm_head(full_out[:, start : start + len(candidate) + 1, :])
                verify_out = full_out[:, start + 1 : start + len(candidate) + 1, :]
                verify_pad_mask = full_pad_masks
            else:
                verify_past_key_values = clone_kv(past_key_values)
                draft_embs = self.model.paligemma_with_expert.embed_language_tokens(draft_tensor)
                draft_embs = draft_embs * math.sqrt(draft_embs.shape[-1])
                draft_embs = draft_embs.to(dtype=prefix_embs.dtype)
                verify_pad_mask = torch.cat(
                    [current_pad_mask, torch.ones((bsize, len(candidate)), dtype=torch.bool, device=device)],
                    dim=1,
                )
                verify_mask = torch.zeros(
                    (bsize, len(candidate), old_mask_len + len(candidate)),
                    dtype=torch.bool,
                    device=device,
                )
                for row in range(len(candidate)):
                    verify_mask[:, row, : old_mask_len + row + 1] = verify_pad_mask[:, : old_mask_len + row + 1]
                verify_att_mask = self.model._prepare_attention_masks_4d(verify_mask, dtype=draft_embs.dtype)
                position_start = int(torch.sum(current_pad_mask, dim=1).item())
                verify_position_ids = torch.arange(
                    position_start,
                    position_start + len(candidate),
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                (verify_out, _), verify_kv = self._forward_prefix_language_model(
                    attention_mask=verify_att_mask,
                    position_ids=verify_position_ids,
                    past_key_values=verify_past_key_values,
                    inputs_embeds=draft_embs,
                    use_cache=True,
                    cache_position=verify_position_ids.squeeze(0),
                )
                verify_logits = lm_head(verify_out)
                verify_logits_all = torch.cat([prev_logits, verify_logits], dim=1)
            target_forwards += 1
            verify_forwards += 1

            verify_needs_scores = min_verify_confidence > 0.0 or min_verify_margin > 0.0
            if not verify_needs_scores:
                step_predictions = torch.argmax(verify_logits_all[:, : len(candidate), :], dim=-1).squeeze(0)
                matches = step_predictions.eq(draft_tensor[0])
                mismatches = torch.nonzero(~matches, as_tuple=False)
                block_accepted = len(candidate) if mismatches.numel() == 0 else int(mismatches[0].item())
            else:
                block_accepted = 0
                for idx, token in enumerate(candidate):
                    step_logits = verify_logits_all[:, idx].float()
                    step_top2 = torch.topk(step_logits, k=2, dim=-1)
                    step_prediction = step_top2.indices[:, 0]
                    step_logit_margin = step_top2.values[:, 0] - step_top2.values[:, 1]
                    step_probs = F.softmax(step_logits, dim=-1)
                    step_confidence = step_probs.gather(-1, step_prediction.unsqueeze(-1)).squeeze(-1)
                    predicted_candidate = int(step_prediction.item())
                    if predicted_candidate != int(token):
                        break
                    if idx > 0 and float(step_confidence.item()) < min_verify_confidence:
                        verify_confidence_rejects += 1
                        break
                    if idx > 0 and float(step_logit_margin.item()) < min_verify_margin:
                        verify_margin_rejects += 1
                        break
                    block_accepted += 1
            if block_accepted == 0:
                block_accepted = 1 if int(candidate[0]) == int(target_next.item()) else 0
            if full_block_only and block_accepted < len(candidate):
                full_block_rejects += 1
                consecutive_spec_fallbacks = 0
                consecutive_full_block_rejects += 1
                if (
                    reject_cooldown_steps > 0
                    and consecutive_full_block_rejects >= max(int(reject_cooldown_after), 1)
                ):
                    reject_cooldown_remaining = int(reject_cooldown_steps)
                    consecutive_full_block_rejects = 0
                if len(debug_events) < debug_event_limit:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "full_block_reject",
                            "candidate_len": len(candidate),
                            "matched_prefix": block_accepted,
                            "cooldown_remaining": reject_cooldown_remaining,
                        }
                    )
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue
            if min_future_accept > 0 and block_accepted < 1 + int(min_future_accept):
                short_accept_rejects += 1
                mark_spec_fallback()
                if len(debug_events) < debug_event_limit:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens),
                            "kind": "short_accept_reject",
                            "candidate_len": len(candidate),
                            "matched_prefix": block_accepted,
                            "min_future_accept": int(min_future_accept),
                        }
                    )
                prev_logits, prev_hidden = advance_one(target_next, prev_logits)
                continue
            consecutive_full_block_rejects = 0
            consecutive_spec_fallbacks = 0
            if max_future_accept is not None and max_future_accept >= 0:
                block_accepted = min(block_accepted, 1 + int(max_future_accept))
            if accept_partial_blocks:
                accepted_emit = block_accepted
            else:
                accepted_emit = block_accepted if block_accepted == len(candidate) else min(block_accepted, 1)
            if action_end_token_id is not None:
                for idx in range(accepted_emit):
                    if int(candidate[idx]) == action_end_token_id:
                        accepted_emit = idx + 1
                        break
            if len(debug_events) < debug_event_limit:
                debug_events.append(
                    {
                        "pos": len(generated_tokens),
                        "kind": "verify",
                        "source": draft_source,
                        "candidate_len": len(candidate),
                        "future_len": len(draft_future),
                        "accepted": accepted_emit,
                        "accepted_future": max(accepted_emit - 1, 0),
                        "matched_prefix": block_accepted,
                        "max_future_accept": max_future_accept,
                    }
                )
            for idx in range(accepted_emit):
                logits_by_step.append(verify_logits_all[:, idx : idx + 1, :])
                generated_tokens.append(int(candidate[idx]))

            accepted_tokens += accepted_emit
            accepted_future_tokens += max(accepted_emit - 1, 0)
            if draft_source == "repeat":
                repeat_accepted_future_tokens += max(accepted_emit - 1, 0)
            if draft_source == "pattern":
                pattern_accepted_future_tokens += max(accepted_emit - 1, 0)
            if resync_accepted_cache and accepted_emit > 0:
                prev_logits, prev_hidden = resync_from_generated()
            elif accepted_emit > 0:
                if accepted_emit == len(candidate):
                    past_key_values = verify_kv
                    current_pad_mask = verify_pad_mask
                else:
                    past_key_values = trim_kv(verify_kv, old_mask_len + accepted_emit)
                    current_pad_mask = verify_pad_mask[:, : old_mask_len + accepted_emit]
                prev_hidden = verify_out[:, accepted_emit - 1 : accepted_emit, :]
                prev_logits = verify_logits_all[:, accepted_emit : accepted_emit + 1, :]
            if len(generated_tokens) >= max_decoding_steps:
                break
            if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                break
            if accepted_emit == len(candidate):
                continue

            correction_logits = prev_logits
            correction_token = torch.argmax(correction_logits[:, -1], dim=-1, keepdim=True)
            prev_logits, prev_hidden = advance_one(correction_token, correction_logits)
            if resync_accepted_cache:
                prev_logits, prev_hidden = resync_from_generated()

        generated = torch.tensor([generated_tokens[:max_decoding_steps]], dtype=torch.long, device=device)
        stats = {
            "prefix_hidden": prefix_out[:, -1, :].detach().float(),
            "target_forwards": target_forwards,
            "verify_forwards": verify_forwards,
            "fallback_forwards": fallback_forwards,
            "resync_forwards": resync_forwards,
            "known_token_advances": known_token_advances,
            "drafted_tokens": drafted_tokens,
            "accepted_tokens": accepted_tokens,
            "accepted_future_tokens": accepted_future_tokens,
            "full_block_rejects": full_block_rejects,
            "confidence_rejects": confidence_rejects,
            "gate_rejects": gate_rejects,
            "verify_confidence_rejects": verify_confidence_rejects,
            "verify_margin_rejects": verify_margin_rejects,
            "short_accept_rejects": short_accept_rejects,
            "cooldown_fallbacks": cooldown_fallbacks,
            "unknown_context_tokens": unknown_context_tokens,
            "repeat_draft_attempts": repeat_draft_attempts,
            "repeat_drafted_tokens": repeat_drafted_tokens,
            "repeat_accepted_future_tokens": repeat_accepted_future_tokens,
            "pattern_draft_attempts": pattern_draft_attempts,
            "pattern_drafted_tokens": pattern_drafted_tokens,
            "pattern_accepted_future_tokens": pattern_accepted_future_tokens,
            "unverified_pattern_tokens": unverified_pattern_tokens,
            "unverified_pattern_eos_tokens": unverified_pattern_eos_tokens,
            "acceptance_rate": accepted_tokens / max(drafted_tokens, 1),
            "future_acceptance_rate": accepted_future_tokens
            / max(drafted_tokens if draft_after_known_token else drafted_tokens - verify_forwards, 1),
            "tokens_per_target_forward": len(generated_tokens) / max(target_forwards, 1),
            "known_first_token": True,
            "draft_after_known_token": bool(draft_after_known_token),
            "verify_from_scratch": verify_from_scratch,
            "resync_accepted_cache": resync_accepted_cache,
            "min_verify_margin": min_verify_margin,
            "block_gate_threshold": block_gate_threshold,
            "max_future_accept": max_future_accept,
            "min_future_accept": int(min_future_accept),
            "full_block_only": full_block_only,
            "allow_unknown_context": allow_unknown_context,
            "repeat_token_draft": bool(repeat_token_draft),
            "repeat_token_min_run": int(repeat_token_min_run),
            "repeat_pattern_draft": bool(repeat_pattern_draft),
            "repeat_pattern_max_period": int(repeat_pattern_max_period),
            "repeat_pattern_min_position": int(repeat_pattern_min_position),
            "pattern_only": bool(pattern_only),
            "unverified_pattern_tail": bool(unverified_pattern_tail),
            "unverified_pattern_eos": bool(unverified_pattern_eos),
            "reject_cooldown_steps": int(reject_cooldown_steps),
            "reject_cooldown_after": int(reject_cooldown_after),
            "spec_fallback_cooldown_steps": int(spec_fallback_cooldown_steps),
            "spec_fallback_cooldown_after": int(spec_fallback_cooldown_after),
            "debug_events": debug_events,
        }
        return generated, torch.cat(logits_by_step, dim=1), stats

    def _language_tokens(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        except ImportError as exc:
            raise RuntimeError("LeRobot must be installed to use PI0FastTokenLogitAdapter") from exc
        return batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

    def _action_key(self) -> str:
        try:
            from lerobot.utils.constants import ACTION
        except ImportError as exc:
            raise RuntimeError("LeRobot must be installed to use PI0FastTokenLogitAdapter") from exc
        return ACTION

    def _detokenize_generated_actions(self, token_ids: torch.Tensor) -> torch.Tensor:
        action_horizon = self.policy.config.n_action_steps
        action_dim = self.policy.config.output_features[self._action_key()].shape[0]
        try:
            return self.policy.detokenize_actions(token_ids, action_horizon=action_horizon, action_dim=action_dim)
        except AssertionError as exc:
            # Some target generations omit the literal "Action: " prefix while
            # still producing a valid FAST action body ending in "|". LeRobot's
            # detokenizer only uses the prefix as a framing marker and removes
            # it before FAST decoding, so add it only for action recovery. Keep
            # returned trace token ids raw for SD training/evaluation.
            prefix = self.model._paligemma_tokenizer.encode("Action: ", add_special_tokens=False)
            if len(prefix) < 2:
                raise exc
            prefix_tensor = torch.tensor(prefix, dtype=token_ids.dtype, device=token_ids.device).unsqueeze(0)
            prefix_tensor = prefix_tensor.expand(token_ids.shape[0], -1)
            framed = torch.cat([prefix_tensor, token_ids], dim=1)
            return self.policy.detokenize_actions(framed, action_horizon=action_horizon, action_dim=action_dim)

    @property
    def action_end_token_id(self) -> int:
        return self._action_end_token_id()

    def _action_end_token_id(self) -> int:
        token_id = self.model._paligemma_tokenizer.convert_tokens_to_ids("|")
        if token_id is None or token_id < 0:
            raise RuntimeError("Could not resolve PI0-FAST action end token '|'")
        return int(token_id)

    def _match_model_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        paligemma = self.model.paligemma_with_expert.paligemma
        language_model = getattr(paligemma, "language_model", None)
        if language_model is None:
            language_model = paligemma.model.language_model
        first_layer = language_model.layers[0]
        dtype = first_layer.self_attn.q_proj.weight.dtype
        return tensor.to(dtype=dtype) if tensor.dtype != dtype else tensor

    def _forward_prefix_language_model(
        self,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Any,
        inputs_embeds: torch.Tensor,
        use_cache: bool,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor | None], Any]:
        """Run the prefix-only PaliGemma path used by LeRobot's target decoder."""

        return self.model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[inputs_embeds, None],
            use_cache=use_cache,
            adarms_cond=[None, None],
        )

    @staticmethod
    def _select_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature > 0:
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True)

    @staticmethod
    def _ensure_2d_long(token_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        tokens = token_ids.to(device=device, dtype=torch.long)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 2:
            raise ValueError(f"Expected draft token shape [B, K], got {tuple(tokens.shape)}")
        return tokens

    @staticmethod
    def _accepted_prefix(target_token_ids: torch.Tensor, draft_token_ids: torch.Tensor) -> int:
        matches = (target_token_ids[0] == draft_token_ids[0]).detach().cpu().tolist()
        accepted = 0
        for matched in matches:
            if not matched:
                break
            accepted += 1
        return accepted
