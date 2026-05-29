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

from serving.kv_cache_manager import clone_kv, trim_kv
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
        )

    @torch.no_grad()
    def predict_action_chunk_action_end(
        self,
        batch: dict[str, torch.Tensor],
        temperature: float | None = None,
        max_decoding_steps: int | None = None,
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

        if self.policy.config.use_kv_cache:
            token_ids = self.sample_actions_fast_kv_cache_action_end(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_steps,
                temperature=decode_temperature,
            )
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

        actions = self._detokenize_generated_actions(token_ids)
        empty_logits = torch.empty((token_ids.shape[0], 0, 0), dtype=torch.float32, device=token_ids.device)
        return PI0FastGenerationTrace(
            actions=actions,
            token_ids=token_ids,
            logits=empty_logits,
            stats={
                "mode": "action_end_no_logits",
                "emitted_tokens": int(token_ids.shape[1]),
                "row_token_counts": getattr(self, "_last_action_end_row_token_counts", None),
                "stopped_on_action_end": int(token_ids.shape[1]) < int(max_steps),
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
        verify_from_scratch: bool = False,
        early_stop_action_end: bool = False,
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
            verify_from_scratch=verify_from_scratch,
            early_stop_action_end=early_stop_action_end,
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
    ) -> PI0FastGenerationTrace:
        """Decode a bounded FAST-token prefix, then append action-end if needed."""

        self.policy.eval()
        images, img_masks = self.policy._preprocess_images(batch)
        tokens, masks = self._language_tokens(batch)
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
        action_end = self._action_end_token_id()
        if token_ids.shape[1] == 0 or int(token_ids[0, -1].item()) != action_end:
            eos = torch.tensor([[action_end]], dtype=token_ids.dtype, device=token_ids.device)
            token_ids = torch.cat([token_ids, eos], dim=1)
        actions = self._detokenize_generated_actions(token_ids)
        stats = {"cutoff_tokens": int(cutoff_tokens), "emitted_tokens": int(token_ids.shape[1])}
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
        max_stable_checkpoint: int | None = None,
        skip_unproductive_checks: bool = False,
        skip_unproductive_after_checkpoint: int = 0,
        continue_to_action_end_on_unstable: bool = False,
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

        def candidate_trace() -> tuple[torch.Tensor, torch.Tensor]:
            token_ids = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            if token_ids.shape[1] == 0 or int(token_ids[0, -1].item()) != action_end:
                eos = torch.tensor([[action_end]], dtype=token_ids.dtype, device=device)
                token_ids = torch.cat([token_ids, eos], dim=1)
            return token_ids, self._detokenize_generated_actions(token_ids)

        def gate_probability(checkpoint: int, token_ids: torch.Tensor, actions: torch.Tensor) -> float | None:
            if prefix_gate is None:
                return None
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
            features = torch.tensor(
                [[float(row.get(name, 0.0)) for name in PREFIX_GATE_FEATURES]],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                return float(prefix_gate.probability(features).item())

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
                }
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
                can_stop_at_checkpoint = max_stable_checkpoint is None or checkpoint <= max_stable_checkpoint
                if stable_count >= required_stable_checks and can_stop_at_checkpoint:
                    gate_prob = gate_probability(checkpoint, token_ids, actions)
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

        if continue_to_action_end_on_unstable:
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
            }
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
        next_token = self._select_next_token(lm_head(prefix_out[:, -1:, :]), temperature)
        generated = torch.full((bsize, max_decoding_steps), action_end_token_id, dtype=torch.long, device=device)
        generated[:, 0] = next_token.squeeze(-1)
        current_pad_mask = prefix_pad_masks
        active_indices = torch.arange(bsize, dtype=torch.long, device=device)
        emitted_lengths = torch.ones((bsize,), dtype=torch.long, device=device)

        finished = next_token.squeeze(-1) == action_end_token_id
        if bool(torch.all(finished)):
            self._last_action_end_row_token_counts = emitted_lengths.detach().cpu().tolist()
            return generated[:, :1]
        keep = torch.nonzero(~finished, as_tuple=False).flatten()
        if keep.numel() != bsize:
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
            next_token = self._select_next_token(lm_head(step_out[:, -1:, :]), temperature)
            selected = next_token.squeeze(-1)
            generated[active_indices, t] = selected
            emitted_lengths[active_indices] = t + 1
            finished = selected == action_end_token_id
            if bool(torch.all(finished)):
                max_len = int(emitted_lengths.max().item())
                self._last_action_end_row_token_counts = emitted_lengths.detach().cpu().tolist()
                return generated[:, :max_len]
            if bool(torch.any(finished)):
                keep = torch.nonzero(~finished, as_tuple=False).flatten()
                past_key_values = past_key_values.batch_select_indices(keep)
                current_pad_mask = current_pad_mask[keep]
                next_token = next_token[keep]
                active_indices = active_indices[keep]

        self._last_action_end_row_token_counts = emitted_lengths.detach().cpu().tolist()
        return generated

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
        verify_from_scratch: bool = False,
        early_stop_action_end: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Greedy exact FAST-token decode with n-gram speculative verification.

        This is a conservative exact-SD path: accepted drafted tokens are added
        to the target KV cache, while mismatches fall back to the target token.
        It does not emit the unprocessed bonus token, so the generated token
        stream remains straightforward to reason about.
        """

        if temperature != 0.0:
            raise ValueError("ngram speculative PI0-FAST decode currently supports greedy temperature=0 only")
        if max_decoding_steps is None:
            max_decoding_steps = self.model.config.max_action_tokens
        bsize = tokens.shape[0]
        if bsize != 1:
            raise ValueError("ngram speculative PI0-FAST decode currently supports batch size 1")

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
        full_block_reuses = 0
        drafted_tokens = 0
        accepted_tokens = 0
        debug_events: list[dict[str, Any]] = []

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

        while len(generated_tokens) < max_decoding_steps:
            remaining = max_decoding_steps - len(generated_tokens)
            target_next = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)
            if action_end_token_id is not None and int(target_next.item()) == action_end_token_id:
                logits_by_step.append(prev_logits)
                generated_tokens.append(int(target_next.item()))
                if len(debug_events) < 64:
                    debug_events.append(
                        {
                            "pos": len(generated_tokens) - 1,
                            "kind": "action_end",
                            "token": int(target_next.item()),
                        }
                    )
                break
            draft = drafter.draft(generated_tokens, lookahead=min(lookahead, remaining))
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
                prev_logits = advance_one(target_next, prev_logits, fallback=True)
                continue

            draft = draft[:remaining]
            draft_tensor = torch.tensor([draft], dtype=torch.long, device=device)
            drafted_tokens += len(draft)
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

            block_accepted = 0
            for idx in range(len(draft)):
                logits_for_candidate = verify_logits_all[:, idx : idx + 1, :]
                predicted_candidate = int(torch.argmax(logits_for_candidate[:, -1], dim=-1).item())
                if predicted_candidate != int(draft[idx]):
                    break
                block_accepted += 1
            block_accepted = min(block_accepted, remaining)
            if len(debug_events) < 64:
                debug_events.append(
                    {
                        "pos": len(generated_tokens),
                        "kind": "verify",
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
                past_key_values = verify_kv
                if accepted == len(draft):
                    current_pad_mask = verify_pad_mask
                else:
                    keep_len = old_mask_len + accepted
                    past_key_values = trim_kv(verify_kv, keep_len)
                    current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()
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
                        raise RuntimeError("Internal SD invariant failed: accepted prefix stopped before matching token")
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
                    prev_logits = advance_one(correction_token, correction_logits, fallback=True)
            accepted_tokens += accepted
            if len(generated_tokens) >= max_decoding_steps:
                break
            if action_end_token_id is not None and generated_tokens and generated_tokens[-1] == action_end_token_id:
                break

        generated = torch.tensor([generated_tokens[:max_decoding_steps]], dtype=torch.long, device=device)
        stats = {
            "prefix_hidden": prefix_out[:, -1, :].detach().float(),
            "target_forwards": target_forwards,
            "verify_forwards": verify_forwards,
            "fallback_forwards": fallback_forwards,
            "replay_forwards": replay_forwards,
            "full_block_reuses": full_block_reuses,
            "reuse_full_blocks": reuse_full_blocks,
            "verify_from_scratch": verify_from_scratch,
            "drafted_tokens": drafted_tokens,
            "accepted_tokens": accepted_tokens,
            "acceptance_rate": accepted_tokens / max(drafted_tokens, 1),
            "tokens_per_target_forward": len(generated_tokens) / max(target_forwards, 1),
            "debug_events": debug_events,
        }
        return generated, torch.cat(logits_by_step, dim=1), stats

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
        """Run the prefix-only PaliGemma language model with explicit cache positions.

        LeRobot's ``PI0FastPaliGemma.forward`` does not expose ``cache_position``.
        Speculative verification depends on batched cache updates after partial
        acceptance, so we bypass the thin wrapper here and call the underlying
        PiGemma/Gemma model directly.
        """

        language_model = self.model.paligemma_with_expert.paligemma.model.language_model
        output = language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            adarms_cond=None,
        )
        return [output.last_hidden_state, None], output.past_key_values

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
