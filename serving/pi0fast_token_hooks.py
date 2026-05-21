"""Token/logit accessors for LeRobot π0-FAST policies.

LeRobot's public ``PI0FastPolicy.predict_action_chunk`` returns continuous
actions. Internally, π0-FAST autoregressively emits PaliGemma-token-space FAST
action tokens and computes logits at every decode step. This adapter mirrors the
upstream decode paths so experiments can log tokens/logits and run exact
draft/verify checks without vendoring or forking LeRobot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from serving.kv_cache_manager import clone_kv, trim_kv


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

        actions = self.policy.detokenize_actions(
            token_ids,
            action_horizon=self.policy.config.n_action_steps,
            action_dim=self.policy.config.output_features[self._action_key()].shape[0],
        )
        return PI0FastGenerationTrace(
            actions=actions,
            token_ids=token_ids,
            logits=logits,
            hidden_states=hidden_states,
        )

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
    def predict_action_chunk_ngram_speculative(
        self,
        batch: dict[str, torch.Tensor],
        drafter: Any,
        lookahead: int = 8,
        reuse_full_blocks: bool = False,
        verify_from_scratch: bool = False,
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
        )
        actions = self.policy.detokenize_actions(
            token_ids,
            action_horizon=self.policy.config.n_action_steps,
            action_dim=self.policy.config.output_features[self._action_key()].shape[0],
        )
        return PI0FastGenerationTrace(actions=actions, token_ids=token_ids, logits=logits, stats=stats)

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
                    cache_position=torch.arange(
                        old_mask_len,
                        old_mask_len + len(draft),
                        device=device,
                        dtype=torch.long,
                    ),
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
            if reuse_full_blocks and block_accepted == len(draft):
                full_block_reuses += 1
                for idx in range(block_accepted):
                    logits_for_token = verify_logits_all[:, idx : idx + 1, :]
                    logits_by_step.append(logits_for_token)
                    generated_tokens.append(int(draft[idx]))
                accepted = block_accepted
                past_key_values = verify_kv
                current_pad_mask = verify_pad_mask
                prev_logits = verify_logits_all[:, accepted : accepted + 1, :]
            else:
                accepted = block_accepted
                for idx in range(accepted):
                    logits_for_token = verify_logits_all[:, idx : idx + 1, :]
                    logits_by_step.append(logits_for_token)
                    generated_tokens.append(int(draft[idx]))

                keep_len = old_mask_len + accepted
                past_key_values = trim_kv(verify_kv, keep_len)
                current_pad_mask = verify_pad_mask[:, :keep_len].contiguous()

                if len(generated_tokens) < max_decoding_steps:
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

        generated = torch.tensor([generated_tokens[:max_decoding_steps]], dtype=torch.long, device=device)
        stats = {
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
