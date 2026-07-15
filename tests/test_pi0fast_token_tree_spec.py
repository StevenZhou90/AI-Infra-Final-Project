from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from serving.pi0fast_pattern_drafter import PatternDraftConfig, PatternFastTokenDrafter
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter


class _IdentityHead(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _FakeTokenEmbedding:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(tokens.clamp(min=0, max=self.vocab_size - 1), num_classes=self.vocab_size).float()


class _FakeModel:
    def __init__(self, vocab_size: int = 128) -> None:
        self.config = SimpleNamespace(max_action_tokens=8)
        self._paligemma_tokenizer = SimpleNamespace(bos_token_id=1, vocab_size=vocab_size)
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(lm_head=_IdentityHead()),
            embed_language_tokens=_FakeTokenEmbedding(vocab_size),
        )
        self.vocab_size = vocab_size

    def embed_prefix_fast(self, images, img_masks, tokens, masks, fast_action_tokens=None, fast_action_masks=None):
        batch = int(tokens.shape[0])
        prefix_len = 3
        if fast_action_tokens is not None:
            prefix_len += int(fast_action_tokens.shape[1])
        embs = torch.zeros((batch, prefix_len, self.vocab_size), dtype=torch.float32)
        pad = torch.ones((batch, prefix_len), dtype=torch.bool)
        return embs, pad, pad.unsqueeze(1), None, int(fast_action_tokens.shape[1]) if fast_action_tokens is not None else 0

    def _prepare_attention_masks_4d(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return mask


class _FakePolicy:
    def __init__(self) -> None:
        self.model = _FakeModel()

    def eval(self) -> None:
        return None


class _FakePast:
    def batch_select_indices(self, _indices: torch.Tensor) -> "_FakePast":
        return self


class _CharStopTokenizer:
    bos_token_id = 1
    eos_token_id = 5
    vocab_size = 200

    def convert_tokens_to_ids(self, token: str):
        if token == "|":
            return 99
        return 0

    def encode(self, text: str, add_special_tokens: bool = False):
        if text == "Action: ":
            return [2, 3, 4]
        if text == "|":
            return [99]
        return []


class _CharStopBpe:
    def __init__(self) -> None:
        self.decode_calls = 0

    def decode(self, token_id: int) -> str:
        self.decode_calls += 1
        return {0: "abcd", 1: "efgh", 2: "ijkl"}.get(int(token_id), "")


class _CharStopActionTokenizer:
    def __init__(self) -> None:
        self.bpe_tokenizer = _CharStopBpe()


class _CharStopModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_action_tokens=8)
        self._paligemma_tokenizer = _CharStopTokenizer()
        self._targets = [90, 91, 92, 99]
        self._forward_calls = 0
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(lm_head=_IdentityHead()),
            embed_language_tokens=_FakeTokenEmbedding(200),
            forward=self.forward,
        )

    def embed_prefix_fast(self, images, img_masks, tokens, masks, fast_action_tokens=None, fast_action_masks=None):
        batch = int(tokens.shape[0])
        embs = torch.zeros((batch, 3, 200), dtype=torch.float32)
        pad = torch.ones((batch, 3), dtype=torch.bool)
        return embs, pad, pad.unsqueeze(1), None, 0

    def _prepare_attention_masks_4d(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return mask

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond):
        batch = int(inputs_embeds[0].shape[0])
        seq_len = int(inputs_embeds[0].shape[1])
        hidden = torch.zeros((batch, seq_len, 200), dtype=torch.float32)
        token = self._targets[min(self._forward_calls, len(self._targets) - 1)]
        hidden[:, -1, token] = 100.0
        self._forward_calls += 1
        return (hidden, None), _FakePast()


class _CharStopPolicy:
    def __init__(self) -> None:
        self.model = _CharStopModel()
        self.action_tokenizer = _CharStopActionTokenizer()
        self.config = SimpleNamespace(
            n_action_steps=2,
            output_features={"action": SimpleNamespace(shape=(4,))},
            fast_skip_tokens=0,
        )

    def eval(self) -> None:
        return None

    def _paligemma_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        mapping = {90: 0, 91: 1, 92: 2}
        return tokens.new_tensor([mapping.get(int(token.item()), 10_000) for token in tokens])

    def detokenize_actions(self, token_ids: torch.Tensor, *, action_horizon: int, action_dim: int) -> torch.Tensor:
        action_token_count = torch.isin(token_ids.cpu(), torch.tensor([90, 91, 92])).sum(dim=1).float()
        return action_token_count.view(-1, 1, 1).expand(-1, action_horizon, action_dim).clone()


def _install_identity_linear_head(policy: _CharStopPolicy) -> None:
    head = torch.nn.Linear(200, 200, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(200))
    policy.model.paligemma_with_expert.paligemma.lm_head = head


def test_pi0fast_action_end_decode_can_stop_after_action_chars() -> None:
    adapter = PI0FastTokenLogitAdapter(_CharStopPolicy())
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
    )

    assert token_ids.tolist() == [[90, 91, 99]]
    assert adapter._last_action_char_target == 8
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_action_char_target_override_stops_before_full_action() -> None:
    adapter = PI0FastTokenLogitAdapter(_CharStopPolicy())
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_target_chars=4,
    )

    assert token_ids.tolist() == [[90, 99]]
    assert adapter._last_action_char_target == 4
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_constrained_action_char_target_override_stops_before_full_action() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [0, 0, 0, 90, 91, 92, 99]
    _install_identity_linear_head(policy)
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_target_chars=4,
    )

    assert token_ids.tolist() == [[2, 3, 4, 90, 99]]
    assert adapter._last_action_char_target == 4
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_constrained_prefilled_action_prefix_stops_before_full_action() -> None:
    policy = _CharStopPolicy()
    _install_identity_linear_head(policy)
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        prefill_action_prefix=True,
        stop_on_action_chars=True,
        action_char_target_chars=4,
    )

    assert token_ids.tolist() == [[2, 3, 4, 90, 99]]
    assert adapter._last_constrained_prefill_action_prefix_tokens == 3
    assert adapter._last_constrained_restricted_head_calls == 1
    assert adapter._last_action_char_target == 4
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_strict_action_char_target_waits_for_extra_action_token() -> None:
    adapter = PI0FastTokenLogitAdapter(_CharStopPolicy())
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_strict_target_stop=True,
    )

    assert token_ids.tolist() == [[90, 91, 92, 99]]
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_count_mean == 12.0
    assert adapter._last_action_char_strict_target_stop is True


def test_pi0fast_action_char_target_confirmation_waits_after_target_hit() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 91, 80, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_target_confirm_tokens=1,
    )

    assert token_ids.tolist() == [[90, 91, 80, 99]]
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_target_confirm_tokens == 1
    assert adapter._last_action_char_target_confirm_block_count == 1
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_action_char_lookup_avoids_hot_loop_tokenizer_decode() -> None:
    policy = _CharStopPolicy()
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    adapter.prepare_action_char_length_lookup(device="cpu")
    warmed_decode_calls = policy.action_tokenizer.bpe_tokenizer.decode_calls
    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
    )

    assert token_ids.tolist() == [[90, 91, 99]]
    assert policy.action_tokenizer.bpe_tokenizer.decode_calls == warmed_decode_calls


def test_pi0fast_action_char_plateau_can_stop_after_decoded_chars_stabilize() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 80, 80, 80, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
    )

    assert token_ids.tolist() == [[90, 80, 80, 99]]
    assert adapter._last_action_char_target == 8
    assert adapter._last_action_char_min_chars == 4
    assert adapter._last_action_char_plateau_tokens == 2
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_plateau_stop_count == 1
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_action_char_plateau_rejects_eos_restart() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 5, 80, 80, 91, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_plateau_reject_eos_restart=True,
    )

    assert token_ids.tolist() == [[90, 5, 80, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_plateau_eos_restart_block_count == 1
    assert adapter._last_action_char_restart_fallback_count == 1
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_action_char_plateau_rejects_eos_action_restart() -> None:
    policy = _CharStopPolicy()
    policy.config.n_action_steps = 3
    policy.model._targets = [90, 5, 91, 80, 80, 92, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_plateau_reject_eos_restart=True,
    )

    assert token_ids.tolist() == [[90, 5, 91, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_plateau_eos_restart_block_count == 0
    assert adapter._last_action_char_restart_fallback_count == 1
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_action_char_plateau_rejects_bos_action_restart() -> None:
    policy = _CharStopPolicy()
    policy.config.n_action_steps = 3
    policy.model._targets = [90, 1, 91, 80, 80, 92, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_plateau_reject_eos_restart=True,
    )

    assert token_ids.tolist() == [[90, 1, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_plateau_eos_restart_block_count == 0
    assert adapter._last_action_char_restart_fallback_count == 1
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_action_char_plateau_ignores_action_end_after_restart() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 5, 1, 99, 91, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_plateau_reject_eos_restart=True,
    )

    assert token_ids.tolist() == [[90, 5, 1, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_plateau_eos_restart_block_count == 1
    assert adapter._last_action_char_restart_fallback_count == 1
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_action_char_restart_continue_allows_action_end() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 5, 1, 99, 91, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_continue=True,
    )

    assert token_ids.tolist() == [[90, 5, 1, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_restart_fallback_count == 0
    assert adapter._last_action_char_restart_continue_count == 1
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_action_char_stop_ignores_target_after_restart() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 1, 91, 92, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=6,
        stop_on_action_chars=True,
        action_char_plateau_reject_eos_restart=True,
    )

    assert token_ids.tolist() == [[90, 1, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_restart_fallback_count == 1
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_action_char_restart_continue_blocks_target_stop() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 1, 91, 92, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=6,
        stop_on_action_chars=True,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_continue=True,
    )

    assert token_ids.tolist() == [[90, 1, 91, 92, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_restart_fallback_count == 0
    assert adapter._last_action_char_restart_continue_count == 1
    assert adapter._last_action_char_count_mean == 12.0


def test_pi0fast_action_char_restart_reset_allows_new_target_stop() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 5, 91, 92, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_reset=True,
    )

    assert token_ids.tolist() == [[90, 5, 91, 92, 99]]
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_restart_fallback_count == 0
    assert adapter._last_action_char_restart_reset_count == 1
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_action_char_restart_reset_can_wait_for_low_text() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 1, 2, 80, 91, 92, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_reset=True,
        action_char_restart_reset_low_text_tokens=1,
    )

    assert token_ids.tolist() == [[90, 1, 2, 80, 91, 92, 99]]
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_restart_fallback_count == 0
    assert adapter._last_action_char_restart_reset_count == 1
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_action_char_reset_requires_action_end_waits_after_reset() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 5, 91, 92, 80, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_reset=True,
        action_char_reset_requires_action_end=True,
    )

    assert token_ids.tolist() == [[90, 5, 91, 92, 80, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_restart_fallback_count == 0
    assert adapter._last_action_char_restart_reset_count == 1
    assert adapter._last_action_char_reset_requires_action_end is True
    assert adapter._last_action_char_reset_requires_action_end_block_count == 1
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_action_char_plateau_allows_repeated_eos_tail() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 5, 5, 5, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_plateau_reject_eos_restart=True,
    )

    assert token_ids.tolist() == [[90, 5, 5, 99]]
    assert adapter._last_action_char_plateau_stop_count == 1
    assert adapter._last_action_char_plateau_eos_restart_block_count == 0
    assert adapter._last_action_char_count_mean == 4.0


def test_pi0fast_action_char_stability_guard_blocks_unstable_plateau() -> None:
    policy = _CharStopPolicy()
    policy.config.n_action_steps = 3
    policy.model._targets = [90, 80, 80, 91, 80, 80, 92]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_stable_checks=1,
    )

    assert token_ids.tolist() == [[90, 80, 80, 91, 80, 80, 92, 99]]
    assert adapter._last_action_char_target == 12
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_plateau_stability_block_count == 1
    assert adapter._last_action_char_stable_snapshot_count_mean == 3.0
    assert adapter._last_action_char_stable_count_mean == 0.0
    assert adapter._last_action_char_count_mean == 12.0


def test_pi0fast_action_char_stability_guard_allows_stable_plateau() -> None:
    policy = _CharStopPolicy()
    policy.config.n_action_steps = 3
    policy.model._targets = [90, 91, 80, 80, 92]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"
    adapter._detokenize_generated_actions = lambda token_ids: torch.zeros((1, 3, 4), dtype=torch.float32)

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_stable_checks=1,
    )

    assert token_ids.tolist() == [[90, 91, 80, 80, 99]]
    assert adapter._last_action_char_target == 12
    assert adapter._last_action_char_plateau_stop_count == 1
    assert adapter._last_action_char_plateau_stability_block_count == 0
    assert adapter._last_action_char_stable_snapshot_count_mean == 2.0
    assert adapter._last_action_char_stable_count_mean == 1.0
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_action_char_stability_guard_can_target_early_plateaus() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 80, 80, 91, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_stable_checks=1,
        action_char_stable_max_chars=8,
    )

    assert token_ids.tolist() == [[90, 80, 80, 91, 99]]
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_plateau_stability_block_count == 1
    assert adapter._last_action_char_stable_max_chars == 8
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_action_char_plateau_low_text_tail_guard_blocks_stop() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 80, 80, 91, 99]
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_plateau_low_text_tail_tokens=2,
    )

    assert token_ids.tolist() == [[90, 80, 80, 91, 99]]
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_plateau_low_text_tail_block_count == 1
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_constrained_action_char_plateau_can_stop_after_decoded_chars_stabilize() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 80, 80, 80, 99]
    _install_identity_linear_head(policy)
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        force_action_prefix=False,
        action_vocab_size=128,
        text_vocab_size=128,
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
    )

    assert token_ids.tolist() == [[90, 80, 80, 99]]
    assert adapter._last_action_char_target == 8
    assert adapter._last_action_char_plateau_stop_count == 1
    assert adapter._last_action_char_count_mean == 4.0
    assert adapter._last_constrained_restricted_head_calls == 3


def test_pi0fast_constrained_extra_tokens_extend_candidate_set() -> None:
    policy = _CharStopPolicy()
    policy.action_tokenizer.vocab_size = 16
    policy.model._targets = [150, 99]
    _install_identity_linear_head(policy)
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=4,
        force_action_prefix=False,
        action_vocab_size=16,
        text_vocab_size=128,
        structural_token_radius=0,
        extra_token_ids=[150, 150, 300],
    )

    assert token_ids.tolist() == [[150, 99]]
    assert adapter._last_constrained_extra_token_count == 1
    assert adapter._last_constrained_candidate_size == 145


def test_pi0fast_constrained_head_respects_lm_head_bias() -> None:
    policy = _CharStopPolicy()
    policy.action_tokenizer.vocab_size = 16
    policy.model._targets = [90]
    head = torch.nn.Linear(200, 200, bias=True)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[150] = 1.0
    policy.model.paligemma_with_expert.paligemma.lm_head = head
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=1,
        force_action_prefix=False,
        action_vocab_size=16,
        text_vocab_size=0,
        structural_token_radius=0,
        extra_token_ids=[150],
    )

    assert token_ids.tolist() == [[150]]
    assert adapter._last_constrained_extra_token_count == 1


def test_pi0fast_constrained_head_uses_argmax_tie_break() -> None:
    policy = _CharStopPolicy()
    policy.action_tokenizer.vocab_size = 0
    policy.model._targets = [90]
    head = torch.nn.Linear(200, 200, bias=True)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    policy.model.paligemma_with_expert.paligemma.lm_head = head
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=1,
        force_action_prefix=False,
        action_vocab_size=0,
        text_vocab_size=0,
        structural_token_radius=0,
        extra_token_ids=[151, 150],
    )

    assert token_ids.tolist() == [[2]]


def test_pi0fast_constrained_action_char_restart_continue_blocks_target_stop() -> None:
    policy = _CharStopPolicy()
    policy.config.n_action_steps = 3
    policy.model._targets = [90, 1, 91, 92, 99]
    _install_identity_linear_head(policy)
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=6,
        force_action_prefix=False,
        action_vocab_size=128,
        text_vocab_size=128,
        stop_on_action_chars=True,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_continue=True,
    )

    assert token_ids.tolist() == [[90, 1, 91, 92, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_plateau_stop_count == 0
    assert adapter._last_action_char_restart_fallback_count == 0
    assert adapter._last_action_char_restart_continue_count == 1
    assert adapter._last_action_char_count_mean == 12.0


def test_pi0fast_constrained_action_char_restart_reset_can_wait_for_low_text() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 1, 2, 80, 91, 92, 99]
    _install_identity_linear_head(policy)
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        force_action_prefix=False,
        action_vocab_size=128,
        text_vocab_size=128,
        stop_on_action_chars=True,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_reset=True,
        action_char_restart_reset_low_text_tokens=1,
        action_char_plateau_low_text_tail_tokens=2,
    )

    assert token_ids.tolist() == [[90, 1, 2, 80, 91, 92, 99]]
    assert adapter._last_action_char_stop_count == 1
    assert adapter._last_action_char_restart_reset_low_text_tokens == 1
    assert adapter._last_action_char_plateau_low_text_tail_tokens == 2
    assert adapter._last_action_char_restart_reset_count == 1
    assert adapter._last_action_char_count_mean == 8.0


def test_pi0fast_constrained_action_char_reset_requires_action_end_waits_after_reset() -> None:
    policy = _CharStopPolicy()
    policy.model._targets = [90, 5, 91, 92, 80, 99]
    _install_identity_linear_head(policy)
    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._match_model_precision = lambda tensor: tensor
    adapter._action_key = lambda: "action"

    token_ids = adapter.sample_actions_fast_kv_cache_action_end_constrained(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 2), dtype=torch.long),
        masks=torch.ones((1, 2), dtype=torch.bool),
        max_decoding_steps=8,
        force_action_prefix=False,
        action_vocab_size=128,
        text_vocab_size=128,
        stop_on_action_chars=True,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_reset=True,
        action_char_reset_requires_action_end=True,
    )

    assert token_ids.tolist() == [[90, 5, 91, 92, 80, 99]]
    assert adapter._last_action_char_stop_count == 0
    assert adapter._last_action_char_restart_fallback_count == 0
    assert adapter._last_action_char_restart_reset_count == 1
    assert adapter._last_action_char_reset_requires_action_end is True
    assert adapter._last_action_char_reset_requires_action_end_block_count == 1
    assert adapter._last_action_char_count_mean == 8.0


class _BranchingDrafter:
    def __init__(self, target: list[int]) -> None:
        self.target = target
        self._last_many_sources: list[list[str]] = []

    def draft(self, prefix: list[int], lookahead: int | None = None) -> list[int]:
        pos = len(prefix)
        if pos >= len(self.target):
            return []
        steps = max(1, int(lookahead or 1))
        return self.target[pos : pos + steps]

    def draft_many(
        self,
        prefix: list[int],
        lookahead: int | None = None,
        *,
        max_candidates: int,
        branch_width: int,
    ) -> list[list[int]]:
        pos = len(prefix)
        if pos >= len(self.target):
            return []
        steps = max(1, int(lookahead or 1))
        good = self.target[pos : pos + steps]
        if len(good) < 2:
            rows = [good]
            self._last_many_sources = [["previous_chunk_position"] * len(row) for row in rows]
            return rows
        bad = [good[0], 127, *good[2:]]
        rows = [bad, good][:max_candidates]
        self._last_many_sources = [["previous_chunk_position"] * len(row) for row in rows]
        return rows

    def last_many_sources(self) -> list[list[str]]:
        return [list(row) for row in self._last_many_sources]


class _SecondTokenMissDrafter:
    def __init__(self, target: list[int]) -> None:
        self.target = target
        self._last_sources: list[str] = []

    def draft(self, prefix: list[int], lookahead: int | None = None) -> list[int]:
        pos = len(prefix)
        if pos >= len(self.target):
            self._last_sources = []
            return []
        steps = max(1, int(lookahead or 1))
        if pos == 0:
            row = [self.target[0], self.target[1] + 1][:steps]
        else:
            row = self.target[pos : pos + steps]
        self._last_sources = ["ngram_continuation"] * len(row)
        return row

    def last_draft_sources(self) -> list[str]:
        return list(self._last_sources)


class _AnchorMissDrafter:
    def __init__(self, target: list[int]) -> None:
        self.target = target
        self._last_many_sources: list[list[str]] = []

    def draft(self, prefix: list[int], lookahead: int | None = None) -> list[int]:
        pos = len(prefix)
        if pos >= len(self.target):
            return []
        steps = max(1, int(lookahead or 1))
        return self.target[pos : pos + steps]

    def draft_many(
        self,
        prefix: list[int],
        lookahead: int | None = None,
        *,
        max_candidates: int,
        branch_width: int,
    ) -> list[list[int]]:
        pos = len(prefix)
        steps = max(1, int(lookahead or 1))
        if pos >= len(self.target):
            self._last_many_sources = []
            return []
        if not prefix:
            row = [self.target[0] + 1, self.target[1]][:steps]
            self._last_many_sources = [["action_transition_histogram"] * len(row)]
            return [row]
        row = self.target[pos : pos + steps]
        self._last_many_sources = [["action_transition_histogram"] * len(row)]
        return [row]

    def last_many_sources(self) -> list[list[str]]:
        return [list(row) for row in self._last_many_sources]


def test_pi0fast_online_tree_spec_selects_best_verified_candidate() -> None:
    target = [10, 20, 30, 40, 50]
    adapter = PI0FastTokenLogitAdapter(_FakePolicy())
    adapter._match_model_precision = lambda tensor: tensor
    prefix_len = 3

    def fake_forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position=None,
    ):
        batch, seq_len, vocab = inputs_embeds.shape
        hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
        for row in range(batch):
            for col in range(seq_len):
                pos = int(position_ids[row, col].item()) if position_ids is not None else col
                target_idx = max(0, pos - prefix_len + 1)
                token = target[min(target_idx, len(target) - 1)]
                hidden[row, col, token] = 100.0
        cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
        key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
        value = torch.zeros_like(key)
        return [hidden, None], ((key, value),)

    adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)

    token_ids, _logits, stats = adapter.sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=_BranchingDrafter(target),
        max_decoding_steps=len(target),
        lookahead=2,
        reuse_full_blocks=True,
        tree_width=2,
        tree_branch_width=2,
    )

    assert token_ids.tolist()[0] == target
    assert stats["tree_verifies"] > 0
    assert stats["tree_candidates"] > stats["tree_verifies"]
    assert stats["tree_accepted_tokens"] >= 2
    assert stats["previous_chunk_position_drafted_tokens"] > 0
    assert stats["previous_chunk_position_accepted_tokens"] >= 2


def test_pi0fast_online_can_defer_rejected_correction_token() -> None:
    target = [10, 20, 30, 40, 50]
    prefix_len = 3

    def make_adapter() -> PI0FastTokenLogitAdapter:
        adapter = PI0FastTokenLogitAdapter(_FakePolicy())
        adapter._match_model_precision = lambda tensor: tensor

        def fake_forward(
            self,
            *,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            cache_position=None,
        ):
            batch, seq_len, vocab = inputs_embeds.shape
            hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
            for row in range(batch):
                for col in range(seq_len):
                    pos = int(position_ids[row, col].item()) if position_ids is not None else col
                    target_idx = max(0, pos - prefix_len + 1)
                    token = target[min(target_idx, len(target) - 1)]
                    hidden[row, col, token] = 100.0
            cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
            key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
            value = torch.zeros_like(key)
            return [hidden, None], ((key, value),)

        adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)
        return adapter

    plain_tokens, _plain_logits, plain_stats = make_adapter().sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=_SecondTokenMissDrafter(target),
        max_decoding_steps=len(target),
        lookahead=2,
        reuse_full_blocks=True,
    )
    deferred_tokens, _deferred_logits, deferred_stats = make_adapter().sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=_SecondTokenMissDrafter(target),
        max_decoding_steps=len(target),
        lookahead=2,
        reuse_full_blocks=True,
        defer_correction_token=True,
    )

    assert plain_tokens.tolist()[0] == target
    assert deferred_tokens.tolist()[0] == target
    assert deferred_stats["target_forwards"] < plain_stats["target_forwards"]
    assert deferred_stats["fallback_forwards"] < plain_stats["fallback_forwards"]
    assert deferred_stats["pending_corrections"] > plain_stats["pending_corrections"]
    assert deferred_stats["defer_correction_token"] is True


def test_pi0fast_online_tree_anchor_verifies_future_after_first_token_miss() -> None:
    target = [10, 20, 30, 40]
    adapter = PI0FastTokenLogitAdapter(_FakePolicy())
    adapter._match_model_precision = lambda tensor: tensor
    prefix_len = 3

    def fake_forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position=None,
    ):
        batch, seq_len, vocab = inputs_embeds.shape
        hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
        for row in range(batch):
            for col in range(seq_len):
                pos = int(position_ids[row, col].item()) if position_ids is not None else col
                target_idx = max(0, pos - prefix_len + 1)
                token = target[min(target_idx, len(target) - 1)]
                hidden[row, col, token] = 100.0
        cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
        key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
        value = torch.zeros_like(key)
        return [hidden, None], ((key, value),)

    adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)

    token_ids, _logits, stats = adapter.sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=_AnchorMissDrafter(target),
        max_decoding_steps=len(target),
        lookahead=2,
        reuse_full_blocks=True,
        tree_width=2,
        tree_branch_width=2,
        tree_anchor_target_token=True,
    )

    assert token_ids.tolist()[0] == target
    assert stats["tree_first_token_misses"] == 1
    assert stats["tree_anchor_verifies"] == 1
    assert stats["tree_anchor_accepted_tokens"] >= 1
    assert stats["action_transition_histogram_accepted_tokens"] >= 1


def test_pi0fast_online_tree_anchor_continuation_verifies_after_target_token() -> None:
    target = [10, 20, 30, 40]
    adapter = PI0FastTokenLogitAdapter(_FakePolicy())
    adapter._match_model_precision = lambda tensor: tensor
    prefix_len = 3

    def fake_forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position=None,
    ):
        batch, seq_len, vocab = inputs_embeds.shape
        hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
        for row in range(batch):
            for col in range(seq_len):
                pos = int(position_ids[row, col].item()) if position_ids is not None else col
                target_idx = max(0, pos - prefix_len + 1)
                token = target[min(target_idx, len(target) - 1)]
                hidden[row, col, token] = 100.0
        cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
        key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
        value = torch.zeros_like(key)
        return [hidden, None], ((key, value),)

    adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)

    token_ids, _logits, stats = adapter.sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=_AnchorMissDrafter(target),
        max_decoding_steps=len(target),
        lookahead=2,
        reuse_full_blocks=True,
        tree_width=2,
        tree_branch_width=2,
        tree_anchor_target_continuation=True,
    )

    assert token_ids.tolist()[0] == target
    assert stats["tree_anchor_target_continuation"] is True
    assert stats["tree_first_token_misses"] == 0
    assert stats["tree_anchor_verifies"] > 0
    assert stats["tree_anchor_accepted_tokens"] >= 1
    assert stats["action_transition_histogram_accepted_tokens"] >= 1


def test_pi0fast_online_pattern_trend_regression_reports_source_stats() -> None:
    target = [10, 12, 14, 16]
    adapter = PI0FastTokenLogitAdapter(_FakePolicy())
    adapter._match_model_precision = lambda tensor: tensor
    prefix_len = 3

    def fake_forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position=None,
    ):
        batch, seq_len, vocab = inputs_embeds.shape
        hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
        for row in range(batch):
            for col in range(seq_len):
                pos = int(position_ids[row, col].item()) if position_ids is not None else col
                target_idx = max(0, pos - prefix_len + 1)
                token = target[min(target_idx, len(target) - 1)]
                hidden[row, col, token] = 100.0
        cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
        key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
        value = torch.zeros_like(key)
        return [hidden, None], ((key, value),)

    adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=128,
            enable_linear_action_extrapolation=False,
            enable_action_trend_regression=True,
            action_trend_history=4,
            action_trend_top_k=3,
            action_trend_max_abs=6,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_trend_regression",),
        )
    )

    token_ids, _logits, stats = adapter.sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=drafter,
        max_decoding_steps=len(target),
        lookahead=2,
        reuse_full_blocks=True,
    )

    assert token_ids.tolist()[0] == target
    assert stats["action_trend_regression_drafted_tokens"] > 0
    assert stats["action_trend_regression_accepted_tokens"] > 0
    assert stats["action_trend_regression_acceptance_rate"] > 0.0


def test_pi0fast_online_pattern_action_prefix_lookup_reports_source_stats() -> None:
    target = [5, 6, 42, 5, 6, 42]
    adapter = PI0FastTokenLogitAdapter(_FakePolicy())
    adapter._match_model_precision = lambda tensor: tensor
    prefix_len = 3

    def fake_forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position=None,
    ):
        batch, seq_len, vocab = inputs_embeds.shape
        hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
        for row in range(batch):
            for col in range(seq_len):
                pos = int(position_ids[row, col].item()) if position_ids is not None else col
                target_idx = max(0, pos - prefix_len + 1)
                token = target[min(target_idx, len(target) - 1)]
                hidden[row, col, token] = 100.0
        cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
        key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
        value = torch.zeros_like(key)
        return [hidden, None], ((key, value),)

    adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=1,
            vocab_size=128,
            enable_linear_action_extrapolation=False,
            enable_action_prefix_lookup=True,
            action_prefix_history_size=2,
            action_prefix_top_k=3,
            action_prefix_min_prefix=1,
            action_prefix_max_mismatches=0,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_prefix_lookup",),
        )
    )

    token_ids, _logits, stats = adapter.sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=drafter,
        max_decoding_steps=len(target),
        lookahead=1,
        reuse_full_blocks=True,
    )

    assert token_ids.tolist()[0] == target
    assert stats["action_prefix_lookup_drafted_tokens"] > 0
    assert stats["action_prefix_lookup_accepted_tokens"] > 0
    assert stats["action_prefix_lookup_acceptance_rate"] > 0.0


def test_pi0fast_online_pattern_action_repeat_vector_reports_source_stats() -> None:
    target = [10, 50, 10, 50, 10, 50]
    adapter = PI0FastTokenLogitAdapter(_FakePolicy())
    adapter._match_model_precision = lambda tensor: tensor
    prefix_len = 3

    def fake_forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position=None,
    ):
        batch, seq_len, vocab = inputs_embeds.shape
        hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
        for row in range(batch):
            for col in range(seq_len):
                pos = int(position_ids[row, col].item()) if position_ids is not None else col
                target_idx = max(0, pos - prefix_len + 1)
                token = target[min(target_idx, len(target) - 1)]
                hidden[row, col, token] = 100.0
        cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
        key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
        value = torch.zeros_like(key)
        return [hidden, None], ((key, value),)

    adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=4,
            vocab_size=128,
            enable_linear_action_extrapolation=False,
            enable_action_repeat_vector=True,
            action_repeat_min_repeats=2,
            action_repeat_max_delta=0,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_repeat_vector",),
        )
    )

    token_ids, _logits, stats = adapter.sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=drafter,
        max_decoding_steps=len(target),
        lookahead=4,
        reuse_full_blocks=True,
    )

    assert token_ids.tolist()[0] == target
    assert stats["action_repeat_vector_drafted_tokens"] > 0
    assert stats["action_repeat_vector_accepted_tokens"] > 0
    assert stats["action_repeat_vector_acceptance_rate"] > 0.0


def test_pi0fast_online_pattern_chunk_length_stop_reports_source_stats() -> None:
    target = [10, 20, 30, 99]
    adapter = PI0FastTokenLogitAdapter(_FakePolicy())
    adapter._match_model_precision = lambda tensor: tensor
    prefix_len = 3

    def fake_forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position=None,
    ):
        batch, seq_len, vocab = inputs_embeds.shape
        hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
        for row in range(batch):
            for col in range(seq_len):
                pos = int(position_ids[row, col].item()) if position_ids is not None else col
                target_idx = max(0, pos - prefix_len + 1)
                token = target[min(target_idx, len(target) - 1)]
                hidden[row, col, token] = 100.0
        cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
        key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
        value = torch.zeros_like(key)
        return [hidden, None], ((key, value),)

    adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            vocab_size=128,
            enable_linear_action_extrapolation=False,
            enable_chunk_length_stop=True,
            chunk_length_stop_history_size=2,
            chunk_length_stop_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
            source_priority=("chunk_length_stop",),
        )
    )
    drafter.observe([1, 2, 3, 99])

    token_ids, _logits, stats = adapter.sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=drafter,
        max_decoding_steps=len(target),
        lookahead=2,
        reuse_full_blocks=True,
    )

    assert token_ids.tolist()[0] == target
    assert stats["chunk_length_stop_drafted_tokens"] > 0
    assert stats["chunk_length_stop_accepted_tokens"] > 0
    assert stats["chunk_length_stop_acceptance_rate"] > 0.0


def test_pi0fast_online_pattern_global_position_mode_reports_source_stats() -> None:
    target = [12, 20, 32, 99]
    adapter = PI0FastTokenLogitAdapter(_FakePolicy())
    adapter._match_model_precision = lambda tensor: tensor
    prefix_len = 3

    def fake_forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        cache_position=None,
    ):
        batch, seq_len, vocab = inputs_embeds.shape
        hidden = torch.zeros((batch, seq_len, vocab), dtype=torch.float32)
        for row in range(batch):
            for col in range(seq_len):
                pos = int(position_ids[row, col].item()) if position_ids is not None else col
                target_idx = max(0, pos - prefix_len + 1)
                token = target[min(target_idx, len(target) - 1)]
                hidden[row, col, token] = 100.0
        cache_len = int(cache_position.max().item()) + 1 if cache_position is not None else seq_len
        key = torch.zeros((batch, 1, cache_len, 1), dtype=torch.float32)
        value = torch.zeros_like(key)
        return [hidden, None], ((key, value),)

    adapter._forward_prefix_language_model = MethodType(fake_forward, adapter)
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            vocab_size=128,
            enable_linear_action_extrapolation=False,
            enable_global_position_mode=True,
            global_position_history_size=4,
            global_position_top_k=2,
            global_position_min_count=2,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
            source_priority=("global_position_mode",),
        )
    )
    drafter.observe([10, 20, 30, 99])
    drafter.observe([11, 20, 31, 99])

    token_ids, _logits, stats = adapter.sample_actions_fast_ngram_speculative(
        images=torch.empty(1),
        img_masks=torch.empty(1),
        tokens=torch.zeros((1, 3), dtype=torch.long),
        masks=torch.ones((1, 3), dtype=torch.bool),
        drafter=drafter,
        max_decoding_steps=len(target),
        lookahead=2,
        reuse_full_blocks=True,
    )

    assert token_ids.tolist()[0] == target
    assert stats["global_position_mode_drafted_tokens"] > 0
    assert stats["global_position_mode_accepted_tokens"] > 0
    assert stats["global_position_mode_acceptance_rate"] > 0.0
