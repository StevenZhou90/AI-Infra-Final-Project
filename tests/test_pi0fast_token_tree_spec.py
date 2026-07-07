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
        )

    def eval(self) -> None:
        return None

    def _paligemma_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        mapping = {90: 0, 91: 1, 92: 2}
        return tokens.new_tensor([mapping.get(int(token.item()), 10_000) for token in tokens])


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
