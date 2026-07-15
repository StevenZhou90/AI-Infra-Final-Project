from __future__ import annotations

import torch
from types import SimpleNamespace

from serving.pi0fast_pattern_drafter import PatternDraftConfig, PatternFastTokenDrafter, parse_source_priority
from serving.pi0fast_pattern_drafter import (
    evaluate_pattern_drafter,
    simulate_exact_pattern_spec_decode,
    simulate_exact_pattern_tree_spec_decode,
)


def test_pattern_drafter_extrapolates_action_dimension_tokens() -> None:
    drafter = PatternFastTokenDrafter(PatternDraftConfig(action_dim=2, lookahead=4, vocab_size=100))

    assert drafter.draft([10, 50, 12, 50], lookahead=4) == [14, 50, 16, 50]


def test_pattern_drafter_extrapolates_second_order_action_tokens() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=4,
            vocab_size=100,
            enable_second_order_action_extrapolation=True,
            second_order_max_accel=8,
        )
    )

    assert drafter.draft([10, 50, 13, 51, 18, 53], lookahead=2) == [25, 56]


def test_pattern_drafter_uses_action_delta_histogram_candidates() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_delta_histogram=True,
            action_delta_history=4,
            action_delta_top_k=2,
            action_delta_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
        )
    )

    assert drafter.draft([10, 12, 14, 16, 17], lookahead=1) == [19]
    assert drafter.last_draft_sources() == ["action_delta_histogram"]

    drafts = drafter.draft_many([10, 12, 14, 16, 17], lookahead=1, max_candidates=4, branch_width=4)
    assert [19] in drafts
    assert [18] in drafts
    assert ["action_delta_histogram"] in drafter.last_many_sources()


def test_pattern_drafter_uses_action_delta_histogram_from_recent_chunks() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_delta_histogram=True,
            action_delta_history=4,
            action_delta_top_k=2,
            action_delta_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_delta_histogram",),
        )
    )
    drafter.observe([10, 14, 18])

    assert drafter.draft([20], lookahead=1) == [24]
    assert drafter.last_draft_sources() == ["action_delta_histogram"]
    assert drafter.draft_many([20], lookahead=1, max_candidates=2, branch_width=2) == [[24]]
    assert drafter.last_many_sources() == [["action_delta_histogram"]]


def test_pattern_drafter_action_delta_histogram_min_count_filters_singletons() -> None:
    loose = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_delta_histogram=True,
            action_delta_history=4,
            action_delta_top_k=2,
            action_delta_min_count=1,
            action_delta_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_delta_histogram",),
        )
    )
    strict = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_delta_histogram=True,
            action_delta_history=4,
            action_delta_top_k=2,
            action_delta_min_count=2,
            action_delta_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_delta_histogram",),
        )
    )
    loose.observe([10, 14])
    strict.observe([10, 14])

    assert loose.draft([20], lookahead=1) == [24]
    assert loose.last_draft_sources() == ["action_delta_histogram"]
    assert strict.draft([20], lookahead=1) == []
    assert strict.last_draft_sources() == []


def test_pattern_drafter_uses_action_delta_ngram_from_recent_chunks() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_delta_ngram=True,
            action_delta_ngram_history_size=2,
            action_delta_ngram_min_context=2,
            action_delta_ngram_max_context=3,
            action_delta_ngram_top_k=2,
            action_delta_ngram_min_count=1,
            action_delta_ngram_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_delta_ngram",),
        )
    )
    drafter.observe([10, 12, 15, 17])

    assert drafter.draft([20, 22, 25], lookahead=1) == [27]
    assert drafter.last_draft_sources() == ["action_delta_ngram"]
    assert drafter.draft_many([20, 22, 25], lookahead=1, max_candidates=2, branch_width=2) == [[27]]
    assert drafter.last_many_sources() == [["action_delta_ngram"]]


def test_pattern_drafter_uses_action_trend_regression_candidates() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=100,
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

    assert drafter.draft([10, 12, 14], lookahead=2) == [16, 18]
    assert drafter.last_draft_sources() == ["action_trend_regression", "action_trend_regression"]
    assert drafter.draft_many([10, 12, 14], lookahead=1, max_candidates=3, branch_width=3) == [
        [16],
        [15],
        [17],
    ]
    assert drafter.last_many_sources() == [
        ["action_trend_regression"],
        ["action_trend_regression"],
        ["action_trend_regression"],
    ]


def test_pattern_drafter_uses_action_prefix_lookup_candidates() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_prefix_lookup=True,
            action_prefix_history_size=2,
            action_prefix_top_k=3,
            action_prefix_min_prefix=2,
            action_prefix_max_mismatches=0,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_prefix_lookup",),
        )
    )

    assert drafter.draft([5, 6, 42, 5, 6], lookahead=1) == [42]
    assert drafter.last_draft_sources() == ["action_prefix_lookup"]
    assert drafter.draft_many([5, 6, 42, 5, 6], lookahead=1, max_candidates=3, branch_width=3) == [[42]]
    assert drafter.last_many_sources() == [["action_prefix_lookup"]]


def test_pattern_drafter_uses_action_vector_suffix_lookup_from_current_tokens() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=2,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_vector_suffix_lookup=True,
            action_vector_suffix_top_k=2,
            action_vector_suffix_min_prefix=2,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_vector_suffix_lookup",),
        )
    )

    assert drafter.draft([5, 6, 42, 5, 6], lookahead=1) == [42]
    assert drafter.last_draft_sources() == ["action_vector_suffix_lookup"]
    assert drafter.draft_many([5, 6, 42, 5, 6], lookahead=1, max_candidates=2, branch_width=2) == [[42]]
    assert drafter.last_many_sources() == [["action_vector_suffix_lookup"]]


def test_pattern_drafter_action_vector_suffix_lookup_uses_recent_chunks_and_min_count() -> None:
    loose = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_vector_suffix_lookup=True,
            action_vector_suffix_history_size=2,
            action_vector_suffix_min_prefix=2,
            action_vector_suffix_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_vector_suffix_lookup",),
        )
    )
    strict = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_vector_suffix_lookup=True,
            action_vector_suffix_history_size=2,
            action_vector_suffix_min_prefix=2,
            action_vector_suffix_min_count=2,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_vector_suffix_lookup",),
        )
    )
    loose.observe([7, 8, 55])
    strict.observe([7, 8, 55])

    assert loose.draft([7, 8], lookahead=1) == [55]
    assert loose.last_draft_sources() == ["action_vector_suffix_lookup"]
    assert strict.draft([7, 8], lookahead=1) == []
    assert strict.last_draft_sources() == []


def test_pattern_drafter_uses_chunk_position_delta_from_recent_chunks() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_chunk_position_delta=True,
            chunk_position_delta_history_size=2,
            chunk_position_delta_top_k=2,
            chunk_position_delta_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("chunk_position_delta",),
        )
    )
    drafter.observe([10, 14, 18])

    assert drafter.draft([20, 24], lookahead=1) == [28]
    assert drafter.last_draft_sources() == ["chunk_position_delta"]
    assert drafter.draft_many([20, 24], lookahead=1, max_candidates=2, branch_width=2) == [[28]]
    assert drafter.last_many_sources() == [["chunk_position_delta"]]


def test_pattern_drafter_ranks_repeated_chunk_position_delta_mode() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_chunk_position_delta=True,
            chunk_position_delta_history_size=3,
            chunk_position_delta_top_k=2,
            chunk_position_delta_min_count=2,
            chunk_position_delta_max_abs=8,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("chunk_position_delta",),
        )
    )
    drafter.observe([10, 12, 14])
    drafter.observe([10, 14, 18])
    drafter.observe([10, 14, 18])

    assert drafter.draft([20, 24], lookahead=1) == [28]
    assert drafter.last_draft_sources() == ["chunk_position_delta"]
    assert drafter.draft_many([20, 24], lookahead=1, max_candidates=2, branch_width=2) == [[28]]
    assert drafter.last_many_sources() == [["chunk_position_delta"]]


def test_pattern_drafter_chunk_position_delta_min_count_filters_singletons() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_chunk_position_delta=True,
            chunk_position_delta_history_size=3,
            chunk_position_delta_top_k=2,
            chunk_position_delta_min_count=2,
            chunk_position_delta_max_abs=8,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("chunk_position_delta",),
        )
    )
    drafter.observe([10, 14, 18])

    assert drafter.draft([20, 24], lookahead=1) == []
    assert drafter.last_draft_sources() == []
    assert drafter.draft_many([20, 24], lookahead=1, max_candidates=2, branch_width=2) == []
    assert drafter.last_many_sources() == []


def test_pattern_drafter_uses_chunk_delta_template_from_recent_chunks() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_chunk_delta_template=True,
            chunk_delta_template_history_size=2,
            chunk_delta_template_top_k=2,
            chunk_delta_template_min_prefix_deltas=1,
            chunk_delta_template_max_delta_mismatch=0,
            chunk_delta_template_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("chunk_delta_template",),
        )
    )
    drafter.observe([10, 14, 18])

    assert drafter.draft([20, 24], lookahead=1) == [28]
    assert drafter.last_draft_sources() == ["chunk_delta_template"]
    assert drafter.draft_many([20, 24], lookahead=1, max_candidates=2, branch_width=2) == [[28]]
    assert drafter.last_many_sources() == [["chunk_delta_template"]]


def test_pattern_drafter_uses_action_token_neighborhood_candidates() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_action_token_neighborhood=True,
            action_token_neighborhood_radius=1,
            action_token_neighborhood_top_k=3,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_token_neighborhood",),
        )
    )

    assert drafter.draft([10, 12], lookahead=1) == [14]
    assert drafter.last_draft_sources() == ["action_token_neighborhood"]
    assert drafter.draft_many([10, 12], lookahead=1, max_candidates=3, branch_width=3) == [[14], [13], [15]]
    assert drafter.last_many_sources() == [
        ["action_token_neighborhood"],
        ["action_token_neighborhood"],
        ["action_token_neighborhood"],
    ]


def test_pattern_drafter_expands_action_token_neighborhood_around_enabled_source_centers() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            previous_chunk_history_size=1,
            enable_action_token_neighborhood=True,
            action_token_neighborhood_radius=1,
            action_token_neighborhood_top_k=3,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_token_neighborhood",),
        )
    )
    drafter.observe([10, 12, 20])

    assert drafter.draft_many([10, 12], lookahead=1, max_candidates=3, branch_width=3) == [[14], [20], [13]]
    assert drafter.last_many_sources() == [
        ["action_token_neighborhood"],
        ["action_token_neighborhood"],
        ["action_token_neighborhood"],
    ]


def test_pattern_drafter_uses_action_transition_histogram_candidates() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_transition_histogram=True,
            action_transition_top_k=2,
            action_transition_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
        )
    )

    assert drafter.draft([1, 10, 2, 10, 1, 10], lookahead=1) == [2]
    assert drafter.last_draft_sources() == ["action_transition_histogram"]

    drafts = drafter.draft_many([1, 10, 2, 10, 1, 10], lookahead=1, max_candidates=4, branch_width=4)
    assert [2] in drafts
    assert ["action_transition_histogram"] in drafter.last_many_sources()


def test_pattern_drafter_uses_action_vector_transition_candidates() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=3,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_vector_transition=True,
            action_vector_transition_top_k=2,
            action_vector_transition_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_vector_transition",),
        )
    )

    assert drafter.draft([1, 2, 3, 4, 5, 6, 1, 2, 3], lookahead=3) == [4, 5, 6]
    assert drafter.last_draft_sources() == ["action_vector_transition"] * 3
    assert drafter.draft_many(
        [1, 2, 3, 4, 5, 6, 1, 2, 3],
        lookahead=1,
        max_candidates=2,
        branch_width=2,
    ) == [[4]]
    assert drafter.last_many_sources() == [["action_vector_transition"]]


def test_pattern_drafter_action_vector_transition_requires_prefix_match() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_vector_transition=True,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_vector_transition",),
        )
    )

    assert drafter.draft([1, 2, 3, 4, 5, 6, 1, 2, 3, 7], lookahead=1) == []
    assert drafter.last_draft_sources() == []


def test_pattern_drafter_action_vector_transition_min_count_filters_singletons() -> None:
    loose = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_vector_transition=True,
            action_vector_transition_min_count=1,
            action_vector_transition_history_size=2,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_vector_transition",),
        )
    )
    strict = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_vector_transition=True,
            action_vector_transition_min_count=2,
            action_vector_transition_history_size=2,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_vector_transition",),
        )
    )
    loose.observe([1, 2, 3, 4, 5, 6])
    strict.observe([1, 2, 3, 4, 5, 6])

    assert loose.draft([1, 2, 3], lookahead=1) == [4]
    assert loose.last_draft_sources() == ["action_vector_transition"]
    assert strict.draft([1, 2, 3], lookahead=1) == []
    assert strict.last_draft_sources() == []


def test_pattern_drafter_uses_action_context_tree_candidates() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_context_tree=True,
            action_context_tree_max_context=2,
            action_context_tree_top_k=2,
            action_context_tree_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
        )
    )

    assert drafter.draft([1, 50, 2, 50, 1, 50, 2, 50, 1, 50], lookahead=1) == [2]
    assert drafter.last_draft_sources() == ["action_context_tree"]

    drafts = drafter.draft_many(
        [1, 50, 2, 50, 1, 50, 2, 50, 1, 50],
        lookahead=1,
        max_candidates=4,
        branch_width=4,
    )
    assert [2] in drafts
    assert ["action_context_tree"] in drafter.last_many_sources()


def test_tree_speculation_can_adapt_tree_width_from_acceptance() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_context_tree=True,
            action_context_tree_max_context=2,
            action_context_tree_top_k=2,
            max_period=0,
            repeat_token_min_run=99,
        )
    )

    result = simulate_exact_pattern_tree_spec_decode(
        [1, 2, 1, 2, 1, 2, 1, 2],
        drafter,
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        dynamic_tree_width=True,
        min_tree_width=1,
        tree_width_growth=1,
        tree_width_shrink=1,
        reuse_full_blocks=True,
    )

    assert result.max_tree_width > result.min_tree_width
    assert result.mean_tree_width > 1.0
    assert result.accepted_tokens > 0


def test_pattern_drafter_second_order_accel_guard_falls_back_to_linear() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=2,
            vocab_size=100,
            enable_second_order_action_extrapolation=True,
            second_order_max_accel=4,
        )
    )

    assert drafter.draft([10, 50, 13, 50, 30, 50], lookahead=1) == [47]


def test_pattern_drafter_uses_periodic_tail_when_linear_disabled() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=5,
            max_period=3,
            min_period_repeats=2,
            enable_linear_action_extrapolation=False,
        )
    )

    assert drafter.draft([1, 2, 3, 1, 2, 3], lookahead=5) == [1, 2, 3, 1, 2]


def test_pattern_drafter_repeats_stable_token_and_stops_after_eos() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=3,
            repeat_token_min_run=3,
            enable_linear_action_extrapolation=False,
            max_period=0,
            stop_token_ids=(99,),
        )
    )

    assert drafter.draft([7, 7, 7], lookahead=3) == [7, 7, 7]
    assert drafter.draft([7, 7, 99], lookahead=3) == []


def test_pattern_drafter_draft_many_keeps_primary_chain_first() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=2,
            vocab_size=100,
            enable_second_order_action_extrapolation=True,
            second_order_max_accel=8,
        )
    )

    drafts = drafter.draft_many([10, 50, 13, 50, 18, 50], lookahead=2, max_candidates=4)

    assert drafts[0] == drafter.draft([10, 50, 13, 50, 18, 50], lookahead=2)
    assert len(drafts) > 1
    assert len({tuple(row) for row in drafts}) == len(drafts)


def test_pattern_drafter_can_use_previous_chunk_token_positions() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_previous_chunk_position=True,
            previous_chunk_history_size=2,
            enable_linear_action_extrapolation=False,
            max_period=0,
        )
    )
    drafter.observe([10, 20, 30, 99])

    assert drafter.draft([], lookahead=4) == [10, 20, 30, 99]
    assert drafter.last_draft_sources() == ["previous_chunk_position"] * 4

    drafter.observe([11, 21, 31, 99])
    drafts = drafter.draft_many([], lookahead=2, max_candidates=4, branch_width=4)
    sources = drafter.last_many_sources()

    assert drafts[0] == [11, 21]
    assert [10, 20] in drafts
    assert sources[0] == ["previous_chunk_position", "previous_chunk_position"]


def test_pattern_drafter_uses_chunk_prefix_retrieval_with_small_prefix_mismatch() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            enable_chunk_prefix_retrieval=True,
            chunk_prefix_history_size=2,
            chunk_prefix_min_matches=1,
            chunk_prefix_max_mismatches=1,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("chunk_prefix_retrieval",),
        )
    )
    drafter.observe([10, 20, 30, 40])

    assert drafter.draft([10, 21], lookahead=2) == [30, 40]
    assert drafter.last_draft_sources() == ["chunk_prefix_retrieval", "chunk_prefix_retrieval"]
    assert drafter.draft_many([10, 21], lookahead=2, max_candidates=2, branch_width=2) == [[30, 40]]
    assert drafter.last_many_sources() == [["chunk_prefix_retrieval", "chunk_prefix_retrieval"]]


def test_pattern_drafter_previous_chunk_prior_stops_after_prefix_divergence() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_previous_chunk_position=True,
            previous_chunk_history_size=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            stop_token_ids=(99,),
        )
    )
    drafter.observe([10, 20, 30, 99])

    assert drafter.draft([10], lookahead=2) == [20, 30]
    assert drafter.last_draft_sources() == ["previous_chunk_position", "previous_chunk_position"]
    assert drafter.draft([11], lookahead=2) == []
    assert drafter.last_draft_sources() == []


def test_pattern_drafter_previous_chunk_tree_candidates_are_prefix_compatible() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            enable_previous_chunk_position=True,
            previous_chunk_history_size=2,
            enable_linear_action_extrapolation=False,
            max_period=0,
        )
    )
    drafter.observe([10, 20, 30])
    drafter.observe([11, 21, 31])

    first_token_drafts = drafter.draft_many([], lookahead=1, max_candidates=4, branch_width=4)
    assert [11] in first_token_drafts
    assert [10] in first_token_drafts

    ten_prefix_drafts = drafter.draft_many([10], lookahead=1, max_candidates=4, branch_width=4)
    eleven_prefix_drafts = drafter.draft_many([11], lookahead=1, max_candidates=4, branch_width=4)

    assert [20] in ten_prefix_drafts
    assert [21] not in ten_prefix_drafts
    assert [21] in eleven_prefix_drafts
    assert [20] not in eleven_prefix_drafts


def test_pattern_drafter_uses_position_mode_histogram() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=3,
            enable_position_mode_histogram=True,
            position_mode_history_size=3,
            position_mode_top_k=2,
            position_mode_min_count=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
        )
    )
    drafter.observe([10, 20, 30, 99])
    drafter.observe([10, 21, 30, 99])
    drafter.observe([10, 20, 30, 99])

    assert drafter.draft([], lookahead=3) == [10, 20, 30]
    assert drafter.last_draft_sources() == ["position_mode_histogram"] * 3

    drafts = drafter.draft_many([10], lookahead=1, max_candidates=4, branch_width=4)
    assert [20] in drafts
    assert [21] in drafts
    assert ["position_mode_histogram"] in drafter.last_many_sources()


def test_pattern_drafter_uses_global_position_mode_without_prefix_match() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            enable_global_position_mode=True,
            global_position_history_size=4,
            global_position_top_k=2,
            global_position_min_count=2,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
        )
    )
    drafter.observe([10, 20, 30, 99])
    drafter.observe([11, 20, 31, 99])

    assert drafter.draft([77], lookahead=2) == [20]
    assert drafter.last_draft_sources() == ["global_position_mode"]

    drafts = drafter.draft_many([77], lookahead=1, max_candidates=4, branch_width=4)
    assert [20] in drafts
    assert ["global_position_mode"] in drafter.last_many_sources()


def test_pattern_drafter_uses_action_dimension_mode() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=3,
            enable_action_dimension_mode=True,
            action_dimension_mode_history_size=3,
            action_dimension_mode_top_k=2,
            action_dimension_mode_min_count=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
        )
    )
    drafter.observe([10, 20, 10, 21, 99])
    drafter.observe([11, 20, 11, 22, 99])
    drafter.observe([10, 20, 10, 23, 99])

    assert drafter.draft([], lookahead=3) == [10, 20, 10]
    assert drafter.last_draft_sources() == ["action_dimension_mode"] * 3

    drafts = drafter.draft_many([], lookahead=1, max_candidates=4, branch_width=4)
    assert [10] in drafts
    assert [11] in drafts
    assert ["action_dimension_mode"] in drafter.last_many_sources()


def test_pattern_drafter_uses_hold_action_token() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=4,
            enable_hold_action_token=True,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
        )
    )

    assert drafter.draft([10, 50], lookahead=4) == [10, 50, 10, 50]
    assert drafter.last_draft_sources() == ["hold_action_token"] * 4

    drafts = drafter.draft_many([10, 50], lookahead=1, max_candidates=4, branch_width=4)
    assert [10] in drafts
    assert ["hold_action_token"] in drafter.last_many_sources()


def test_pattern_drafter_uses_action_repeat_vector() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=4,
            enable_action_repeat_vector=True,
            action_repeat_min_repeats=2,
            action_repeat_max_delta=0,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_repeat_vector",),
        )
    )

    assert drafter.draft([10, 50, 10, 50], lookahead=4) == [10, 50, 10, 50]
    assert drafter.last_draft_sources() == ["action_repeat_vector"] * 4
    assert drafter.draft([10, 50, 10, 50, 11], lookahead=1) == []

    drafts = drafter.draft_many([10, 50, 10, 50], lookahead=1, max_candidates=4, branch_width=4)
    assert [10] in drafts
    assert ["action_repeat_vector"] in drafter.last_many_sources()


def test_pattern_drafter_uses_chunk_length_stop() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            enable_chunk_length_stop=True,
            chunk_length_stop_history_size=4,
            chunk_length_stop_min_count=2,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
            source_priority=("chunk_length_stop",),
        )
    )
    drafter.observe([1, 2, 3, 99])
    drafter.observe([4, 5, 6, 99])

    assert drafter.draft([10, 11], lookahead=1) == []
    assert drafter.draft([10, 11, 12], lookahead=2) == [99]
    assert drafter.last_draft_sources() == ["chunk_length_stop"]

    drafts = drafter.draft_many([10, 11, 12], lookahead=1, max_candidates=4, branch_width=4)
    assert [99] in drafts
    assert ["chunk_length_stop"] in drafter.last_many_sources()


def test_pattern_drafter_uses_ngram_continuation_from_verified_prefix() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_ngram_continuation=True,
            ngram_min_context=2,
            ngram_max_context=4,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
        )
    )

    assert drafter.draft([1, 2, 3, 4, 1, 2], lookahead=4) == [3, 4, 1, 2]
    assert drafter.last_draft_sources() == ["ngram_continuation"] * 4


def test_pattern_drafter_uses_ngram_continuation_from_previous_chunks() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=3,
            enable_ngram_continuation=True,
            ngram_min_context=2,
            ngram_max_context=4,
            ngram_history_size=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
        )
    )
    drafter.observe([10, 11, 12, 13, 14])

    assert drafter.draft([20, 11, 12], lookahead=2) == [13, 14]
    assert drafter.last_draft_sources() == ["ngram_continuation", "ngram_continuation"]


def test_pattern_drafter_source_priority_changes_primary_proposal() -> None:
    base = dict(
        lookahead=1,
        action_dim=2,
        enable_ngram_continuation=True,
        ngram_min_context=2,
        ngram_max_context=4,
        enable_linear_action_extrapolation=True,
        max_period=0,
        vocab_size=100,
    )
    lookup_first = PatternFastTokenDrafter(
        PatternDraftConfig(**base, source_priority=parse_source_priority("lookup_first"))
    )
    smooth_first = PatternFastTokenDrafter(
        PatternDraftConfig(**base, source_priority=parse_source_priority("smooth_first"))
    )

    assert lookup_first.draft([1, 2, 3, 4, 1, 2], lookahead=1) == [3]
    assert lookup_first.last_draft_sources() == ["ngram_continuation"]
    assert smooth_first.draft([1, 2, 3, 4, 1, 2], lookahead=1) == [0]
    assert smooth_first.last_draft_sources() == ["linear_action_extrapolation"]


def test_pattern_drafter_prefers_source_agreement_when_sources_match() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=1,
            action_dim=1,
            enable_ngram_continuation=True,
            ngram_min_context=2,
            ngram_max_context=4,
            max_period=0,
            repeat_token_min_run=99,
            min_source_agreement=2,
            vocab_size=100,
            source_priority=parse_source_priority("lookup_first"),
        )
    )

    assert drafter.draft([1, 2, 3, 1, 2], lookahead=1) == [3]
    assert drafter.last_draft_sources() == ["source_agreement"]

    drafts = drafter.draft_many([1, 2, 3, 1, 2], lookahead=1, max_candidates=4, branch_width=4)
    assert drafts[0] == [3]
    assert drafter.last_many_sources()[0] == ["source_agreement"]


def test_pattern_drafter_source_cooldown_skips_rejected_source_once() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=1,
            enable_ngram_continuation=True,
            ngram_min_context=1,
            ngram_max_context=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            enable_source_cooldown=True,
            source_cooldown_after=1,
            source_cooldown_steps=1,
            source_priority=parse_source_priority("lookup_first"),
        )
    )

    assert drafter.draft([1, 2, 1], lookahead=1) == [2]
    assert drafter.last_draft_sources() == ["ngram_continuation"]

    drafter.record_source_feedback(drafter.last_draft_sources(), accepted=0)
    assert drafter.draft([1, 2, 1], lookahead=1) == []
    assert drafter.source_cooldown_stats()["source_cooldown_events"] == 1
    assert drafter.source_cooldown_stats()["source_cooldown_skipped_sources"] > 0

    drafter.record_source_miss()
    assert drafter.draft([1, 2, 1], lookahead=1) == [2]


def test_pattern_drafter_source_acceptance_bias_promotes_verified_source() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=1,
            action_dim=2,
            enable_ngram_continuation=True,
            ngram_min_context=2,
            ngram_max_context=4,
            enable_linear_action_extrapolation=True,
            max_period=0,
            repeat_token_min_run=99,
            enable_source_acceptance_bias=True,
            source_acceptance_bias_min_observations=1,
            source_priority=parse_source_priority("lookup_first"),
            vocab_size=100,
        )
    )

    assert drafter.draft([1, 2, 3, 4, 1, 2], lookahead=1) == [3]
    assert drafter.last_draft_sources() == ["ngram_continuation"]

    drafter.record_source_feedback(["ngram_continuation"], accepted=0)
    drafter.record_source_feedback(["linear_action_extrapolation"], accepted=1)

    assert drafter.draft([1, 2, 3, 4, 1, 2], lookahead=1) == [0]
    assert drafter.last_draft_sources() == ["linear_action_extrapolation"]
    stats = drafter.source_cooldown_stats()
    assert stats["source_acceptance_bias_events"] == 2
    assert stats["source_acceptance_bias_reorders"] > 0


def test_pattern_drafter_source_agreement_falls_back_when_sources_disagree() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=1,
            action_dim=2,
            enable_ngram_continuation=True,
            ngram_min_context=2,
            ngram_max_context=4,
            max_period=0,
            repeat_token_min_run=99,
            min_source_agreement=2,
            vocab_size=100,
            source_priority=parse_source_priority("lookup_first"),
        )
    )

    assert drafter.draft([1, 2, 3, 4, 1, 2], lookahead=1) == [3]
    assert drafter.last_draft_sources() == ["ngram_continuation"]


def test_pattern_drafter_reset_history_clears_previous_chunk_prior() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            enable_previous_chunk_position=True,
            enable_linear_action_extrapolation=False,
            max_period=0,
        )
    )
    drafter.observe([10, 20])
    drafter.reset_history()

    assert drafter.draft([], lookahead=2) == []


def test_pattern_spec_decode_simulation_is_exact_and_reduces_forwards() -> None:
    drafter = PatternFastTokenDrafter(PatternDraftConfig(action_dim=2, lookahead=4, vocab_size=100))

    result = simulate_exact_pattern_spec_decode([10, 50, 12, 50, 14, 50, 16, 50], drafter, lookahead=4)

    assert result.tokens == 8
    assert result.accepted_tokens > 0
    assert result.target_forwards < result.tokens
    assert result.target_forward_reduction > 1.0


def test_pattern_tree_simulation_uses_alternate_robot_candidates() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=2,
            vocab_size=100,
            enable_second_order_action_extrapolation=True,
            second_order_max_accel=8,
        )
    )
    tokens = [10, 50, 13, 50, 18, 50, 23, 50]

    chain = simulate_exact_pattern_spec_decode(tokens, drafter, lookahead=2, reuse_full_blocks=True)
    tree = simulate_exact_pattern_tree_spec_decode(
        tokens,
        drafter,
        lookahead=2,
        tree_width=4,
        tree_branch_width=4,
        reuse_full_blocks=True,
    )

    assert tree.tree_verifies > 0
    assert tree.tree_candidates > tree.tree_verifies
    assert tree.accepted_tokens >= chain.accepted_tokens
    assert tree.target_forwards <= chain.target_forwards


def test_pattern_tree_simulation_reports_first_token_misses() -> None:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_second_order_action_extrapolation=False,
            enable_action_transition_histogram=True,
            action_transition_top_k=3,
            action_transition_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_transition_histogram",),
        )
    )

    result = simulate_exact_pattern_tree_spec_decode(
        [1, 2, 1, 2, 1, 3, 1, 3],
        drafter,
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        reuse_full_blocks=True,
    )

    assert result.tree_first_token_checks > 0
    assert result.tree_first_token_misses > 0
    assert result.tree_first_token_miss_rate > 0.0


def test_pattern_tree_anchor_target_token_verifies_future_after_first_token_miss() -> None:
    config = PatternDraftConfig(
        action_dim=1,
        lookahead=2,
        vocab_size=100,
        enable_linear_action_extrapolation=False,
        enable_second_order_action_extrapolation=False,
        enable_action_transition_histogram=True,
        action_transition_top_k=3,
        action_transition_min_count=1,
        max_period=0,
        repeat_token_min_run=99,
        source_priority=("action_transition_histogram",),
    )
    tokens = [3, 4, 1, 2, 1, 3, 4]

    plain = simulate_exact_pattern_tree_spec_decode(
        tokens,
        PatternFastTokenDrafter(config),
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        reuse_full_blocks=True,
    )
    anchored = simulate_exact_pattern_tree_spec_decode(
        tokens,
        PatternFastTokenDrafter(config),
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        tree_anchor_target_token=True,
        reuse_full_blocks=True,
    )

    assert anchored.tree_first_token_misses > 0
    assert anchored.tree_anchor_verifies > 0
    assert anchored.tree_anchor_accepted_tokens > 0
    assert anchored.target_forwards < plain.target_forwards


def test_pattern_tree_anchor_target_continuation_verifies_after_target_token() -> None:
    config = PatternDraftConfig(
        action_dim=1,
        lookahead=2,
        vocab_size=100,
        enable_linear_action_extrapolation=False,
        enable_ngram_continuation=True,
        ngram_min_context=1,
        ngram_max_context=3,
        max_period=0,
        repeat_token_min_run=99,
        source_priority=("ngram_continuation",),
    )
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3]

    plain = simulate_exact_pattern_tree_spec_decode(
        tokens,
        PatternFastTokenDrafter(config),
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        reuse_full_blocks=True,
    )
    continuation = simulate_exact_pattern_tree_spec_decode(
        tokens,
        PatternFastTokenDrafter(config),
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        tree_anchor_target_continuation=True,
        reuse_full_blocks=True,
    )

    assert continuation.tree_anchor_verifies > 0
    assert continuation.tree_anchor_accepted_tokens > 0
    assert continuation.tree_first_token_misses == 0
    assert continuation.target_forwards < plain.target_forwards


def test_pattern_spec_decode_can_reuse_full_verified_blocks() -> None:
    drafter = PatternFastTokenDrafter(PatternDraftConfig(action_dim=2, lookahead=4, vocab_size=100))
    tokens = [10, 50, 12, 50, 14, 50, 16, 50, 18, 50]

    plain = simulate_exact_pattern_spec_decode(tokens, drafter, lookahead=2, reuse_full_blocks=False)
    reused = simulate_exact_pattern_spec_decode(tokens, drafter, lookahead=2, reuse_full_blocks=True)

    assert reused.full_block_reuses > 0
    assert reused.target_forwards <= plain.target_forwards
    assert reused.target_forward_reduction >= plain.target_forward_reduction


def test_pattern_spec_decode_can_emit_standard_bonus_token() -> None:
    drafter = PatternFastTokenDrafter(PatternDraftConfig(action_dim=2, lookahead=4, vocab_size=100))
    tokens = [10, 50, 12, 50, 14, 50, 16, 50, 18, 50]

    reused = simulate_exact_pattern_spec_decode(tokens, drafter, lookahead=2, reuse_full_blocks=True)
    bonus = simulate_exact_pattern_spec_decode(
        tokens,
        drafter,
        lookahead=2,
        reuse_full_blocks=True,
        emit_bonus_token=True,
    )

    assert bonus.bonus_tokens > 0
    assert bonus.target_forwards <= reused.target_forwards
    assert bonus.target_forward_reduction >= reused.target_forward_reduction


def test_pattern_spec_decode_can_defer_rejected_correction_token() -> None:
    tokens = [10, 20, 30, 10, 20, 31, 10, 20, 32]
    config = PatternDraftConfig(action_dim=2, lookahead=4, vocab_size=100)

    plain = simulate_exact_pattern_spec_decode(
        tokens,
        PatternFastTokenDrafter(config),
        lookahead=2,
        reuse_full_blocks=True,
    )
    deferred = simulate_exact_pattern_spec_decode(
        tokens,
        PatternFastTokenDrafter(config),
        lookahead=2,
        reuse_full_blocks=True,
        defer_correction_token=True,
    )

    assert deferred.rejected_blocks == plain.rejected_blocks
    assert deferred.deferred_correction_tokens == plain.rejected_blocks
    assert deferred.target_forwards < plain.target_forwards
    assert deferred.target_forward_reduction > plain.target_forward_reduction


def test_pattern_spec_decode_dynamic_lookahead_records_schedule() -> None:
    drafter = PatternFastTokenDrafter(PatternDraftConfig(action_dim=2, lookahead=4, vocab_size=100))

    result = simulate_exact_pattern_spec_decode(
        [10, 50, 12, 50, 14, 50, 30, 50, 32, 50],
        drafter,
        lookahead=4,
        dynamic_lookahead=True,
        min_lookahead=1,
        lookahead_growth=1,
        lookahead_shrink=2,
    )

    assert result.min_lookahead >= 1
    assert result.max_lookahead <= 4
    assert result.mean_lookahead > 0


def test_evaluate_pattern_drafter_reports_trace_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 50, 12, 50, 14, 50, 16, 50]),
            task_id=3,
            seed=7,
            trace_id="smooth",
        )
    ]
    drafter = PatternFastTokenDrafter(PatternDraftConfig(action_dim=2, lookahead=4, vocab_size=100))

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4)

    assert metrics["traces"] == 1
    assert metrics["tokens"] == 8
    assert metrics["target_forward_reduction"] > 1.0
    assert metrics["modeled_speedup"] == metrics["target_forward_reduction"]
    assert metrics["full_block_reuses"] == 0
    assert metrics["tree_width"] == 1
    assert metrics["tree_candidates"] == 0
    assert metrics["per_task"]["3"]["traces"] == 1


def test_evaluate_pattern_drafter_uses_suite_scoped_task_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 50, 12, 50, 14, 50, 16, 50]),
            task="libero_goal",
            task_id=3,
            seed=7,
            trace_id="goal",
        ),
        SimpleNamespace(
            token_ids=torch.tensor([1, 5, 9]),
            task="libero_spatial",
            task_id=3,
            seed=7,
            trace_id="spatial",
        ),
    ]
    drafter = PatternFastTokenDrafter(PatternDraftConfig(action_dim=2, lookahead=4, vocab_size=100))

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4)

    assert metrics["task_count"] == 2
    assert metrics["per_trace"][0]["task_key"] == "libero_goal:3"
    assert metrics["per_trace"][1]["task_key"] == "libero_spatial:3"
    assert "3" not in metrics["per_task"]
    assert metrics["per_task"]["libero_goal:3"]["target_forward_reduction"] > 1.0
    assert metrics["per_task"]["libero_spatial:3"]["target_forward_reduction"] == 1.0
    assert metrics["min_task_target_forward_reduction"] == 1.0


def test_evaluate_pattern_drafter_reports_source_agreement_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3]),
            task_id=3,
            seed=7,
            trace_id="agreement",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=3,
            action_dim=1,
            enable_ngram_continuation=True,
            ngram_min_context=2,
            ngram_max_context=4,
            max_period=0,
            repeat_token_min_run=99,
            min_source_agreement=2,
            vocab_size=100,
            source_priority=parse_source_priority("lookup_first"),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=3, reuse_full_blocks=True)

    assert metrics["source_agreement_drafted_tokens"] > 0
    assert metrics["source_agreement_accepted_tokens"] > 0
    assert metrics["source_agreement_acceptance_rate"] > 0.0
    assert metrics["per_task"]["3"]["source_agreement_accepted_tokens"] > 0
    assert metrics["min_task_source_agreement_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_delta_histogram_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([1, 3, 5, 7, 9, 11]),
            task_id=3,
            seed=7,
            trace_id="delta",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            action_dim=1,
            enable_linear_action_extrapolation=False,
            enable_action_delta_histogram=True,
            action_delta_history=4,
            action_delta_top_k=2,
            action_delta_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            vocab_size=100,
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["action_delta_histogram_drafted_tokens"] > 0
    assert metrics["action_delta_histogram_accepted_tokens"] > 0
    assert metrics["action_delta_histogram_acceptance_rate"] > 0.0
    assert metrics["per_task"]["3"]["action_delta_histogram_accepted_tokens"] > 0
    assert metrics["min_task_action_delta_histogram_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_trend_regression_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 12, 14, 16, 18, 20]),
            task_id=3,
            seed=1,
            trace_id="trend",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=100,
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

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["action_trend_regression_drafted_tokens"] > 0
    assert metrics["action_trend_regression_accepted_tokens"] > 0
    assert metrics["action_trend_regression_acceptance_rate"] > 0.0
    assert metrics["per_trace"][0]["action_trend_regression_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["action_trend_regression_accepted_tokens"] > 0
    assert metrics["min_task_action_trend_regression_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_prefix_lookup_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([5, 6, 42, 5, 6, 42]),
            task_id=3,
            seed=1,
            trace_id="action-prefix",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=3,
            lookahead=1,
            vocab_size=100,
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

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=1, reuse_full_blocks=True)

    assert metrics["action_prefix_lookup_drafted_tokens"] > 0
    assert metrics["action_prefix_lookup_accepted_tokens"] > 0
    assert metrics["action_prefix_lookup_acceptance_rate"] > 0.0
    assert metrics["per_trace"][0]["action_prefix_lookup_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["action_prefix_lookup_accepted_tokens"] > 0
    assert metrics["min_task_action_prefix_lookup_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_repeat_vector_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 50, 10, 50, 10, 50, 99]),
            task_id=3,
            seed=1,
            trace_id="action-repeat-vector",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=4,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_repeat_vector=True,
            action_repeat_min_repeats=2,
            action_repeat_max_delta=0,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
            source_priority=("action_repeat_vector",),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["action_repeat_vector_drafted_tokens"] > 0
    assert metrics["action_repeat_vector_accepted_tokens"] > 0
    assert metrics["action_repeat_vector_acceptance_rate"] > 0.0
    assert metrics["per_trace"][0]["action_repeat_vector_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["action_repeat_vector_accepted_tokens"] > 0
    assert metrics["min_task_action_repeat_vector_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_chunk_length_stop_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 30, 99]),
            task_id=3,
            seed=1,
            trace_id="chunk-length-stop-seed",
        ),
        SimpleNamespace(
            token_ids=torch.tensor([11, 21, 31, 99]),
            task_id=3,
            seed=1,
            trace_id="chunk-length-stop-repeat",
        ),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            vocab_size=100,
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

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["chunk_length_stop_drafted_tokens"] > 0
    assert metrics["chunk_length_stop_accepted_tokens"] > 0
    assert metrics["chunk_length_stop_acceptance_rate"] > 0.0
    assert metrics["per_trace"][1]["chunk_length_stop_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["chunk_length_stop_accepted_tokens"] > 0
    assert metrics["min_task_chunk_length_stop_accepted_tokens"] > 0


def test_action_delta_histogram_accepts_from_recent_chunk_delta() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 14, 18]), task_id=3, seed=7, trace_id="seed"),
        SimpleNamespace(token_ids=torch.tensor([20, 24]), task_id=3, seed=7, trace_id="repeat-delta"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_delta_histogram=True,
            action_delta_history=4,
            action_delta_top_k=2,
            action_delta_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_delta_histogram",),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=1, reuse_full_blocks=True)

    assert metrics["per_trace"][1]["action_delta_histogram_accepted_tokens"] > 0
    assert metrics["action_delta_histogram_accepted_tokens"] > 0
    assert metrics["min_task_action_delta_histogram_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_transition_histogram_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([1, 10, 2, 10, 1, 10, 2, 10, 1, 10, 2, 10]),
            task_id=3,
            seed=7,
            trace_id="transition",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            action_dim=2,
            enable_linear_action_extrapolation=False,
            enable_action_transition_histogram=True,
            action_transition_top_k=2,
            action_transition_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
            vocab_size=100,
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["action_transition_histogram_drafted_tokens"] > 0
    assert metrics["action_transition_histogram_accepted_tokens"] > 0
    assert metrics["action_transition_histogram_acceptance_rate"] > 0.0
    assert metrics["per_task"]["3"]["action_transition_histogram_accepted_tokens"] > 0
    assert metrics["min_task_action_transition_histogram_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_context_tree_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([1, 50, 2, 50, 1, 50, 2, 50, 1, 50, 2, 50]),
            task_id=3,
            seed=7,
            trace_id="context-tree",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            action_dim=2,
            enable_linear_action_extrapolation=False,
            enable_action_context_tree=True,
            action_context_tree_max_context=2,
            action_context_tree_top_k=2,
            action_context_tree_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
            vocab_size=100,
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["action_context_tree_drafted_tokens"] > 0
    assert metrics["action_context_tree_accepted_tokens"] > 0
    assert metrics["action_context_tree_acceptance_rate"] > 0.0
    assert metrics["per_task"]["3"]["action_context_tree_accepted_tokens"] > 0
    assert metrics["min_task_action_context_tree_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_source_cooldown_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([1, 2, 1, 3, 1, 3]),
            task_id=3,
            seed=7,
            trace_id="cooldown",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=1,
            enable_ngram_continuation=True,
            ngram_min_context=1,
            ngram_max_context=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            enable_source_cooldown=True,
            source_cooldown_after=1,
            source_cooldown_steps=1,
            source_priority=parse_source_priority("lookup_first"),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=1)

    assert metrics["source_cooldown_events"] > 0
    assert metrics["source_cooldown_skipped_sources"] > 0
    assert metrics["per_task"]["3"]["source_cooldown_events"] > 0
    assert metrics["min_task_source_cooldown_events"] > 0


def test_evaluate_pattern_drafter_reports_source_acceptance_bias_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([1, 2, 3, 4, 1, 2, 0, 0]),
            task_id=3,
            seed=7,
            trace_id="acceptance-bias",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=1,
            action_dim=2,
            enable_ngram_continuation=True,
            ngram_min_context=2,
            ngram_max_context=4,
            enable_linear_action_extrapolation=True,
            max_period=0,
            repeat_token_min_run=99,
            enable_source_acceptance_bias=True,
            source_acceptance_bias_min_observations=1,
            source_priority=parse_source_priority("lookup_first"),
            vocab_size=100,
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=1)

    assert metrics["source_acceptance_bias_events"] > 0
    assert metrics["source_acceptance_bias_reorders"] > 0
    assert metrics["per_task"]["3"]["source_acceptance_bias_events"] > 0
    assert metrics["min_task_source_acceptance_bias_reorders"] > 0


def test_evaluate_pattern_drafter_uses_previous_chunk_history_within_task() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 30, 99]),
            task_id=3,
            seed=7,
            trace_id="first",
        ),
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 31, 99]),
            task_id=3,
            seed=7,
            trace_id="second",
        ),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_previous_chunk_position=True,
            previous_chunk_history_size=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            stop_token_ids=(99,),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["traces"] == 2
    assert metrics["accepted_tokens"] == 2
    assert metrics["per_trace"][1]["accepted_tokens"] >= 2
    assert metrics["previous_chunk_position_drafted_tokens"] >= 3
    assert metrics["previous_chunk_position_accepted_tokens"] >= 2
    assert metrics["previous_chunk_position_acceptance_rate"] > 0.0
    assert metrics["per_trace"][1]["previous_chunk_position_accepted_tokens"] >= 2
    assert metrics["per_task"]["3"]["previous_chunk_position_accepted_tokens"] >= 2
    assert metrics["target_forward_reduction"] > 1.0


def test_evaluate_pattern_drafter_reports_chunk_prefix_retrieval_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 20, 30, 40]), task_id=3, seed=7, trace_id="base"),
        SimpleNamespace(token_ids=torch.tensor([10, 21, 30, 40]), task_id=3, seed=7, trace_id="near-repeat"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=2,
            enable_chunk_prefix_retrieval=True,
            chunk_prefix_history_size=2,
            chunk_prefix_min_matches=1,
            chunk_prefix_max_mismatches=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("chunk_prefix_retrieval",),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["chunk_prefix_retrieval_drafted_tokens"] > 0
    assert metrics["chunk_prefix_retrieval_accepted_tokens"] > 0
    assert metrics["chunk_prefix_retrieval_acceptance_rate"] > 0.0
    assert metrics["per_trace"][1]["chunk_prefix_retrieval_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["chunk_prefix_retrieval_accepted_tokens"] > 0
    assert metrics["min_task_chunk_prefix_retrieval_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_token_neighborhood_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 12, 13, 15]), task_id=3, seed=7, trace_id="near-linear")
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            enable_action_token_neighborhood=True,
            action_token_neighborhood_radius=1,
            action_token_neighborhood_top_k=3,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_token_neighborhood",),
        )
    )

    metrics = evaluate_pattern_drafter(
        drafter,
        traces,
        lookahead=1,
        tree_width=3,
        tree_branch_width=3,
        reuse_full_blocks=True,
    )

    assert metrics["action_token_neighborhood_drafted_tokens"] > 0
    assert metrics["action_token_neighborhood_accepted_tokens"] > 0
    assert metrics["action_token_neighborhood_acceptance_rate"] > 0.0
    assert metrics["per_trace"][0]["action_token_neighborhood_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["action_token_neighborhood_accepted_tokens"] > 0
    assert metrics["min_task_action_token_neighborhood_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_chunk_position_delta_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 14, 18]), task_id=3, seed=7, trace_id="seed"),
        SimpleNamespace(token_ids=torch.tensor([20, 24, 28]), task_id=3, seed=7, trace_id="repeat-delta"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_chunk_position_delta=True,
            chunk_position_delta_history_size=2,
            chunk_position_delta_top_k=2,
            chunk_position_delta_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("chunk_position_delta",),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["chunk_position_delta_drafted_tokens"] > 0
    assert metrics["chunk_position_delta_accepted_tokens"] > 0
    assert metrics["chunk_position_delta_acceptance_rate"] > 0.0
    assert metrics["per_trace"][1]["chunk_position_delta_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["chunk_position_delta_accepted_tokens"] > 0
    assert metrics["min_task_chunk_position_delta_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_delta_ngram_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 12, 15, 17]), task_id=3, seed=7, trace_id="seed"),
        SimpleNamespace(token_ids=torch.tensor([20, 22, 25, 27]), task_id=3, seed=7, trace_id="repeat-deltas"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_action_delta_ngram=True,
            action_delta_ngram_history_size=2,
            action_delta_ngram_min_context=1,
            action_delta_ngram_max_context=3,
            action_delta_ngram_top_k=2,
            action_delta_ngram_min_count=1,
            action_delta_ngram_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_delta_ngram",),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["action_delta_ngram_drafted_tokens"] > 0
    assert metrics["action_delta_ngram_accepted_tokens"] > 0
    assert metrics["action_delta_ngram_acceptance_rate"] > 0.0
    assert metrics["per_trace"][1]["action_delta_ngram_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["action_delta_ngram_accepted_tokens"] > 0
    assert metrics["min_task_action_delta_ngram_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_chunk_delta_template_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 14, 18]), task_id=3, seed=7, trace_id="seed"),
        SimpleNamespace(token_ids=torch.tensor([20, 24, 28]), task_id=3, seed=7, trace_id="repeat-template"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_chunk_delta_template=True,
            chunk_delta_template_history_size=2,
            chunk_delta_template_top_k=2,
            chunk_delta_template_min_prefix_deltas=1,
            chunk_delta_template_max_delta_mismatch=0,
            chunk_delta_template_max_abs=4,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("chunk_delta_template",),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, reuse_full_blocks=True)

    assert metrics["chunk_delta_template_drafted_tokens"] > 0
    assert metrics["chunk_delta_template_accepted_tokens"] > 0
    assert metrics["chunk_delta_template_acceptance_rate"] > 0.0
    assert metrics["per_trace"][1]["chunk_delta_template_accepted_tokens"] > 0
    assert metrics["per_task"]["3"]["chunk_delta_template_accepted_tokens"] > 0
    assert metrics["min_task_chunk_delta_template_accepted_tokens"] > 0


def test_action_token_neighborhood_accepts_from_previous_chunk_center() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 12, 20]), task_id=3, seed=7, trace_id="seed"),
        SimpleNamespace(token_ids=torch.tensor([10, 12, 20]), task_id=3, seed=7, trace_id="repeat"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=1,
            vocab_size=100,
            previous_chunk_history_size=1,
            enable_action_token_neighborhood=True,
            action_token_neighborhood_radius=1,
            action_token_neighborhood_top_k=3,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_token_neighborhood",),
        )
    )

    metrics = evaluate_pattern_drafter(
        drafter,
        traces,
        lookahead=1,
        tree_width=3,
        tree_branch_width=3,
        reuse_full_blocks=True,
    )

    assert metrics["per_trace"][1]["action_token_neighborhood_accepted_tokens"] > 0
    assert metrics["action_token_neighborhood_accepted_tokens"] > 0
    assert metrics["min_task_action_token_neighborhood_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_position_mode_histogram_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 20, 30, 99]), task_id=3, seed=7, trace_id="a"),
        SimpleNamespace(token_ids=torch.tensor([10, 20, 31, 99]), task_id=3, seed=7, trace_id="b"),
        SimpleNamespace(token_ids=torch.tensor([10, 20, 30, 99]), task_id=3, seed=7, trace_id="c"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_position_mode_histogram=True,
            position_mode_history_size=3,
            position_mode_top_k=2,
            position_mode_min_count=2,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["position_mode_histogram_drafted_tokens"] > 0
    assert metrics["position_mode_histogram_accepted_tokens"] > 0
    assert metrics["position_mode_histogram_acceptance_rate"] > 0.0
    assert metrics["per_task"]["3"]["position_mode_histogram_accepted_tokens"] > 0
    assert metrics["min_task_position_mode_histogram_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_global_position_mode_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 20, 30, 99]), task_id=3, seed=7, trace_id="a"),
        SimpleNamespace(token_ids=torch.tensor([11, 20, 31, 99]), task_id=3, seed=7, trace_id="b"),
        SimpleNamespace(token_ids=torch.tensor([12, 20, 32, 99]), task_id=3, seed=7, trace_id="c"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_global_position_mode=True,
            global_position_history_size=3,
            global_position_top_k=2,
            global_position_min_count=2,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
            source_priority=("global_position_mode",),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["global_position_mode_drafted_tokens"] > 0
    assert metrics["global_position_mode_accepted_tokens"] > 0
    assert metrics["global_position_mode_acceptance_rate"] > 0.0
    assert metrics["per_task"]["3"]["global_position_mode_accepted_tokens"] > 0
    assert metrics["min_task_global_position_mode_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_action_dimension_mode_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 50, 10, 50, 10, 50, 99]), task_id=3, seed=7, trace_id="a"),
        SimpleNamespace(token_ids=torch.tensor([11, 51, 11, 51, 11, 51, 99]), task_id=3, seed=7, trace_id="b"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=4,
            enable_action_dimension_mode=True,
            action_dimension_mode_history_size=1,
            action_dimension_mode_top_k=2,
            action_dimension_mode_min_count=2,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["action_dimension_mode_drafted_tokens"] > 0
    assert metrics["action_dimension_mode_accepted_tokens"] > 0
    assert metrics["action_dimension_mode_acceptance_rate"] > 0.0
    assert metrics["per_task"]["3"]["action_dimension_mode_accepted_tokens"] > 0
    assert metrics["min_task_action_dimension_mode_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_hold_action_token_metrics() -> None:
    traces = [
        SimpleNamespace(token_ids=torch.tensor([10, 50, 10, 50, 10, 50, 99]), task_id=3, seed=7, trace_id="a"),
        SimpleNamespace(token_ids=torch.tensor([11, 51, 11, 51, 11, 51, 99]), task_id=3, seed=8, trace_id="b"),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=4,
            enable_hold_action_token=True,
            enable_linear_action_extrapolation=False,
            max_period=0,
            repeat_token_min_run=99,
            stop_token_ids=(99,),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["hold_action_token_drafted_tokens"] > 0
    assert metrics["hold_action_token_accepted_tokens"] > 0
    assert metrics["hold_action_token_acceptance_rate"] > 0.0
    assert metrics["per_task"]["3"]["hold_action_token_accepted_tokens"] > 0
    assert metrics["min_task_hold_action_token_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_resets_previous_chunk_history_on_seed_change() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 30, 99]),
            task_id=3,
            seed=7,
            trace_id="first",
        ),
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 31, 99]),
            task_id=3,
            seed=8,
            trace_id="new-seed",
        ),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_previous_chunk_position=True,
            previous_chunk_history_size=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            stop_token_ids=(99,),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["history_reset"] == "task_seed"
    assert metrics["history_resets"] == 1
    assert metrics["per_trace"][1]["accepted_tokens"] == 0
    assert metrics["per_trace"][1]["previous_chunk_position_drafted_tokens"] == 0


def test_evaluate_pattern_drafter_resets_previous_chunk_history_on_suite_change() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 30, 99]),
            task="libero_goal",
            task_id=3,
            seed=7,
            trace_id="first",
        ),
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 31, 99]),
            task="libero_spatial",
            task_id=3,
            seed=7,
            trace_id="same-id-seed",
        ),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_previous_chunk_position=True,
            previous_chunk_history_size=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            stop_token_ids=(99,),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["history_resets"] == 1
    assert metrics["per_trace"][1]["accepted_tokens"] == 0
    assert metrics["per_trace"][1]["previous_chunk_position_drafted_tokens"] == 0
    assert metrics["per_task"]["libero_spatial:3"]["previous_chunk_position_accepted_tokens"] == 0


def test_evaluate_pattern_drafter_resets_previous_chunk_history_on_task_change() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 30, 99]),
            task_id=3,
            seed=7,
            trace_id="first",
        ),
        SimpleNamespace(
            token_ids=torch.tensor([10, 20, 31, 99]),
            task_id=4,
            seed=8,
            trace_id="other-task",
        ),
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=4,
            enable_previous_chunk_position=True,
            previous_chunk_history_size=1,
            enable_linear_action_extrapolation=False,
            max_period=0,
            stop_token_ids=(99,),
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=4, reuse_full_blocks=True)

    assert metrics["per_trace"][1]["accepted_tokens"] == 0
    assert metrics["per_trace"][1]["previous_chunk_position_drafted_tokens"] == 0


def test_evaluate_pattern_drafter_reports_tree_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([10, 50, 13, 50, 18, 50, 23, 50]),
            task_id=3,
            seed=7,
            trace_id="smooth",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=2,
            lookahead=2,
            vocab_size=100,
            enable_second_order_action_extrapolation=True,
            second_order_max_accel=8,
        )
    )

    metrics = evaluate_pattern_drafter(drafter, traces, lookahead=2, tree_width=4, tree_branch_width=4)

    assert metrics["tree_width"] == 4
    assert metrics["tree_candidates"] > 0
    assert metrics["mean_tree_candidates"] > 1.0
    assert metrics["per_task"]["3"]["tree_verifies"] > 0


def test_evaluate_pattern_drafter_reports_tree_first_token_misses() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([1, 2, 1, 2, 1, 3, 1, 3]),
            task_id=9,
            seed=1,
            trace_id="first-token-miss",
        )
    ]
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            action_dim=1,
            lookahead=2,
            vocab_size=100,
            enable_linear_action_extrapolation=False,
            enable_second_order_action_extrapolation=False,
            enable_action_transition_histogram=True,
            action_transition_top_k=3,
            action_transition_min_count=1,
            max_period=0,
            repeat_token_min_run=99,
            source_priority=("action_transition_histogram",),
        )
    )

    metrics = evaluate_pattern_drafter(
        drafter,
        traces,
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        reuse_full_blocks=True,
    )

    assert metrics["tree_first_token_checks"] > 0
    assert metrics["tree_first_token_misses"] > 0
    assert metrics["tree_first_token_miss_rate"] > 0.0
    assert metrics["per_task"]["9"]["tree_first_token_misses"] > 0


def test_evaluate_pattern_drafter_reports_tree_anchor_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([3, 4, 1, 2, 1, 3, 4]),
            task_id=9,
            seed=1,
            trace_id="anchor",
        )
    ]
    config = PatternDraftConfig(
        action_dim=1,
        lookahead=2,
        vocab_size=100,
        enable_linear_action_extrapolation=False,
        enable_second_order_action_extrapolation=False,
        enable_action_transition_histogram=True,
        action_transition_top_k=3,
        action_transition_min_count=1,
        max_period=0,
        repeat_token_min_run=99,
        source_priority=("action_transition_histogram",),
    )

    plain = evaluate_pattern_drafter(
        PatternFastTokenDrafter(config),
        traces,
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        reuse_full_blocks=True,
    )
    anchored = evaluate_pattern_drafter(
        PatternFastTokenDrafter(config),
        traces,
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        tree_anchor_target_token=True,
        reuse_full_blocks=True,
    )

    assert anchored["tree_anchor_verifies"] > 0
    assert anchored["tree_anchor_accepted_tokens"] > 0
    assert anchored["target_forwards"] < plain["target_forwards"]
    assert anchored["tree_anchor_target_token"] is True
    assert anchored["per_task"]["9"]["tree_anchor_accepted_tokens"] > 0


def test_evaluate_pattern_drafter_reports_tree_anchor_continuation_metrics() -> None:
    traces = [
        SimpleNamespace(
            token_ids=torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3]),
            task_id=9,
            seed=1,
            trace_id="anchor-continuation",
        )
    ]
    config = PatternDraftConfig(
        action_dim=1,
        lookahead=2,
        vocab_size=100,
        enable_linear_action_extrapolation=False,
        enable_ngram_continuation=True,
        ngram_min_context=1,
        ngram_max_context=3,
        max_period=0,
        repeat_token_min_run=99,
        source_priority=("ngram_continuation",),
    )

    metrics = evaluate_pattern_drafter(
        PatternFastTokenDrafter(config),
        traces,
        lookahead=2,
        tree_width=3,
        tree_branch_width=3,
        tree_anchor_target_continuation=True,
        reuse_full_blocks=True,
    )

    assert metrics["tree_anchor_target_continuation"] is True
    assert metrics["tree_anchor_verifies"] > 0
    assert metrics["tree_anchor_accepted_tokens"] > 0
    assert metrics["tree_first_token_misses"] == 0
    assert metrics["per_task"]["9"]["tree_anchor_accepted_tokens"] > 0
