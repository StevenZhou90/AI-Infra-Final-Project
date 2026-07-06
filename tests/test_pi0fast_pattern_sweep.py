from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.sweep_pi0fast_pattern_offline import (
    parse_bool_modes,
    parse_int_csv,
    parse_optional_int_csv,
    parse_source_priority_modes,
    run_pattern_sweep,
)
from serving.pi0fast_pattern_drafter import parse_source_priority


def _trace(tokens: list[int], task_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=torch.tensor(tokens),
        task_id=task_id,
        seed=task_id,
        trace_id=f"trace-{task_id}",
    )


def _tiny_sweep(records: list[SimpleNamespace], **overrides):
    params = {
        "lookaheads": [2],
        "action_dims": [2],
        "max_periods": [0],
        "min_period_repeats": [2],
        "repeat_token_min_runs": [99],
        "linear_action_extrapolations": [False],
        "second_order_action_extrapolations": [False],
        "second_order_max_accels": [8],
        "action_transition_histogram_values": [False],
        "action_transition_history_sizes": [4],
        "action_transition_top_ks": [3],
        "action_transition_min_counts": [1],
        "action_delta_histogram_values": [False],
        "action_delta_histories": [8],
        "action_delta_top_ks": [3],
        "action_delta_max_abs_values": [12],
        "previous_chunk_position_values": [False],
        "previous_chunk_history_sizes": [1],
        "position_mode_histogram_values": [False],
        "position_mode_history_sizes": [4],
        "position_mode_top_ks": [3],
        "position_mode_min_counts": [2],
        "action_dimension_mode_values": [False],
        "action_dimension_mode_history_sizes": [4],
        "action_dimension_mode_top_ks": [3],
        "action_dimension_mode_min_counts": [2],
        "hold_action_token_values": [False],
        "ngram_continuation_values": [False],
        "ngram_min_contexts": [3],
        "ngram_max_contexts": [16],
        "ngram_history_sizes": [4],
        "min_source_agreements": [1],
        "source_priorities": [parse_source_priority("default")],
        "reuse_full_blocks_values": [True],
        "emit_bonus_token_values": [False],
        "dynamic_lookahead_values": [False],
        "min_lookaheads": [1],
        "lookahead_growths": [1],
        "lookahead_shrinks": [4],
        "history_reset": "task_seed",
        "tree_widths": [1],
        "tree_branch_widths": [4],
        "vocab_size": 100,
        "stop_token_ids": (),
        "target_forward_ms": 1.0,
        "draft_token_ms": 0.0,
        "top_k": 8,
    }
    params.update(overrides)
    return run_pattern_sweep(records, **params)


def test_sweep_parsers() -> None:
    assert parse_int_csv("4,8, 12") == [4, 8, 12]
    assert parse_bool_modes("true") == [True]
    assert parse_bool_modes("false") == [False]
    assert parse_bool_modes("both") == [True, False]
    assert parse_optional_int_csv("4, none,8") == [4, None, 8]
    lookup_first = parse_source_priority_modes("lookup_first")[0]
    assert lookup_first[0] == "chunk_length_stop"
    assert lookup_first.index("ngram_continuation") < lookup_first.index("linear_action_extrapolation")


def test_pattern_sweep_can_cap_enabled_source_count() -> None:
    summary = _tiny_sweep(
        [_trace([10, 12, 14, 16], task_id=0)],
        second_order_action_extrapolations=[False, True],
        action_transition_histogram_values=[False, True],
        action_delta_histogram_values=[False, True],
        max_enabled_sources=1,
    )

    assert summary["sweep_count"] == 4
    assert summary["max_enabled_sources"] == 1
    assert summary["skipped_source_budget_configs"] == 4
    assert summary["evaluated_enabled_source_count_histogram"] == {"0": 1, "1": 3}
    assert summary["evaluated_source_counts"]["second_order_action_extrapolation"] == 1
    assert summary["evaluated_source_counts"]["action_transition_histogram"] == 1
    assert summary["evaluated_source_counts"]["action_delta_histogram"] == 1
    assert "second_order_action_extrapolation" in summary["evaluated_source_coverage"]
    assert "action_transition_histogram" in summary["evaluated_source_coverage"]
    assert "action_delta_histogram" in summary["evaluated_source_coverage"]
    assert all(row["config"]["enabled_source_count"] <= 1 for row in summary["top"])


def test_pattern_sweep_can_cap_evaluated_configs() -> None:
    summary = _tiny_sweep(
        [_trace([10, 12, 14, 16], task_id=0)],
        second_order_action_extrapolations=[False, True],
        action_transition_histogram_values=[False, True],
        max_configs=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["max_configs"] == 2
    assert summary["sweep_truncated"] is True


def test_pattern_sweep_reports_second_order_action_extrapolation_usage() -> None:
    summary = _tiny_sweep(
        [_trace([0, 1, 3, 6, 10], task_id=0)],
        action_dims=[1],
        second_order_action_extrapolations=[True],
        linear_action_extrapolations=[False],
    )

    row = summary["top"][0]
    assert row["config"]["second_order_action_extrapolation"] is True
    assert row["second_order_action_extrapolation_drafted_tokens"] >= 1
    assert row["second_order_action_extrapolation_accepted_tokens"] >= 1
    assert row["min_task_second_order_action_extrapolation_accepted_tokens"] >= 1


def test_pattern_sweep_ranks_best_settings() -> None:
    records = [
        _trace([10, 50, 12, 50, 14, 50, 16, 50], task_id=0),
        _trace([20, 40, 22, 40, 24, 40, 26, 40], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2, 4],
        action_dims=[2],
        max_periods=[4],
        min_period_repeats=[2],
        repeat_token_min_runs=[3],
        linear_action_extrapolations=[True, False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[4, 8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True, False],
        emit_bonus_token_values=[True, False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=3,
    )

    assert summary["sweep_count"] == 16
    assert summary["top_k"] == 3
    assert summary["best"]["modeled_speedup"] >= summary["top"][-1]["modeled_speedup"]
    assert summary["best"]["config"]["linear_action_extrapolation"] is True
    assert summary["best"]["config"]["second_order_action_extrapolation"] is False
    assert summary["best"]["config"]["tree_width"] == 1
    assert "reuse_full_blocks" in summary["best"]["config"]


def test_pattern_sweep_can_enable_action_token_neighborhood() -> None:
    records = [_trace([10, 12, 13, 15], task_id=0)]

    summary = run_pattern_sweep(
        records,
        lookaheads=[1],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[("action_token_neighborhood",)],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[3],
        tree_branch_widths=[3],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=1,
        action_token_neighborhood_values=[True],
        action_token_neighborhood_radii=[1],
        action_token_neighborhood_top_ks=[3],
    )

    assert summary["sweep_count"] == 1
    assert summary["best"]["config"]["action_token_neighborhood"] is True
    assert summary["best"]["action_token_neighborhood_accepted_tokens"] > 0


def test_pattern_sweep_can_enable_action_vector_transition_prior() -> None:
    summary = _tiny_sweep(
        [_trace([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], task_id=0)],
        lookaheads=[3],
        action_dims=[3],
        action_vector_transition_values=[True],
        action_vector_transition_history_sizes=[4],
        action_vector_transition_top_ks=[2],
        action_vector_transition_min_counts=[1],
        action_vector_transition_max_prev_delta_values=[0],
        action_vector_transition_max_prefix_delta_values=[0],
        source_priorities=[parse_source_priority("action_vector_transition")],
    )

    assert summary["sweep_count"] == 1
    assert summary["best"]["config"]["action_vector_transition"] is True
    assert summary["best"]["action_vector_transition_drafted_tokens"] > 0
    assert summary["best"]["action_vector_transition_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_vector_transition_accepted_tokens"] > 0
    assert summary["evaluated_source_counts"]["action_vector_transition"] == 1
    assert "action_vector_transition" in summary["evaluated_source_coverage"]


def test_pattern_sweep_reports_heldout_metrics_for_top_rows() -> None:
    train = [
        _trace([10, 50, 12, 50, 14, 50, 16, 50], task_id=0),
        _trace([20, 40, 22, 40, 24, 40, 26, 40], task_id=1),
    ]
    heldout = [_trace([30, 60, 32, 60, 34, 60, 36, 60], task_id=2)]

    summary = run_pattern_sweep(
        train,
        lookaheads=[2],
        action_dims=[2],
        max_periods=[4],
        min_period_repeats=[2],
        repeat_token_min_runs=[3],
        linear_action_extrapolations=[True],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[True],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=1,
        heldout_records=heldout,
    )

    assert summary["best"]["heldout"]["traces"] == 1
    assert summary["best"]["heldout_traces"] == 1
    assert summary["best"]["heldout_task_count"] == 1
    assert summary["best"]["heldout_target_forward_reduction"] > 1.0


def test_pattern_sweep_can_rank_offline_tree_candidates() -> None:
    records = [
        _trace([10, 50, 13, 50, 18, 50, 23, 50], task_id=0),
        _trace([20, 40, 23, 40, 28, 40, 33, 40], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[2],
        max_periods=[4],
        min_period_repeats=[2],
        repeat_token_min_runs=[3],
        linear_action_extrapolations=[True],
        second_order_action_extrapolations=[True],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1, 4],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert any(row["config"]["tree_width"] == 4 for row in summary["top"])
    tree_row = next(row for row in summary["top"] if row["config"]["tree_width"] == 4)
    assert tree_row["tree_candidates"] > 0


def test_pattern_sweep_can_include_dynamic_tree_width_configs() -> None:
    records = [
        _trace([1, 2, 1, 2, 1, 2, 1, 2], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1, 3],
        tree_branch_widths=[3],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=4,
        action_context_tree_values=[True],
        action_context_tree_history_sizes=[4],
        action_context_tree_max_contexts=[2],
        action_context_tree_top_ks=[2],
        action_context_tree_min_counts=[1],
        dynamic_tree_width_values=[False, True],
        min_tree_widths=[1],
        tree_width_growths=[1],
        tree_width_shrinks=[1],
    )

    configs = [row["config"] for row in summary["top"]]
    assert summary["sweep_count"] == 3
    assert any(config["tree_width"] == 3 and config["dynamic_tree_width"] for config in configs)
    dynamic_row = next(row for row in summary["top"] if row["config"]["dynamic_tree_width"])
    assert dynamic_row["mean_tree_width"] > 1.0


def test_pattern_sweep_can_include_tree_anchor_target_token_configs() -> None:
    records = [
        _trace([3, 4, 1, 2, 1, 3, 4], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[True],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1, 3],
        tree_branch_widths=[3],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=3,
        tree_anchor_target_token_values=[False, True],
    )

    configs = [row["config"] for row in summary["top"]]
    assert summary["sweep_count"] == 3
    assert any(config["tree_width"] == 3 and config["tree_anchor_target_token"] for config in configs)
    anchor_row = next(row for row in summary["top"] if row["config"]["tree_anchor_target_token"])
    assert anchor_row["tree_anchor_verifies"] > 0
    assert anchor_row["tree_anchor_accepted_tokens"] > 0
    assert anchor_row["min_task_tree_anchor_accepted_tokens"] > 0


def test_pattern_sweep_can_include_tree_anchor_target_continuation_configs() -> None:
    records = [
        _trace([1, 2, 3, 1, 2, 3, 1, 2, 3], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[True],
        ngram_min_contexts=[1],
        ngram_max_contexts=[3],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("ngram_continuation")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1, 3],
        tree_branch_widths=[3],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=4,
        tree_anchor_target_token_values=[False, True],
        tree_anchor_target_continuation_values=[False, True],
    )

    configs = [row["config"] for row in summary["top"]]
    assert summary["sweep_count"] == 4
    assert any(
        config["tree_width"] == 3 and config["tree_anchor_target_continuation"]
        for config in configs
    )
    assert not any(
        config["tree_anchor_target_token"] and config["tree_anchor_target_continuation"]
        for config in configs
    )
    continuation_row = next(row for row in summary["top"] if row["config"]["tree_anchor_target_continuation"])
    assert continuation_row["tree_anchor_verifies"] > 0
    assert continuation_row["tree_anchor_accepted_tokens"] > 0
    assert continuation_row["min_task_tree_anchor_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_previous_chunk_position_prior() -> None:
    records = [
        _trace([10, 20, 30, 99], task_id=0),
        _trace([10, 20, 31, 99], task_id=0),
        _trace([10, 20, 32, 99], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[4],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[3],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[True, False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(99,),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["previous_chunk_position"] is True
    assert summary["best"]["accepted_tokens"] > 0


def test_pattern_sweep_can_rank_position_mode_histogram_prior() -> None:
    records = [
        _trace([10, 20, 30, 99], task_id=0),
        _trace([10, 20, 31, 99], task_id=0),
        _trace([10, 20, 30, 99], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[4],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[True, False],
        position_mode_history_sizes=[3],
        position_mode_top_ks=[2],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(99,),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["best"]["config"]["position_mode_histogram"] is True
    assert summary["best"]["position_mode_histogram_accepted_tokens"] > 0
    assert summary["best"]["min_task_position_mode_histogram_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_global_position_mode_prior() -> None:
    records = [
        _trace([10, 20, 30, 99], task_id=0),
        _trace([11, 20, 31, 99], task_id=0),
        _trace([12, 20, 32, 99], task_id=0),
    ]

    summary = _tiny_sweep(
        records,
        lookaheads=[4],
        global_position_mode_values=[True, False],
        global_position_history_sizes=[3],
        global_position_top_ks=[2],
        global_position_min_counts=[2],
        stop_token_ids=(99,),
        top_k=2,
    )

    assert summary["best"]["config"]["global_position_mode"] is True
    assert summary["best"]["global_position_mode_accepted_tokens"] > 0
    assert summary["best"]["min_task_global_position_mode_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_action_vector_suffix_lookup_prior() -> None:
    records = [
        _trace([5, 6, 42, 5, 6, 42, 99], task_id=0),
        _trace([7, 8, 43, 7, 8, 43, 99], task_id=1),
    ]

    summary = _tiny_sweep(
        records,
        lookaheads=[2],
        action_dims=[3],
        action_vector_suffix_lookup_values=[True, False],
        action_vector_suffix_history_sizes=[2],
        action_vector_suffix_top_ks=[2],
        action_vector_suffix_min_prefixes=[2],
        action_vector_suffix_min_counts=[1],
        action_vector_suffix_max_prefix_delta_values=[0],
        stop_token_ids=(99,),
        top_k=2,
    )

    assert summary["best"]["config"]["action_vector_suffix_lookup"] is True
    assert summary["best"]["action_vector_suffix_lookup_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_vector_suffix_lookup_accepted_tokens"] > 0
    assert summary["evaluated_source_counts"]["action_vector_suffix_lookup"] == 1
    assert "action_vector_suffix_lookup" in summary["evaluated_source_coverage"]


def test_pattern_sweep_can_rank_action_dimension_mode_prior() -> None:
    records = [
        _trace([10, 50, 10, 50, 10, 50, 99], task_id=0),
        _trace([11, 51, 11, 51, 11, 51, 99], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[4],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[True, False],
        action_dimension_mode_history_sizes=[1],
        action_dimension_mode_top_ks=[2],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(99,),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["best"]["config"]["action_dimension_mode"] is True
    assert summary["best"]["action_dimension_mode_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_dimension_mode_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_hold_action_token_prior() -> None:
    records = [
        _trace([10, 50, 10, 50, 10, 50, 99], task_id=0),
        _trace([11, 51, 11, 51, 11, 51, 99], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[4],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[True, False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(99,),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["best"]["config"]["hold_action_token"] is True
    assert summary["best"]["hold_action_token_accepted_tokens"] > 0
    assert summary["best"]["min_task_hold_action_token_accepted_tokens"] > 0


def test_pattern_sweep_can_include_source_cooldown_configs() -> None:
    records = [_trace([1, 2, 1, 3, 1, 3], task_id=0)]

    summary = run_pattern_sweep(
        records,
        lookaheads=[1],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[True],
        ngram_min_contexts=[1],
        ngram_max_contexts=[1],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("lookup_first")],
        reuse_full_blocks_values=[False],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
        source_cooldown_values=[False, True],
        source_cooldown_afters=[1],
        source_cooldown_steps_values=[1],
    )

    assert summary["sweep_count"] == 2
    assert {row["config"]["source_cooldown"] for row in summary["top"]} == {False, True}
    cooldown_row = next(row for row in summary["top"] if row["config"]["source_cooldown"])
    assert "source_cooldown_events" in cooldown_row


def test_pattern_sweep_can_include_source_acceptance_bias_configs() -> None:
    records = [_trace([1, 2, 3, 4, 1, 2, 0, 0], task_id=0)]

    summary = run_pattern_sweep(
        records,
        lookaheads=[1],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[True],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[True],
        ngram_min_contexts=[2],
        ngram_max_contexts=[4],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("lookup_first")],
        reuse_full_blocks_values=[False],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
        source_acceptance_bias_values=[False, True],
        source_acceptance_bias_history_sizes=[8],
        source_acceptance_bias_min_observations_values=[1],
    )

    assert summary["sweep_count"] == 2
    assert {row["config"]["source_acceptance_bias"] for row in summary["top"]} == {False, True}
    bias_row = next(row for row in summary["top"] if row["config"]["source_acceptance_bias"])
    assert bias_row["source_acceptance_bias_events"] > 0
    assert bias_row["source_acceptance_bias_reorders"] > 0


def test_pattern_sweep_can_rank_ngram_continuation_prior() -> None:
    records = [
        _trace([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], task_id=0),
        _trace([5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[3],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[True, False],
        ngram_min_contexts=[2],
        ngram_max_contexts=[4],
        ngram_history_sizes=[1],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["ngram_continuation"] is True
    assert summary["best"]["ngram_continuation_accepted_tokens"] > 0
    assert summary["best"]["min_task_ngram_continuation_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_action_delta_histogram_prior() -> None:
    records = [
        _trace([1, 3, 5, 7, 9, 11], task_id=0),
        _trace([10, 13, 16, 19, 22, 25], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[True, False],
        action_delta_histories=[4],
        action_delta_top_ks=[2],
        action_delta_min_counts=[2],
        action_delta_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("smooth_first")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["action_delta_histogram"] is True
    assert summary["best"]["config"]["action_delta_min_count"] == 2
    assert summary["best"]["action_delta_histogram_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_delta_histogram_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_action_trend_regression_prior() -> None:
    records = [
        _trace([10, 12, 14, 16, 18, 20], task_id=0),
        _trace([30, 33, 36, 39, 42, 45], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[4],
        action_delta_top_ks=[2],
        action_delta_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("action_trend_regression")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
        action_trend_regression_values=[True, False],
        action_trend_histories=[4],
        action_trend_top_ks=[3],
        action_trend_max_abs_values=[6],
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["action_trend_regression"] is True
    assert summary["best"]["action_trend_regression_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_trend_regression_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_action_prefix_lookup_prior() -> None:
    records = [
        _trace([5, 6, 42, 5, 6, 42], task_id=0),
        _trace([7, 8, 50, 7, 8, 50], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[1],
        action_dims=[3],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[4],
        action_delta_top_ks=[2],
        action_delta_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("action_prefix_lookup")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
        action_prefix_lookup_values=[True, False],
        action_prefix_history_sizes=[2],
        action_prefix_top_ks=[3],
        action_prefix_min_prefixes=[1],
        action_prefix_max_mismatches_values=[0],
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["action_prefix_lookup"] is True
    assert summary["best"]["action_prefix_lookup_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_prefix_lookup_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_action_repeat_vector_prior() -> None:
    records = [
        _trace([10, 50, 10, 50, 10, 50], task_id=0),
        _trace([20, 60, 20, 60, 20, 60], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[4],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[4],
        action_delta_top_ks=[2],
        action_delta_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("action_repeat_vector")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
        action_repeat_vector_values=[True, False],
        action_repeat_min_repeats_values=[2],
        action_repeat_max_delta_values=[0],
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["action_repeat_vector"] is True
    assert summary["best"]["action_repeat_vector_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_repeat_vector_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_chunk_length_stop_prior() -> None:
    records = [
        _trace([10, 20, 30, 99], task_id=0),
        _trace([11, 21, 31, 99], task_id=0),
        _trace([40, 50, 60, 99], task_id=1),
        _trace([41, 51, 61, 99], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[4],
        action_delta_top_ks=[2],
        action_delta_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("chunk_length_stop")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(99,),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
        chunk_length_stop_values=[True, False],
        chunk_length_stop_history_sizes=[2],
        chunk_length_stop_min_counts=[1],
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["chunk_length_stop"] is True
    assert summary["best"]["chunk_length_stop_accepted_tokens"] > 0
    assert summary["best"]["min_task_chunk_length_stop_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_action_delta_histogram_from_recent_chunks() -> None:
    records = [
        _trace([10, 14, 18], task_id=0),
        _trace([20, 24], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[1],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[True, False],
        action_delta_histories=[4],
        action_delta_top_ks=[2],
        action_delta_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("smooth_first")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["action_delta_histogram"] is True
    assert summary["best"]["action_delta_histogram_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_delta_histogram_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_action_delta_ngram_from_recent_chunks() -> None:
    records = [
        _trace([10, 12, 15, 17], task_id=0),
        _trace([20, 22, 25, 27], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        action_delta_ngram_values=[True, False],
        action_delta_ngram_history_sizes=[2],
        action_delta_ngram_min_contexts=[1],
        action_delta_ngram_max_contexts=[3],
        action_delta_ngram_top_ks=[2],
        action_delta_ngram_min_counts=[1],
        action_delta_ngram_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("smooth_first")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["action_delta_ngram"] is True
    assert summary["best"]["action_delta_ngram_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_delta_ngram_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_chunk_position_delta_from_recent_chunks() -> None:
    records = [
        _trace([10, 14, 18], task_id=0),
        _trace([20, 24, 28], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        chunk_position_delta_values=[True, False],
        chunk_position_delta_history_sizes=[2],
        chunk_position_delta_top_ks=[2],
        chunk_position_delta_min_counts=[1],
        chunk_position_delta_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("smooth_first")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["chunk_position_delta"] is True
    assert summary["best"]["config"]["chunk_position_delta_min_count"] == 1
    assert summary["best"]["chunk_position_delta_accepted_tokens"] > 0
    assert summary["best"]["min_task_chunk_position_delta_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_chunk_delta_template_from_recent_chunks() -> None:
    records = [
        _trace([10, 14, 18], task_id=0),
        _trace([20, 24, 28], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        chunk_delta_template_values=[True, False],
        chunk_delta_template_history_sizes=[2],
        chunk_delta_template_top_ks=[2],
        chunk_delta_template_min_prefix_deltas_values=[1],
        chunk_delta_template_max_delta_mismatch_values=[0],
        chunk_delta_template_max_abs_values=[4],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("smooth_first")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["chunk_delta_template"] is True
    assert summary["best"]["chunk_delta_template_accepted_tokens"] > 0
    assert summary["best"]["min_task_chunk_delta_template_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_action_transition_histogram_prior() -> None:
    records = [
        _trace([1, 10, 2, 10, 1, 10, 2, 10, 1, 10, 2, 10], task_id=0),
        _trace([5, 20, 6, 20, 5, 20, 6, 20, 5, 20, 6, 20], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[True, False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[2],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("smooth_first")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    assert summary["best"]["config"]["action_transition_histogram"] is True
    assert summary["best"]["action_transition_histogram_accepted_tokens"] > 0
    assert summary["best"]["min_task_action_transition_histogram_accepted_tokens"] > 0


def test_pattern_sweep_reports_source_agreement_metrics() -> None:
    records = [
        _trace([1, 2, 3, 1, 2, 3, 1, 2, 3], task_id=0),
        _trace([5, 6, 7, 5, 6, 7, 5, 6, 7], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[3],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[True],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[True],
        ngram_min_contexts=[2],
        ngram_max_contexts=[4],
        ngram_history_sizes=[1],
        min_source_agreements=[1, 2],
        source_priorities=[parse_source_priority("lookup_first")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    agreement_row = next(row for row in summary["top"] if row["config"]["min_source_agreement"] == 2)
    assert agreement_row["source_agreement_drafted_tokens"] > 0
    assert agreement_row["source_agreement_accepted_tokens"] > 0
    assert agreement_row["min_task_source_agreement_accepted_tokens"] > 0


def test_pattern_sweep_flattens_heldout_source_metrics() -> None:
    train = [_trace([1, 2, 3, 4, 1, 2, 3, 4, 99], task_id=0)]
    heldout = [_trace([5, 6, 7, 8, 5, 6, 7, 8, 99], task_id=1)]

    summary = run_pattern_sweep(
        train,
        lookaheads=[3],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[True],
        ngram_min_contexts=[2],
        ngram_max_contexts=[4],
        ngram_history_sizes=[1],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(99,),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=1,
        heldout_records=heldout,
    )

    assert summary["best"]["heldout_ngram_continuation_accepted_tokens"] > 0
    assert summary["best"]["heldout_ngram_continuation_drafted_tokens"] > 0
    assert summary["best"]["heldout_min_task_ngram_continuation_accepted_tokens"] > 0


def test_pattern_sweep_flattens_heldout_tree_anchor_metrics() -> None:
    train = [_trace([1, 2, 3, 1, 2, 3, 1, 2, 3], task_id=0)]
    heldout = [_trace([4, 5, 6, 4, 5, 6, 4, 5, 6], task_id=1)]

    summary = run_pattern_sweep(
        train,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[True],
        ngram_min_contexts=[1],
        ngram_max_contexts=[3],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("ngram_continuation")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[3],
        tree_branch_widths=[3],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=1,
        heldout_records=heldout,
        tree_anchor_target_continuation_values=[True],
    )

    assert summary["best"]["heldout_tree_anchor_accepted_tokens"] > 0
    assert summary["best"]["heldout_min_task_tree_anchor_accepted_tokens"] > 0


def test_pattern_sweep_flattens_heldout_action_trend_metrics() -> None:
    train = [_trace([10, 12, 14, 16, 18, 20], task_id=0)]
    heldout = [_trace([30, 33, 36, 39, 42, 45], task_id=1)]

    summary = run_pattern_sweep(
        train,
        lookaheads=[2],
        action_dims=[1],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("action_trend_regression")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=1,
        heldout_records=heldout,
        action_trend_regression_values=[True],
        action_trend_histories=[4],
        action_trend_top_ks=[3],
        action_trend_max_abs_values=[6],
    )

    assert summary["best"]["heldout_action_trend_regression_accepted_tokens"] > 0
    assert summary["best"]["heldout_action_trend_regression_drafted_tokens"] > 0
    assert summary["best"]["heldout_min_task_action_trend_regression_accepted_tokens"] > 0


def test_pattern_sweep_can_rank_source_priority_modes() -> None:
    records = [
        _trace([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], task_id=0),
        _trace([5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], task_id=1),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[3],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[True],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[True],
        ngram_min_contexts=[2],
        ngram_max_contexts=[4],
        ngram_history_sizes=[1],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("lookup_first"), parse_source_priority("smooth_first")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
    )

    assert summary["sweep_count"] == 2
    priority = summary["best"]["config"]["source_priority"]
    assert priority[0] == "chunk_length_stop"
    assert priority.index("ngram_continuation") < priority.index("linear_action_extrapolation")
    assert summary["best"]["ngram_continuation_accepted_tokens"] > 0


def test_pattern_sweep_can_include_action_context_tree_configs() -> None:
    records = [
        _trace([1, 50, 2, 50, 1, 50, 2, 50, 1, 50, 2, 50], task_id=0),
    ]

    summary = run_pattern_sweep(
        records,
        lookaheads=[2],
        action_dims=[2],
        max_periods=[0],
        min_period_repeats=[2],
        repeat_token_min_runs=[99],
        linear_action_extrapolations=[False],
        second_order_action_extrapolations=[False],
        second_order_max_accels=[8],
        action_transition_histogram_values=[False],
        action_transition_history_sizes=[4],
        action_transition_top_ks=[3],
        action_transition_min_counts=[1],
        action_delta_histogram_values=[False],
        action_delta_histories=[8],
        action_delta_top_ks=[3],
        action_delta_max_abs_values=[12],
        previous_chunk_position_values=[False],
        previous_chunk_history_sizes=[1],
        position_mode_histogram_values=[False],
        position_mode_history_sizes=[4],
        position_mode_top_ks=[3],
        position_mode_min_counts=[2],
        action_dimension_mode_values=[False],
        action_dimension_mode_history_sizes=[4],
        action_dimension_mode_top_ks=[3],
        action_dimension_mode_min_counts=[2],
        hold_action_token_values=[False],
        ngram_continuation_values=[False],
        ngram_min_contexts=[3],
        ngram_max_contexts=[16],
        ngram_history_sizes=[4],
        min_source_agreements=[1],
        source_priorities=[parse_source_priority("default")],
        reuse_full_blocks_values=[True],
        emit_bonus_token_values=[False],
        dynamic_lookahead_values=[False],
        min_lookaheads=[1],
        lookahead_growths=[1],
        lookahead_shrinks=[4],
        history_reset="task_seed",
        tree_widths=[1],
        tree_branch_widths=[4],
        vocab_size=100,
        stop_token_ids=(),
        target_forward_ms=1.0,
        draft_token_ms=0.0,
        top_k=2,
        action_context_tree_values=[False, True],
        action_context_tree_history_sizes=[4],
        action_context_tree_max_contexts=[2, 3],
        action_context_tree_top_ks=[2],
        action_context_tree_min_counts=[1],
    )

    assert summary["sweep_count"] == 3
    assert summary["best"]["config"]["action_context_tree"] is True
    assert summary["best"]["action_context_tree_accepted_tokens"] > 0
