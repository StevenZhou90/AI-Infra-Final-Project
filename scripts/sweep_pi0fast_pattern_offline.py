#!/usr/bin/env python3
"""Sweep checkpoint-free PI0-FAST pattern speculation settings on trace shards."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_pi0fast_pattern_offline import (  # noqa: E402
    infer_stop_token_ids,
    load_lightweight_trace_records,
    parse_stop_tokens,
    record_suite_key,
    record_task_key,
    split_eval_indices,
)
from serving.pi0fast_pattern_drafter import (  # noqa: E402
    PatternDraftConfig,
    PatternFastTokenDrafter,
    evaluate_pattern_drafter,
    parse_source_priority,
)


SOURCE_COVERAGE_CONFIG_KEYS = (
    ("second_order_action_extrapolation", "second_order_action_extrapolation"),
    ("action_trend_regression", "action_trend_regression"),
    ("action_prefix_lookup", "action_prefix_lookup"),
    ("action_vector_suffix_lookup", "action_vector_suffix_lookup"),
    ("action_vector_transition", "action_vector_transition"),
    ("action_repeat_vector", "action_repeat_vector"),
    ("chunk_length_stop", "chunk_length_stop"),
    ("action_context_tree", "action_context_tree"),
    ("action_transition_histogram", "action_transition_histogram"),
    ("action_delta_histogram", "action_delta_histogram"),
    ("action_delta_ngram", "action_delta_ngram"),
    ("chunk_position_delta", "chunk_position_delta"),
    ("chunk_delta_template", "chunk_delta_template"),
    ("previous_chunk_position", "previous_chunk_position"),
    ("chunk_prefix_retrieval", "chunk_prefix_retrieval"),
    ("action_token_neighborhood", "action_token_neighborhood"),
    ("position_mode_histogram", "position_mode_histogram"),
    ("global_position_mode", "global_position_mode"),
    ("action_dimension_mode", "action_dimension_mode"),
    ("hold_action_token", "hold_action_token"),
    ("ngram_continuation", "ngram_continuation"),
    ("source_agreement", "source_agreement"),
)


def parse_int_csv(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_bool_modes(value: str) -> list[bool]:
    normalized = value.strip().lower()
    if normalized == "both":
        return [True, False]
    if normalized in {"true", "1", "yes", "y"}:
        return [True]
    if normalized in {"false", "0", "no", "n"}:
        return [False]
    raise ValueError("Boolean sweep values must be true, false, or both")


def parse_optional_int_csv(value: str) -> list[int | None]:
    values: list[int | None] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "off"}:
            values.append(None)
        else:
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value or none")
    return values


def parse_source_priority_modes(value: str) -> list[tuple[str, ...]]:
    values = [parse_source_priority(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one source-priority mode")
    return values


def compact_metrics(metrics: dict[str, Any], *, include_per_trace: bool) -> dict[str, Any]:
    if include_per_trace:
        return metrics
    return {key: value for key, value in metrics.items() if key != "per_trace"}


def complement_indices(total: int, heldout_indices: list[int]) -> list[int]:
    heldout = set(heldout_indices)
    return [idx for idx in range(total) if idx not in heldout]


def task_key_metadata(records: list[Any], *, rank_indices: list[int], heldout_indices: list[int]) -> dict[str, Any]:
    rank_task_keys = sorted({str(record_task_key(records[idx])) for idx in rank_indices})
    heldout_task_keys = sorted({str(record_task_key(records[idx])) for idx in heldout_indices})
    overlap = sorted(set(rank_task_keys) & set(heldout_task_keys))
    rank_suite_keys = sorted(
        {suite for idx in rank_indices if (suite := record_suite_key(records[idx])) is not None}
    )
    heldout_suite_keys = sorted(
        {suite for idx in heldout_indices if (suite := record_suite_key(records[idx])) is not None}
    )
    return {
        "rank_task_keys": rank_task_keys,
        "rank_task_count": len(rank_task_keys),
        "heldout_task_keys": heldout_task_keys,
        "heldout_task_count": len(heldout_task_keys),
        "heldout_task_overlap": overlap,
        "heldout_task_overlap_count": len(overlap),
        "heldout_task_disjoint": not overlap,
        "rank_suite_keys": rank_suite_keys,
        "rank_suite_count": len(rank_suite_keys),
        "heldout_suite_keys": heldout_suite_keys,
        "heldout_suite_count": len(heldout_suite_keys),
    }


def _first(values: list[Any]) -> Any:
    if not values:
        raise ValueError("sweep option lists must not be empty")
    return values[0]


def _source_enabled(config: dict[str, Any], config_key: str) -> bool:
    if config_key == "source_agreement":
        return int(config.get("min_source_agreement", 1)) > 1
    return bool(config.get(config_key, False))


def evaluated_source_coverage(rows: list[dict[str, Any]]) -> tuple[dict[str, int], list[str], dict[str, int]]:
    source_counts = {source: 0 for source, _config_key in SOURCE_COVERAGE_CONFIG_KEYS}
    enabled_source_count_histogram: dict[str, int] = {}
    for row in rows:
        config = row.get("config") or {}
        if not isinstance(config, dict):
            continue
        enabled_count = str(int(config.get("enabled_source_count", 0)))
        enabled_source_count_histogram[enabled_count] = enabled_source_count_histogram.get(enabled_count, 0) + 1
        for source, config_key in SOURCE_COVERAGE_CONFIG_KEYS:
            if _source_enabled(config, config_key):
                source_counts[source] += 1
    coverage = [source for source, count in source_counts.items() if count > 0]
    return source_counts, coverage, dict(sorted(enabled_source_count_histogram.items(), key=lambda item: int(item[0])))


def evaluate_pattern_config(
    records: list[Any],
    config: dict[str, Any],
    *,
    include_per_trace: bool = False,
) -> dict[str, Any]:
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=int(config["lookahead"]),
            action_dim=int(config["action_dim"]),
            max_period=int(config["max_period"]),
            min_period_repeats=int(config["min_period_repeats"]),
            repeat_token_min_run=int(config["repeat_token_min_run"]),
            enable_linear_action_extrapolation=bool(config["linear_action_extrapolation"]),
            enable_second_order_action_extrapolation=bool(config.get("second_order_action_extrapolation", False)),
            second_order_max_accel=config.get("second_order_max_accel", 8),
            enable_action_trend_regression=bool(config.get("action_trend_regression", False)),
            action_trend_history=int(config.get("action_trend_history", 4)),
            action_trend_top_k=int(config.get("action_trend_top_k", 3)),
            action_trend_max_abs=config.get("action_trend_max_abs", 12),
            enable_action_prefix_lookup=bool(config.get("action_prefix_lookup", False)),
            action_prefix_history_size=int(config.get("action_prefix_history_size", 4)),
            action_prefix_top_k=int(config.get("action_prefix_top_k", 3)),
            action_prefix_min_prefix=int(config.get("action_prefix_min_prefix", 1)),
            action_prefix_max_mismatches=int(config.get("action_prefix_max_mismatches", 0)),
            enable_action_vector_suffix_lookup=bool(config.get("action_vector_suffix_lookup", False)),
            action_vector_suffix_history_size=int(config.get("action_vector_suffix_history_size", 4)),
            action_vector_suffix_top_k=int(config.get("action_vector_suffix_top_k", 3)),
            action_vector_suffix_min_prefix=int(config.get("action_vector_suffix_min_prefix", 1)),
            action_vector_suffix_min_count=int(config.get("action_vector_suffix_min_count", 1)),
            action_vector_suffix_max_prefix_delta=int(config.get("action_vector_suffix_max_prefix_delta", 0)),
            enable_action_vector_transition=bool(config.get("action_vector_transition", False)),
            action_vector_transition_history_size=int(config.get("action_vector_transition_history_size", 4)),
            action_vector_transition_top_k=int(config.get("action_vector_transition_top_k", 3)),
            action_vector_transition_min_count=int(config.get("action_vector_transition_min_count", 1)),
            action_vector_transition_max_prev_delta=int(config.get("action_vector_transition_max_prev_delta", 0)),
            action_vector_transition_max_prefix_delta=int(
                config.get("action_vector_transition_max_prefix_delta", 0)
            ),
            enable_action_repeat_vector=bool(config.get("action_repeat_vector", False)),
            action_repeat_min_repeats=int(config.get("action_repeat_min_repeats", 2)),
            action_repeat_max_delta=int(config.get("action_repeat_max_delta", 0)),
            enable_chunk_length_stop=bool(config.get("chunk_length_stop", False)),
            chunk_length_stop_history_size=int(config.get("chunk_length_stop_history_size", 4)),
            chunk_length_stop_min_count=int(config.get("chunk_length_stop_min_count", 2)),
            enable_action_context_tree=bool(config.get("action_context_tree", False)),
            action_context_tree_history_size=int(config.get("action_context_tree_history_size", 4)),
            action_context_tree_max_context=int(config.get("action_context_tree_max_context", 3)),
            action_context_tree_top_k=int(config.get("action_context_tree_top_k", 3)),
            action_context_tree_min_count=int(config.get("action_context_tree_min_count", 1)),
            enable_action_transition_histogram=bool(config.get("action_transition_histogram", False)),
            action_transition_history_size=int(config.get("action_transition_history_size", 4)),
            action_transition_top_k=int(config.get("action_transition_top_k", 3)),
            action_transition_min_count=int(config.get("action_transition_min_count", 1)),
            enable_action_delta_histogram=bool(config.get("action_delta_histogram", False)),
            action_delta_history=int(config.get("action_delta_history", 8)),
            action_delta_top_k=int(config.get("action_delta_top_k", 3)),
            action_delta_min_count=int(config.get("action_delta_min_count", 1)),
            action_delta_max_abs=config.get("action_delta_max_abs", 12),
            enable_action_delta_ngram=bool(config.get("action_delta_ngram", False)),
            action_delta_ngram_history_size=int(config.get("action_delta_ngram_history_size", 4)),
            action_delta_ngram_min_context=int(config.get("action_delta_ngram_min_context", 1)),
            action_delta_ngram_max_context=int(config.get("action_delta_ngram_max_context", 4)),
            action_delta_ngram_top_k=int(config.get("action_delta_ngram_top_k", 3)),
            action_delta_ngram_min_count=int(config.get("action_delta_ngram_min_count", 1)),
            action_delta_ngram_max_abs=config.get("action_delta_ngram_max_abs", 12),
            enable_chunk_position_delta=bool(config.get("chunk_position_delta", False)),
            chunk_position_delta_history_size=int(config.get("chunk_position_delta_history_size", 4)),
            chunk_position_delta_top_k=int(config.get("chunk_position_delta_top_k", 3)),
            chunk_position_delta_min_count=int(config.get("chunk_position_delta_min_count", 1)),
            chunk_position_delta_max_abs=config.get("chunk_position_delta_max_abs", 12),
            enable_chunk_delta_template=bool(config.get("chunk_delta_template", False)),
            chunk_delta_template_history_size=int(config.get("chunk_delta_template_history_size", 4)),
            chunk_delta_template_top_k=int(config.get("chunk_delta_template_top_k", 3)),
            chunk_delta_template_min_prefix_deltas=int(config.get("chunk_delta_template_min_prefix_deltas", 1)),
            chunk_delta_template_max_delta_mismatch=config.get("chunk_delta_template_max_delta_mismatch", 0),
            chunk_delta_template_max_abs=config.get("chunk_delta_template_max_abs", 12),
            enable_previous_chunk_position=bool(config.get("previous_chunk_position", False)),
            previous_chunk_history_size=int(config.get("previous_chunk_history_size", 1)),
            enable_chunk_prefix_retrieval=bool(config.get("chunk_prefix_retrieval", False)),
            chunk_prefix_history_size=int(config.get("chunk_prefix_history_size", 4)),
            chunk_prefix_top_k=int(config.get("chunk_prefix_top_k", 3)),
            chunk_prefix_min_matches=int(config.get("chunk_prefix_min_matches", 2)),
            chunk_prefix_max_mismatches=int(config.get("chunk_prefix_max_mismatches", 1)),
            enable_action_token_neighborhood=bool(config.get("action_token_neighborhood", False)),
            action_token_neighborhood_radius=int(config.get("action_token_neighborhood_radius", 1)),
            action_token_neighborhood_top_k=int(config.get("action_token_neighborhood_top_k", 3)),
            enable_position_mode_histogram=bool(config.get("position_mode_histogram", False)),
            position_mode_history_size=int(config.get("position_mode_history_size", 4)),
            position_mode_top_k=int(config.get("position_mode_top_k", 3)),
            position_mode_min_count=int(config.get("position_mode_min_count", 2)),
            enable_global_position_mode=bool(config.get("global_position_mode", False)),
            global_position_history_size=int(config.get("global_position_history_size", 16)),
            global_position_top_k=int(config.get("global_position_top_k", 3)),
            global_position_min_count=int(config.get("global_position_min_count", 3)),
            enable_action_dimension_mode=bool(config.get("action_dimension_mode", False)),
            action_dimension_mode_history_size=int(config.get("action_dimension_mode_history_size", 4)),
            action_dimension_mode_top_k=int(config.get("action_dimension_mode_top_k", 3)),
            action_dimension_mode_min_count=int(config.get("action_dimension_mode_min_count", 2)),
            enable_hold_action_token=bool(config.get("hold_action_token", False)),
            enable_ngram_continuation=bool(config.get("ngram_continuation", False)),
            ngram_min_context=int(config.get("ngram_min_context", 3)),
            ngram_max_context=int(config.get("ngram_max_context", 16)),
            ngram_history_size=int(config.get("ngram_history_size", 4)),
            min_source_agreement=int(config.get("min_source_agreement", 1)),
            source_priority=parse_source_priority(config.get("source_priority", "default")),
            enable_source_cooldown=bool(config.get("source_cooldown", False)),
            source_cooldown_after=int(config.get("source_cooldown_after", 1)),
            source_cooldown_steps=int(config.get("source_cooldown_steps", 1)),
            enable_source_acceptance_bias=bool(config.get("source_acceptance_bias", False)),
            source_acceptance_bias_history_size=int(config.get("source_acceptance_bias_history_size", 32)),
            source_acceptance_bias_min_observations=int(
                config.get("source_acceptance_bias_min_observations", 2)
            ),
            vocab_size=config.get("vocab_size"),
            stop_token_ids=tuple(int(token) for token in config.get("stop_token_ids", [])),
        )
    )
    return compact_metrics(
        evaluate_pattern_drafter(
            drafter,
            records,
            lookahead=int(config["lookahead"]),
            reuse_full_blocks=bool(config.get("reuse_full_blocks", False)),
            emit_bonus_token=bool(config.get("emit_bonus_token", False)),
            dynamic_lookahead=bool(config.get("dynamic_lookahead", False)),
            min_lookahead=int(config.get("min_lookahead", 1)),
            lookahead_growth=int(config.get("lookahead_growth", 1)),
            lookahead_shrink=int(config.get("lookahead_shrink", 4)),
            history_reset=str(config.get("history_reset", "task_seed")),
            tree_width=int(config.get("tree_width", 1)),
            tree_branch_width=int(config.get("tree_branch_width", 4)),
            dynamic_tree_width=bool(config.get("dynamic_tree_width", False)),
            min_tree_width=int(config.get("min_tree_width", 1)),
            tree_width_growth=int(config.get("tree_width_growth", 1)),
            tree_width_shrink=int(config.get("tree_width_shrink", 1)),
            tree_anchor_target_token=bool(config.get("tree_anchor_target_token", False)),
            tree_anchor_target_continuation=bool(config.get("tree_anchor_target_continuation", False)),
            target_forward_ms=float(config.get("target_forward_ms", 1.0)),
            draft_token_ms=float(config.get("draft_token_ms", 0.0)),
        ),
        include_per_trace=include_per_trace,
    )


def run_pattern_sweep(
    records: list[Any],
    *,
    lookaheads: list[int],
    action_dims: list[int],
    max_periods: list[int],
    min_period_repeats: list[int],
    repeat_token_min_runs: list[int],
    linear_action_extrapolations: list[bool],
    second_order_action_extrapolations: list[bool],
    second_order_max_accels: list[int | None],
    action_transition_histogram_values: list[bool],
    action_transition_history_sizes: list[int],
    action_transition_top_ks: list[int],
    action_transition_min_counts: list[int],
    action_delta_histogram_values: list[bool],
    action_delta_histories: list[int],
    action_delta_top_ks: list[int],
    action_delta_min_counts: list[int] | None = None,
    action_delta_max_abs_values: list[int | None],
    previous_chunk_position_values: list[bool],
    previous_chunk_history_sizes: list[int],
    position_mode_histogram_values: list[bool],
    position_mode_history_sizes: list[int],
    position_mode_top_ks: list[int],
    position_mode_min_counts: list[int],
    action_dimension_mode_values: list[bool],
    action_dimension_mode_history_sizes: list[int],
    action_dimension_mode_top_ks: list[int],
    action_dimension_mode_min_counts: list[int],
    hold_action_token_values: list[bool],
    ngram_continuation_values: list[bool],
    ngram_min_contexts: list[int],
    ngram_max_contexts: list[int],
    ngram_history_sizes: list[int],
    min_source_agreements: list[int],
    source_priorities: list[tuple[str, ...]],
    reuse_full_blocks_values: list[bool],
    emit_bonus_token_values: list[bool],
    dynamic_lookahead_values: list[bool],
    min_lookaheads: list[int],
    lookahead_growths: list[int],
    lookahead_shrinks: list[int],
    history_reset: str,
    tree_widths: list[int],
    tree_branch_widths: list[int],
    vocab_size: int | None,
    stop_token_ids: tuple[int, ...],
    target_forward_ms: float,
    draft_token_ms: float,
    top_k: int,
    include_per_trace: bool = False,
    heldout_records: list[Any] | None = None,
    action_trend_regression_values: list[bool] | None = None,
    action_trend_histories: list[int] | None = None,
    action_trend_top_ks: list[int] | None = None,
    action_trend_max_abs_values: list[int | None] | None = None,
    action_prefix_lookup_values: list[bool] | None = None,
    action_prefix_history_sizes: list[int] | None = None,
    action_prefix_top_ks: list[int] | None = None,
    action_prefix_min_prefixes: list[int] | None = None,
    action_prefix_max_mismatches_values: list[int] | None = None,
    action_vector_suffix_lookup_values: list[bool] | None = None,
    action_vector_suffix_history_sizes: list[int] | None = None,
    action_vector_suffix_top_ks: list[int] | None = None,
    action_vector_suffix_min_prefixes: list[int] | None = None,
    action_vector_suffix_min_counts: list[int] | None = None,
    action_vector_suffix_max_prefix_delta_values: list[int] | None = None,
    action_vector_transition_values: list[bool] | None = None,
    action_vector_transition_history_sizes: list[int] | None = None,
    action_vector_transition_top_ks: list[int] | None = None,
    action_vector_transition_min_counts: list[int] | None = None,
    action_vector_transition_max_prev_delta_values: list[int] | None = None,
    action_vector_transition_max_prefix_delta_values: list[int] | None = None,
    action_repeat_vector_values: list[bool] | None = None,
    action_repeat_min_repeats_values: list[int] | None = None,
    action_repeat_max_delta_values: list[int] | None = None,
    chunk_length_stop_values: list[bool] | None = None,
    chunk_length_stop_history_sizes: list[int] | None = None,
    chunk_length_stop_min_counts: list[int] | None = None,
    action_context_tree_values: list[bool] | None = None,
    action_context_tree_history_sizes: list[int] | None = None,
    action_context_tree_max_contexts: list[int] | None = None,
    action_context_tree_top_ks: list[int] | None = None,
    action_context_tree_min_counts: list[int] | None = None,
    chunk_prefix_retrieval_values: list[bool] | None = None,
    chunk_prefix_history_sizes: list[int] | None = None,
    chunk_prefix_top_ks: list[int] | None = None,
    chunk_prefix_min_matches_values: list[int] | None = None,
    chunk_prefix_max_mismatches_values: list[int] | None = None,
    action_token_neighborhood_values: list[bool] | None = None,
    action_token_neighborhood_radii: list[int] | None = None,
    action_token_neighborhood_top_ks: list[int] | None = None,
    global_position_mode_values: list[bool] | None = None,
    global_position_history_sizes: list[int] | None = None,
    global_position_top_ks: list[int] | None = None,
    global_position_min_counts: list[int] | None = None,
    chunk_position_delta_values: list[bool] | None = None,
    chunk_position_delta_history_sizes: list[int] | None = None,
    chunk_position_delta_top_ks: list[int] | None = None,
    chunk_position_delta_min_counts: list[int] | None = None,
    chunk_position_delta_max_abs_values: list[int | None] | None = None,
    action_delta_ngram_values: list[bool] | None = None,
    action_delta_ngram_history_sizes: list[int] | None = None,
    action_delta_ngram_min_contexts: list[int] | None = None,
    action_delta_ngram_max_contexts: list[int] | None = None,
    action_delta_ngram_top_ks: list[int] | None = None,
    action_delta_ngram_min_counts: list[int] | None = None,
    action_delta_ngram_max_abs_values: list[int | None] | None = None,
    chunk_delta_template_values: list[bool] | None = None,
    chunk_delta_template_history_sizes: list[int] | None = None,
    chunk_delta_template_top_ks: list[int] | None = None,
    chunk_delta_template_min_prefix_deltas_values: list[int] | None = None,
    chunk_delta_template_max_delta_mismatch_values: list[int | None] | None = None,
    chunk_delta_template_max_abs_values: list[int | None] | None = None,
    dynamic_tree_width_values: list[bool] | None = None,
    min_tree_widths: list[int] | None = None,
    tree_width_growths: list[int] | None = None,
    tree_width_shrinks: list[int] | None = None,
    tree_anchor_target_token_values: list[bool] | None = None,
    tree_anchor_target_continuation_values: list[bool] | None = None,
    source_cooldown_values: list[bool] | None = None,
    source_cooldown_afters: list[int] | None = None,
    source_cooldown_steps_values: list[int] | None = None,
    source_acceptance_bias_values: list[bool] | None = None,
    source_acceptance_bias_history_sizes: list[int] | None = None,
    source_acceptance_bias_min_observations_values: list[int] | None = None,
    max_enabled_sources: int | None = None,
    max_configs: int | None = None,
) -> dict[str, Any]:
    action_trend_regression_values = (
        [False] if action_trend_regression_values is None else action_trend_regression_values
    )
    action_trend_histories = [4] if action_trend_histories is None else action_trend_histories
    action_trend_top_ks = [3] if action_trend_top_ks is None else action_trend_top_ks
    action_trend_max_abs_values = (
        [12] if action_trend_max_abs_values is None else action_trend_max_abs_values
    )
    action_prefix_lookup_values = [False] if action_prefix_lookup_values is None else action_prefix_lookup_values
    action_prefix_history_sizes = [4] if action_prefix_history_sizes is None else action_prefix_history_sizes
    action_prefix_top_ks = [3] if action_prefix_top_ks is None else action_prefix_top_ks
    action_prefix_min_prefixes = [1] if action_prefix_min_prefixes is None else action_prefix_min_prefixes
    action_prefix_max_mismatches_values = (
        [0] if action_prefix_max_mismatches_values is None else action_prefix_max_mismatches_values
    )
    action_vector_suffix_lookup_values = (
        [False] if action_vector_suffix_lookup_values is None else action_vector_suffix_lookup_values
    )
    action_vector_suffix_history_sizes = (
        [4] if action_vector_suffix_history_sizes is None else action_vector_suffix_history_sizes
    )
    action_vector_suffix_top_ks = (
        [3] if action_vector_suffix_top_ks is None else action_vector_suffix_top_ks
    )
    action_vector_suffix_min_prefixes = (
        [1] if action_vector_suffix_min_prefixes is None else action_vector_suffix_min_prefixes
    )
    action_vector_suffix_min_counts = (
        [1] if action_vector_suffix_min_counts is None else action_vector_suffix_min_counts
    )
    action_vector_suffix_max_prefix_delta_values = (
        [0]
        if action_vector_suffix_max_prefix_delta_values is None
        else action_vector_suffix_max_prefix_delta_values
    )
    action_vector_transition_values = (
        [False] if action_vector_transition_values is None else action_vector_transition_values
    )
    action_vector_transition_history_sizes = (
        [4] if action_vector_transition_history_sizes is None else action_vector_transition_history_sizes
    )
    action_vector_transition_top_ks = (
        [3] if action_vector_transition_top_ks is None else action_vector_transition_top_ks
    )
    action_vector_transition_min_counts = (
        [1] if action_vector_transition_min_counts is None else action_vector_transition_min_counts
    )
    action_vector_transition_max_prev_delta_values = (
        [0]
        if action_vector_transition_max_prev_delta_values is None
        else action_vector_transition_max_prev_delta_values
    )
    action_vector_transition_max_prefix_delta_values = (
        [0]
        if action_vector_transition_max_prefix_delta_values is None
        else action_vector_transition_max_prefix_delta_values
    )
    action_repeat_vector_values = [False] if action_repeat_vector_values is None else action_repeat_vector_values
    action_repeat_min_repeats_values = (
        [2] if action_repeat_min_repeats_values is None else action_repeat_min_repeats_values
    )
    action_repeat_max_delta_values = [0] if action_repeat_max_delta_values is None else action_repeat_max_delta_values
    chunk_length_stop_values = [False] if chunk_length_stop_values is None else chunk_length_stop_values
    chunk_length_stop_history_sizes = (
        [4] if chunk_length_stop_history_sizes is None else chunk_length_stop_history_sizes
    )
    chunk_length_stop_min_counts = [2] if chunk_length_stop_min_counts is None else chunk_length_stop_min_counts
    action_context_tree_values = [False] if action_context_tree_values is None else action_context_tree_values
    action_context_tree_history_sizes = (
        [4] if action_context_tree_history_sizes is None else action_context_tree_history_sizes
    )
    action_context_tree_max_contexts = [3] if action_context_tree_max_contexts is None else action_context_tree_max_contexts
    action_context_tree_top_ks = [3] if action_context_tree_top_ks is None else action_context_tree_top_ks
    action_context_tree_min_counts = [1] if action_context_tree_min_counts is None else action_context_tree_min_counts
    action_delta_min_counts = [1] if action_delta_min_counts is None else action_delta_min_counts
    chunk_prefix_retrieval_values = [False] if chunk_prefix_retrieval_values is None else chunk_prefix_retrieval_values
    chunk_prefix_history_sizes = [4] if chunk_prefix_history_sizes is None else chunk_prefix_history_sizes
    chunk_prefix_top_ks = [3] if chunk_prefix_top_ks is None else chunk_prefix_top_ks
    chunk_prefix_min_matches_values = (
        [2] if chunk_prefix_min_matches_values is None else chunk_prefix_min_matches_values
    )
    chunk_prefix_max_mismatches_values = (
        [1] if chunk_prefix_max_mismatches_values is None else chunk_prefix_max_mismatches_values
    )
    action_token_neighborhood_values = (
        [False] if action_token_neighborhood_values is None else action_token_neighborhood_values
    )
    action_token_neighborhood_radii = (
        [1] if action_token_neighborhood_radii is None else action_token_neighborhood_radii
    )
    action_token_neighborhood_top_ks = (
        [3] if action_token_neighborhood_top_ks is None else action_token_neighborhood_top_ks
    )
    global_position_mode_values = [False] if global_position_mode_values is None else global_position_mode_values
    global_position_history_sizes = [16] if global_position_history_sizes is None else global_position_history_sizes
    global_position_top_ks = [3] if global_position_top_ks is None else global_position_top_ks
    global_position_min_counts = [3] if global_position_min_counts is None else global_position_min_counts
    chunk_position_delta_values = [False] if chunk_position_delta_values is None else chunk_position_delta_values
    chunk_position_delta_history_sizes = (
        [4] if chunk_position_delta_history_sizes is None else chunk_position_delta_history_sizes
    )
    chunk_position_delta_top_ks = [3] if chunk_position_delta_top_ks is None else chunk_position_delta_top_ks
    chunk_position_delta_min_counts = (
        [1] if chunk_position_delta_min_counts is None else chunk_position_delta_min_counts
    )
    chunk_position_delta_max_abs_values = (
        [12] if chunk_position_delta_max_abs_values is None else chunk_position_delta_max_abs_values
    )
    action_delta_ngram_values = [False] if action_delta_ngram_values is None else action_delta_ngram_values
    action_delta_ngram_history_sizes = (
        [4] if action_delta_ngram_history_sizes is None else action_delta_ngram_history_sizes
    )
    action_delta_ngram_min_contexts = [1] if action_delta_ngram_min_contexts is None else action_delta_ngram_min_contexts
    action_delta_ngram_max_contexts = [4] if action_delta_ngram_max_contexts is None else action_delta_ngram_max_contexts
    action_delta_ngram_top_ks = [3] if action_delta_ngram_top_ks is None else action_delta_ngram_top_ks
    action_delta_ngram_min_counts = [1] if action_delta_ngram_min_counts is None else action_delta_ngram_min_counts
    action_delta_ngram_max_abs_values = (
        [12] if action_delta_ngram_max_abs_values is None else action_delta_ngram_max_abs_values
    )
    chunk_delta_template_values = [False] if chunk_delta_template_values is None else chunk_delta_template_values
    chunk_delta_template_history_sizes = (
        [4] if chunk_delta_template_history_sizes is None else chunk_delta_template_history_sizes
    )
    chunk_delta_template_top_ks = [3] if chunk_delta_template_top_ks is None else chunk_delta_template_top_ks
    chunk_delta_template_min_prefix_deltas_values = (
        [1] if chunk_delta_template_min_prefix_deltas_values is None else chunk_delta_template_min_prefix_deltas_values
    )
    chunk_delta_template_max_delta_mismatch_values = (
        [0]
        if chunk_delta_template_max_delta_mismatch_values is None
        else chunk_delta_template_max_delta_mismatch_values
    )
    chunk_delta_template_max_abs_values = (
        [12] if chunk_delta_template_max_abs_values is None else chunk_delta_template_max_abs_values
    )
    source_cooldown_values = [False] if source_cooldown_values is None else source_cooldown_values
    source_cooldown_afters = [1] if source_cooldown_afters is None else source_cooldown_afters
    source_cooldown_steps_values = [1] if source_cooldown_steps_values is None else source_cooldown_steps_values
    source_acceptance_bias_values = (
        [False] if source_acceptance_bias_values is None else source_acceptance_bias_values
    )
    source_acceptance_bias_history_sizes = (
        [32] if source_acceptance_bias_history_sizes is None else source_acceptance_bias_history_sizes
    )
    source_acceptance_bias_min_observations_values = (
        [2]
        if source_acceptance_bias_min_observations_values is None
        else source_acceptance_bias_min_observations_values
    )
    dynamic_tree_width_values = [False] if dynamic_tree_width_values is None else dynamic_tree_width_values
    min_tree_widths = [1] if min_tree_widths is None else min_tree_widths
    tree_width_growths = [1] if tree_width_growths is None else tree_width_growths
    tree_width_shrinks = [1] if tree_width_shrinks is None else tree_width_shrinks
    tree_anchor_target_token_values = (
        [False] if tree_anchor_target_token_values is None else tree_anchor_target_token_values
    )
    tree_anchor_target_continuation_values = (
        [False] if tree_anchor_target_continuation_values is None else tree_anchor_target_continuation_values
    )
    default_second_order_max_accel = _first(second_order_max_accels)
    default_action_trend_history = _first(action_trend_histories)
    default_action_trend_top_k = _first(action_trend_top_ks)
    default_action_trend_max_abs = _first(action_trend_max_abs_values)
    default_action_prefix_history_size = _first(action_prefix_history_sizes)
    default_action_prefix_top_k = _first(action_prefix_top_ks)
    default_action_prefix_min_prefix = _first(action_prefix_min_prefixes)
    default_action_prefix_max_mismatches = _first(action_prefix_max_mismatches_values)
    default_action_vector_suffix_history_size = _first(action_vector_suffix_history_sizes)
    default_action_vector_suffix_top_k = _first(action_vector_suffix_top_ks)
    default_action_vector_suffix_min_prefix = _first(action_vector_suffix_min_prefixes)
    default_action_vector_suffix_min_count = _first(action_vector_suffix_min_counts)
    default_action_vector_suffix_max_prefix_delta = _first(action_vector_suffix_max_prefix_delta_values)
    default_action_vector_transition_history_size = _first(action_vector_transition_history_sizes)
    default_action_vector_transition_top_k = _first(action_vector_transition_top_ks)
    default_action_vector_transition_min_count = _first(action_vector_transition_min_counts)
    default_action_vector_transition_max_prev_delta = _first(action_vector_transition_max_prev_delta_values)
    default_action_vector_transition_max_prefix_delta = _first(action_vector_transition_max_prefix_delta_values)
    default_action_repeat_min_repeats = _first(action_repeat_min_repeats_values)
    default_action_repeat_max_delta = _first(action_repeat_max_delta_values)
    default_chunk_length_stop_history_size = _first(chunk_length_stop_history_sizes)
    default_chunk_length_stop_min_count = _first(chunk_length_stop_min_counts)
    default_action_context_tree_history_size = _first(action_context_tree_history_sizes)
    default_action_context_tree_max_context = _first(action_context_tree_max_contexts)
    default_action_context_tree_top_k = _first(action_context_tree_top_ks)
    default_action_context_tree_min_count = _first(action_context_tree_min_counts)
    default_action_transition_history_size = _first(action_transition_history_sizes)
    default_action_transition_top_k = _first(action_transition_top_ks)
    default_action_transition_min_count = _first(action_transition_min_counts)
    default_action_delta_history = _first(action_delta_histories)
    default_action_delta_top_k = _first(action_delta_top_ks)
    default_action_delta_min_count = _first(action_delta_min_counts)
    default_action_delta_max_abs = _first(action_delta_max_abs_values)
    default_action_delta_ngram_history_size = _first(action_delta_ngram_history_sizes)
    default_action_delta_ngram_min_context = _first(action_delta_ngram_min_contexts)
    default_action_delta_ngram_max_context = _first(action_delta_ngram_max_contexts)
    default_action_delta_ngram_top_k = _first(action_delta_ngram_top_ks)
    default_action_delta_ngram_min_count = _first(action_delta_ngram_min_counts)
    default_action_delta_ngram_max_abs = _first(action_delta_ngram_max_abs_values)
    default_chunk_position_delta_history_size = _first(chunk_position_delta_history_sizes)
    default_chunk_position_delta_top_k = _first(chunk_position_delta_top_ks)
    default_chunk_position_delta_min_count = _first(chunk_position_delta_min_counts)
    default_chunk_position_delta_max_abs = _first(chunk_position_delta_max_abs_values)
    default_chunk_delta_template_history_size = _first(chunk_delta_template_history_sizes)
    default_chunk_delta_template_top_k = _first(chunk_delta_template_top_ks)
    default_chunk_delta_template_min_prefix_deltas = _first(chunk_delta_template_min_prefix_deltas_values)
    default_chunk_delta_template_max_delta_mismatch = _first(chunk_delta_template_max_delta_mismatch_values)
    default_chunk_delta_template_max_abs = _first(chunk_delta_template_max_abs_values)
    default_previous_chunk_history_size = _first(previous_chunk_history_sizes)
    default_chunk_prefix_history_size = _first(chunk_prefix_history_sizes)
    default_chunk_prefix_top_k = _first(chunk_prefix_top_ks)
    default_chunk_prefix_min_matches = _first(chunk_prefix_min_matches_values)
    default_chunk_prefix_max_mismatches = _first(chunk_prefix_max_mismatches_values)
    default_action_token_neighborhood_radius = _first(action_token_neighborhood_radii)
    default_action_token_neighborhood_top_k = _first(action_token_neighborhood_top_ks)
    default_position_mode_history_size = _first(position_mode_history_sizes)
    default_position_mode_top_k = _first(position_mode_top_ks)
    default_position_mode_min_count = _first(position_mode_min_counts)
    default_global_position_history_size = _first(global_position_history_sizes)
    default_global_position_top_k = _first(global_position_top_ks)
    default_global_position_min_count = _first(global_position_min_counts)
    default_action_dimension_mode_history_size = _first(action_dimension_mode_history_sizes)
    default_action_dimension_mode_top_k = _first(action_dimension_mode_top_ks)
    default_action_dimension_mode_min_count = _first(action_dimension_mode_min_counts)
    default_ngram_min_context = _first(ngram_min_contexts)
    default_ngram_max_context = _first(ngram_max_contexts)
    default_ngram_history_size = _first(ngram_history_sizes)
    default_source_cooldown_after = _first(source_cooldown_afters)
    default_source_cooldown_steps = _first(source_cooldown_steps_values)
    default_source_acceptance_bias_history_size = _first(source_acceptance_bias_history_sizes)
    default_source_acceptance_bias_min_observations = _first(source_acceptance_bias_min_observations_values)
    default_min_lookahead = _first(min_lookaheads)
    default_lookahead_growth = _first(lookahead_growths)
    default_lookahead_shrink = _first(lookahead_shrinks)
    default_tree_branch_width = _first(tree_branch_widths)
    default_min_tree_width = _first(min_tree_widths)
    default_tree_width_growth = _first(tree_width_growths)
    default_tree_width_shrink = _first(tree_width_shrinks)

    def _feature_options(
        enabled_values: list[bool],
        value_lists: list[list[Any]],
        default_values: tuple[Any, ...],
    ) -> list[tuple[Any, ...]]:
        options: list[tuple[Any, ...]] = []
        for enabled in enabled_values:
            if enabled:
                for values in product(*value_lists):
                    options.append((True, *values, 1))
            else:
                options.append((False, *default_values, 0))
        return options

    source_settings: list[tuple[Any, ...]] = []
    skipped_source_budget_configs = 0
    for (
        second_order_setting,
        action_trend_setting,
        action_prefix_setting,
        action_vector_suffix_setting,
        action_vector_setting,
        action_repeat_setting,
        chunk_length_stop_setting,
        action_context_tree_setting,
        action_transition_setting,
        action_delta_setting,
        action_delta_ngram_setting,
        chunk_position_delta_setting,
        chunk_delta_template_setting,
        previous_chunk_position_setting,
        chunk_prefix_retrieval_setting,
        action_token_neighborhood_setting,
        position_mode_histogram_setting,
        global_position_mode_setting,
        action_dimension_mode_setting,
        hold_action_token_setting,
        ngram_continuation_setting,
        min_source_agreement,
    ) in product(
        _feature_options(second_order_action_extrapolations, [second_order_max_accels], (default_second_order_max_accel,)),
        _feature_options(
            action_trend_regression_values,
            [action_trend_histories, action_trend_top_ks, action_trend_max_abs_values],
            (default_action_trend_history, default_action_trend_top_k, default_action_trend_max_abs),
        ),
        _feature_options(
            action_prefix_lookup_values,
            [
                action_prefix_history_sizes,
                action_prefix_top_ks,
                action_prefix_min_prefixes,
                action_prefix_max_mismatches_values,
            ],
            (
                default_action_prefix_history_size,
                default_action_prefix_top_k,
                default_action_prefix_min_prefix,
                default_action_prefix_max_mismatches,
            ),
        ),
        _feature_options(
            action_vector_suffix_lookup_values,
            [
                action_vector_suffix_history_sizes,
                action_vector_suffix_top_ks,
                action_vector_suffix_min_prefixes,
                action_vector_suffix_min_counts,
                action_vector_suffix_max_prefix_delta_values,
            ],
            (
                default_action_vector_suffix_history_size,
                default_action_vector_suffix_top_k,
                default_action_vector_suffix_min_prefix,
                default_action_vector_suffix_min_count,
                default_action_vector_suffix_max_prefix_delta,
            ),
        ),
        _feature_options(
            action_vector_transition_values,
            [
                action_vector_transition_history_sizes,
                action_vector_transition_top_ks,
                action_vector_transition_min_counts,
                action_vector_transition_max_prev_delta_values,
                action_vector_transition_max_prefix_delta_values,
            ],
            (
                default_action_vector_transition_history_size,
                default_action_vector_transition_top_k,
                default_action_vector_transition_min_count,
                default_action_vector_transition_max_prev_delta,
                default_action_vector_transition_max_prefix_delta,
            ),
        ),
        _feature_options(
            action_repeat_vector_values,
            [action_repeat_min_repeats_values, action_repeat_max_delta_values],
            (default_action_repeat_min_repeats, default_action_repeat_max_delta),
        ),
        _feature_options(
            chunk_length_stop_values,
            [chunk_length_stop_history_sizes, chunk_length_stop_min_counts],
            (default_chunk_length_stop_history_size, default_chunk_length_stop_min_count),
        ),
        _feature_options(
            action_context_tree_values,
            [
                action_context_tree_history_sizes,
                action_context_tree_max_contexts,
                action_context_tree_top_ks,
                action_context_tree_min_counts,
            ],
            (
                default_action_context_tree_history_size,
                default_action_context_tree_max_context,
                default_action_context_tree_top_k,
                default_action_context_tree_min_count,
            ),
        ),
        _feature_options(
            action_transition_histogram_values,
            [action_transition_history_sizes, action_transition_top_ks, action_transition_min_counts],
            (
                default_action_transition_history_size,
                default_action_transition_top_k,
                default_action_transition_min_count,
            ),
        ),
        _feature_options(
            action_delta_histogram_values,
            [action_delta_histories, action_delta_top_ks, action_delta_min_counts, action_delta_max_abs_values],
            (
                default_action_delta_history,
                default_action_delta_top_k,
                default_action_delta_min_count,
                default_action_delta_max_abs,
            ),
        ),
        _feature_options(
            action_delta_ngram_values,
            [
                action_delta_ngram_history_sizes,
                action_delta_ngram_min_contexts,
                action_delta_ngram_max_contexts,
                action_delta_ngram_top_ks,
                action_delta_ngram_min_counts,
                action_delta_ngram_max_abs_values,
            ],
            (
                default_action_delta_ngram_history_size,
                default_action_delta_ngram_min_context,
                default_action_delta_ngram_max_context,
                default_action_delta_ngram_top_k,
                default_action_delta_ngram_min_count,
                default_action_delta_ngram_max_abs,
            ),
        ),
        _feature_options(
            chunk_position_delta_values,
            [
                chunk_position_delta_history_sizes,
                chunk_position_delta_top_ks,
                chunk_position_delta_min_counts,
                chunk_position_delta_max_abs_values,
            ],
            (
                default_chunk_position_delta_history_size,
                default_chunk_position_delta_top_k,
                default_chunk_position_delta_min_count,
                default_chunk_position_delta_max_abs,
            ),
        ),
        _feature_options(
            chunk_delta_template_values,
            [
                chunk_delta_template_history_sizes,
                chunk_delta_template_top_ks,
                chunk_delta_template_min_prefix_deltas_values,
                chunk_delta_template_max_delta_mismatch_values,
                chunk_delta_template_max_abs_values,
            ],
            (
                default_chunk_delta_template_history_size,
                default_chunk_delta_template_top_k,
                default_chunk_delta_template_min_prefix_deltas,
                default_chunk_delta_template_max_delta_mismatch,
                default_chunk_delta_template_max_abs,
            ),
        ),
        _feature_options(
            previous_chunk_position_values,
            [previous_chunk_history_sizes],
            (default_previous_chunk_history_size,),
        ),
        _feature_options(
            chunk_prefix_retrieval_values,
            [
                chunk_prefix_history_sizes,
                chunk_prefix_top_ks,
                chunk_prefix_min_matches_values,
                chunk_prefix_max_mismatches_values,
            ],
            (
                default_chunk_prefix_history_size,
                default_chunk_prefix_top_k,
                default_chunk_prefix_min_matches,
                default_chunk_prefix_max_mismatches,
            ),
        ),
        _feature_options(
            action_token_neighborhood_values,
            [action_token_neighborhood_radii, action_token_neighborhood_top_ks],
            (default_action_token_neighborhood_radius, default_action_token_neighborhood_top_k),
        ),
        _feature_options(
            position_mode_histogram_values,
            [position_mode_history_sizes, position_mode_top_ks, position_mode_min_counts],
            (default_position_mode_history_size, default_position_mode_top_k, default_position_mode_min_count),
        ),
        _feature_options(
            global_position_mode_values,
            [global_position_history_sizes, global_position_top_ks, global_position_min_counts],
            (
                default_global_position_history_size,
                default_global_position_top_k,
                default_global_position_min_count,
            ),
        ),
        _feature_options(
            action_dimension_mode_values,
            [action_dimension_mode_history_sizes, action_dimension_mode_top_ks, action_dimension_mode_min_counts],
            (
                default_action_dimension_mode_history_size,
                default_action_dimension_mode_top_k,
                default_action_dimension_mode_min_count,
            ),
        ),
        _feature_options(hold_action_token_values, [], ()),
        _feature_options(
            ngram_continuation_values,
            [ngram_min_contexts, ngram_max_contexts, ngram_history_sizes],
            (default_ngram_min_context, default_ngram_max_context, default_ngram_history_size),
        ),
        min_source_agreements,
    ):
        enabled_source_count = sum(
            int(setting[-1])
            for setting in (
                second_order_setting,
                action_trend_setting,
                action_prefix_setting,
                action_vector_suffix_setting,
                action_vector_setting,
                action_repeat_setting,
                chunk_length_stop_setting,
                action_context_tree_setting,
                action_transition_setting,
                action_delta_setting,
                action_delta_ngram_setting,
                chunk_position_delta_setting,
                chunk_delta_template_setting,
                previous_chunk_position_setting,
                chunk_prefix_retrieval_setting,
                action_token_neighborhood_setting,
                position_mode_histogram_setting,
                global_position_mode_setting,
                action_dimension_mode_setting,
                hold_action_token_setting,
                ngram_continuation_setting,
            )
        )
        if int(min_source_agreement) > 1:
            enabled_source_count += 1
        if max_enabled_sources is not None and enabled_source_count > max_enabled_sources:
            skipped_source_budget_configs += 1
            continue
        source_settings.append(
            (
                second_order_setting,
                action_trend_setting,
                action_prefix_setting,
                action_vector_suffix_setting,
                action_vector_setting,
                action_repeat_setting,
                chunk_length_stop_setting,
                action_context_tree_setting,
                action_transition_setting,
                action_delta_setting,
                action_delta_ngram_setting,
                chunk_position_delta_setting,
                chunk_delta_template_setting,
                previous_chunk_position_setting,
                chunk_prefix_retrieval_setting,
                action_token_neighborhood_setting,
                position_mode_histogram_setting,
                global_position_mode_setting,
                action_dimension_mode_setting,
                hold_action_token_setting,
                ngram_continuation_setting,
                int(min_source_agreement),
                enabled_source_count,
            )
        )
    source_settings.sort(key=lambda setting: (int(setting[-1]), repr(setting)))

    rows: list[dict[str, Any]] = []
    sweep_truncated = False
    for (
        lookahead,
        action_dim,
        max_period,
        min_repeats,
        repeat_run,
        linear,
        source_priority,
        source_cooldown,
        source_cooldown_after,
        source_cooldown_steps,
        source_acceptance_bias,
        source_acceptance_bias_history_size,
        source_acceptance_bias_min_observations,
        reuse_full_blocks,
        emit_bonus_token,
        dynamic_lookahead,
        min_lookahead,
        lookahead_growth,
        lookahead_shrink,
        tree_width,
        tree_branch_width,
        dynamic_tree_width,
        min_tree_width,
        tree_width_growth,
        tree_width_shrink,
        tree_anchor_target_token,
        tree_anchor_target_continuation,
        source_setting,
    ) in product(
        lookaheads,
        action_dims,
        max_periods,
        min_period_repeats,
        repeat_token_min_runs,
        linear_action_extrapolations,
        source_priorities,
        source_cooldown_values,
        source_cooldown_afters,
        source_cooldown_steps_values,
        source_acceptance_bias_values,
        source_acceptance_bias_history_sizes,
        source_acceptance_bias_min_observations_values,
        reuse_full_blocks_values,
        emit_bonus_token_values,
        dynamic_lookahead_values,
        min_lookaheads,
        lookahead_growths,
        lookahead_shrinks,
        tree_widths,
        tree_branch_widths,
        dynamic_tree_width_values,
        min_tree_widths,
        tree_width_growths,
        tree_width_shrinks,
        tree_anchor_target_token_values,
        tree_anchor_target_continuation_values,
        source_settings,
    ):
        (
            second_order_setting,
            action_trend_setting,
            action_prefix_setting,
            action_vector_suffix_setting,
            action_vector_setting,
            action_repeat_setting,
            chunk_length_stop_setting,
            action_context_tree_setting,
            action_transition_setting,
            action_delta_setting,
            action_delta_ngram_setting,
            chunk_position_delta_setting,
            chunk_delta_template_setting,
            previous_chunk_position_setting,
            chunk_prefix_retrieval_setting,
            action_token_neighborhood_setting,
            position_mode_histogram_setting,
            global_position_mode_setting,
            action_dimension_mode_setting,
            hold_action_token_setting,
            ngram_continuation_setting,
            min_source_agreement,
            enabled_source_count,
        ) = source_setting
        second_order, second_order_max_accel, _second_order_count = second_order_setting
        (
            action_trend_regression,
            action_trend_history,
            action_trend_top_k,
            action_trend_max_abs,
            _action_trend_count,
        ) = action_trend_setting
        (
            action_prefix_lookup,
            action_prefix_history_size,
            action_prefix_top_k,
            action_prefix_min_prefix,
            action_prefix_max_mismatches,
            _action_prefix_count,
        ) = action_prefix_setting
        (
            action_vector_suffix_lookup,
            action_vector_suffix_history_size,
            action_vector_suffix_top_k,
            action_vector_suffix_min_prefix,
            action_vector_suffix_min_count,
            action_vector_suffix_max_prefix_delta,
            _action_vector_suffix_count,
        ) = action_vector_suffix_setting
        (
            action_vector_transition,
            action_vector_transition_history_size,
            action_vector_transition_top_k,
            action_vector_transition_min_count,
            action_vector_transition_max_prev_delta,
            action_vector_transition_max_prefix_delta,
            _action_vector_transition_count,
        ) = action_vector_setting
        action_repeat_vector, action_repeat_min_repeats, action_repeat_max_delta, _action_repeat_count = (
            action_repeat_setting
        )
        chunk_length_stop, chunk_length_stop_history_size, chunk_length_stop_min_count, _chunk_length_stop_count = (
            chunk_length_stop_setting
        )
        (
            action_context_tree,
            action_context_tree_history_size,
            action_context_tree_max_context,
            action_context_tree_top_k,
            action_context_tree_min_count,
            _action_context_tree_count,
        ) = action_context_tree_setting
        (
            action_transition_histogram,
            action_transition_history_size,
            action_transition_top_k,
            action_transition_min_count,
            _action_transition_count,
        ) = action_transition_setting
        (
            action_delta_histogram,
            action_delta_history,
            action_delta_top_k,
            action_delta_min_count,
            action_delta_max_abs,
            _action_delta_count,
        ) = action_delta_setting
        (
            action_delta_ngram,
            action_delta_ngram_history_size,
            action_delta_ngram_min_context,
            action_delta_ngram_max_context,
            action_delta_ngram_top_k,
            action_delta_ngram_min_count,
            action_delta_ngram_max_abs,
            _action_delta_ngram_count,
        ) = action_delta_ngram_setting
        (
            chunk_position_delta,
            chunk_position_delta_history_size,
            chunk_position_delta_top_k,
            chunk_position_delta_min_count,
            chunk_position_delta_max_abs,
            _chunk_position_delta_count,
        ) = chunk_position_delta_setting
        (
            chunk_delta_template,
            chunk_delta_template_history_size,
            chunk_delta_template_top_k,
            chunk_delta_template_min_prefix_deltas,
            chunk_delta_template_max_delta_mismatch,
            chunk_delta_template_max_abs,
            _chunk_delta_template_count,
        ) = chunk_delta_template_setting
        previous_chunk_position, previous_chunk_history_size, _previous_chunk_position_count = (
            previous_chunk_position_setting
        )
        (
            chunk_prefix_retrieval,
            chunk_prefix_history_size,
            chunk_prefix_top_k,
            chunk_prefix_min_matches,
            chunk_prefix_max_mismatches,
            _chunk_prefix_retrieval_count,
        ) = chunk_prefix_retrieval_setting
        action_token_neighborhood, action_token_neighborhood_radius, action_token_neighborhood_top_k, _neighborhood_count = (
            action_token_neighborhood_setting
        )
        (
            position_mode_histogram,
            position_mode_history_size,
            position_mode_top_k,
            position_mode_min_count,
            _position_mode_count,
        ) = position_mode_histogram_setting
        (
            global_position_mode,
            global_position_history_size,
            global_position_top_k,
            global_position_min_count,
            _global_position_count,
        ) = global_position_mode_setting
        (
            action_dimension_mode,
            action_dimension_mode_history_size,
            action_dimension_mode_top_k,
            action_dimension_mode_min_count,
            _action_dimension_mode_count,
        ) = action_dimension_mode_setting
        hold_action_token, _hold_action_token_count = hold_action_token_setting
        ngram_continuation, ngram_min_context, ngram_max_context, ngram_history_size, _ngram_count = (
            ngram_continuation_setting
        )
        if not second_order and second_order_max_accel != default_second_order_max_accel:
            continue
        if not action_trend_regression and (
            action_trend_history != default_action_trend_history
            or action_trend_top_k != default_action_trend_top_k
            or action_trend_max_abs != default_action_trend_max_abs
        ):
            continue
        if not action_prefix_lookup and (
            action_prefix_history_size != default_action_prefix_history_size
            or action_prefix_top_k != default_action_prefix_top_k
            or action_prefix_min_prefix != default_action_prefix_min_prefix
            or action_prefix_max_mismatches != default_action_prefix_max_mismatches
        ):
            continue
        if not action_vector_suffix_lookup and (
            action_vector_suffix_history_size != default_action_vector_suffix_history_size
            or action_vector_suffix_top_k != default_action_vector_suffix_top_k
            or action_vector_suffix_min_prefix != default_action_vector_suffix_min_prefix
            or action_vector_suffix_min_count != default_action_vector_suffix_min_count
            or action_vector_suffix_max_prefix_delta != default_action_vector_suffix_max_prefix_delta
        ):
            continue
        if not action_vector_transition and (
            action_vector_transition_history_size != default_action_vector_transition_history_size
            or action_vector_transition_top_k != default_action_vector_transition_top_k
            or action_vector_transition_min_count != default_action_vector_transition_min_count
            or action_vector_transition_max_prev_delta != default_action_vector_transition_max_prev_delta
            or action_vector_transition_max_prefix_delta != default_action_vector_transition_max_prefix_delta
        ):
            continue
        if not action_repeat_vector and (
            action_repeat_min_repeats != default_action_repeat_min_repeats
            or action_repeat_max_delta != default_action_repeat_max_delta
        ):
            continue
        if not chunk_length_stop and (
            chunk_length_stop_history_size != default_chunk_length_stop_history_size
            or chunk_length_stop_min_count != default_chunk_length_stop_min_count
        ):
            continue
        if not action_context_tree and (
            action_context_tree_history_size != default_action_context_tree_history_size
            or action_context_tree_max_context != default_action_context_tree_max_context
            or action_context_tree_top_k != default_action_context_tree_top_k
            or action_context_tree_min_count != default_action_context_tree_min_count
        ):
            continue
        if not action_transition_histogram and (
            action_transition_history_size != default_action_transition_history_size
            or action_transition_top_k != default_action_transition_top_k
            or action_transition_min_count != default_action_transition_min_count
        ):
            continue
        if not action_delta_histogram and (
            action_delta_history != default_action_delta_history
            or action_delta_top_k != default_action_delta_top_k
            or action_delta_min_count != default_action_delta_min_count
            or action_delta_max_abs != default_action_delta_max_abs
        ):
            continue
        if not action_delta_ngram and (
            action_delta_ngram_history_size != default_action_delta_ngram_history_size
            or action_delta_ngram_min_context != default_action_delta_ngram_min_context
            or action_delta_ngram_max_context != default_action_delta_ngram_max_context
            or action_delta_ngram_top_k != default_action_delta_ngram_top_k
            or action_delta_ngram_min_count != default_action_delta_ngram_min_count
            or action_delta_ngram_max_abs != default_action_delta_ngram_max_abs
        ):
            continue
        if not chunk_position_delta and (
            chunk_position_delta_history_size != default_chunk_position_delta_history_size
            or chunk_position_delta_top_k != default_chunk_position_delta_top_k
            or chunk_position_delta_min_count != default_chunk_position_delta_min_count
            or chunk_position_delta_max_abs != default_chunk_position_delta_max_abs
        ):
            continue
        if not chunk_delta_template and (
            chunk_delta_template_history_size != default_chunk_delta_template_history_size
            or chunk_delta_template_top_k != default_chunk_delta_template_top_k
            or chunk_delta_template_min_prefix_deltas != default_chunk_delta_template_min_prefix_deltas
            or chunk_delta_template_max_delta_mismatch != default_chunk_delta_template_max_delta_mismatch
            or chunk_delta_template_max_abs != default_chunk_delta_template_max_abs
        ):
            continue
        if not previous_chunk_position and previous_chunk_history_size != default_previous_chunk_history_size:
            continue
        if not chunk_prefix_retrieval and (
            chunk_prefix_history_size != default_chunk_prefix_history_size
            or chunk_prefix_top_k != default_chunk_prefix_top_k
            or chunk_prefix_min_matches != default_chunk_prefix_min_matches
            or chunk_prefix_max_mismatches != default_chunk_prefix_max_mismatches
        ):
            continue
        if not action_token_neighborhood and (
            action_token_neighborhood_radius != default_action_token_neighborhood_radius
            or action_token_neighborhood_top_k != default_action_token_neighborhood_top_k
        ):
            continue
        if not position_mode_histogram and (
            position_mode_history_size != default_position_mode_history_size
            or position_mode_top_k != default_position_mode_top_k
            or position_mode_min_count != default_position_mode_min_count
        ):
            continue
        if not global_position_mode and (
            global_position_history_size != default_global_position_history_size
            or global_position_top_k != default_global_position_top_k
            or global_position_min_count != default_global_position_min_count
        ):
            continue
        if not action_dimension_mode and (
            action_dimension_mode_history_size != default_action_dimension_mode_history_size
            or action_dimension_mode_top_k != default_action_dimension_mode_top_k
            or action_dimension_mode_min_count != default_action_dimension_mode_min_count
        ):
            continue
        if not ngram_continuation and (
            ngram_min_context != default_ngram_min_context
            or ngram_max_context != default_ngram_max_context
            or ngram_history_size != default_ngram_history_size
        ):
            continue
        if not source_cooldown and (
            source_cooldown_after != default_source_cooldown_after
            or source_cooldown_steps != default_source_cooldown_steps
        ):
            continue
        if not source_acceptance_bias and (
            source_acceptance_bias_history_size != default_source_acceptance_bias_history_size
            or source_acceptance_bias_min_observations != default_source_acceptance_bias_min_observations
        ):
            continue
        if not dynamic_lookahead and (
            min_lookahead != default_min_lookahead
            or lookahead_growth != default_lookahead_growth
            or lookahead_shrink != default_lookahead_shrink
        ):
            continue
        if tree_width <= 1 and tree_branch_width != default_tree_branch_width:
            continue
        if tree_width <= 1 and dynamic_tree_width:
            continue
        if tree_width <= 1 and tree_anchor_target_token:
            continue
        if tree_width <= 1 and tree_anchor_target_continuation:
            continue
        if tree_anchor_target_token and tree_anchor_target_continuation:
            continue
        if (tree_width <= 1 or not dynamic_tree_width) and (
            min_tree_width != default_min_tree_width
            or tree_width_growth != default_tree_width_growth
            or tree_width_shrink != default_tree_width_shrink
        ):
            continue
        if min_lookahead > lookahead:
            continue
        if min_tree_width > tree_width:
            continue
        if ngram_min_context > ngram_max_context:
            continue
        if action_delta_ngram_min_context > action_delta_ngram_max_context:
            continue
        if max_configs is not None and len(rows) >= max_configs:
            sweep_truncated = True
            break
        drafter = PatternFastTokenDrafter(
            PatternDraftConfig(
                lookahead=lookahead,
                action_dim=action_dim,
                max_period=max_period,
                min_period_repeats=min_repeats,
                repeat_token_min_run=repeat_run,
                enable_linear_action_extrapolation=linear,
                enable_second_order_action_extrapolation=second_order,
                second_order_max_accel=second_order_max_accel,
                enable_action_trend_regression=action_trend_regression,
                action_trend_history=action_trend_history,
                action_trend_top_k=action_trend_top_k,
                action_trend_max_abs=action_trend_max_abs,
                enable_action_prefix_lookup=action_prefix_lookup,
                action_prefix_history_size=action_prefix_history_size,
                action_prefix_top_k=action_prefix_top_k,
                action_prefix_min_prefix=action_prefix_min_prefix,
                action_prefix_max_mismatches=action_prefix_max_mismatches,
                enable_action_vector_suffix_lookup=action_vector_suffix_lookup,
                action_vector_suffix_history_size=action_vector_suffix_history_size,
                action_vector_suffix_top_k=action_vector_suffix_top_k,
                action_vector_suffix_min_prefix=action_vector_suffix_min_prefix,
                action_vector_suffix_min_count=action_vector_suffix_min_count,
                action_vector_suffix_max_prefix_delta=action_vector_suffix_max_prefix_delta,
                enable_action_vector_transition=action_vector_transition,
                action_vector_transition_history_size=action_vector_transition_history_size,
                action_vector_transition_top_k=action_vector_transition_top_k,
                action_vector_transition_min_count=action_vector_transition_min_count,
                action_vector_transition_max_prev_delta=action_vector_transition_max_prev_delta,
                action_vector_transition_max_prefix_delta=action_vector_transition_max_prefix_delta,
                enable_action_repeat_vector=action_repeat_vector,
                action_repeat_min_repeats=action_repeat_min_repeats,
                action_repeat_max_delta=action_repeat_max_delta,
                enable_chunk_length_stop=chunk_length_stop,
                chunk_length_stop_history_size=chunk_length_stop_history_size,
                chunk_length_stop_min_count=chunk_length_stop_min_count,
                enable_action_context_tree=action_context_tree,
                action_context_tree_history_size=action_context_tree_history_size,
                action_context_tree_max_context=action_context_tree_max_context,
                action_context_tree_top_k=action_context_tree_top_k,
                action_context_tree_min_count=action_context_tree_min_count,
                enable_action_transition_histogram=action_transition_histogram,
                action_transition_history_size=action_transition_history_size,
                action_transition_top_k=action_transition_top_k,
                action_transition_min_count=action_transition_min_count,
                enable_action_delta_histogram=action_delta_histogram,
                action_delta_history=action_delta_history,
                action_delta_top_k=action_delta_top_k,
                action_delta_min_count=action_delta_min_count,
                action_delta_max_abs=action_delta_max_abs,
                enable_action_delta_ngram=action_delta_ngram,
                action_delta_ngram_history_size=action_delta_ngram_history_size,
                action_delta_ngram_min_context=action_delta_ngram_min_context,
                action_delta_ngram_max_context=action_delta_ngram_max_context,
                action_delta_ngram_top_k=action_delta_ngram_top_k,
                action_delta_ngram_min_count=action_delta_ngram_min_count,
                action_delta_ngram_max_abs=action_delta_ngram_max_abs,
                enable_chunk_position_delta=chunk_position_delta,
                chunk_position_delta_history_size=chunk_position_delta_history_size,
                chunk_position_delta_top_k=chunk_position_delta_top_k,
                chunk_position_delta_min_count=chunk_position_delta_min_count,
                chunk_position_delta_max_abs=chunk_position_delta_max_abs,
                enable_chunk_delta_template=chunk_delta_template,
                chunk_delta_template_history_size=chunk_delta_template_history_size,
                chunk_delta_template_top_k=chunk_delta_template_top_k,
                chunk_delta_template_min_prefix_deltas=chunk_delta_template_min_prefix_deltas,
                chunk_delta_template_max_delta_mismatch=chunk_delta_template_max_delta_mismatch,
                chunk_delta_template_max_abs=chunk_delta_template_max_abs,
                enable_previous_chunk_position=previous_chunk_position,
                previous_chunk_history_size=previous_chunk_history_size,
                enable_chunk_prefix_retrieval=chunk_prefix_retrieval,
                chunk_prefix_history_size=chunk_prefix_history_size,
                chunk_prefix_top_k=chunk_prefix_top_k,
                chunk_prefix_min_matches=chunk_prefix_min_matches,
                chunk_prefix_max_mismatches=chunk_prefix_max_mismatches,
                enable_action_token_neighborhood=action_token_neighborhood,
                action_token_neighborhood_radius=action_token_neighborhood_radius,
                action_token_neighborhood_top_k=action_token_neighborhood_top_k,
                enable_position_mode_histogram=position_mode_histogram,
                position_mode_history_size=position_mode_history_size,
                position_mode_top_k=position_mode_top_k,
                position_mode_min_count=position_mode_min_count,
                enable_global_position_mode=global_position_mode,
                global_position_history_size=global_position_history_size,
                global_position_top_k=global_position_top_k,
                global_position_min_count=global_position_min_count,
                enable_action_dimension_mode=action_dimension_mode,
                action_dimension_mode_history_size=action_dimension_mode_history_size,
                action_dimension_mode_top_k=action_dimension_mode_top_k,
                action_dimension_mode_min_count=action_dimension_mode_min_count,
                enable_hold_action_token=hold_action_token,
                enable_ngram_continuation=ngram_continuation,
                ngram_min_context=ngram_min_context,
                ngram_max_context=ngram_max_context,
                ngram_history_size=ngram_history_size,
                min_source_agreement=min_source_agreement,
                source_priority=source_priority,
                enable_source_cooldown=source_cooldown,
                source_cooldown_after=source_cooldown_after,
                source_cooldown_steps=source_cooldown_steps,
                enable_source_acceptance_bias=source_acceptance_bias,
                source_acceptance_bias_history_size=source_acceptance_bias_history_size,
                source_acceptance_bias_min_observations=source_acceptance_bias_min_observations,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
            )
        )
        metrics = compact_metrics(
            evaluate_pattern_drafter(
                drafter,
                records,
                lookahead=lookahead,
                reuse_full_blocks=reuse_full_blocks,
                emit_bonus_token=emit_bonus_token,
                dynamic_lookahead=dynamic_lookahead,
                min_lookahead=min_lookahead,
                lookahead_growth=lookahead_growth,
                lookahead_shrink=lookahead_shrink,
                history_reset=history_reset,
                tree_width=tree_width,
                tree_branch_width=tree_branch_width,
                dynamic_tree_width=dynamic_tree_width,
                min_tree_width=min_tree_width,
                tree_width_growth=tree_width_growth,
                tree_width_shrink=tree_width_shrink,
                tree_anchor_target_token=tree_anchor_target_token,
                tree_anchor_target_continuation=tree_anchor_target_continuation,
                target_forward_ms=target_forward_ms,
                draft_token_ms=draft_token_ms,
            ),
            include_per_trace=include_per_trace,
        )
        metrics["config"] = {
            "lookahead": lookahead,
            "action_dim": action_dim,
            "max_period": max_period,
            "min_period_repeats": min_repeats,
            "repeat_token_min_run": repeat_run,
            "linear_action_extrapolation": linear,
            "second_order_action_extrapolation": second_order,
            "second_order_max_accel": second_order_max_accel,
            "action_trend_regression": action_trend_regression,
            "action_trend_history": action_trend_history,
            "action_trend_top_k": action_trend_top_k,
            "action_trend_max_abs": action_trend_max_abs,
            "action_prefix_lookup": action_prefix_lookup,
            "action_prefix_history_size": action_prefix_history_size,
            "action_prefix_top_k": action_prefix_top_k,
            "action_prefix_min_prefix": action_prefix_min_prefix,
            "action_prefix_max_mismatches": action_prefix_max_mismatches,
            "action_vector_suffix_lookup": action_vector_suffix_lookup,
            "action_vector_suffix_history_size": action_vector_suffix_history_size,
            "action_vector_suffix_top_k": action_vector_suffix_top_k,
            "action_vector_suffix_min_prefix": action_vector_suffix_min_prefix,
            "action_vector_suffix_min_count": action_vector_suffix_min_count,
            "action_vector_suffix_max_prefix_delta": action_vector_suffix_max_prefix_delta,
            "action_vector_transition": action_vector_transition,
            "action_vector_transition_history_size": action_vector_transition_history_size,
            "action_vector_transition_top_k": action_vector_transition_top_k,
            "action_vector_transition_min_count": action_vector_transition_min_count,
            "action_vector_transition_max_prev_delta": action_vector_transition_max_prev_delta,
            "action_vector_transition_max_prefix_delta": action_vector_transition_max_prefix_delta,
            "action_repeat_vector": action_repeat_vector,
            "action_repeat_min_repeats": action_repeat_min_repeats,
            "action_repeat_max_delta": action_repeat_max_delta,
            "chunk_length_stop": chunk_length_stop,
            "chunk_length_stop_history_size": chunk_length_stop_history_size,
            "chunk_length_stop_min_count": chunk_length_stop_min_count,
            "action_context_tree": action_context_tree,
            "action_context_tree_history_size": action_context_tree_history_size,
            "action_context_tree_max_context": action_context_tree_max_context,
            "action_context_tree_top_k": action_context_tree_top_k,
            "action_context_tree_min_count": action_context_tree_min_count,
            "action_transition_histogram": action_transition_histogram,
            "action_transition_history_size": action_transition_history_size,
            "action_transition_top_k": action_transition_top_k,
            "action_transition_min_count": action_transition_min_count,
            "action_delta_histogram": action_delta_histogram,
            "action_delta_history": action_delta_history,
            "action_delta_top_k": action_delta_top_k,
            "action_delta_min_count": action_delta_min_count,
            "action_delta_max_abs": action_delta_max_abs,
            "action_delta_ngram": action_delta_ngram,
            "action_delta_ngram_history_size": action_delta_ngram_history_size,
            "action_delta_ngram_min_context": action_delta_ngram_min_context,
            "action_delta_ngram_max_context": action_delta_ngram_max_context,
            "action_delta_ngram_top_k": action_delta_ngram_top_k,
            "action_delta_ngram_min_count": action_delta_ngram_min_count,
            "action_delta_ngram_max_abs": action_delta_ngram_max_abs,
            "chunk_position_delta": chunk_position_delta,
            "chunk_position_delta_history_size": chunk_position_delta_history_size,
            "chunk_position_delta_top_k": chunk_position_delta_top_k,
            "chunk_position_delta_min_count": chunk_position_delta_min_count,
            "chunk_position_delta_max_abs": chunk_position_delta_max_abs,
            "chunk_delta_template": chunk_delta_template,
            "chunk_delta_template_history_size": chunk_delta_template_history_size,
            "chunk_delta_template_top_k": chunk_delta_template_top_k,
            "chunk_delta_template_min_prefix_deltas": chunk_delta_template_min_prefix_deltas,
            "chunk_delta_template_max_delta_mismatch": chunk_delta_template_max_delta_mismatch,
            "chunk_delta_template_max_abs": chunk_delta_template_max_abs,
            "previous_chunk_position": previous_chunk_position,
            "previous_chunk_history_size": previous_chunk_history_size,
            "chunk_prefix_retrieval": chunk_prefix_retrieval,
            "chunk_prefix_history_size": chunk_prefix_history_size,
            "chunk_prefix_top_k": chunk_prefix_top_k,
            "chunk_prefix_min_matches": chunk_prefix_min_matches,
            "chunk_prefix_max_mismatches": chunk_prefix_max_mismatches,
            "action_token_neighborhood": action_token_neighborhood,
            "action_token_neighborhood_radius": action_token_neighborhood_radius,
            "action_token_neighborhood_top_k": action_token_neighborhood_top_k,
            "position_mode_histogram": position_mode_histogram,
            "position_mode_history_size": position_mode_history_size,
            "position_mode_top_k": position_mode_top_k,
            "position_mode_min_count": position_mode_min_count,
            "global_position_mode": global_position_mode,
            "global_position_history_size": global_position_history_size,
            "global_position_top_k": global_position_top_k,
            "global_position_min_count": global_position_min_count,
            "action_dimension_mode": action_dimension_mode,
            "action_dimension_mode_history_size": action_dimension_mode_history_size,
            "action_dimension_mode_top_k": action_dimension_mode_top_k,
            "action_dimension_mode_min_count": action_dimension_mode_min_count,
            "hold_action_token": hold_action_token,
            "ngram_continuation": ngram_continuation,
            "ngram_min_context": ngram_min_context,
            "ngram_max_context": ngram_max_context,
            "ngram_history_size": ngram_history_size,
            "min_source_agreement": min_source_agreement,
            "source_priority": list(source_priority),
            "source_cooldown": source_cooldown,
            "source_cooldown_after": source_cooldown_after,
            "source_cooldown_steps": source_cooldown_steps,
            "source_acceptance_bias": source_acceptance_bias,
            "source_acceptance_bias_history_size": source_acceptance_bias_history_size,
            "source_acceptance_bias_min_observations": source_acceptance_bias_min_observations,
            "enabled_source_count": enabled_source_count,
            "reuse_full_blocks": reuse_full_blocks,
            "emit_bonus_token": emit_bonus_token,
            "dynamic_lookahead": dynamic_lookahead,
            "min_lookahead": min_lookahead,
            "lookahead_growth": lookahead_growth,
            "lookahead_shrink": lookahead_shrink,
            "history_reset": history_reset,
            "tree_width": tree_width,
            "tree_branch_width": tree_branch_width,
            "dynamic_tree_width": dynamic_tree_width,
            "min_tree_width": min_tree_width,
            "tree_width_growth": tree_width_growth,
            "tree_width_shrink": tree_width_shrink,
            "tree_anchor_target_token": tree_anchor_target_token,
            "tree_anchor_target_continuation": tree_anchor_target_continuation,
            "vocab_size": vocab_size,
            "stop_token_ids": list(stop_token_ids),
            "target_forward_ms": target_forward_ms,
            "draft_token_ms": draft_token_ms,
        }
        rows.append(metrics)

    if not rows:
        raise ValueError("No valid pattern sweep settings; every min_lookahead exceeded its lookahead")

    rows.sort(
        key=lambda row: (
            float(row["modeled_speedup"]),
            float(row["min_task_target_forward_reduction"]),
            float(row["target_forward_reduction"]),
            float(row["acceptance_rate"]),
        ),
        reverse=True,
    )
    top = rows[: max(int(top_k), 1)]
    if heldout_records:
        for row in top:
            heldout = evaluate_pattern_config(
                heldout_records,
                row["config"],
                include_per_trace=include_per_trace,
            )
            row["heldout"] = heldout
            row["heldout_traces"] = heldout["traces"]
            row["heldout_modeled_speedup"] = heldout["modeled_speedup"]
            row["heldout_target_forward_reduction"] = heldout["target_forward_reduction"]
            row["heldout_min_task_target_forward_reduction"] = heldout["min_task_target_forward_reduction"]
            row["heldout_min_task_acceptance_rate"] = heldout["min_task_acceptance_rate"]
            row["heldout_task_count"] = heldout["task_count"]
            row["heldout_min_task_traces"] = heldout["min_task_traces"]
            for metric_name in (
                "tree_anchor_accepted_tokens",
                "min_task_tree_anchor_accepted_tokens",
                "previous_chunk_position_drafted_tokens",
                "previous_chunk_position_accepted_tokens",
                "previous_chunk_position_acceptance_rate",
                "min_task_previous_chunk_position_drafted_tokens",
                "min_task_previous_chunk_position_accepted_tokens",
                "min_task_previous_chunk_position_acceptance_rate",
                "mean_task_previous_chunk_position_acceptance_rate",
                "chunk_prefix_retrieval_drafted_tokens",
                "chunk_prefix_retrieval_accepted_tokens",
                "chunk_prefix_retrieval_acceptance_rate",
                "min_task_chunk_prefix_retrieval_drafted_tokens",
                "min_task_chunk_prefix_retrieval_accepted_tokens",
                "min_task_chunk_prefix_retrieval_acceptance_rate",
                "mean_task_chunk_prefix_retrieval_acceptance_rate",
                "action_token_neighborhood_drafted_tokens",
                "action_token_neighborhood_accepted_tokens",
                "action_token_neighborhood_acceptance_rate",
                "min_task_action_token_neighborhood_drafted_tokens",
                "min_task_action_token_neighborhood_accepted_tokens",
                "min_task_action_token_neighborhood_acceptance_rate",
                "mean_task_action_token_neighborhood_acceptance_rate",
                "position_mode_histogram_drafted_tokens",
                "position_mode_histogram_accepted_tokens",
                "position_mode_histogram_acceptance_rate",
                "min_task_position_mode_histogram_drafted_tokens",
                "min_task_position_mode_histogram_accepted_tokens",
                "min_task_position_mode_histogram_acceptance_rate",
                "mean_task_position_mode_histogram_acceptance_rate",
                "global_position_mode_drafted_tokens",
                "global_position_mode_accepted_tokens",
                "global_position_mode_acceptance_rate",
                "min_task_global_position_mode_drafted_tokens",
                "min_task_global_position_mode_accepted_tokens",
                "min_task_global_position_mode_acceptance_rate",
                "mean_task_global_position_mode_acceptance_rate",
                "action_dimension_mode_drafted_tokens",
                "action_dimension_mode_accepted_tokens",
                "action_dimension_mode_acceptance_rate",
                "min_task_action_dimension_mode_drafted_tokens",
                "min_task_action_dimension_mode_accepted_tokens",
                "min_task_action_dimension_mode_acceptance_rate",
                "mean_task_action_dimension_mode_acceptance_rate",
                "hold_action_token_drafted_tokens",
                "hold_action_token_accepted_tokens",
                "hold_action_token_acceptance_rate",
                "min_task_hold_action_token_drafted_tokens",
                "min_task_hold_action_token_accepted_tokens",
                "min_task_hold_action_token_acceptance_rate",
                "mean_task_hold_action_token_acceptance_rate",
                "ngram_continuation_drafted_tokens",
                "ngram_continuation_accepted_tokens",
                "ngram_continuation_acceptance_rate",
                "min_task_ngram_continuation_drafted_tokens",
                "min_task_ngram_continuation_accepted_tokens",
                "min_task_ngram_continuation_acceptance_rate",
                "mean_task_ngram_continuation_acceptance_rate",
                "source_agreement_drafted_tokens",
                "source_agreement_accepted_tokens",
                "source_agreement_acceptance_rate",
                "min_task_source_agreement_drafted_tokens",
                "min_task_source_agreement_accepted_tokens",
                "min_task_source_agreement_acceptance_rate",
                "mean_task_source_agreement_acceptance_rate",
                "action_trend_regression_drafted_tokens",
                "action_trend_regression_accepted_tokens",
                "action_trend_regression_acceptance_rate",
                "min_task_action_trend_regression_drafted_tokens",
                "min_task_action_trend_regression_accepted_tokens",
                "min_task_action_trend_regression_acceptance_rate",
                "mean_task_action_trend_regression_acceptance_rate",
                "action_prefix_lookup_drafted_tokens",
                "action_prefix_lookup_accepted_tokens",
                "action_prefix_lookup_acceptance_rate",
                "min_task_action_prefix_lookup_drafted_tokens",
                "min_task_action_prefix_lookup_accepted_tokens",
                "min_task_action_prefix_lookup_acceptance_rate",
                "mean_task_action_prefix_lookup_acceptance_rate",
                "action_vector_suffix_lookup_drafted_tokens",
                "action_vector_suffix_lookup_accepted_tokens",
                "action_vector_suffix_lookup_acceptance_rate",
                "min_task_action_vector_suffix_lookup_drafted_tokens",
                "min_task_action_vector_suffix_lookup_accepted_tokens",
                "min_task_action_vector_suffix_lookup_acceptance_rate",
                "mean_task_action_vector_suffix_lookup_acceptance_rate",
                "action_vector_transition_drafted_tokens",
                "action_vector_transition_accepted_tokens",
                "action_vector_transition_acceptance_rate",
                "min_task_action_vector_transition_drafted_tokens",
                "min_task_action_vector_transition_accepted_tokens",
                "min_task_action_vector_transition_acceptance_rate",
                "mean_task_action_vector_transition_acceptance_rate",
                "action_repeat_vector_drafted_tokens",
                "action_repeat_vector_accepted_tokens",
                "action_repeat_vector_acceptance_rate",
                "min_task_action_repeat_vector_drafted_tokens",
                "min_task_action_repeat_vector_accepted_tokens",
                "min_task_action_repeat_vector_acceptance_rate",
                "mean_task_action_repeat_vector_acceptance_rate",
                "chunk_length_stop_drafted_tokens",
                "chunk_length_stop_accepted_tokens",
                "chunk_length_stop_acceptance_rate",
                "min_task_chunk_length_stop_drafted_tokens",
                "min_task_chunk_length_stop_accepted_tokens",
                "min_task_chunk_length_stop_acceptance_rate",
                "mean_task_chunk_length_stop_acceptance_rate",
                "action_context_tree_drafted_tokens",
                "action_context_tree_accepted_tokens",
                "action_context_tree_acceptance_rate",
                "min_task_action_context_tree_drafted_tokens",
                "min_task_action_context_tree_accepted_tokens",
                "min_task_action_context_tree_acceptance_rate",
                "mean_task_action_context_tree_acceptance_rate",
                "action_transition_histogram_drafted_tokens",
                "action_transition_histogram_accepted_tokens",
                "action_transition_histogram_acceptance_rate",
                "min_task_action_transition_histogram_drafted_tokens",
                "min_task_action_transition_histogram_accepted_tokens",
                "min_task_action_transition_histogram_acceptance_rate",
                "mean_task_action_transition_histogram_acceptance_rate",
                "action_delta_histogram_drafted_tokens",
                "action_delta_histogram_accepted_tokens",
                "action_delta_histogram_acceptance_rate",
                "min_task_action_delta_histogram_drafted_tokens",
                "min_task_action_delta_histogram_accepted_tokens",
                "min_task_action_delta_histogram_acceptance_rate",
                "mean_task_action_delta_histogram_acceptance_rate",
                "action_delta_ngram_drafted_tokens",
                "action_delta_ngram_accepted_tokens",
                "action_delta_ngram_acceptance_rate",
                "min_task_action_delta_ngram_drafted_tokens",
                "min_task_action_delta_ngram_accepted_tokens",
                "min_task_action_delta_ngram_acceptance_rate",
                "mean_task_action_delta_ngram_acceptance_rate",
                "chunk_delta_template_drafted_tokens",
                "chunk_delta_template_accepted_tokens",
                "chunk_delta_template_acceptance_rate",
                "min_task_chunk_delta_template_drafted_tokens",
                "min_task_chunk_delta_template_accepted_tokens",
                "min_task_chunk_delta_template_acceptance_rate",
                "mean_task_chunk_delta_template_acceptance_rate",
                "chunk_position_delta_drafted_tokens",
                "chunk_position_delta_accepted_tokens",
                "chunk_position_delta_acceptance_rate",
                "min_task_chunk_position_delta_drafted_tokens",
                "min_task_chunk_position_delta_accepted_tokens",
                "min_task_chunk_position_delta_acceptance_rate",
                "mean_task_chunk_position_delta_acceptance_rate",
                "source_cooldown_events",
                "source_cooldown_skipped_sources",
                "min_task_source_cooldown_events",
                "mean_task_source_cooldown_events",
                "min_task_source_cooldown_skipped_sources",
                "mean_task_source_cooldown_skipped_sources",
                "source_acceptance_bias_events",
                "source_acceptance_bias_reorders",
                "min_task_source_acceptance_bias_events",
                "mean_task_source_acceptance_bias_events",
                "min_task_source_acceptance_bias_reorders",
                "mean_task_source_acceptance_bias_reorders",
            ):
                if metric_name in heldout:
                    row[f"heldout_{metric_name}"] = heldout[metric_name]
    source_counts, source_coverage, enabled_source_count_histogram = evaluated_source_coverage(rows)
    return {
        "sweep_count": len(rows),
        "top_k": len(top),
        "sweep_truncated": sweep_truncated,
        "max_configs": max_configs,
        "max_enabled_sources": max_enabled_sources,
        "skipped_source_budget_configs": skipped_source_budget_configs,
        "evaluated_source_counts": source_counts,
        "evaluated_source_coverage": source_coverage,
        "evaluated_enabled_source_count_histogram": enabled_source_count_histogram,
        "best": top[0],
        "top": top,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    has_heldout = any("heldout" in row for row in summary["top"])
    lines = [
        "PI0-FAST pattern drafter offline sweep",
        "",
    ]
    if has_heldout:
        header = "| rank | train speedup | heldout speedup | train worst-task reduction | heldout worst-task reduction | train accept | heldout accept | full reuses | bonus | tree | mean tree cands | anchor token | anchor cont | mean lookahead | lookahead | action dim | period | repeats | run | linear | second order | accel | context tree | context accept | transition hist | transition accept | delta hist | delta accept | delta ngram | delta ngram accept | chunk template | chunk template accept | chunk-pos delta | chunk-pos accept | pos mode | pos accept | global pos | global pos accept | action-dim mode | action-dim accept | hold | hold accept | ngram | ngram ctx | ngram accept | source agreement | source priority | cooldown | cooldown events | acceptance bias | bias reorders | reuse | bonus mode | dynamic |"
        lines.extend(
            [
                header,
                "| " + " | ".join(["---"] * (header.count("|") - 1)) + " |",
            ]
        )
    else:
        header = "| rank | speedup | forward reduction | worst-task reduction | accept | worst-task accept | full reuses | bonus | tree | mean tree cands | anchor token | anchor cont | mean lookahead | lookahead | action dim | period | repeats | run | linear | second order | accel | context tree | context accept | transition hist | transition accept | delta hist | delta accept | delta ngram | delta ngram accept | chunk template | chunk template accept | chunk-pos delta | chunk-pos accept | pos mode | pos accept | global pos | global pos accept | action-dim mode | action-dim accept | hold | hold accept | ngram | ngram ctx | ngram accept | source agreement | source priority | cooldown | cooldown events | acceptance bias | bias reorders | reuse | bonus mode | dynamic |"
        lines.extend(
            [
                header,
                "| " + " | ".join(["---"] * (header.count("|") - 1)) + " |",
            ]
        )
    for idx, row in enumerate(summary["top"], start=1):
        cfg = row["config"]
        metric_cells = [
            str(idx),
            f"{row['modeled_speedup']:.2f}x",
        ]
        if has_heldout:
            metric_cells.extend(
                [
                    f"{row.get('heldout_modeled_speedup', 0.0):.2f}x",
                    f"{row['min_task_target_forward_reduction']:.2f}x",
                    f"{row.get('heldout_min_task_target_forward_reduction', 0.0):.2f}x",
                    f"{row['acceptance_rate']:.2%}",
                    f"{row.get('heldout_min_task_acceptance_rate', 0.0):.2%}",
                ]
            )
        else:
            metric_cells.extend(
                [
                    f"{row['target_forward_reduction']:.2f}x",
                    f"{row['min_task_target_forward_reduction']:.2f}x",
                    f"{row['acceptance_rate']:.2%}",
                    f"{row['min_task_acceptance_rate']:.2%}",
                ]
            )
        lines.append(
            "| "
            + " | ".join(
                metric_cells
                + [
                    str(row["full_block_reuses"]),
                    str(row["bonus_tokens"]),
                    str(cfg.get("tree_width", 1)),
                    f"{row.get('mean_tree_candidates', 0.0):.1f}",
                    str(cfg.get("tree_anchor_target_token", False)),
                    str(cfg.get("tree_anchor_target_continuation", False)),
                    f"{row['mean_lookahead']:.1f}",
                    str(cfg["lookahead"]),
                    str(cfg["action_dim"]),
                    str(cfg["max_period"]),
                    str(cfg["min_period_repeats"]),
                    str(cfg["repeat_token_min_run"]),
                    str(cfg["linear_action_extrapolation"]),
                    str(cfg["second_order_action_extrapolation"]),
                    "none" if cfg["second_order_max_accel"] is None else str(cfg["second_order_max_accel"]),
                    str(cfg.get("action_context_tree", False)),
                    f"{row.get('action_context_tree_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("action_transition_histogram", False)),
                    f"{row.get('action_transition_histogram_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("action_delta_histogram", False)),
                    f"{row.get('action_delta_histogram_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("action_delta_ngram", False)),
                    f"{row.get('action_delta_ngram_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("chunk_delta_template", False)),
                    f"{row.get('chunk_delta_template_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("chunk_position_delta", False)),
                    f"{row.get('chunk_position_delta_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("position_mode_histogram", False)),
                    f"{row.get('position_mode_histogram_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("global_position_mode", False)),
                    f"{row.get('global_position_mode_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("action_dimension_mode", False)),
                    f"{row.get('action_dimension_mode_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("hold_action_token", False)),
                    f"{row.get('hold_action_token_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("ngram_continuation", False)),
                    f"{cfg.get('ngram_min_context', 3)}-{cfg.get('ngram_max_context', 16)}",
                    f"{row.get('ngram_continuation_acceptance_rate', 0.0):.2%}",
                    str(cfg.get("min_source_agreement", 1)),
                    ",".join(str(source) for source in cfg.get("source_priority", [])),
                    str(cfg.get("source_cooldown", False)),
                    str(row.get("source_cooldown_events", 0)),
                    str(cfg.get("source_acceptance_bias", False)),
                    str(row.get("source_acceptance_bias_reorders", 0)),
                    str(cfg["reuse_full_blocks"]),
                    str(cfg["emit_bonus_token"]),
                    str(cfg["dynamic_lookahead"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep pattern_sd settings on PI0-FAST trace shards.")
    parser.add_argument("--data-dir", required=True, help="Directory containing shard_*.pt trace files.")
    parser.add_argument("--split", choices=["trace", "task", "seed", "task_seed", "all"], default="all")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction for trace splits, or group fraction for task/seed/task_seed splits.",
    )
    parser.add_argument("--heldout-task-id", type=int, default=None)
    parser.add_argument("--heldout-seed", type=int, default=None)
    parser.add_argument(
        "--heldout-split",
        choices=["none", "trace", "task", "seed", "task_seed"],
        default="none",
        help="Rank on the complement of this split and report heldout metrics for top rows.",
    )
    parser.add_argument(
        "--heldout-val-fraction",
        type=float,
        default=None,
        help="Heldout trace or group fraction for --heldout-split. Defaults to --val-fraction.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookaheads", default="4,8,12")
    parser.add_argument("--action-dims", default="7")
    parser.add_argument("--max-periods", default="8,16")
    parser.add_argument("--min-period-repeats", default="2")
    parser.add_argument("--repeat-token-min-runs", default="2,3")
    parser.add_argument("--linear-action-extrapolation", default="both")
    parser.add_argument(
        "--second-order-action-extrapolation",
        default="both",
        help="true, false, or both. Uses per-action-dimension acceleration when recent token motion is smooth.",
    )
    parser.add_argument(
        "--second-order-max-accels",
        default="4,8",
        help="Comma-separated max second-order token accelerations, or none for no guard.",
    )
    parser.add_argument(
        "--action-trend-regression",
        default="false",
        help="true, false, or both. Draft from short per-action-dimension regression trends.",
    )
    parser.add_argument("--action-trend-histories", default="4")
    parser.add_argument("--action-trend-top-ks", default="3")
    parser.add_argument(
        "--action-trend-max-abs-values",
        default="12",
        help="Comma-separated max projected same-dimension token steps, or none for no guard.",
    )
    parser.add_argument(
        "--action-prefix-lookup",
        default="false",
        help="true, false, or both. Draft later action dimensions from prior actions sharing the verified prefix.",
    )
    parser.add_argument("--action-prefix-history-sizes", default="4")
    parser.add_argument("--action-prefix-top-ks", default="3")
    parser.add_argument("--action-prefix-min-prefixes", default="1")
    parser.add_argument("--action-prefix-max-mismatches-values", default="0")
    parser.add_argument(
        "--action-vector-suffix-lookup",
        default="false",
        help="true, false, or both. Draft later action dimensions from prior full action vectors with matching prefixes.",
    )
    parser.add_argument("--action-vector-suffix-history-sizes", default="4")
    parser.add_argument("--action-vector-suffix-top-ks", default="3")
    parser.add_argument("--action-vector-suffix-min-prefixes", default="1")
    parser.add_argument("--action-vector-suffix-min-counts", default="1")
    parser.add_argument("--action-vector-suffix-max-prefix-delta-values", default="0")
    parser.add_argument(
        "--action-vector-transition",
        default="false",
        help="true, false, or both. Draft from prior full action-vector transitions.",
    )
    parser.add_argument("--action-vector-transition-history-sizes", default="4")
    parser.add_argument("--action-vector-transition-top-ks", default="3")
    parser.add_argument("--action-vector-transition-min-counts", default="1")
    parser.add_argument("--action-vector-transition-max-prev-delta-values", default="0")
    parser.add_argument("--action-vector-transition-max-prefix-delta-values", default="0")
    parser.add_argument(
        "--action-repeat-vector",
        default="false",
        help="true, false, or both. Draft from repeated full action vectors.",
    )
    parser.add_argument("--action-repeat-min-repeats-values", default="2")
    parser.add_argument("--action-repeat-max-delta-values", default="0")
    parser.add_argument(
        "--chunk-length-stop",
        default="false",
        help="true, false, or both. Draft stop tokens from repeated verified chunk lengths.",
    )
    parser.add_argument("--chunk-length-stop-history-sizes", default="4")
    parser.add_argument("--chunk-length-stop-min-counts", default="2")
    parser.add_argument(
        "--action-transition-histogram",
        default="false",
        help="true, false, or both. Draft from same-action-dimension transition histograms.",
    )
    parser.add_argument(
        "--action-context-tree",
        default="false",
        help="true, false, or both. Draft from same-action-dimension context-tree histograms.",
    )
    parser.add_argument("--action-context-tree-history-sizes", default="4")
    parser.add_argument("--action-context-tree-max-contexts", default="2,3")
    parser.add_argument("--action-context-tree-top-ks", default="2,3")
    parser.add_argument("--action-context-tree-min-counts", default="1,2")
    parser.add_argument("--action-transition-history-sizes", default="4")
    parser.add_argument("--action-transition-top-ks", default="3")
    parser.add_argument("--action-transition-min-counts", default="1")
    parser.add_argument(
        "--action-delta-histogram",
        default="false",
        help="true, false, or both. Draft from frequent recent same-action-dimension token deltas.",
    )
    parser.add_argument("--action-delta-histories", default="8")
    parser.add_argument("--action-delta-top-ks", default="3")
    parser.add_argument("--action-delta-min-counts", default="1")
    parser.add_argument(
        "--action-delta-max-abs-values",
        default="12",
        help="Comma-separated max absolute token deltas, or none for no guard.",
    )
    parser.add_argument(
        "--action-delta-ngram",
        default="false",
        help="true, false, or both. Draft from same-action-dimension token-delta n-gram continuations.",
    )
    parser.add_argument("--action-delta-ngram-history-sizes", default="4")
    parser.add_argument("--action-delta-ngram-min-contexts", default="1")
    parser.add_argument("--action-delta-ngram-max-contexts", default="4")
    parser.add_argument("--action-delta-ngram-top-ks", default="3")
    parser.add_argument("--action-delta-ngram-min-counts", default="1")
    parser.add_argument(
        "--action-delta-ngram-max-abs-values",
        default="12",
        help="Comma-separated max absolute token deltas for delta n-grams, or none for no guard.",
    )
    parser.add_argument(
        "--chunk-position-delta",
        default="false",
        help="true, false, or both. Draft per-position FAST-token deltas from recent verified chunks.",
    )
    parser.add_argument("--chunk-position-delta-history-sizes", default="4")
    parser.add_argument("--chunk-position-delta-top-ks", default="3")
    parser.add_argument("--chunk-position-delta-min-counts", default="1")
    parser.add_argument(
        "--chunk-position-delta-max-abs-values",
        default="12",
        help="Comma-separated max absolute chunk-position deltas, or none for no guard.",
    )
    parser.add_argument(
        "--chunk-delta-template",
        default="false",
        help="true, false, or both. Draft from recent chunks with matching same-dimension delta prefixes.",
    )
    parser.add_argument("--chunk-delta-template-history-sizes", default="4")
    parser.add_argument("--chunk-delta-template-top-ks", default="3")
    parser.add_argument("--chunk-delta-template-min-prefix-deltas", default="1")
    parser.add_argument(
        "--chunk-delta-template-max-delta-mismatch-values",
        default="0",
        help="Comma-separated allowed total delta-prefix mismatch values, or none for no guard.",
    )
    parser.add_argument(
        "--chunk-delta-template-max-abs-values",
        default="12",
        help="Comma-separated max absolute chunk-template deltas, or none for no guard.",
    )
    parser.add_argument(
        "--previous-chunk-position",
        default="false",
        help="true, false, or both. Draft token i from token i of recent verified chunks from the same task.",
    )
    parser.add_argument(
        "--previous-chunk-history-sizes",
        default="1",
        help="Comma-separated history sizes for --previous-chunk-position.",
    )
    parser.add_argument(
        "--chunk-prefix-retrieval",
        default="false",
        help="true, false, or both. Draft from recent chunks with mostly matching partial prefixes.",
    )
    parser.add_argument("--chunk-prefix-history-sizes", default="4")
    parser.add_argument("--chunk-prefix-top-ks", default="3")
    parser.add_argument("--chunk-prefix-min-matches", default="2")
    parser.add_argument("--chunk-prefix-max-mismatches", default="1")
    parser.add_argument(
        "--action-token-neighborhood",
        default="false",
        help="true, false, or both. Draft token neighborhoods around smooth action extrapolations.",
    )
    parser.add_argument("--action-token-neighborhood-radii", default="1")
    parser.add_argument("--action-token-neighborhood-top-ks", default="3")
    parser.add_argument(
        "--position-mode-histogram",
        default="false",
        help="true, false, or both. Draft per-position mode tokens from recent verified chunks.",
    )
    parser.add_argument("--position-mode-history-sizes", default="4")
    parser.add_argument("--position-mode-top-ks", default="3")
    parser.add_argument("--position-mode-min-counts", default="2")
    parser.add_argument(
        "--global-position-mode",
        default="false",
        help="true, false, or both. Draft per-position modal tokens without requiring a matching current prefix.",
    )
    parser.add_argument("--global-position-history-sizes", default="16")
    parser.add_argument("--global-position-top-ks", default="3")
    parser.add_argument("--global-position-min-counts", default="3")
    parser.add_argument(
        "--action-dimension-mode",
        default="false",
        help="true, false, or both. Draft the per-action-dimension modal verified token.",
    )
    parser.add_argument("--action-dimension-mode-history-sizes", default="4")
    parser.add_argument("--action-dimension-mode-top-ks", default="3")
    parser.add_argument("--action-dimension-mode-min-counts", default="2")
    parser.add_argument(
        "--hold-action-token",
        default="false",
        help="true, false, or both. Draft the previous token from the same action dimension.",
    )
    parser.add_argument(
        "--ngram-continuation",
        default="false",
        help="true, false, or both. Prompt-lookup suffix continuation from verified tokens and recent chunks.",
    )
    parser.add_argument("--ngram-min-contexts", default="3")
    parser.add_argument("--ngram-max-contexts", default="16")
    parser.add_argument("--ngram-history-sizes", default="4")
    parser.add_argument(
        "--min-source-agreements",
        default="1",
        help="Comma-separated source agreement thresholds. 1 disables source-agreement preference.",
    )
    parser.add_argument(
        "--source-priority-modes",
        default="default",
        help="Comma-separated source priority modes: default, lookup_first, smooth_first, repeat_first.",
    )
    parser.add_argument(
        "--source-cooldown",
        default="false",
        help="true, false, or both. Temporarily skip sources after verified rejections.",
    )
    parser.add_argument("--source-cooldown-afters", default="1")
    parser.add_argument("--source-cooldown-steps", default="1")
    parser.add_argument(
        "--source-acceptance-bias",
        default="false",
        help="true, false, or both. Reorder sources by recent verified source acceptance.",
    )
    parser.add_argument("--source-acceptance-bias-history-sizes", default="32")
    parser.add_argument("--source-acceptance-bias-min-observations", default="2")
    parser.add_argument(
        "--reuse-full-blocks",
        default="both",
        help="Whether to keep fully verified draft blocks in the target KV cache without immediately emitting a bonus token.",
    )
    parser.add_argument(
        "--emit-bonus-token",
        default="both",
        help="Whether a full verified block also emits the verifier's next greedy token with no extra target forward.",
    )
    parser.add_argument(
        "--dynamic-lookahead",
        default="false",
        help="true, false, or both. Shrinks/grows the pattern draft window from recent exact acceptance.",
    )
    parser.add_argument("--min-lookaheads", default="1")
    parser.add_argument("--lookahead-growths", default="1")
    parser.add_argument("--lookahead-shrinks", default="4")
    parser.add_argument(
        "--history-reset",
        choices=["none", "task", "task_seed"],
        default="task_seed",
        help="When previous-chunk history resets during offline simulation. task_seed matches per-episode runner resets.",
    )
    parser.add_argument(
        "--tree-widths",
        default="1",
        help="Exact tree candidate counts. Keep 1 for the single-chain verifier.",
    )
    parser.add_argument("--tree-branch-widths", default="4")
    parser.add_argument(
        "--dynamic-tree-width",
        default="false",
        help="true, false, or both. Grow or shrink exact tree candidate count from recent acceptance.",
    )
    parser.add_argument("--min-tree-widths", default="1")
    parser.add_argument("--tree-width-growths", default="1")
    parser.add_argument("--tree-width-shrinks", default="1")
    parser.add_argument(
        "--tree-anchor-target-token",
        default="false",
        help="true, false, or both. On tree first-token misses, anchor on the target token and verify continuations.",
    )
    parser.add_argument(
        "--tree-anchor-target-continuation",
        default="false",
        help="true, false, or both. Always anchor on the target token and verify drafted continuations.",
    )
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--stop-token-ids", default="")
    parser.add_argument("--target-forward-ms", type=float, default=1.0)
    parser.add_argument("--draft-token-ms", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--max-enabled-sources",
        type=int,
        default=0,
        help=(
            "Maximum optional proposal-source priors enabled in one row. "
            "0 disables the cap. Source agreement with threshold >1 counts as one source."
        ),
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Maximum valid configs to evaluate after pruning. 0 disables the cap.",
    )
    parser.add_argument("--include-per-trace", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_lightweight_trace_records(args.data_dir)
    heldout_indices: list[int] = []
    if args.heldout_split != "none":
        heldout_indices = split_eval_indices(
            records,
            split=args.heldout_split,
            val_fraction=args.heldout_val_fraction if args.heldout_val_fraction is not None else args.val_fraction,
            seed=args.seed,
            heldout_task_id=args.heldout_task_id,
            heldout_seed=args.heldout_seed,
        )
        indices = complement_indices(len(records), heldout_indices)
        if not indices:
            raise ValueError("--heldout-split left no traces for ranking")
    else:
        indices = split_eval_indices(
            records,
            split=args.split,
            val_fraction=args.val_fraction,
            seed=args.seed,
            heldout_task_id=args.heldout_task_id,
            heldout_seed=args.heldout_seed,
        )
    selected = [records[idx] for idx in indices]
    heldout_records = [records[idx] for idx in heldout_indices] if heldout_indices else None
    stop_token_ids = parse_stop_tokens(args.stop_token_ids) or infer_stop_token_ids(records)
    summary = run_pattern_sweep(
        selected,
        lookaheads=parse_int_csv(args.lookaheads),
        action_dims=parse_int_csv(args.action_dims),
        max_periods=parse_int_csv(args.max_periods),
        min_period_repeats=parse_int_csv(args.min_period_repeats),
        repeat_token_min_runs=parse_int_csv(args.repeat_token_min_runs),
        linear_action_extrapolations=parse_bool_modes(args.linear_action_extrapolation),
        second_order_action_extrapolations=parse_bool_modes(args.second_order_action_extrapolation),
        second_order_max_accels=parse_optional_int_csv(args.second_order_max_accels),
        action_trend_regression_values=parse_bool_modes(args.action_trend_regression),
        action_trend_histories=parse_int_csv(args.action_trend_histories),
        action_trend_top_ks=parse_int_csv(args.action_trend_top_ks),
        action_trend_max_abs_values=parse_optional_int_csv(args.action_trend_max_abs_values),
        action_prefix_lookup_values=parse_bool_modes(args.action_prefix_lookup),
        action_prefix_history_sizes=parse_int_csv(args.action_prefix_history_sizes),
        action_prefix_top_ks=parse_int_csv(args.action_prefix_top_ks),
        action_prefix_min_prefixes=parse_int_csv(args.action_prefix_min_prefixes),
        action_prefix_max_mismatches_values=parse_int_csv(args.action_prefix_max_mismatches_values),
        action_vector_suffix_lookup_values=parse_bool_modes(args.action_vector_suffix_lookup),
        action_vector_suffix_history_sizes=parse_int_csv(args.action_vector_suffix_history_sizes),
        action_vector_suffix_top_ks=parse_int_csv(args.action_vector_suffix_top_ks),
        action_vector_suffix_min_prefixes=parse_int_csv(args.action_vector_suffix_min_prefixes),
        action_vector_suffix_min_counts=parse_int_csv(args.action_vector_suffix_min_counts),
        action_vector_suffix_max_prefix_delta_values=parse_int_csv(
            args.action_vector_suffix_max_prefix_delta_values
        ),
        action_vector_transition_values=parse_bool_modes(args.action_vector_transition),
        action_vector_transition_history_sizes=parse_int_csv(args.action_vector_transition_history_sizes),
        action_vector_transition_top_ks=parse_int_csv(args.action_vector_transition_top_ks),
        action_vector_transition_min_counts=parse_int_csv(args.action_vector_transition_min_counts),
        action_vector_transition_max_prev_delta_values=parse_int_csv(
            args.action_vector_transition_max_prev_delta_values
        ),
        action_vector_transition_max_prefix_delta_values=parse_int_csv(
            args.action_vector_transition_max_prefix_delta_values
        ),
        action_repeat_vector_values=parse_bool_modes(args.action_repeat_vector),
        action_repeat_min_repeats_values=parse_int_csv(args.action_repeat_min_repeats_values),
        action_repeat_max_delta_values=parse_int_csv(args.action_repeat_max_delta_values),
        chunk_length_stop_values=parse_bool_modes(args.chunk_length_stop),
        chunk_length_stop_history_sizes=parse_int_csv(args.chunk_length_stop_history_sizes),
        chunk_length_stop_min_counts=parse_int_csv(args.chunk_length_stop_min_counts),
        action_context_tree_values=parse_bool_modes(args.action_context_tree),
        action_context_tree_history_sizes=parse_int_csv(args.action_context_tree_history_sizes),
        action_context_tree_max_contexts=parse_int_csv(args.action_context_tree_max_contexts),
        action_context_tree_top_ks=parse_int_csv(args.action_context_tree_top_ks),
        action_context_tree_min_counts=parse_int_csv(args.action_context_tree_min_counts),
        action_transition_histogram_values=parse_bool_modes(args.action_transition_histogram),
        action_transition_history_sizes=parse_int_csv(args.action_transition_history_sizes),
        action_transition_top_ks=parse_int_csv(args.action_transition_top_ks),
        action_transition_min_counts=parse_int_csv(args.action_transition_min_counts),
        action_delta_histogram_values=parse_bool_modes(args.action_delta_histogram),
        action_delta_histories=parse_int_csv(args.action_delta_histories),
        action_delta_top_ks=parse_int_csv(args.action_delta_top_ks),
        action_delta_min_counts=parse_int_csv(args.action_delta_min_counts),
        action_delta_max_abs_values=parse_optional_int_csv(args.action_delta_max_abs_values),
        action_delta_ngram_values=parse_bool_modes(args.action_delta_ngram),
        action_delta_ngram_history_sizes=parse_int_csv(args.action_delta_ngram_history_sizes),
        action_delta_ngram_min_contexts=parse_int_csv(args.action_delta_ngram_min_contexts),
        action_delta_ngram_max_contexts=parse_int_csv(args.action_delta_ngram_max_contexts),
        action_delta_ngram_top_ks=parse_int_csv(args.action_delta_ngram_top_ks),
        action_delta_ngram_min_counts=parse_int_csv(args.action_delta_ngram_min_counts),
        action_delta_ngram_max_abs_values=parse_optional_int_csv(args.action_delta_ngram_max_abs_values),
        chunk_position_delta_values=parse_bool_modes(args.chunk_position_delta),
        chunk_position_delta_history_sizes=parse_int_csv(args.chunk_position_delta_history_sizes),
        chunk_position_delta_top_ks=parse_int_csv(args.chunk_position_delta_top_ks),
        chunk_position_delta_min_counts=parse_int_csv(args.chunk_position_delta_min_counts),
        chunk_position_delta_max_abs_values=parse_optional_int_csv(args.chunk_position_delta_max_abs_values),
        chunk_delta_template_values=parse_bool_modes(args.chunk_delta_template),
        chunk_delta_template_history_sizes=parse_int_csv(args.chunk_delta_template_history_sizes),
        chunk_delta_template_top_ks=parse_int_csv(args.chunk_delta_template_top_ks),
        chunk_delta_template_min_prefix_deltas_values=parse_int_csv(args.chunk_delta_template_min_prefix_deltas),
        chunk_delta_template_max_delta_mismatch_values=parse_optional_int_csv(
            args.chunk_delta_template_max_delta_mismatch_values
        ),
        chunk_delta_template_max_abs_values=parse_optional_int_csv(args.chunk_delta_template_max_abs_values),
        previous_chunk_position_values=parse_bool_modes(args.previous_chunk_position),
        previous_chunk_history_sizes=parse_int_csv(args.previous_chunk_history_sizes),
        chunk_prefix_retrieval_values=parse_bool_modes(args.chunk_prefix_retrieval),
        chunk_prefix_history_sizes=parse_int_csv(args.chunk_prefix_history_sizes),
        chunk_prefix_top_ks=parse_int_csv(args.chunk_prefix_top_ks),
        chunk_prefix_min_matches_values=parse_int_csv(args.chunk_prefix_min_matches),
        chunk_prefix_max_mismatches_values=parse_int_csv(args.chunk_prefix_max_mismatches),
        action_token_neighborhood_values=parse_bool_modes(args.action_token_neighborhood),
        action_token_neighborhood_radii=parse_int_csv(args.action_token_neighborhood_radii),
        action_token_neighborhood_top_ks=parse_int_csv(args.action_token_neighborhood_top_ks),
        position_mode_histogram_values=parse_bool_modes(args.position_mode_histogram),
        position_mode_history_sizes=parse_int_csv(args.position_mode_history_sizes),
        position_mode_top_ks=parse_int_csv(args.position_mode_top_ks),
        position_mode_min_counts=parse_int_csv(args.position_mode_min_counts),
        global_position_mode_values=parse_bool_modes(args.global_position_mode),
        global_position_history_sizes=parse_int_csv(args.global_position_history_sizes),
        global_position_top_ks=parse_int_csv(args.global_position_top_ks),
        global_position_min_counts=parse_int_csv(args.global_position_min_counts),
        action_dimension_mode_values=parse_bool_modes(args.action_dimension_mode),
        action_dimension_mode_history_sizes=parse_int_csv(args.action_dimension_mode_history_sizes),
        action_dimension_mode_top_ks=parse_int_csv(args.action_dimension_mode_top_ks),
        action_dimension_mode_min_counts=parse_int_csv(args.action_dimension_mode_min_counts),
        hold_action_token_values=parse_bool_modes(args.hold_action_token),
        ngram_continuation_values=parse_bool_modes(args.ngram_continuation),
        ngram_min_contexts=parse_int_csv(args.ngram_min_contexts),
        ngram_max_contexts=parse_int_csv(args.ngram_max_contexts),
        ngram_history_sizes=parse_int_csv(args.ngram_history_sizes),
        min_source_agreements=parse_int_csv(args.min_source_agreements),
        source_priorities=parse_source_priority_modes(args.source_priority_modes),
        source_cooldown_values=parse_bool_modes(args.source_cooldown),
        source_cooldown_afters=parse_int_csv(args.source_cooldown_afters),
        source_cooldown_steps_values=parse_int_csv(args.source_cooldown_steps),
        source_acceptance_bias_values=parse_bool_modes(args.source_acceptance_bias),
        source_acceptance_bias_history_sizes=parse_int_csv(args.source_acceptance_bias_history_sizes),
        source_acceptance_bias_min_observations_values=parse_int_csv(
            args.source_acceptance_bias_min_observations
        ),
        reuse_full_blocks_values=parse_bool_modes(args.reuse_full_blocks),
        emit_bonus_token_values=parse_bool_modes(args.emit_bonus_token),
        dynamic_lookahead_values=parse_bool_modes(args.dynamic_lookahead),
        min_lookaheads=parse_int_csv(args.min_lookaheads),
        lookahead_growths=parse_int_csv(args.lookahead_growths),
        lookahead_shrinks=parse_int_csv(args.lookahead_shrinks),
        history_reset=args.history_reset,
        tree_widths=parse_int_csv(args.tree_widths),
        tree_branch_widths=parse_int_csv(args.tree_branch_widths),
        dynamic_tree_width_values=parse_bool_modes(args.dynamic_tree_width),
        min_tree_widths=parse_int_csv(args.min_tree_widths),
        tree_width_growths=parse_int_csv(args.tree_width_growths),
        tree_width_shrinks=parse_int_csv(args.tree_width_shrinks),
        tree_anchor_target_token_values=parse_bool_modes(args.tree_anchor_target_token),
        tree_anchor_target_continuation_values=parse_bool_modes(args.tree_anchor_target_continuation),
        vocab_size=args.vocab_size,
        stop_token_ids=stop_token_ids,
        target_forward_ms=args.target_forward_ms,
        draft_token_ms=args.draft_token_ms,
        top_k=args.top_k,
        include_per_trace=args.include_per_trace,
        heldout_records=heldout_records,
        max_enabled_sources=args.max_enabled_sources if args.max_enabled_sources > 0 else None,
        max_configs=args.max_configs if args.max_configs > 0 else None,
    )
    summary["selection"] = {
        "data_dir": args.data_dir,
        "split": args.split if args.heldout_split == "none" else "train_complement",
        "eval_trace_indices": indices,
        "rank_trace_indices": indices,
        "heldout_split": args.heldout_split,
        "heldout_trace_indices": heldout_indices,
        **task_key_metadata(records, rank_indices=indices, heldout_indices=heldout_indices),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(format_markdown(summary) if args.markdown else json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
