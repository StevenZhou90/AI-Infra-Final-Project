#!/usr/bin/env python3
"""Collect PI0-FAST target-EOS traces, sweep exact pattern drafts, and stage a 120-task gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def bool_mode_enables_true(value: str) -> bool:
    return any(part.strip().lower() in {"both", "true", "1", "yes", "y"} for part in parse_csv(value))


def int_csv_has_value_above(value: str, threshold: int) -> bool:
    for part in parse_csv(value):
        try:
            if int(part) > int(threshold):
                return True
        except ValueError:
            continue
    return False


def resolve_required_source_coverage(args: argparse.Namespace) -> list[str]:
    raw = str(args.pattern_required_source_coverage).strip()
    if raw.lower() in {"", "none", "off", "false", "0"}:
        return []
    if raw.lower() != "auto":
        values: list[str] = []
        seen: set[str] = set()
        for part in parse_csv(raw):
            if part not in seen:
                values.append(part)
                seen.add(part)
        return values

    source_modes = (
        ("second_order_action_extrapolation", args.second_order_action_extrapolation),
        ("action_trend_regression", args.action_trend_regression),
        ("action_prefix_lookup", args.action_prefix_lookup),
        ("action_vector_suffix_lookup", args.action_vector_suffix_lookup),
        ("action_vector_transition", args.action_vector_transition),
        ("action_repeat_vector", args.action_repeat_vector),
        ("chunk_length_stop", args.chunk_length_stop),
        ("action_context_tree", args.action_context_tree),
        ("action_transition_histogram", args.action_transition_histogram),
        ("action_delta_histogram", args.action_delta_histogram),
        ("action_delta_ngram", args.action_delta_ngram),
        ("chunk_position_delta", args.chunk_position_delta),
        ("chunk_delta_template", args.chunk_delta_template),
        ("previous_chunk_position", args.previous_chunk_position),
        ("chunk_prefix_retrieval", args.chunk_prefix_retrieval),
        ("action_token_neighborhood", args.action_token_neighborhood),
        ("position_mode_histogram", args.position_mode_histogram),
        ("global_position_mode", args.global_position_mode),
        ("action_dimension_mode", args.action_dimension_mode),
        ("hold_action_token", args.hold_action_token),
        ("ngram_continuation", args.ngram_continuation),
    )
    coverage = [source for source, mode in source_modes if bool_mode_enables_true(mode)]
    if int_csv_has_value_above(args.min_source_agreements, 1):
        coverage.append("source_agreement")
    return coverage


def _task_ids_arg(task_ids: str) -> str:
    values = parse_csv(task_ids)
    if not values:
        raise ValueError("--task-ids must contain at least one id")
    return ",".join(values)


def resolve_pattern_task_count_thresholds(
    *,
    suite_count: int,
    task_id_count: int,
    heldout_split: str,
    heldout_val_fraction: float | None,
    min_task_count: int | None,
    min_heldout_task_count: int | None,
) -> tuple[int, int]:
    planned_task_count = max(int(suite_count) * int(task_id_count), 1)
    fraction = 0.0 if heldout_val_fraction is None else max(0.0, min(float(heldout_val_fraction), 1.0))
    default_heldout_task_count = 0
    if heldout_split != "none":
        if heldout_split == "task" and int(suite_count) > 1:
            per_suite_heldout = max(1, min(int(task_id_count), int(round(int(task_id_count) * fraction))))
            default_heldout_task_count = int(suite_count) * per_suite_heldout
        else:
            default_heldout_task_count = max(1, min(planned_task_count, math.ceil(planned_task_count * fraction)))

    if min_task_count is None:
        if heldout_split == "task":
            resolved_min_task_count = max(1, planned_task_count - default_heldout_task_count)
        else:
            resolved_min_task_count = planned_task_count
    else:
        resolved_min_task_count = int(min_task_count)

    if min_heldout_task_count is None:
        resolved_min_heldout_task_count = default_heldout_task_count
    else:
        resolved_min_heldout_task_count = int(min_heldout_task_count)

    return resolved_min_task_count, resolved_min_heldout_task_count


def build_trace_command(
    *,
    python: str,
    eval_script: str,
    suite: str,
    task_ids: str,
    episodes: int,
    steps: int,
    seed: int,
    output_dir: Path,
    trace_dir: Path,
    trace_max_rows_per_shard: int,
    device: str,
    dtype: str,
    policy_kind: str,
    policy: str | None,
    num_inference_steps: int | None,
    smooth_position_delta: float,
    smooth_rotation_delta: float,
) -> list[str]:
    cmd = [
        python,
        eval_script,
        "--task",
        suite,
        "--task-ids",
        task_ids,
        "--episodes",
        str(episodes),
        "--steps",
        str(steps),
        "--modes",
        "target_eos",
        "--summary-baseline-mode",
        "target_eos",
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--dtype",
        dtype,
        "--policy-kind",
        policy_kind,
        "--smooth-position-delta",
        str(smooth_position_delta),
        "--smooth-rotation-delta",
        str(smooth_rotation_delta),
        "--enable-fast-token-hooks",
        "--token-trace-output-dir",
        str(trace_dir),
        "--token-trace-modes",
        "target_eos",
        "--token-trace-max-rows-per-shard",
        str(trace_max_rows_per_shard),
    ]
    if policy is not None:
        cmd.extend(["--policy", policy])
    if num_inference_steps is not None:
        cmd.extend(["--num-inference-steps", str(num_inference_steps)])
    return cmd


def build_sweep_command(
    *,
    python: str,
    sweep_script: str,
    trace_dir: Path,
    output: Path,
    seed: int,
    heldout_split: str,
    heldout_val_fraction: float,
    lookaheads: str,
    action_dims: str,
    max_periods: str,
    min_period_repeats: str,
    repeat_token_min_runs: str,
    second_order_action_extrapolation: str,
    second_order_max_accels: str,
    action_trend_regression: str,
    action_trend_histories: str,
    action_trend_top_ks: str,
    action_trend_max_abs_values: str,
    action_prefix_lookup: str,
    action_prefix_history_sizes: str,
    action_prefix_top_ks: str,
    action_prefix_min_prefixes: str,
    action_prefix_max_mismatches_values: str,
    action_vector_suffix_lookup: str,
    action_vector_suffix_history_sizes: str,
    action_vector_suffix_top_ks: str,
    action_vector_suffix_min_prefixes: str,
    action_vector_suffix_min_counts: str,
    action_vector_suffix_max_prefix_delta_values: str,
    action_vector_transition: str,
    action_vector_transition_history_sizes: str,
    action_vector_transition_top_ks: str,
    action_vector_transition_min_counts: str,
    action_vector_transition_max_prev_delta_values: str,
    action_vector_transition_max_prefix_delta_values: str,
    action_repeat_vector: str,
    action_repeat_min_repeats_values: str,
    action_repeat_max_delta_values: str,
    chunk_length_stop: str,
    chunk_length_stop_history_sizes: str,
    chunk_length_stop_min_counts: str,
    action_context_tree: str,
    action_context_tree_history_sizes: str,
    action_context_tree_max_contexts: str,
    action_context_tree_top_ks: str,
    action_context_tree_min_counts: str,
    action_transition_histogram: str,
    action_transition_history_sizes: str,
    action_transition_top_ks: str,
    action_delta_histogram: str,
    action_delta_histories: str,
    action_delta_top_ks: str,
    action_delta_min_counts: str,
    action_delta_max_abs_values: str,
    action_delta_ngram: str,
    action_delta_ngram_history_sizes: str,
    action_delta_ngram_min_contexts: str,
    action_delta_ngram_max_contexts: str,
    action_delta_ngram_top_ks: str,
    action_delta_ngram_min_counts: str,
    action_delta_ngram_max_abs_values: str,
    chunk_position_delta: str,
    chunk_position_delta_history_sizes: str,
    chunk_position_delta_top_ks: str,
    chunk_position_delta_min_counts: str,
    chunk_position_delta_max_abs_values: str,
    chunk_delta_template: str,
    chunk_delta_template_history_sizes: str,
    chunk_delta_template_top_ks: str,
    chunk_delta_template_min_prefix_deltas: str,
    chunk_delta_template_max_delta_mismatch_values: str,
    chunk_delta_template_max_abs_values: str,
    previous_chunk_position: str,
    previous_chunk_history_sizes: str,
    chunk_prefix_retrieval: str,
    chunk_prefix_history_sizes: str,
    chunk_prefix_top_ks: str,
    chunk_prefix_min_matches: str,
    chunk_prefix_max_mismatches: str,
    action_token_neighborhood: str,
    action_token_neighborhood_radii: str,
    action_token_neighborhood_top_ks: str,
    position_mode_histogram: str,
    position_mode_history_sizes: str,
    position_mode_top_ks: str,
    position_mode_min_counts: str,
    global_position_mode: str,
    global_position_history_sizes: str,
    global_position_top_ks: str,
    global_position_min_counts: str,
    action_dimension_mode: str,
    action_dimension_mode_history_sizes: str,
    action_dimension_mode_top_ks: str,
    action_dimension_mode_min_counts: str,
    hold_action_token: str,
    ngram_continuation: str,
    ngram_min_contexts: str,
    ngram_max_contexts: str,
    ngram_history_sizes: str,
    min_source_agreements: str,
    source_priority_modes: str,
    source_cooldown: str,
    source_cooldown_afters: str,
    source_cooldown_steps: str,
    source_acceptance_bias: str,
    source_acceptance_bias_history_sizes: str,
    source_acceptance_bias_min_observations: str,
    reuse_full_blocks: str,
    emit_bonus_token: str,
    defer_correction_token: str,
    dynamic_lookahead: str,
    min_lookaheads: str,
    lookahead_growths: str,
    lookahead_shrinks: str,
    history_reset: str,
    tree_widths: str,
    tree_branch_widths: str,
    dynamic_tree_width: str,
    min_tree_widths: str,
    tree_width_growths: str,
    tree_width_shrinks: str,
    tree_anchor_target_token: str,
    tree_anchor_target_continuation: str,
    stop_token_ids: str,
    target_forward_ms: float,
    draft_token_ms: float,
    top_k: int,
    max_enabled_sources: int,
    max_configs: int,
) -> list[str]:
    cmd = [
        python,
        sweep_script,
        "--data-dir",
        str(trace_dir),
        "--heldout-split",
        heldout_split,
        "--heldout-val-fraction",
        str(heldout_val_fraction),
        "--seed",
        str(seed),
        "--lookaheads",
        lookaheads,
        "--action-dims",
        action_dims,
        "--max-periods",
        max_periods,
        "--min-period-repeats",
        min_period_repeats,
        "--repeat-token-min-runs",
        repeat_token_min_runs,
        "--second-order-action-extrapolation",
        second_order_action_extrapolation,
        "--second-order-max-accels",
        second_order_max_accels,
        "--action-trend-regression",
        action_trend_regression,
        "--action-trend-histories",
        action_trend_histories,
        "--action-trend-top-ks",
        action_trend_top_ks,
        "--action-trend-max-abs-values",
        action_trend_max_abs_values,
        "--action-prefix-lookup",
        action_prefix_lookup,
        "--action-prefix-history-sizes",
        action_prefix_history_sizes,
        "--action-prefix-top-ks",
        action_prefix_top_ks,
        "--action-prefix-min-prefixes",
        action_prefix_min_prefixes,
        "--action-prefix-max-mismatches-values",
        action_prefix_max_mismatches_values,
        "--action-vector-suffix-lookup",
        action_vector_suffix_lookup,
        "--action-vector-suffix-history-sizes",
        action_vector_suffix_history_sizes,
        "--action-vector-suffix-top-ks",
        action_vector_suffix_top_ks,
        "--action-vector-suffix-min-prefixes",
        action_vector_suffix_min_prefixes,
        "--action-vector-suffix-min-counts",
        action_vector_suffix_min_counts,
        "--action-vector-suffix-max-prefix-delta-values",
        action_vector_suffix_max_prefix_delta_values,
        "--action-vector-transition",
        action_vector_transition,
        "--action-vector-transition-history-sizes",
        action_vector_transition_history_sizes,
        "--action-vector-transition-top-ks",
        action_vector_transition_top_ks,
        "--action-vector-transition-min-counts",
        action_vector_transition_min_counts,
        "--action-vector-transition-max-prev-delta-values",
        action_vector_transition_max_prev_delta_values,
        "--action-vector-transition-max-prefix-delta-values",
        action_vector_transition_max_prefix_delta_values,
        "--action-repeat-vector",
        action_repeat_vector,
        "--action-repeat-min-repeats-values",
        action_repeat_min_repeats_values,
        "--action-repeat-max-delta-values",
        action_repeat_max_delta_values,
        "--chunk-length-stop",
        chunk_length_stop,
        "--chunk-length-stop-history-sizes",
        chunk_length_stop_history_sizes,
        "--chunk-length-stop-min-counts",
        chunk_length_stop_min_counts,
        "--action-context-tree",
        action_context_tree,
        "--action-context-tree-history-sizes",
        action_context_tree_history_sizes,
        "--action-context-tree-max-contexts",
        action_context_tree_max_contexts,
        "--action-context-tree-top-ks",
        action_context_tree_top_ks,
        "--action-context-tree-min-counts",
        action_context_tree_min_counts,
        "--action-transition-histogram",
        action_transition_histogram,
        "--action-transition-history-sizes",
        action_transition_history_sizes,
        "--action-transition-top-ks",
        action_transition_top_ks,
        "--action-delta-histogram",
        action_delta_histogram,
        "--action-delta-histories",
        action_delta_histories,
        "--action-delta-top-ks",
        action_delta_top_ks,
        "--action-delta-min-counts",
        action_delta_min_counts,
        "--action-delta-max-abs-values",
        action_delta_max_abs_values,
        "--action-delta-ngram",
        action_delta_ngram,
        "--action-delta-ngram-history-sizes",
        action_delta_ngram_history_sizes,
        "--action-delta-ngram-min-contexts",
        action_delta_ngram_min_contexts,
        "--action-delta-ngram-max-contexts",
        action_delta_ngram_max_contexts,
        "--action-delta-ngram-top-ks",
        action_delta_ngram_top_ks,
        "--action-delta-ngram-min-counts",
        action_delta_ngram_min_counts,
        "--action-delta-ngram-max-abs-values",
        action_delta_ngram_max_abs_values,
        "--chunk-position-delta",
        chunk_position_delta,
        "--chunk-position-delta-history-sizes",
        chunk_position_delta_history_sizes,
        "--chunk-position-delta-top-ks",
        chunk_position_delta_top_ks,
        "--chunk-position-delta-min-counts",
        chunk_position_delta_min_counts,
        "--chunk-position-delta-max-abs-values",
        chunk_position_delta_max_abs_values,
        "--chunk-delta-template",
        chunk_delta_template,
        "--chunk-delta-template-history-sizes",
        chunk_delta_template_history_sizes,
        "--chunk-delta-template-top-ks",
        chunk_delta_template_top_ks,
        "--chunk-delta-template-min-prefix-deltas",
        chunk_delta_template_min_prefix_deltas,
        "--chunk-delta-template-max-delta-mismatch-values",
        chunk_delta_template_max_delta_mismatch_values,
        "--chunk-delta-template-max-abs-values",
        chunk_delta_template_max_abs_values,
        "--previous-chunk-position",
        previous_chunk_position,
        "--previous-chunk-history-sizes",
        previous_chunk_history_sizes,
        "--chunk-prefix-retrieval",
        chunk_prefix_retrieval,
        "--chunk-prefix-history-sizes",
        chunk_prefix_history_sizes,
        "--chunk-prefix-top-ks",
        chunk_prefix_top_ks,
        "--chunk-prefix-min-matches",
        chunk_prefix_min_matches,
        "--chunk-prefix-max-mismatches",
        chunk_prefix_max_mismatches,
        "--action-token-neighborhood",
        action_token_neighborhood,
        "--action-token-neighborhood-radii",
        action_token_neighborhood_radii,
        "--action-token-neighborhood-top-ks",
        action_token_neighborhood_top_ks,
        "--position-mode-histogram",
        position_mode_histogram,
        "--position-mode-history-sizes",
        position_mode_history_sizes,
        "--position-mode-top-ks",
        position_mode_top_ks,
        "--position-mode-min-counts",
        position_mode_min_counts,
        "--global-position-mode",
        global_position_mode,
        "--global-position-history-sizes",
        global_position_history_sizes,
        "--global-position-top-ks",
        global_position_top_ks,
        "--global-position-min-counts",
        global_position_min_counts,
        "--action-dimension-mode",
        action_dimension_mode,
        "--action-dimension-mode-history-sizes",
        action_dimension_mode_history_sizes,
        "--action-dimension-mode-top-ks",
        action_dimension_mode_top_ks,
        "--action-dimension-mode-min-counts",
        action_dimension_mode_min_counts,
        "--hold-action-token",
        hold_action_token,
        "--ngram-continuation",
        ngram_continuation,
        "--ngram-min-contexts",
        ngram_min_contexts,
        "--ngram-max-contexts",
        ngram_max_contexts,
        "--ngram-history-sizes",
        ngram_history_sizes,
        "--min-source-agreements",
        min_source_agreements,
        "--source-priority-modes",
        source_priority_modes,
        "--source-cooldown",
        source_cooldown,
        "--source-cooldown-afters",
        source_cooldown_afters,
        "--source-cooldown-steps",
        source_cooldown_steps,
        "--source-acceptance-bias",
        source_acceptance_bias,
        "--source-acceptance-bias-history-sizes",
        source_acceptance_bias_history_sizes,
        "--source-acceptance-bias-min-observations",
        source_acceptance_bias_min_observations,
        "--reuse-full-blocks",
        reuse_full_blocks,
        "--emit-bonus-token",
        emit_bonus_token,
        "--defer-correction-token",
        defer_correction_token,
        "--dynamic-lookahead",
        dynamic_lookahead,
        "--min-lookaheads",
        min_lookaheads,
        "--lookahead-growths",
        lookahead_growths,
        "--lookahead-shrinks",
        lookahead_shrinks,
        "--history-reset",
        history_reset,
        "--tree-widths",
        tree_widths,
        "--tree-branch-widths",
        tree_branch_widths,
        "--dynamic-tree-width",
        dynamic_tree_width,
        "--min-tree-widths",
        min_tree_widths,
        "--tree-width-growths",
        tree_width_growths,
        "--tree-width-shrinks",
        tree_width_shrinks,
        "--tree-anchor-target-token",
        tree_anchor_target_token,
        "--tree-anchor-target-continuation",
        tree_anchor_target_continuation,
        "--target-forward-ms",
        str(target_forward_ms),
        "--draft-token-ms",
        str(draft_token_ms),
        "--top-k",
        str(top_k),
        "--output",
        str(output),
    ]
    if max_enabled_sources > 0:
        cmd.extend(["--max-enabled-sources", str(max_enabled_sources)])
    if max_configs > 0:
        cmd.extend(["--max-configs", str(max_configs)])
    if stop_token_ids:
        cmd.extend(["--stop-token-ids", stop_token_ids])
    return cmd


def build_gate_command(
    *,
    python: str,
    gate_runner_script: str,
    root: Path,
    suites: str,
    task_ids: str,
    episodes: int,
    steps: int,
    seed: int,
    device: str,
    dtype: str,
    policy_kind: str,
    policy: str | None,
    num_inference_steps: int | None,
    smooth_position_delta: float,
    smooth_rotation_delta: float,
    sweep_json: Path,
    pattern_sweep_rank: int,
    pattern_min_modeled_speedup: float,
    pattern_min_forward_reduction: float,
    pattern_min_task_forward_reduction: float,
    pattern_min_task_acceptance_rate: float,
    pattern_min_task_count: int,
    pattern_min_heldout_modeled_speedup: float,
    pattern_min_heldout_forward_reduction: float,
    pattern_min_heldout_task_forward_reduction: float,
    pattern_min_heldout_task_acceptance_rate: float,
    pattern_min_heldout_task_count: int,
    pattern_min_metric: list[str],
    pattern_min_heldout_metric: list[str],
    pattern_auto_source_min_metrics: bool,
    pattern_required_source_coverage: list[str],
    min_pairs: int,
    min_speedup: float,
    max_success_drop: float,
    max_baseline_success_regressions: int,
    min_validation_episodes: int,
    min_exact_verifies: int,
    max_action_diff: float,
    gate_dry_run: bool,
) -> list[str]:
    cmd = [
        python,
        gate_runner_script,
        "--root",
        str(root),
        "--suites",
        suites,
        "--task-ids",
        task_ids,
        "--episodes",
        str(episodes),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--device",
        device,
        "--dtype",
        dtype,
        "--smooth-position-delta",
        str(smooth_position_delta),
        "--smooth-rotation-delta",
        str(smooth_rotation_delta),
        "--speed-modes",
        "baseline,target_eos,pattern_sd_direct",
        "--candidate-mode",
        "pattern_sd_direct",
        "--reference-mode",
        "target_eos",
        "--validation-modes",
        "auto",
        "--baseline-mode",
        "baseline",
        "--min-pairs",
        str(min_pairs),
        "--min-speedup",
        str(min_speedup),
        "--max-success-drop",
        str(max_success_drop),
        "--max-baseline-success-regressions",
        str(max_baseline_success_regressions),
        "--min-validation-episodes",
        str(min_validation_episodes),
        "--min-exact-verifies",
        str(min_exact_verifies),
        "--max-action-diff",
        str(max_action_diff),
        "--pattern-sweep-json",
        str(sweep_json),
        "--pattern-sweep-rank",
        str(pattern_sweep_rank),
        "--pattern-min-modeled-speedup",
        str(pattern_min_modeled_speedup),
        "--pattern-min-forward-reduction",
        str(pattern_min_forward_reduction),
        "--pattern-min-task-forward-reduction",
        str(pattern_min_task_forward_reduction),
        "--pattern-min-task-acceptance-rate",
        str(pattern_min_task_acceptance_rate),
        "--pattern-min-task-count",
        str(pattern_min_task_count),
        "--pattern-min-heldout-modeled-speedup",
        str(pattern_min_heldout_modeled_speedup),
        "--pattern-min-heldout-forward-reduction",
        str(pattern_min_heldout_forward_reduction),
        "--pattern-min-heldout-task-forward-reduction",
        str(pattern_min_heldout_task_forward_reduction),
        "--pattern-min-heldout-task-acceptance-rate",
        str(pattern_min_heldout_task_acceptance_rate),
        "--pattern-min-heldout-task-count",
        str(pattern_min_heldout_task_count),
        "--enable-fast-token-hooks",
    ]
    for threshold in pattern_min_metric:
        cmd.extend(["--pattern-min-metric", threshold])
    for threshold in pattern_min_heldout_metric:
        cmd.extend(["--pattern-min-heldout-metric", threshold])
    if pattern_auto_source_min_metrics:
        cmd.append("--pattern-auto-source-min-metrics")
    for source in pattern_required_source_coverage:
        cmd.extend(["--pattern-required-source-coverage", source])
    if gate_dry_run:
        cmd.append("--dry-run")
    eval_extra_args: list[str] = []
    if policy is not None:
        eval_extra_args.extend(["--policy", policy])
    if policy_kind != "pi0fast" or num_inference_steps is not None:
        eval_extra_args.extend(["--policy-kind", policy_kind])
    if num_inference_steps is not None:
        eval_extra_args.extend(["--num-inference-steps", str(num_inference_steps)])
    if eval_extra_args:
        cmd.append("--")
        cmd.extend(eval_extra_args)
    return cmd


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    suites = parse_csv(args.suites)
    if not suites:
        raise ValueError("--suites must contain at least one suite")
    task_ids = _task_ids_arg(args.task_ids)
    root = args.root
    trace_dir = args.trace_dir or (root / "target_eos_traces")
    trace_eval_root = args.trace_eval_root or (root / "trace_eval")
    sweep_json = args.sweep_output or (root / "pattern_sweep.json")
    gate_root = args.gate_root or (root / "target_eos_120_gate")
    resolved_pattern_min_task_count, resolved_pattern_min_heldout_task_count = (
        resolve_pattern_task_count_thresholds(
            suite_count=len(suites),
            task_id_count=len(parse_csv(task_ids)),
            heldout_split=args.heldout_split,
            heldout_val_fraction=args.heldout_val_fraction,
            min_task_count=args.pattern_min_task_count,
            min_heldout_task_count=args.pattern_min_heldout_task_count,
        )
    )
    pattern_required_source_coverage = resolve_required_source_coverage(args)
    trace_commands = [
        build_trace_command(
            python=args.python,
            eval_script=args.eval_script,
            suite=suite,
            task_ids=task_ids,
            episodes=args.trace_episodes,
            steps=args.steps,
            seed=args.seed,
            output_dir=trace_eval_root / suite,
            trace_dir=trace_dir / suite,
            trace_max_rows_per_shard=args.trace_max_rows_per_shard,
            device=args.device,
            dtype=args.dtype,
            policy_kind=args.policy_kind,
            policy=args.policy,
            num_inference_steps=args.num_inference_steps,
            smooth_position_delta=args.smooth_position_delta,
            smooth_rotation_delta=args.smooth_rotation_delta,
        )
        for suite in suites
    ]
    sweep_command = build_sweep_command(
        python=args.python,
        sweep_script=args.sweep_script,
        trace_dir=trace_dir,
        output=sweep_json,
        seed=args.seed,
        heldout_split=args.heldout_split,
        heldout_val_fraction=args.heldout_val_fraction,
        lookaheads=args.lookaheads,
        action_dims=args.action_dims,
        max_periods=args.max_periods,
        min_period_repeats=args.min_period_repeats,
        repeat_token_min_runs=args.repeat_token_min_runs,
        second_order_action_extrapolation=args.second_order_action_extrapolation,
        second_order_max_accels=args.second_order_max_accels,
        action_trend_regression=args.action_trend_regression,
        action_trend_histories=args.action_trend_histories,
        action_trend_top_ks=args.action_trend_top_ks,
        action_trend_max_abs_values=args.action_trend_max_abs_values,
        action_prefix_lookup=args.action_prefix_lookup,
        action_prefix_history_sizes=args.action_prefix_history_sizes,
        action_prefix_top_ks=args.action_prefix_top_ks,
        action_prefix_min_prefixes=args.action_prefix_min_prefixes,
        action_prefix_max_mismatches_values=args.action_prefix_max_mismatches_values,
        action_vector_suffix_lookup=args.action_vector_suffix_lookup,
        action_vector_suffix_history_sizes=args.action_vector_suffix_history_sizes,
        action_vector_suffix_top_ks=args.action_vector_suffix_top_ks,
        action_vector_suffix_min_prefixes=args.action_vector_suffix_min_prefixes,
        action_vector_suffix_min_counts=args.action_vector_suffix_min_counts,
        action_vector_suffix_max_prefix_delta_values=args.action_vector_suffix_max_prefix_delta_values,
        action_vector_transition=args.action_vector_transition,
        action_vector_transition_history_sizes=args.action_vector_transition_history_sizes,
        action_vector_transition_top_ks=args.action_vector_transition_top_ks,
        action_vector_transition_min_counts=args.action_vector_transition_min_counts,
        action_vector_transition_max_prev_delta_values=args.action_vector_transition_max_prev_delta_values,
        action_vector_transition_max_prefix_delta_values=args.action_vector_transition_max_prefix_delta_values,
        action_repeat_vector=args.action_repeat_vector,
        action_repeat_min_repeats_values=args.action_repeat_min_repeats_values,
        action_repeat_max_delta_values=args.action_repeat_max_delta_values,
        chunk_length_stop=args.chunk_length_stop,
        chunk_length_stop_history_sizes=args.chunk_length_stop_history_sizes,
        chunk_length_stop_min_counts=args.chunk_length_stop_min_counts,
        action_context_tree=args.action_context_tree,
        action_context_tree_history_sizes=args.action_context_tree_history_sizes,
        action_context_tree_max_contexts=args.action_context_tree_max_contexts,
        action_context_tree_top_ks=args.action_context_tree_top_ks,
        action_context_tree_min_counts=args.action_context_tree_min_counts,
        action_transition_histogram=args.action_transition_histogram,
        action_transition_history_sizes=args.action_transition_history_sizes,
        action_transition_top_ks=args.action_transition_top_ks,
        action_delta_histogram=args.action_delta_histogram,
        action_delta_histories=args.action_delta_histories,
        action_delta_top_ks=args.action_delta_top_ks,
        action_delta_min_counts=args.action_delta_min_counts,
        action_delta_max_abs_values=args.action_delta_max_abs_values,
        action_delta_ngram=args.action_delta_ngram,
        action_delta_ngram_history_sizes=args.action_delta_ngram_history_sizes,
        action_delta_ngram_min_contexts=args.action_delta_ngram_min_contexts,
        action_delta_ngram_max_contexts=args.action_delta_ngram_max_contexts,
        action_delta_ngram_top_ks=args.action_delta_ngram_top_ks,
        action_delta_ngram_min_counts=args.action_delta_ngram_min_counts,
        action_delta_ngram_max_abs_values=args.action_delta_ngram_max_abs_values,
        chunk_position_delta=args.chunk_position_delta,
        chunk_position_delta_history_sizes=args.chunk_position_delta_history_sizes,
        chunk_position_delta_top_ks=args.chunk_position_delta_top_ks,
        chunk_position_delta_min_counts=args.chunk_position_delta_min_counts,
        chunk_position_delta_max_abs_values=args.chunk_position_delta_max_abs_values,
        chunk_delta_template=args.chunk_delta_template,
        chunk_delta_template_history_sizes=args.chunk_delta_template_history_sizes,
        chunk_delta_template_top_ks=args.chunk_delta_template_top_ks,
        chunk_delta_template_min_prefix_deltas=args.chunk_delta_template_min_prefix_deltas,
        chunk_delta_template_max_delta_mismatch_values=args.chunk_delta_template_max_delta_mismatch_values,
        chunk_delta_template_max_abs_values=args.chunk_delta_template_max_abs_values,
        previous_chunk_position=args.previous_chunk_position,
        previous_chunk_history_sizes=args.previous_chunk_history_sizes,
        chunk_prefix_retrieval=args.chunk_prefix_retrieval,
        chunk_prefix_history_sizes=args.chunk_prefix_history_sizes,
        chunk_prefix_top_ks=args.chunk_prefix_top_ks,
        chunk_prefix_min_matches=args.chunk_prefix_min_matches,
        chunk_prefix_max_mismatches=args.chunk_prefix_max_mismatches,
        action_token_neighborhood=args.action_token_neighborhood,
        action_token_neighborhood_radii=args.action_token_neighborhood_radii,
        action_token_neighborhood_top_ks=args.action_token_neighborhood_top_ks,
        position_mode_histogram=args.position_mode_histogram,
        position_mode_history_sizes=args.position_mode_history_sizes,
        position_mode_top_ks=args.position_mode_top_ks,
        position_mode_min_counts=args.position_mode_min_counts,
        global_position_mode=args.global_position_mode,
        global_position_history_sizes=args.global_position_history_sizes,
        global_position_top_ks=args.global_position_top_ks,
        global_position_min_counts=args.global_position_min_counts,
        action_dimension_mode=args.action_dimension_mode,
        action_dimension_mode_history_sizes=args.action_dimension_mode_history_sizes,
        action_dimension_mode_top_ks=args.action_dimension_mode_top_ks,
        action_dimension_mode_min_counts=args.action_dimension_mode_min_counts,
        hold_action_token=args.hold_action_token,
        ngram_continuation=args.ngram_continuation,
        ngram_min_contexts=args.ngram_min_contexts,
        ngram_max_contexts=args.ngram_max_contexts,
        ngram_history_sizes=args.ngram_history_sizes,
        min_source_agreements=args.min_source_agreements,
        source_priority_modes=args.source_priority_modes,
        source_cooldown=args.source_cooldown,
        source_cooldown_afters=args.source_cooldown_afters,
        source_cooldown_steps=args.source_cooldown_steps,
        source_acceptance_bias=args.source_acceptance_bias,
        source_acceptance_bias_history_sizes=args.source_acceptance_bias_history_sizes,
        source_acceptance_bias_min_observations=args.source_acceptance_bias_min_observations,
        reuse_full_blocks=args.reuse_full_blocks,
        emit_bonus_token=args.emit_bonus_token,
        defer_correction_token=args.defer_correction_token,
        dynamic_lookahead=args.dynamic_lookahead,
        min_lookaheads=args.min_lookaheads,
        lookahead_growths=args.lookahead_growths,
        lookahead_shrinks=args.lookahead_shrinks,
        history_reset=args.history_reset,
        tree_widths=args.tree_widths,
        tree_branch_widths=args.tree_branch_widths,
        dynamic_tree_width=args.dynamic_tree_width,
        min_tree_widths=args.min_tree_widths,
        tree_width_growths=args.tree_width_growths,
        tree_width_shrinks=args.tree_width_shrinks,
        tree_anchor_target_token=args.tree_anchor_target_token,
        tree_anchor_target_continuation=args.tree_anchor_target_continuation,
        stop_token_ids=args.stop_token_ids,
        target_forward_ms=args.target_forward_ms,
        draft_token_ms=args.draft_token_ms,
        top_k=args.top_k,
        max_enabled_sources=args.max_enabled_sources,
        max_configs=args.max_sweep_configs,
    )
    gate_command = build_gate_command(
        python=args.python,
        gate_runner_script=args.gate_runner_script,
        root=gate_root,
        suites=args.suites,
        task_ids=task_ids,
        episodes=args.gate_episodes,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        policy_kind=args.policy_kind,
        policy=args.policy,
        num_inference_steps=args.num_inference_steps,
        smooth_position_delta=args.smooth_position_delta,
        smooth_rotation_delta=args.smooth_rotation_delta,
        sweep_json=sweep_json,
        pattern_sweep_rank=args.pattern_sweep_rank,
        pattern_min_modeled_speedup=args.pattern_min_modeled_speedup,
        pattern_min_forward_reduction=args.pattern_min_forward_reduction,
        pattern_min_task_forward_reduction=args.pattern_min_task_forward_reduction,
        pattern_min_task_acceptance_rate=args.pattern_min_task_acceptance_rate,
        pattern_min_task_count=resolved_pattern_min_task_count,
        pattern_min_heldout_modeled_speedup=args.pattern_min_heldout_modeled_speedup,
        pattern_min_heldout_forward_reduction=args.pattern_min_heldout_forward_reduction,
        pattern_min_heldout_task_forward_reduction=args.pattern_min_heldout_task_forward_reduction,
        pattern_min_heldout_task_acceptance_rate=args.pattern_min_heldout_task_acceptance_rate,
        pattern_min_heldout_task_count=resolved_pattern_min_heldout_task_count,
        pattern_min_metric=args.pattern_min_metric,
        pattern_min_heldout_metric=args.pattern_min_heldout_metric,
        pattern_auto_source_min_metrics=args.pattern_auto_source_min_metrics,
        pattern_required_source_coverage=pattern_required_source_coverage,
        min_pairs=args.min_pairs,
        min_speedup=args.min_speedup,
        max_success_drop=args.max_success_drop,
        max_baseline_success_regressions=args.max_baseline_success_regressions,
        min_validation_episodes=args.min_validation_episodes,
        min_exact_verifies=args.min_exact_verifies,
        max_action_diff=args.max_action_diff,
        gate_dry_run=args.gate_dry_run,
    )
    return {
        "root": str(root),
        "suites": suites,
        "task_ids": task_ids,
        "trace_dir": str(trace_dir),
        "trace_eval_root": str(trace_eval_root),
        "sweep_json": str(sweep_json),
        "gate_root": str(gate_root),
        "early_stop_reference": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_gate_eval_count": len(suites) * len(parse_csv(task_ids)) * args.gate_episodes,
        "trace_commands": [] if args.skip_trace_collection else trace_commands,
        "sweep_command": None if args.skip_sweep else sweep_command,
        "gate_command": None if args.skip_gate_manifest else gate_command,
        "gate_dry_run": args.gate_dry_run,
        "thresholds": {
            "min_pairs": args.min_pairs,
            "min_speedup": args.min_speedup,
            "max_success_drop": args.max_success_drop,
            "max_baseline_success_regressions": args.max_baseline_success_regressions,
            "pattern_min_modeled_speedup": args.pattern_min_modeled_speedup,
            "pattern_min_forward_reduction": args.pattern_min_forward_reduction,
            "pattern_min_task_forward_reduction": args.pattern_min_task_forward_reduction,
            "pattern_min_task_acceptance_rate": args.pattern_min_task_acceptance_rate,
            "pattern_min_task_count": resolved_pattern_min_task_count,
            "pattern_min_heldout_modeled_speedup": args.pattern_min_heldout_modeled_speedup,
            "pattern_min_heldout_forward_reduction": args.pattern_min_heldout_forward_reduction,
            "pattern_min_heldout_task_forward_reduction": args.pattern_min_heldout_task_forward_reduction,
            "pattern_min_heldout_task_acceptance_rate": args.pattern_min_heldout_task_acceptance_rate,
            "pattern_min_heldout_task_count": resolved_pattern_min_heldout_task_count,
            "planned_task_count": len(suites) * len(parse_csv(task_ids)),
            "heldout_val_fraction": args.heldout_val_fraction,
            "pattern_auto_source_min_metrics": args.pattern_auto_source_min_metrics,
            "pattern_required_source_coverage": pattern_required_source_coverage,
        },
        "sweep_budget": {
            "max_enabled_sources": args.max_enabled_sources,
            "max_configs": args.max_sweep_configs,
        },
    }


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/pi0fast_pattern_candidate"))
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--trace-eval-root", type=Path, default=None)
    parser.add_argument("--sweep-output", type=Path, default=None)
    parser.add_argument("--gate-root", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval-script", default="scripts/run_pi0fast_chunk_eval.py")
    parser.add_argument("--sweep-script", default="scripts/sweep_pi0fast_pattern_offline.py")
    parser.add_argument("--gate-runner-script", default="scripts/run_pi0fast_100_eval_gate.py")
    parser.add_argument("--suites", default="libero_object,libero_spatial,libero_goal")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--trace-episodes", type=int, default=4)
    parser.add_argument("--gate-episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--policy-kind", choices=["pi0fast", "pi05"], default="pi0fast")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--smooth-position-delta", type=float, default=0.06)
    parser.add_argument("--smooth-rotation-delta", type=float, default=0.22)
    parser.add_argument("--trace-max-rows-per-shard", type=int, default=512)
    parser.add_argument("--heldout-split", choices=["none", "trace", "task", "seed", "task_seed"], default="task")
    parser.add_argument(
        "--heldout-val-fraction",
        type=float,
        default=0.2,
        help="Heldout trace or group fraction for the offline sweep.",
    )
    parser.add_argument("--lookaheads", default="4,8")
    parser.add_argument("--action-dims", default="7")
    parser.add_argument("--max-periods", default="8")
    parser.add_argument("--min-period-repeats", default="2")
    parser.add_argument("--repeat-token-min-runs", default="2")
    parser.add_argument("--second-order-action-extrapolation", default="both")
    parser.add_argument("--second-order-max-accels", default="4")
    parser.add_argument("--action-trend-regression", default="both")
    parser.add_argument("--action-trend-histories", default="4")
    parser.add_argument("--action-trend-top-ks", default="3")
    parser.add_argument("--action-trend-max-abs-values", default="8")
    parser.add_argument("--action-prefix-lookup", default="both")
    parser.add_argument("--action-prefix-history-sizes", default="4")
    parser.add_argument("--action-prefix-top-ks", default="3")
    parser.add_argument("--action-prefix-min-prefixes", default="1")
    parser.add_argument("--action-prefix-max-mismatches-values", default="0")
    parser.add_argument("--action-vector-suffix-lookup", default="both")
    parser.add_argument("--action-vector-suffix-history-sizes", default="4")
    parser.add_argument("--action-vector-suffix-top-ks", default="3")
    parser.add_argument("--action-vector-suffix-min-prefixes", default="1,2")
    parser.add_argument("--action-vector-suffix-min-counts", default="1,2")
    parser.add_argument("--action-vector-suffix-max-prefix-delta-values", default="0")
    parser.add_argument("--action-vector-transition", default="both")
    parser.add_argument("--action-vector-transition-history-sizes", default="4")
    parser.add_argument("--action-vector-transition-top-ks", default="3")
    parser.add_argument("--action-vector-transition-min-counts", default="1,2")
    parser.add_argument("--action-vector-transition-max-prev-delta-values", default="0")
    parser.add_argument("--action-vector-transition-max-prefix-delta-values", default="0")
    parser.add_argument("--action-repeat-vector", default="both")
    parser.add_argument("--action-repeat-min-repeats-values", default="2")
    parser.add_argument("--action-repeat-max-delta-values", default="0")
    parser.add_argument("--chunk-length-stop", default="both")
    parser.add_argument("--chunk-length-stop-history-sizes", default="4")
    parser.add_argument("--chunk-length-stop-min-counts", default="2")
    parser.add_argument("--action-context-tree", default="both")
    parser.add_argument("--action-context-tree-history-sizes", default="4")
    parser.add_argument("--action-context-tree-max-contexts", default="2")
    parser.add_argument("--action-context-tree-top-ks", default="2")
    parser.add_argument("--action-context-tree-min-counts", default="1")
    parser.add_argument("--action-transition-histogram", default="both")
    parser.add_argument("--action-transition-history-sizes", default="2")
    parser.add_argument("--action-transition-top-ks", default="2")
    parser.add_argument("--action-delta-histogram", default="both")
    parser.add_argument("--action-delta-histories", default="4")
    parser.add_argument("--action-delta-top-ks", default="2")
    parser.add_argument("--action-delta-min-counts", default="1,2")
    parser.add_argument("--action-delta-max-abs-values", default="8")
    parser.add_argument("--action-delta-ngram", default="both")
    parser.add_argument("--action-delta-ngram-history-sizes", default="4")
    parser.add_argument("--action-delta-ngram-min-contexts", default="1")
    parser.add_argument("--action-delta-ngram-max-contexts", default="2")
    parser.add_argument("--action-delta-ngram-top-ks", default="2")
    parser.add_argument("--action-delta-ngram-min-counts", default="1")
    parser.add_argument("--action-delta-ngram-max-abs-values", default="8")
    parser.add_argument("--chunk-position-delta", default="both")
    parser.add_argument("--chunk-position-delta-history-sizes", default="4")
    parser.add_argument("--chunk-position-delta-top-ks", default="3")
    parser.add_argument("--chunk-position-delta-min-counts", default="1,2")
    parser.add_argument("--chunk-position-delta-max-abs-values", default="12")
    parser.add_argument("--chunk-delta-template", default="both")
    parser.add_argument("--chunk-delta-template-history-sizes", default="4")
    parser.add_argument("--chunk-delta-template-top-ks", default="3")
    parser.add_argument("--chunk-delta-template-min-prefix-deltas", default="1")
    parser.add_argument("--chunk-delta-template-max-delta-mismatch-values", default="0")
    parser.add_argument("--chunk-delta-template-max-abs-values", default="12")
    parser.add_argument("--previous-chunk-position", default="both")
    parser.add_argument("--previous-chunk-history-sizes", default="1,2")
    parser.add_argument("--chunk-prefix-retrieval", default="both")
    parser.add_argument("--chunk-prefix-history-sizes", default="4")
    parser.add_argument("--chunk-prefix-top-ks", default="3")
    parser.add_argument("--chunk-prefix-min-matches", default="2")
    parser.add_argument("--chunk-prefix-max-mismatches", default="1")
    parser.add_argument("--action-token-neighborhood", default="false")
    parser.add_argument("--action-token-neighborhood-radii", default="1")
    parser.add_argument("--action-token-neighborhood-top-ks", default="3")
    parser.add_argument("--position-mode-histogram", default="false")
    parser.add_argument("--position-mode-history-sizes", default="2")
    parser.add_argument("--position-mode-top-ks", default="2")
    parser.add_argument("--position-mode-min-counts", default="2")
    parser.add_argument("--global-position-mode", default="both")
    parser.add_argument("--global-position-history-sizes", default="16")
    parser.add_argument("--global-position-top-ks", default="3")
    parser.add_argument("--global-position-min-counts", default="3")
    parser.add_argument("--action-dimension-mode", default="both")
    parser.add_argument("--action-dimension-mode-history-sizes", default="2")
    parser.add_argument("--action-dimension-mode-top-ks", default="2")
    parser.add_argument("--action-dimension-mode-min-counts", default="2")
    parser.add_argument("--hold-action-token", default="false")
    parser.add_argument("--ngram-continuation", default="both")
    parser.add_argument("--ngram-min-contexts", default="2")
    parser.add_argument("--ngram-max-contexts", default="8")
    parser.add_argument("--ngram-history-sizes", default="4")
    parser.add_argument("--min-source-agreements", default="1,2")
    parser.add_argument("--source-priority-modes", default="smooth_first")
    parser.add_argument("--source-cooldown", default="both")
    parser.add_argument("--source-cooldown-afters", default="1")
    parser.add_argument("--source-cooldown-steps", default="1,2")
    parser.add_argument("--source-acceptance-bias", default="both")
    parser.add_argument("--source-acceptance-bias-history-sizes", default="32")
    parser.add_argument("--source-acceptance-bias-min-observations", default="1")
    parser.add_argument("--reuse-full-blocks", default="true")
    parser.add_argument("--emit-bonus-token", default="both")
    parser.add_argument("--defer-correction-token", default="false")
    parser.add_argument("--dynamic-lookahead", default="both")
    parser.add_argument("--min-lookaheads", default="1,2")
    parser.add_argument("--lookahead-growths", default="1")
    parser.add_argument("--lookahead-shrinks", default="2,4")
    parser.add_argument("--history-reset", choices=["none", "task", "task_seed"], default="task_seed")
    parser.add_argument("--tree-widths", default="1,4")
    parser.add_argument("--tree-branch-widths", default="4")
    parser.add_argument("--dynamic-tree-width", default="both")
    parser.add_argument("--min-tree-widths", default="1")
    parser.add_argument("--tree-width-growths", default="1")
    parser.add_argument("--tree-width-shrinks", default="1")
    parser.add_argument("--tree-anchor-target-token", default="both")
    parser.add_argument("--tree-anchor-target-continuation", default="both")
    parser.add_argument("--stop-token-ids", default="")
    parser.add_argument("--target-forward-ms", type=float, default=1.0)
    parser.add_argument("--draft-token-ms", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--max-enabled-sources",
        type=int,
        default=4,
        help=(
            "Maximum optional proposal-source priors enabled per offline sweep row. "
            "Use 0 to disable the cap for exhaustive research sweeps."
        ),
    )
    parser.add_argument(
        "--max-sweep-configs",
        type=int,
        default=4096,
        help="Maximum valid offline sweep configs to evaluate after pruning. Use 0 for no cap.",
    )
    parser.add_argument(
        "--pattern-sweep-rank",
        type=int,
        default=0,
        help=(
            "1-based row rank from the offline sweep top list. Defaults to 0, "
            "which lets the gate runner auto-select the first row that passes thresholds."
        ),
    )
    parser.add_argument("--pattern-min-modeled-speedup", type=float, default=2.0)
    parser.add_argument("--pattern-min-forward-reduction", type=float, default=2.0)
    parser.add_argument("--pattern-min-task-forward-reduction", type=float, default=1.25)
    parser.add_argument("--pattern-min-task-acceptance-rate", type=float, default=0.30)
    parser.add_argument(
        "--pattern-min-task-count",
        type=int,
        default=None,
        help="Minimum train task groups in the selected sweep row. Defaults to the planned suite x task-id count.",
    )
    parser.add_argument("--pattern-min-heldout-modeled-speedup", type=float, default=1.10)
    parser.add_argument("--pattern-min-heldout-forward-reduction", type=float, default=1.10)
    parser.add_argument("--pattern-min-heldout-task-forward-reduction", type=float, default=1.10)
    parser.add_argument("--pattern-min-heldout-task-acceptance-rate", type=float, default=0.20)
    parser.add_argument(
        "--pattern-min-heldout-task-count",
        type=int,
        default=None,
        help=(
            "Minimum heldout task groups in the selected sweep row. Defaults to the planned "
            "suite x task-id count times --heldout-val-fraction when heldout is enabled."
        ),
    )
    parser.add_argument("--pattern-min-metric", action="append", default=[])
    parser.add_argument("--pattern-min-heldout-metric", action="append", default=[])
    parser.add_argument(
        "--pattern-auto-source-min-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass auto source accepted-token thresholds to the 120-task gate runner.",
    )
    parser.add_argument(
        "--pattern-required-source-coverage",
        default="auto",
        help=(
            "Comma-separated source families that the sweep JSON must report as evaluated before "
            "the gate can select a row. Use auto to derive from enabled sweep modes, or none to disable."
        ),
    )
    parser.add_argument("--min-pairs", type=int, default=120)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--max-success-drop", type=float, default=0.0)
    parser.add_argument("--max-baseline-success-regressions", type=int, default=0)
    parser.add_argument("--min-validation-episodes", type=int, default=120)
    parser.add_argument("--min-exact-verifies", type=int, default=1)
    parser.add_argument("--max-action-diff", type=float, default=0.0)
    parser.add_argument("--gate-dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-trace-collection", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-gate-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "pattern_candidate_pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Gate matched eval count: {manifest['matched_gate_eval_count']}")
    print("Early-stop reference: target_eos")

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    for cmd in manifest["trace_commands"]:
        run_command(cmd, env=env, dry_run=args.dry_run)
    if manifest["sweep_command"] is not None:
        run_command(manifest["sweep_command"], env=env, dry_run=args.dry_run)
    if manifest["gate_command"] is not None:
        run_command(manifest["gate_command"], env=env, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run only; no trace, sweep, or gate commands were executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
