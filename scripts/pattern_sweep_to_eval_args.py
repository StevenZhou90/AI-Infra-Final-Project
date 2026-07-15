#!/usr/bin/env python3
"""Convert a pattern offline sweep row into PI0-FAST eval CLI arguments."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


SOURCE_USAGE_METRIC_PREFIXES = (
    ("second_order_action_extrapolation", "second_order_action_extrapolation"),
    ("previous_chunk_position", "previous_chunk_position"),
    ("chunk_prefix_retrieval", "chunk_prefix_retrieval"),
    ("chunk_length_stop", "chunk_length_stop"),
    ("action_token_neighborhood", "action_token_neighborhood"),
    ("position_mode_histogram", "position_mode_histogram"),
    ("global_position_mode", "global_position_mode"),
    ("action_dimension_mode", "action_dimension_mode"),
    ("hold_action_token", "hold_action_token"),
    ("ngram_continuation", "ngram_continuation"),
    ("action_trend_regression", "action_trend_regression"),
    ("action_prefix_lookup", "action_prefix_lookup"),
    ("action_vector_suffix_lookup", "action_vector_suffix_lookup"),
    ("action_vector_transition", "action_vector_transition"),
    ("action_repeat_vector", "action_repeat_vector"),
    ("action_context_tree", "action_context_tree"),
    ("action_transition_histogram", "action_transition_histogram"),
    ("action_delta_histogram", "action_delta_histogram"),
    ("action_delta_ngram", "action_delta_ngram"),
    ("chunk_delta_template", "chunk_delta_template"),
    ("chunk_position_delta", "chunk_position_delta"),
)
TREE_USAGE_CONFIG_KEYS = (
    "tree_anchor_target_token",
    "tree_anchor_target_continuation",
)


def load_sweep(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text())
    if "top" not in row or not row["top"]:
        raise ValueError(f"{path} does not contain a non-empty top sweep list")
    return row


def select_sweep_row(summary: dict[str, Any], *, rank: int) -> dict[str, Any]:
    if rank < 1:
        raise ValueError("--rank is 1-based and must be >= 1")
    rows = summary.get("top") or []
    if rank > len(rows):
        raise ValueError(f"--rank {rank} exceeds available sweep rows ({len(rows)})")
    return rows[rank - 1]


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    merged = list(values)
    seen = set(merged)
    for item in additions:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def _normalize_source_names(values: list[str] | None) -> list[str]:
    if not values:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            source = part.strip()
            if not source or source in seen:
                continue
            names.append(source)
            seen.add(source)
    return names


def validate_required_source_coverage(summary: dict[str, Any], required_sources: list[str] | None) -> dict[str, int]:
    required = _normalize_source_names(required_sources)
    if not required:
        return {}
    counts = summary.get("evaluated_source_counts")
    coverage = summary.get("evaluated_source_coverage")
    if not isinstance(counts, dict) and not isinstance(coverage, list):
        raise ValueError(
            "sweep summary is missing evaluated_source_counts/evaluated_source_coverage; "
            "rerun sweep_pi0fast_pattern_offline.py with coverage-aware output"
        )
    coverage_set = {str(source) for source in coverage or []}
    normalized_counts: dict[str, int] = {}
    missing: list[str] = []
    for source in required:
        count = 0
        if isinstance(counts, dict) and source in counts:
            count = int(counts.get(source, 0))
        elif source in coverage_set:
            count = 1
        normalized_counts[source] = count
        if count <= 0:
            missing.append(source)
    if missing:
        raise ValueError(
            "sweep summary did not evaluate required source coverage: "
            + ", ".join(missing)
        )
    return normalized_counts


def pattern_eval_args_from_config(config: dict[str, Any]) -> list[str]:
    required = [
        "lookahead",
        "action_dim",
        "max_period",
        "min_period_repeats",
        "repeat_token_min_run",
        "linear_action_extrapolation",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Sweep row config is missing required keys: {', '.join(missing)}")
    args = [
        "--pattern-lookahead",
        str(int(config["lookahead"])),
        "--pattern-action-dim",
        str(int(config["action_dim"])),
        "--pattern-max-period",
        str(int(config["max_period"])),
        "--pattern-min-period-repeats",
        str(int(config["min_period_repeats"])),
        "--pattern-repeat-token-min-run",
        str(int(config["repeat_token_min_run"])),
    ]
    if bool(config["linear_action_extrapolation"]):
        args.append("--pattern-linear-action-extrapolation")
    else:
        args.append("--no-pattern-linear-action-extrapolation")
    if bool(config.get("second_order_action_extrapolation", False)):
        max_accel = config.get("second_order_max_accel", 8)
        args.extend(
            [
                "--pattern-second-order-action-extrapolation",
                "--pattern-second-order-max-accel",
                "-1" if max_accel is None else str(int(max_accel)),
            ]
        )
    if bool(config.get("action_trend_regression", False)):
        max_abs = config.get("action_trend_max_abs", 12)
        args.extend(
            [
                "--pattern-action-trend-regression",
                "--pattern-action-trend-history",
                str(int(config.get("action_trend_history", 4))),
                "--pattern-action-trend-top-k",
                str(int(config.get("action_trend_top_k", 3))),
                "--pattern-action-trend-max-abs",
                "-1" if max_abs is None else str(int(max_abs)),
            ]
        )
    if bool(config.get("action_prefix_lookup", False)):
        args.extend(
            [
                "--pattern-action-prefix-lookup",
                "--pattern-action-prefix-history-size",
                str(int(config.get("action_prefix_history_size", 4))),
                "--pattern-action-prefix-top-k",
                str(int(config.get("action_prefix_top_k", 3))),
                "--pattern-action-prefix-min-prefix",
                str(int(config.get("action_prefix_min_prefix", 1))),
                "--pattern-action-prefix-max-mismatches",
                str(int(config.get("action_prefix_max_mismatches", 0))),
            ]
        )
    if bool(config.get("action_vector_suffix_lookup", False)):
        args.extend(
            [
                "--pattern-action-vector-suffix-lookup",
                "--pattern-action-vector-suffix-history-size",
                str(int(config.get("action_vector_suffix_history_size", 4))),
                "--pattern-action-vector-suffix-top-k",
                str(int(config.get("action_vector_suffix_top_k", 3))),
                "--pattern-action-vector-suffix-min-prefix",
                str(int(config.get("action_vector_suffix_min_prefix", 1))),
                "--pattern-action-vector-suffix-min-count",
                str(int(config.get("action_vector_suffix_min_count", 1))),
                "--pattern-action-vector-suffix-max-prefix-delta",
                str(int(config.get("action_vector_suffix_max_prefix_delta", 0))),
            ]
        )
    if bool(config.get("action_vector_transition", False)):
        args.extend(
            [
                "--pattern-action-vector-transition",
                "--pattern-action-vector-transition-history-size",
                str(int(config.get("action_vector_transition_history_size", 4))),
                "--pattern-action-vector-transition-top-k",
                str(int(config.get("action_vector_transition_top_k", 3))),
                "--pattern-action-vector-transition-min-count",
                str(int(config.get("action_vector_transition_min_count", 1))),
                "--pattern-action-vector-transition-max-prev-delta",
                str(int(config.get("action_vector_transition_max_prev_delta", 0))),
                "--pattern-action-vector-transition-max-prefix-delta",
                str(int(config.get("action_vector_transition_max_prefix_delta", 0))),
            ]
        )
    if bool(config.get("action_repeat_vector", False)):
        args.extend(
            [
                "--pattern-action-repeat-vector",
                "--pattern-action-repeat-min-repeats",
                str(int(config.get("action_repeat_min_repeats", 2))),
                "--pattern-action-repeat-max-delta",
                str(int(config.get("action_repeat_max_delta", 0))),
            ]
        )
    if bool(config.get("chunk_length_stop", False)):
        args.extend(
            [
                "--pattern-chunk-length-stop",
                "--pattern-chunk-length-stop-history-size",
                str(int(config.get("chunk_length_stop_history_size", 4))),
                "--pattern-chunk-length-stop-min-count",
                str(int(config.get("chunk_length_stop_min_count", 2))),
            ]
        )
    if bool(config.get("action_transition_histogram", False)):
        args.extend(
            [
                "--pattern-action-transition-histogram",
                "--pattern-action-transition-history-size",
                str(int(config.get("action_transition_history_size", 4))),
                "--pattern-action-transition-top-k",
                str(int(config.get("action_transition_top_k", 3))),
                "--pattern-action-transition-min-count",
                str(int(config.get("action_transition_min_count", 1))),
            ]
        )
    if bool(config.get("action_delta_histogram", False)):
        max_abs = config.get("action_delta_max_abs", 12)
        args.extend(
            [
                "--pattern-action-delta-histogram",
                "--pattern-action-delta-history",
                str(int(config.get("action_delta_history", 8))),
                "--pattern-action-delta-top-k",
                str(int(config.get("action_delta_top_k", 3))),
                "--pattern-action-delta-min-count",
                str(int(config.get("action_delta_min_count", 1))),
                "--pattern-action-delta-max-abs",
                "-1" if max_abs is None else str(int(max_abs)),
            ]
        )
    if bool(config.get("action_delta_ngram", False)):
        max_abs = config.get("action_delta_ngram_max_abs", 12)
        args.extend(
            [
                "--pattern-action-delta-ngram",
                "--pattern-action-delta-ngram-history-size",
                str(int(config.get("action_delta_ngram_history_size", 4))),
                "--pattern-action-delta-ngram-min-context",
                str(int(config.get("action_delta_ngram_min_context", 1))),
                "--pattern-action-delta-ngram-max-context",
                str(int(config.get("action_delta_ngram_max_context", 4))),
                "--pattern-action-delta-ngram-top-k",
                str(int(config.get("action_delta_ngram_top_k", 3))),
                "--pattern-action-delta-ngram-min-count",
                str(int(config.get("action_delta_ngram_min_count", 1))),
                "--pattern-action-delta-ngram-max-abs",
                "-1" if max_abs is None else str(int(max_abs)),
            ]
        )
    if bool(config.get("chunk_position_delta", False)):
        max_abs = config.get("chunk_position_delta_max_abs", 12)
        args.extend(
            [
                "--pattern-chunk-position-delta",
                "--pattern-chunk-position-delta-history-size",
                str(int(config.get("chunk_position_delta_history_size", 4))),
                "--pattern-chunk-position-delta-top-k",
                str(int(config.get("chunk_position_delta_top_k", 3))),
                "--pattern-chunk-position-delta-min-count",
                str(int(config.get("chunk_position_delta_min_count", 1))),
                "--pattern-chunk-position-delta-max-abs",
                "-1" if max_abs is None else str(int(max_abs)),
            ]
        )
    if bool(config.get("chunk_delta_template", False)):
        max_delta_mismatch = config.get("chunk_delta_template_max_delta_mismatch", 0)
        max_abs = config.get("chunk_delta_template_max_abs", 12)
        args.extend(
            [
                "--pattern-chunk-delta-template",
                "--pattern-chunk-delta-template-history-size",
                str(int(config.get("chunk_delta_template_history_size", 4))),
                "--pattern-chunk-delta-template-top-k",
                str(int(config.get("chunk_delta_template_top_k", 3))),
                "--pattern-chunk-delta-template-min-prefix-deltas",
                str(int(config.get("chunk_delta_template_min_prefix_deltas", 1))),
                "--pattern-chunk-delta-template-max-delta-mismatch",
                "-1" if max_delta_mismatch is None else str(int(max_delta_mismatch)),
                "--pattern-chunk-delta-template-max-abs",
                "-1" if max_abs is None else str(int(max_abs)),
            ]
        )
    if bool(config.get("previous_chunk_position", False)):
        args.extend(
            [
                "--pattern-previous-chunk-position",
                "--pattern-previous-chunk-history-size",
                str(int(config.get("previous_chunk_history_size", 1))),
            ]
        )
    if bool(config.get("chunk_prefix_retrieval", False)):
        args.extend(
            [
                "--pattern-chunk-prefix-retrieval",
                "--pattern-chunk-prefix-history-size",
                str(int(config.get("chunk_prefix_history_size", 4))),
                "--pattern-chunk-prefix-top-k",
                str(int(config.get("chunk_prefix_top_k", 3))),
                "--pattern-chunk-prefix-min-matches",
                str(int(config.get("chunk_prefix_min_matches", 2))),
                "--pattern-chunk-prefix-max-mismatches",
                str(int(config.get("chunk_prefix_max_mismatches", 1))),
            ]
        )
    if bool(config.get("action_token_neighborhood", False)):
        args.extend(
            [
                "--pattern-action-token-neighborhood",
                "--pattern-action-token-neighborhood-radius",
                str(int(config.get("action_token_neighborhood_radius", 1))),
                "--pattern-action-token-neighborhood-top-k",
                str(int(config.get("action_token_neighborhood_top_k", 3))),
            ]
        )
    if bool(config.get("position_mode_histogram", False)):
        args.extend(
            [
                "--pattern-position-mode-histogram",
                "--pattern-position-mode-history-size",
                str(int(config.get("position_mode_history_size", 4))),
                "--pattern-position-mode-top-k",
                str(int(config.get("position_mode_top_k", 3))),
                "--pattern-position-mode-min-count",
                str(int(config.get("position_mode_min_count", 2))),
            ]
        )
    if bool(config.get("global_position_mode", False)):
        args.extend(
            [
                "--pattern-global-position-mode",
                "--pattern-global-position-history-size",
                str(int(config.get("global_position_history_size", 16))),
                "--pattern-global-position-top-k",
                str(int(config.get("global_position_top_k", 3))),
                "--pattern-global-position-min-count",
                str(int(config.get("global_position_min_count", 3))),
            ]
        )
    if bool(config.get("action_dimension_mode", False)):
        args.extend(
            [
                "--pattern-action-dimension-mode",
                "--pattern-action-dimension-mode-history-size",
                str(int(config.get("action_dimension_mode_history_size", 4))),
                "--pattern-action-dimension-mode-top-k",
                str(int(config.get("action_dimension_mode_top_k", 3))),
                "--pattern-action-dimension-mode-min-count",
                str(int(config.get("action_dimension_mode_min_count", 2))),
            ]
        )
    if bool(config.get("hold_action_token", False)):
        args.append("--pattern-hold-action-token")
    if bool(config.get("ngram_continuation", False)):
        args.extend(
            [
                "--pattern-ngram-continuation",
                "--pattern-ngram-min-context",
                str(int(config.get("ngram_min_context", 3))),
                "--pattern-ngram-max-context",
                str(int(config.get("ngram_max_context", 16))),
                "--pattern-ngram-history-size",
                str(int(config.get("ngram_history_size", 4))),
            ]
        )
    if bool(config.get("action_context_tree", False)):
        args.extend(
            [
                "--pattern-action-context-tree",
                "--pattern-action-context-tree-history-size",
                str(int(config.get("action_context_tree_history_size", 4))),
                "--pattern-action-context-tree-max-context",
                str(int(config.get("action_context_tree_max_context", 3))),
                "--pattern-action-context-tree-top-k",
                str(int(config.get("action_context_tree_top_k", 3))),
                "--pattern-action-context-tree-min-count",
                str(int(config.get("action_context_tree_min_count", 1))),
            ]
        )
    if "source_priority" in config:
        args.extend(
            [
                "--pattern-source-priority",
                ",".join(str(source) for source in config.get("source_priority", [])),
            ]
        )
    if int(config.get("min_source_agreement", 1)) > 1:
        args.extend(
            [
                "--pattern-min-source-agreement",
                str(int(config.get("min_source_agreement", 1))),
            ]
        )
    if bool(config.get("source_cooldown", False)):
        args.extend(
            [
                "--pattern-source-cooldown",
                "--pattern-source-cooldown-after",
                str(int(config.get("source_cooldown_after", 1))),
                "--pattern-source-cooldown-steps",
                str(int(config.get("source_cooldown_steps", 1))),
            ]
        )
    if bool(config.get("source_acceptance_bias", False)):
        args.extend(
            [
                "--pattern-source-acceptance-bias",
                "--pattern-source-acceptance-bias-history-size",
                str(int(config.get("source_acceptance_bias_history_size", 32))),
                "--pattern-source-acceptance-bias-min-observations",
                str(int(config.get("source_acceptance_bias_min_observations", 2))),
            ]
        )
    if bool(config.get("reuse_full_blocks", False)):
        args.append("--pattern-reuse-full-blocks")
    if bool(config.get("emit_bonus_token", False)):
        args.append("--pattern-emit-bonus-token")
    if bool(config.get("dynamic_lookahead", False)):
        args.extend(
            [
                "--pattern-dynamic-lookahead",
                "--pattern-min-lookahead",
                str(int(config.get("min_lookahead", 1))),
                "--pattern-lookahead-growth",
                str(int(config.get("lookahead_growth", 1))),
                "--pattern-lookahead-shrink",
                str(int(config.get("lookahead_shrink", 4))),
            ]
        )
    if int(config.get("tree_width", 1)) > 1:
        args.extend(
            [
                "--pattern-tree-width",
                str(int(config.get("tree_width", 1))),
                "--pattern-tree-branch-width",
                str(int(config.get("tree_branch_width", 4))),
            ]
        )
        if bool(config.get("dynamic_tree_width", False)):
            args.extend(
                [
                    "--pattern-dynamic-tree-width",
                    "--pattern-min-tree-width",
                    str(int(config.get("min_tree_width", 1))),
                    "--pattern-tree-width-growth",
                    str(int(config.get("tree_width_growth", 1))),
                    "--pattern-tree-width-shrink",
                    str(int(config.get("tree_width_shrink", 1))),
                ]
            )
        if bool(config.get("tree_anchor_target_token", False)):
            args.append("--pattern-tree-anchor-target-token")
        if bool(config.get("tree_anchor_target_continuation", False)):
            args.append("--pattern-tree-anchor-target-continuation")
    return args


def _require_metric(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise ValueError(f"selected sweep row is missing required metric {key!r}")
    return float(row[key])


def _parse_metric_threshold(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise ValueError(f"Metric threshold must have NAME=VALUE form, got {value!r}")
    name, raw_threshold = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Metric threshold must include a metric name, got {value!r}")
    try:
        threshold = float(raw_threshold)
    except ValueError as exc:
        raise ValueError(f"Metric threshold for {name!r} must be numeric, got {raw_threshold!r}") from exc
    return name, threshold


def _check_metric_thresholds(row: dict[str, Any], thresholds: list[str] | None, *, prefix: str = "") -> None:
    for item in thresholds or []:
        key, threshold = _parse_metric_threshold(item)
        metric_key = f"{prefix}{key}"
        value = _require_metric(row, metric_key)
        if value < threshold:
            raise ValueError(
                f"selected {metric_key}={value:.3f} is below required {threshold:.3f}"
            )


def source_usage_metric_thresholds(
    row: dict[str, Any],
    *,
    include_min_task: bool = True,
    include_heldout: bool = True,
) -> tuple[list[str], list[str]]:
    """Return NAME=VALUE thresholds proving enabled optional sources accepted tokens."""

    config = row.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("selected sweep row config must be an object")
    metric_prefixes = [
        metric_prefix
        for config_key, metric_prefix in SOURCE_USAGE_METRIC_PREFIXES
        if bool(config.get(config_key, False))
    ]
    if int(config.get("min_source_agreement", 1)) > 1:
        metric_prefixes.append("source_agreement")
    if any(bool(config.get(key, False)) for key in TREE_USAGE_CONFIG_KEYS):
        metric_prefixes.append("tree_anchor")

    thresholds: list[str] = []
    for metric_prefix in metric_prefixes:
        thresholds.append(f"{metric_prefix}_accepted_tokens=1")
        if include_min_task:
            thresholds.append(f"min_task_{metric_prefix}_accepted_tokens=1")

    heldout_thresholds: list[str] = []
    has_heldout_metrics = any(str(key).startswith("heldout_") for key in row)
    if include_heldout and has_heldout_metrics:
        heldout_thresholds = list(thresholds)
    return thresholds, heldout_thresholds


def effective_metric_thresholds(
    row: dict[str, Any],
    *,
    min_metric: list[str] | None = None,
    min_heldout_metric: list[str] | None = None,
    auto_source_min_metrics: bool = False,
    auto_source_min_task_metrics: bool = True,
    auto_source_heldout_metrics: bool = True,
) -> tuple[list[str], list[str], list[str], list[str]]:
    auto_min_metric: list[str] = []
    auto_min_heldout_metric: list[str] = []
    if auto_source_min_metrics:
        auto_min_metric, auto_min_heldout_metric = source_usage_metric_thresholds(
            row,
            include_min_task=auto_source_min_task_metrics,
            include_heldout=auto_source_heldout_metrics,
        )
    effective_min_metric = _append_unique(min_metric or [], auto_min_metric)
    effective_min_heldout_metric = _append_unique(min_heldout_metric or [], auto_min_heldout_metric)
    return effective_min_metric, effective_min_heldout_metric, auto_min_metric, auto_min_heldout_metric


def _validate_sweep_row(
    row: dict[str, Any],
    *,
    min_modeled_speedup: float | None,
    min_forward_reduction: float | None,
    min_task_forward_reduction: float | None = None,
    min_task_acceptance_rate: float | None = None,
    min_task_count: int | None = None,
    min_heldout_modeled_speedup: float | None = None,
    min_heldout_forward_reduction: float | None = None,
    min_heldout_task_forward_reduction: float | None = None,
    min_heldout_task_acceptance_rate: float | None = None,
    min_heldout_task_count: int | None = None,
    min_metric: list[str] | None = None,
    min_heldout_metric: list[str] | None = None,
) -> None:
    if min_modeled_speedup is not None and float(row.get("modeled_speedup", 0.0)) < min_modeled_speedup:
        raise ValueError(
            f"selected modeled_speedup={float(row.get('modeled_speedup', 0.0)):.3f} "
            f"is below required {min_modeled_speedup:.3f}"
        )
    if min_forward_reduction is not None and float(row.get("target_forward_reduction", 0.0)) < min_forward_reduction:
        raise ValueError(
            f"selected target_forward_reduction={float(row.get('target_forward_reduction', 0.0)):.3f} "
            f"is below required {min_forward_reduction:.3f}"
        )
    if (
        min_task_forward_reduction is not None
        and float(row.get("min_task_target_forward_reduction", 0.0)) < min_task_forward_reduction
    ):
        raise ValueError(
            f"selected min_task_target_forward_reduction="
            f"{float(row.get('min_task_target_forward_reduction', 0.0)):.3f} "
            f"is below required {min_task_forward_reduction:.3f}"
        )
    if min_task_acceptance_rate is not None and float(row.get("min_task_acceptance_rate", 0.0)) < min_task_acceptance_rate:
        raise ValueError(
            f"selected min_task_acceptance_rate={float(row.get('min_task_acceptance_rate', 0.0)):.3f} "
            f"is below required {min_task_acceptance_rate:.3f}"
        )
    if min_task_count is not None and int(row.get("task_count", 0)) < min_task_count:
        raise ValueError(
            f"selected task_count={int(row.get('task_count', 0))} "
            f"is below required {int(min_task_count)}"
        )
    if (
        min_heldout_modeled_speedup is not None
        and _require_metric(row, "heldout_modeled_speedup") < min_heldout_modeled_speedup
    ):
        raise ValueError(
            f"selected heldout_modeled_speedup={_require_metric(row, 'heldout_modeled_speedup'):.3f} "
            f"is below required {min_heldout_modeled_speedup:.3f}"
        )
    if (
        min_heldout_forward_reduction is not None
        and _require_metric(row, "heldout_target_forward_reduction") < min_heldout_forward_reduction
    ):
        raise ValueError(
            f"selected heldout_target_forward_reduction="
            f"{_require_metric(row, 'heldout_target_forward_reduction'):.3f} "
            f"is below required {min_heldout_forward_reduction:.3f}"
        )
    if (
        min_heldout_task_forward_reduction is not None
        and _require_metric(row, "heldout_min_task_target_forward_reduction") < min_heldout_task_forward_reduction
    ):
        raise ValueError(
            f"selected heldout_min_task_target_forward_reduction="
            f"{_require_metric(row, 'heldout_min_task_target_forward_reduction'):.3f} "
            f"is below required {min_heldout_task_forward_reduction:.3f}"
        )
    if (
        min_heldout_task_acceptance_rate is not None
        and _require_metric(row, "heldout_min_task_acceptance_rate") < min_heldout_task_acceptance_rate
    ):
        raise ValueError(
            f"selected heldout_min_task_acceptance_rate="
            f"{_require_metric(row, 'heldout_min_task_acceptance_rate'):.3f} "
            f"is below required {min_heldout_task_acceptance_rate:.3f}"
        )
    if min_heldout_task_count is not None and int(row.get("heldout_task_count", 0)) < min_heldout_task_count:
        raise ValueError(
            f"selected heldout_task_count={int(row.get('heldout_task_count', 0))} "
            f"is below required {int(min_heldout_task_count)}"
        )
    _check_metric_thresholds(row, min_metric)
    _check_metric_thresholds(row, min_heldout_metric, prefix="heldout_")


def _with_selection_metadata(
    row: dict[str, Any],
    *,
    requested_rank: int,
    selected_rank: int,
    effective_min_metric: list[str],
    effective_min_heldout_metric: list[str],
    auto_min_metric: list[str],
    auto_min_heldout_metric: list[str],
    required_source_coverage: list[str],
    required_source_counts: dict[str, int],
) -> dict[str, Any]:
    selected = dict(row)
    selected["requested_sweep_rank"] = requested_rank
    selected["selected_sweep_rank"] = selected_rank
    selected["effective_min_metric"] = list(effective_min_metric)
    selected["effective_min_heldout_metric"] = list(effective_min_heldout_metric)
    selected["auto_source_min_metric"] = list(auto_min_metric)
    selected["auto_source_min_heldout_metric"] = list(auto_min_heldout_metric)
    selected["required_source_coverage"] = list(required_source_coverage)
    selected["required_source_counts"] = dict(required_source_counts)
    return selected


def build_eval_args(
    summary: dict[str, Any],
    *,
    rank: int,
    min_modeled_speedup: float | None,
    min_forward_reduction: float | None,
    min_task_forward_reduction: float | None = None,
    min_task_acceptance_rate: float | None = None,
    min_task_count: int | None = None,
    min_heldout_modeled_speedup: float | None = None,
    min_heldout_forward_reduction: float | None = None,
    min_heldout_task_forward_reduction: float | None = None,
    min_heldout_task_acceptance_rate: float | None = None,
    min_heldout_task_count: int | None = None,
    min_metric: list[str] | None = None,
    min_heldout_metric: list[str] | None = None,
    auto_source_min_metrics: bool = False,
    auto_source_min_task_metrics: bool = True,
    auto_source_heldout_metrics: bool = True,
    required_source_coverage: list[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    required_source_counts = validate_required_source_coverage(summary, required_source_coverage)
    rows = summary.get("top") or []
    if rank == 0:
        failures: list[str] = []
        for selected_rank, row in enumerate(rows, start=1):
            (
                effective_min_metric,
                effective_min_heldout_metric,
                auto_min_metric,
                auto_min_heldout_metric,
            ) = effective_metric_thresholds(
                row,
                min_metric=min_metric,
                min_heldout_metric=min_heldout_metric,
                auto_source_min_metrics=auto_source_min_metrics,
                auto_source_min_task_metrics=auto_source_min_task_metrics,
                auto_source_heldout_metrics=auto_source_heldout_metrics,
            )
            try:
                _validate_sweep_row(
                    row,
                    min_modeled_speedup=min_modeled_speedup,
                    min_forward_reduction=min_forward_reduction,
                    min_task_forward_reduction=min_task_forward_reduction,
                    min_task_acceptance_rate=min_task_acceptance_rate,
                    min_task_count=min_task_count,
                    min_heldout_modeled_speedup=min_heldout_modeled_speedup,
                    min_heldout_forward_reduction=min_heldout_forward_reduction,
                    min_heldout_task_forward_reduction=min_heldout_task_forward_reduction,
                    min_heldout_task_acceptance_rate=min_heldout_task_acceptance_rate,
                    min_heldout_task_count=min_heldout_task_count,
                    min_metric=effective_min_metric,
                    min_heldout_metric=effective_min_heldout_metric,
                )
            except ValueError as exc:
                failures.append(f"rank {selected_rank}: {exc}")
                continue
            selected = _with_selection_metadata(
                row,
                requested_rank=rank,
                selected_rank=selected_rank,
                effective_min_metric=effective_min_metric,
                effective_min_heldout_metric=effective_min_heldout_metric,
                auto_min_metric=auto_min_metric,
                auto_min_heldout_metric=auto_min_heldout_metric,
                required_source_coverage=_normalize_source_names(required_source_coverage),
                required_source_counts=required_source_counts,
            )
            return pattern_eval_args_from_config(row["config"]), selected
        detail = "; ".join(failures[:5])
        if len(failures) > 5:
            detail = f"{detail}; ... {len(failures) - 5} more rows failed"
        raise ValueError(
            "no sweep rows satisfied thresholds for --rank 0 auto-select"
            + (f": {detail}" if detail else "")
        )

    row = select_sweep_row(summary, rank=rank)
    effective_min_metric, effective_min_heldout_metric, auto_min_metric, auto_min_heldout_metric = (
        effective_metric_thresholds(
            row,
            min_metric=min_metric,
            min_heldout_metric=min_heldout_metric,
            auto_source_min_metrics=auto_source_min_metrics,
            auto_source_min_task_metrics=auto_source_min_task_metrics,
            auto_source_heldout_metrics=auto_source_heldout_metrics,
        )
    )
    _validate_sweep_row(
        row,
        min_modeled_speedup=min_modeled_speedup,
        min_forward_reduction=min_forward_reduction,
        min_task_forward_reduction=min_task_forward_reduction,
        min_task_acceptance_rate=min_task_acceptance_rate,
        min_task_count=min_task_count,
        min_heldout_modeled_speedup=min_heldout_modeled_speedup,
        min_heldout_forward_reduction=min_heldout_forward_reduction,
        min_heldout_task_forward_reduction=min_heldout_task_forward_reduction,
        min_heldout_task_acceptance_rate=min_heldout_task_acceptance_rate,
        min_heldout_task_count=min_heldout_task_count,
        min_metric=effective_min_metric,
        min_heldout_metric=effective_min_heldout_metric,
    )
    selected = _with_selection_metadata(
        row,
        requested_rank=rank,
        selected_rank=rank,
        effective_min_metric=effective_min_metric,
        effective_min_heldout_metric=effective_min_heldout_metric,
        auto_min_metric=auto_min_metric,
        auto_min_heldout_metric=auto_min_heldout_metric,
        required_source_coverage=_normalize_source_names(required_source_coverage),
        required_source_counts=required_source_counts,
    )
    return pattern_eval_args_from_config(row["config"]), selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PI0-FAST pattern eval args from a sweep JSON.")
    parser.add_argument("sweep_json", type=Path)
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1-based row rank from the sweep top list. Use 0 to auto-select the first row that passes thresholds.",
    )
    parser.add_argument("--min-modeled-speedup", type=float, default=None)
    parser.add_argument("--min-forward-reduction", type=float, default=None)
    parser.add_argument("--min-task-forward-reduction", type=float, default=None)
    parser.add_argument("--min-task-acceptance-rate", type=float, default=None)
    parser.add_argument("--min-task-count", type=int, default=None)
    parser.add_argument("--min-heldout-modeled-speedup", type=float, default=None)
    parser.add_argument("--min-heldout-forward-reduction", type=float, default=None)
    parser.add_argument("--min-heldout-task-forward-reduction", type=float, default=None)
    parser.add_argument("--min-heldout-task-acceptance-rate", type=float, default=None)
    parser.add_argument("--min-heldout-task-count", type=int, default=None)
    parser.add_argument(
        "--min-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require selected train metric NAME to be at least VALUE, e.g. ngram_continuation_accepted_tokens=1.",
    )
    parser.add_argument(
        "--min-heldout-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require selected heldout metric NAME to be at least VALUE; heldout_ is added automatically.",
    )
    parser.add_argument(
        "--auto-source-min-metrics",
        action="store_true",
        help="Add accepted-token thresholds for enabled optional proposal sources on each candidate row.",
    )
    parser.add_argument(
        "--no-auto-source-min-task-metrics",
        dest="auto_source_min_task_metrics",
        action="store_false",
        default=True,
        help="Do not add per-task accepted-token thresholds when --auto-source-min-metrics is used.",
    )
    parser.add_argument(
        "--no-auto-source-heldout-metrics",
        dest="auto_source_heldout_metrics",
        action="store_false",
        default=True,
        help="Do not mirror auto source thresholds onto heldout metrics.",
    )
    parser.add_argument(
        "--required-source-coverage",
        action="append",
        default=[],
        metavar="SOURCE",
        help=(
            "Require sweep-level evaluated_source_counts[SOURCE] > 0 before selecting a row. "
            "May be repeated or comma-separated."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a shell-escaped arg line.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = load_sweep(args.sweep_json)
    eval_args, row = build_eval_args(
        summary,
        rank=args.rank,
        min_modeled_speedup=args.min_modeled_speedup,
        min_forward_reduction=args.min_forward_reduction,
        min_task_forward_reduction=args.min_task_forward_reduction,
        min_task_acceptance_rate=args.min_task_acceptance_rate,
        min_task_count=args.min_task_count,
        min_heldout_modeled_speedup=args.min_heldout_modeled_speedup,
        min_heldout_forward_reduction=args.min_heldout_forward_reduction,
        min_heldout_task_forward_reduction=args.min_heldout_task_forward_reduction,
        min_heldout_task_acceptance_rate=args.min_heldout_task_acceptance_rate,
        min_heldout_task_count=args.min_heldout_task_count,
        min_metric=args.min_metric,
        min_heldout_metric=args.min_heldout_metric,
        auto_source_min_metrics=args.auto_source_min_metrics,
        auto_source_min_task_metrics=args.auto_source_min_task_metrics,
        auto_source_heldout_metrics=args.auto_source_heldout_metrics,
        required_source_coverage=args.required_source_coverage,
    )
    if args.json:
        print(json.dumps({"args": eval_args, "selected": row}, indent=2))
    else:
        print(shlex.join(eval_args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
