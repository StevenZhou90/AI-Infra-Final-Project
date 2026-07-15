from __future__ import annotations

from scripts.pattern_sweep_to_eval_args import (
    build_eval_args,
    pattern_eval_args_from_config,
    source_usage_metric_thresholds,
)


def _summary() -> dict:
    return {
        "top": [
            {
                "modeled_speedup": 2.3,
                "target_forward_reduction": 2.5,
                "min_task_target_forward_reduction": 2.1,
                "min_task_acceptance_rate": 0.75,
                "task_count": 10,
                "heldout_modeled_speedup": 2.15,
                "heldout_target_forward_reduction": 2.2,
                "heldout_min_task_target_forward_reduction": 1.9,
                "heldout_min_task_acceptance_rate": 0.7,
                "heldout_task_count": 3,
                "second_order_action_extrapolation_accepted_tokens": 4,
                "min_task_second_order_action_extrapolation_accepted_tokens": 1,
                "heldout_second_order_action_extrapolation_accepted_tokens": 2,
                "heldout_min_task_second_order_action_extrapolation_accepted_tokens": 1,
                "ngram_continuation_drafted_tokens": 12,
                "ngram_continuation_accepted_tokens": 9,
                "min_task_ngram_continuation_accepted_tokens": 2,
                "action_transition_histogram_accepted_tokens": 7,
                "min_task_action_transition_histogram_accepted_tokens": 1,
                "action_delta_histogram_accepted_tokens": 8,
                "min_task_action_delta_histogram_accepted_tokens": 1,
                "action_delta_ngram_accepted_tokens": 4,
                "min_task_action_delta_ngram_accepted_tokens": 1,
                "chunk_position_delta_accepted_tokens": 4,
                "min_task_chunk_position_delta_accepted_tokens": 1,
                "chunk_delta_template_accepted_tokens": 4,
                "min_task_chunk_delta_template_accepted_tokens": 1,
                "source_agreement_accepted_tokens": 6,
                "min_task_source_agreement_accepted_tokens": 1,
                "action_trend_regression_accepted_tokens": 5,
                "min_task_action_trend_regression_accepted_tokens": 1,
                "action_prefix_lookup_accepted_tokens": 5,
                "min_task_action_prefix_lookup_accepted_tokens": 1,
                "action_repeat_vector_accepted_tokens": 5,
                "min_task_action_repeat_vector_accepted_tokens": 1,
                "chunk_length_stop_accepted_tokens": 5,
                "min_task_chunk_length_stop_accepted_tokens": 1,
                "heldout_ngram_continuation_drafted_tokens": 5,
                "heldout_ngram_continuation_accepted_tokens": 3,
                "heldout_min_task_ngram_continuation_accepted_tokens": 1,
                "heldout_source_agreement_accepted_tokens": 2,
                "heldout_min_task_source_agreement_accepted_tokens": 1,
                "heldout_action_trend_regression_accepted_tokens": 2,
                "heldout_min_task_action_trend_regression_accepted_tokens": 1,
                "heldout_action_prefix_lookup_accepted_tokens": 2,
                "heldout_min_task_action_prefix_lookup_accepted_tokens": 1,
                "heldout_action_repeat_vector_accepted_tokens": 2,
                "heldout_min_task_action_repeat_vector_accepted_tokens": 1,
                "heldout_chunk_length_stop_accepted_tokens": 2,
                "heldout_min_task_chunk_length_stop_accepted_tokens": 1,
                "config": {
                    "lookahead": 8,
                    "action_dim": 7,
                    "max_period": 16,
                    "min_period_repeats": 2,
                    "repeat_token_min_run": 3,
                    "linear_action_extrapolation": True,
                    "second_order_action_extrapolation": True,
                    "second_order_max_accel": 4,
                    "action_trend_regression": True,
                    "action_trend_history": 4,
                    "action_trend_top_k": 3,
                    "action_trend_max_abs": 8,
                    "action_prefix_lookup": True,
                    "action_prefix_history_size": 4,
                    "action_prefix_top_k": 3,
                    "action_prefix_min_prefix": 1,
                    "action_prefix_max_mismatches": 0,
                    "action_repeat_vector": True,
                    "action_repeat_min_repeats": 2,
                    "action_repeat_max_delta": 0,
                    "chunk_length_stop": True,
                    "chunk_length_stop_history_size": 4,
                    "chunk_length_stop_min_count": 2,
                    "action_transition_histogram": True,
                    "action_transition_history_size": 3,
                    "action_transition_top_k": 2,
                    "action_transition_min_count": 1,
                    "action_delta_histogram": True,
                    "action_delta_history": 6,
                    "action_delta_top_k": 2,
                    "action_delta_min_count": 2,
                    "action_delta_max_abs": 5,
                    "action_delta_ngram": True,
                    "action_delta_ngram_history_size": 3,
                    "action_delta_ngram_min_context": 1,
                    "action_delta_ngram_max_context": 3,
                    "action_delta_ngram_top_k": 2,
                    "action_delta_ngram_min_count": 1,
                    "action_delta_ngram_max_abs": None,
                    "chunk_position_delta": True,
                    "chunk_position_delta_history_size": 4,
                    "chunk_position_delta_top_k": 2,
                    "chunk_position_delta_min_count": 2,
                    "chunk_position_delta_max_abs": None,
                    "chunk_delta_template": True,
                    "chunk_delta_template_history_size": 4,
                    "chunk_delta_template_top_k": 2,
                    "chunk_delta_template_min_prefix_deltas": 1,
                    "chunk_delta_template_max_delta_mismatch": None,
                    "chunk_delta_template_max_abs": 5,
                    "previous_chunk_position": True,
                    "previous_chunk_history_size": 2,
                    "ngram_continuation": True,
                    "ngram_min_context": 3,
                    "ngram_max_context": 12,
                    "ngram_history_size": 2,
                    "min_source_agreement": 2,
                    "source_priority": ["ngram_continuation", "linear_action_extrapolation"],
                    "reuse_full_blocks": True,
                    "emit_bonus_token": True,
                    "dynamic_lookahead": True,
                    "min_lookahead": 2,
                    "lookahead_growth": 1,
                    "lookahead_shrink": 3,
                },
            },
            {
                "modeled_speedup": 1.7,
                "target_forward_reduction": 1.8,
                "min_task_target_forward_reduction": 1.2,
                "min_task_acceptance_rate": 0.25,
                "task_count": 2,
                "config": {
                    "lookahead": 4,
                    "action_dim": 7,
                    "max_period": 8,
                    "min_period_repeats": 2,
                    "repeat_token_min_run": 2,
                    "linear_action_extrapolation": False,
                    "second_order_action_extrapolation": False,
                    "second_order_max_accel": 8,
                    "reuse_full_blocks": False,
                    "emit_bonus_token": False,
                    "dynamic_lookahead": False,
                },
            },
        ]
    }


def test_pattern_eval_args_from_config() -> None:
    args = pattern_eval_args_from_config(_summary()["top"][0]["config"])

    assert args == [
        "--pattern-lookahead",
        "8",
        "--pattern-action-dim",
        "7",
        "--pattern-max-period",
        "16",
        "--pattern-min-period-repeats",
        "2",
        "--pattern-repeat-token-min-run",
        "3",
        "--pattern-linear-action-extrapolation",
        "--pattern-second-order-action-extrapolation",
        "--pattern-second-order-max-accel",
        "4",
        "--pattern-action-trend-regression",
        "--pattern-action-trend-history",
        "4",
        "--pattern-action-trend-top-k",
        "3",
        "--pattern-action-trend-max-abs",
        "8",
        "--pattern-action-prefix-lookup",
        "--pattern-action-prefix-history-size",
        "4",
        "--pattern-action-prefix-top-k",
        "3",
        "--pattern-action-prefix-min-prefix",
        "1",
        "--pattern-action-prefix-max-mismatches",
        "0",
        "--pattern-action-repeat-vector",
        "--pattern-action-repeat-min-repeats",
        "2",
        "--pattern-action-repeat-max-delta",
        "0",
        "--pattern-chunk-length-stop",
        "--pattern-chunk-length-stop-history-size",
        "4",
        "--pattern-chunk-length-stop-min-count",
        "2",
        "--pattern-action-transition-histogram",
        "--pattern-action-transition-history-size",
        "3",
        "--pattern-action-transition-top-k",
        "2",
        "--pattern-action-transition-min-count",
        "1",
        "--pattern-action-delta-histogram",
        "--pattern-action-delta-history",
        "6",
        "--pattern-action-delta-top-k",
        "2",
        "--pattern-action-delta-min-count",
        "2",
        "--pattern-action-delta-max-abs",
        "5",
        "--pattern-action-delta-ngram",
        "--pattern-action-delta-ngram-history-size",
        "3",
        "--pattern-action-delta-ngram-min-context",
        "1",
        "--pattern-action-delta-ngram-max-context",
        "3",
        "--pattern-action-delta-ngram-top-k",
        "2",
        "--pattern-action-delta-ngram-min-count",
        "1",
        "--pattern-action-delta-ngram-max-abs",
        "-1",
        "--pattern-chunk-position-delta",
        "--pattern-chunk-position-delta-history-size",
        "4",
        "--pattern-chunk-position-delta-top-k",
        "2",
        "--pattern-chunk-position-delta-min-count",
        "2",
        "--pattern-chunk-position-delta-max-abs",
        "-1",
        "--pattern-chunk-delta-template",
        "--pattern-chunk-delta-template-history-size",
        "4",
        "--pattern-chunk-delta-template-top-k",
        "2",
        "--pattern-chunk-delta-template-min-prefix-deltas",
        "1",
        "--pattern-chunk-delta-template-max-delta-mismatch",
        "-1",
        "--pattern-chunk-delta-template-max-abs",
        "5",
        "--pattern-previous-chunk-position",
        "--pattern-previous-chunk-history-size",
        "2",
        "--pattern-ngram-continuation",
        "--pattern-ngram-min-context",
        "3",
        "--pattern-ngram-max-context",
        "12",
        "--pattern-ngram-history-size",
        "2",
        "--pattern-source-priority",
        "ngram_continuation,linear_action_extrapolation",
        "--pattern-min-source-agreement",
        "2",
        "--pattern-reuse-full-blocks",
        "--pattern-emit-bonus-token",
        "--pattern-dynamic-lookahead",
        "--pattern-min-lookahead",
        "2",
        "--pattern-lookahead-growth",
        "1",
        "--pattern-lookahead-shrink",
        "3",
    ]


def test_pattern_eval_args_can_disable_linear_extrapolation() -> None:
    args, selected = build_eval_args(
        _summary(),
        rank=2,
        min_modeled_speedup=None,
        min_forward_reduction=None,
    )

    assert selected["config"]["linear_action_extrapolation"] is False
    assert "--no-pattern-linear-action-extrapolation" in args
    assert "--pattern-reuse-full-blocks" not in args
    assert "--pattern-emit-bonus-token" not in args
    assert "--pattern-dynamic-lookahead" not in args


def test_pattern_eval_args_can_disable_second_order_accel_guard() -> None:
    config = dict(_summary()["top"][0]["config"])
    config["second_order_max_accel"] = None

    args = pattern_eval_args_from_config(config)

    idx = args.index("--pattern-second-order-max-accel")
    assert args[idx + 1] == "-1"


def test_pattern_eval_args_include_position_mode_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "position_mode_histogram": True,
            "position_mode_history_size": 5,
            "position_mode_top_k": 2,
            "position_mode_min_count": 3,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-position-mode-histogram" in args
    idx = args.index("--pattern-position-mode-history-size")
    assert args[idx + 1] == "5"
    idx = args.index("--pattern-position-mode-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-position-mode-min-count")
    assert args[idx + 1] == "3"


def test_pattern_eval_args_include_global_position_mode_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "global_position_mode": True,
            "global_position_history_size": 12,
            "global_position_top_k": 2,
            "global_position_min_count": 4,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-global-position-mode" in args
    idx = args.index("--pattern-global-position-history-size")
    assert args[idx + 1] == "12"
    idx = args.index("--pattern-global-position-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-global-position-min-count")
    assert args[idx + 1] == "4"


def test_pattern_eval_args_include_action_dimension_mode_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "action_dimension_mode": True,
            "action_dimension_mode_history_size": 5,
            "action_dimension_mode_top_k": 2,
            "action_dimension_mode_min_count": 3,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-action-dimension-mode" in args
    idx = args.index("--pattern-action-dimension-mode-history-size")
    assert args[idx + 1] == "5"
    idx = args.index("--pattern-action-dimension-mode-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-dimension-mode-min-count")
    assert args[idx + 1] == "3"


def test_pattern_eval_args_include_hold_action_token_flag() -> None:
    config = dict(_summary()["top"][0]["config"])
    config["hold_action_token"] = True

    args = pattern_eval_args_from_config(config)

    assert "--pattern-hold-action-token" in args


def test_pattern_eval_args_include_action_context_tree_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "action_context_tree": True,
            "action_context_tree_history_size": 5,
            "action_context_tree_max_context": 3,
            "action_context_tree_top_k": 2,
            "action_context_tree_min_count": 2,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-action-context-tree" in args
    idx = args.index("--pattern-action-context-tree-history-size")
    assert args[idx + 1] == "5"
    idx = args.index("--pattern-action-context-tree-max-context")
    assert args[idx + 1] == "3"
    idx = args.index("--pattern-action-context-tree-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-context-tree-min-count")
    assert args[idx + 1] == "2"


def test_pattern_eval_args_include_source_cooldown_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "source_cooldown": True,
            "source_cooldown_after": 2,
            "source_cooldown_steps": 3,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-source-cooldown" in args
    idx = args.index("--pattern-source-cooldown-after")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-source-cooldown-steps")
    assert args[idx + 1] == "3"


def test_pattern_eval_args_include_source_acceptance_bias_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "source_acceptance_bias": True,
            "source_acceptance_bias_history_size": 8,
            "source_acceptance_bias_min_observations": 1,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-source-acceptance-bias" in args
    idx = args.index("--pattern-source-acceptance-bias-history-size")
    assert args[idx + 1] == "8"
    idx = args.index("--pattern-source-acceptance-bias-min-observations")
    assert args[idx + 1] == "1"


def test_pattern_eval_args_include_action_trend_regression_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "action_trend_regression": True,
            "action_trend_history": 5,
            "action_trend_top_k": 2,
            "action_trend_max_abs": None,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-action-trend-regression" in args
    idx = args.index("--pattern-action-trend-history")
    assert args[idx + 1] == "5"
    idx = args.index("--pattern-action-trend-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-trend-max-abs")
    assert args[idx + 1] == "-1"


def test_pattern_eval_args_include_action_prefix_lookup_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "action_prefix_lookup": True,
            "action_prefix_history_size": 5,
            "action_prefix_top_k": 2,
            "action_prefix_min_prefix": 2,
            "action_prefix_max_mismatches": 1,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-action-prefix-lookup" in args
    idx = args.index("--pattern-action-prefix-history-size")
    assert args[idx + 1] == "5"
    idx = args.index("--pattern-action-prefix-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-prefix-min-prefix")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-prefix-max-mismatches")
    assert args[idx + 1] == "1"


def test_pattern_eval_args_include_action_vector_suffix_lookup_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "action_vector_suffix_lookup": True,
            "action_vector_suffix_history_size": 5,
            "action_vector_suffix_top_k": 2,
            "action_vector_suffix_min_prefix": 2,
            "action_vector_suffix_min_count": 2,
            "action_vector_suffix_max_prefix_delta": 1,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-action-vector-suffix-lookup" in args
    idx = args.index("--pattern-action-vector-suffix-history-size")
    assert args[idx + 1] == "5"
    idx = args.index("--pattern-action-vector-suffix-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-vector-suffix-min-prefix")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-vector-suffix-min-count")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-vector-suffix-max-prefix-delta")
    assert args[idx + 1] == "1"


def test_pattern_eval_args_include_action_vector_transition_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "action_vector_transition": True,
            "action_vector_transition_history_size": 5,
            "action_vector_transition_top_k": 2,
            "action_vector_transition_min_count": 2,
            "action_vector_transition_max_prev_delta": 1,
            "action_vector_transition_max_prefix_delta": 0,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-action-vector-transition" in args
    idx = args.index("--pattern-action-vector-transition-history-size")
    assert args[idx + 1] == "5"
    idx = args.index("--pattern-action-vector-transition-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-vector-transition-min-count")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-vector-transition-max-prev-delta")
    assert args[idx + 1] == "1"
    idx = args.index("--pattern-action-vector-transition-max-prefix-delta")
    assert args[idx + 1] == "0"


def test_auto_source_usage_thresholds_include_action_trend_regression() -> None:
    thresholds, heldout_thresholds = source_usage_metric_thresholds(_summary()["top"][0])

    assert "action_trend_regression_accepted_tokens=1" in thresholds
    assert "min_task_action_trend_regression_accepted_tokens=1" in thresholds
    assert "action_trend_regression_accepted_tokens=1" in heldout_thresholds
    assert "min_task_action_trend_regression_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_second_order_action_extrapolation() -> None:
    thresholds, heldout_thresholds = source_usage_metric_thresholds(_summary()["top"][0])

    assert "second_order_action_extrapolation_accepted_tokens=1" in thresholds
    assert "min_task_second_order_action_extrapolation_accepted_tokens=1" in thresholds
    assert "second_order_action_extrapolation_accepted_tokens=1" in heldout_thresholds
    assert "min_task_second_order_action_extrapolation_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_action_prefix_lookup() -> None:
    thresholds, heldout_thresholds = source_usage_metric_thresholds(_summary()["top"][0])

    assert "action_prefix_lookup_accepted_tokens=1" in thresholds
    assert "min_task_action_prefix_lookup_accepted_tokens=1" in thresholds
    assert "action_prefix_lookup_accepted_tokens=1" in heldout_thresholds
    assert "min_task_action_prefix_lookup_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_action_vector_suffix_lookup() -> None:
    row = dict(_summary()["top"][0])
    config = dict(row["config"])
    config["action_vector_suffix_lookup"] = True
    row["config"] = config

    thresholds, heldout_thresholds = source_usage_metric_thresholds(row)

    assert "action_vector_suffix_lookup_accepted_tokens=1" in thresholds
    assert "min_task_action_vector_suffix_lookup_accepted_tokens=1" in thresholds
    assert "action_vector_suffix_lookup_accepted_tokens=1" in heldout_thresholds
    assert "min_task_action_vector_suffix_lookup_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_action_vector_transition() -> None:
    row = dict(_summary()["top"][0])
    config = dict(row["config"])
    config["action_vector_transition"] = True
    row["config"] = config

    thresholds, heldout_thresholds = source_usage_metric_thresholds(row)

    assert "action_vector_transition_accepted_tokens=1" in thresholds
    assert "min_task_action_vector_transition_accepted_tokens=1" in thresholds
    assert "action_vector_transition_accepted_tokens=1" in heldout_thresholds
    assert "min_task_action_vector_transition_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_action_repeat_vector() -> None:
    thresholds, heldout_thresholds = source_usage_metric_thresholds(_summary()["top"][0])

    assert "action_repeat_vector_accepted_tokens=1" in thresholds
    assert "min_task_action_repeat_vector_accepted_tokens=1" in thresholds
    assert "action_repeat_vector_accepted_tokens=1" in heldout_thresholds
    assert "min_task_action_repeat_vector_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_chunk_length_stop() -> None:
    thresholds, heldout_thresholds = source_usage_metric_thresholds(_summary()["top"][0])

    assert "chunk_length_stop_accepted_tokens=1" in thresholds
    assert "min_task_chunk_length_stop_accepted_tokens=1" in thresholds
    assert "chunk_length_stop_accepted_tokens=1" in heldout_thresholds
    assert "min_task_chunk_length_stop_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_chunk_position_delta() -> None:
    row = {
        "chunk_position_delta_accepted_tokens": 4,
        "min_task_chunk_position_delta_accepted_tokens": 1,
        "heldout_chunk_position_delta_accepted_tokens": 2,
        "heldout_min_task_chunk_position_delta_accepted_tokens": 1,
        "config": {
            "chunk_position_delta": True,
            "chunk_position_delta_min_count": 2,
        },
    }

    thresholds, heldout_thresholds = source_usage_metric_thresholds(row)

    assert "chunk_position_delta_accepted_tokens=1" in thresholds
    assert "min_task_chunk_position_delta_accepted_tokens=1" in thresholds
    assert "chunk_position_delta_accepted_tokens=1" in heldout_thresholds
    assert "min_task_chunk_position_delta_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_chunk_delta_template() -> None:
    row = {
        "chunk_delta_template_accepted_tokens": 4,
        "min_task_chunk_delta_template_accepted_tokens": 1,
        "heldout_chunk_delta_template_accepted_tokens": 2,
        "heldout_min_task_chunk_delta_template_accepted_tokens": 1,
        "config": {
            "chunk_delta_template": True,
            "chunk_delta_template_max_delta_mismatch": 0,
        },
    }

    thresholds, heldout_thresholds = source_usage_metric_thresholds(row)

    assert "chunk_delta_template_accepted_tokens=1" in thresholds
    assert "min_task_chunk_delta_template_accepted_tokens=1" in thresholds
    assert "chunk_delta_template_accepted_tokens=1" in heldout_thresholds
    assert "min_task_chunk_delta_template_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_global_position_mode() -> None:
    row = {
        "global_position_mode_accepted_tokens": 4,
        "min_task_global_position_mode_accepted_tokens": 1,
        "heldout_global_position_mode_accepted_tokens": 2,
        "heldout_min_task_global_position_mode_accepted_tokens": 1,
        "config": {
            "global_position_mode": True,
        },
    }

    thresholds, heldout_thresholds = source_usage_metric_thresholds(row)

    assert "global_position_mode_accepted_tokens=1" in thresholds
    assert "min_task_global_position_mode_accepted_tokens=1" in thresholds
    assert "global_position_mode_accepted_tokens=1" in heldout_thresholds
    assert "min_task_global_position_mode_accepted_tokens=1" in heldout_thresholds


def test_auto_source_usage_thresholds_include_tree_anchor_usage() -> None:
    row = {
        "tree_anchor_accepted_tokens": 3,
        "min_task_tree_anchor_accepted_tokens": 1,
        "heldout_tree_anchor_accepted_tokens": 2,
        "heldout_min_task_tree_anchor_accepted_tokens": 1,
        "config": {
            "tree_anchor_target_continuation": True,
        },
    }

    thresholds, heldout_thresholds = source_usage_metric_thresholds(row)

    assert thresholds == [
        "tree_anchor_accepted_tokens=1",
        "min_task_tree_anchor_accepted_tokens=1",
    ]
    assert heldout_thresholds == thresholds


def test_pattern_eval_args_include_online_tree_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config["tree_width"] = 4
    config["tree_branch_width"] = 3

    args = pattern_eval_args_from_config(config)

    idx = args.index("--pattern-tree-width")
    assert args[idx + 1] == "4"
    idx = args.index("--pattern-tree-branch-width")
    assert args[idx + 1] == "3"


def test_pattern_eval_args_include_chunk_prefix_retrieval_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "chunk_prefix_retrieval": True,
            "chunk_prefix_history_size": 5,
            "chunk_prefix_top_k": 2,
            "chunk_prefix_min_matches": 1,
            "chunk_prefix_max_mismatches": 2,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-chunk-prefix-retrieval" in args
    idx = args.index("--pattern-chunk-prefix-history-size")
    assert args[idx + 1] == "5"
    idx = args.index("--pattern-chunk-prefix-top-k")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-chunk-prefix-min-matches")
    assert args[idx + 1] == "1"
    idx = args.index("--pattern-chunk-prefix-max-mismatches")
    assert args[idx + 1] == "2"


def test_pattern_eval_args_include_action_token_neighborhood_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "action_token_neighborhood": True,
            "action_token_neighborhood_radius": 2,
            "action_token_neighborhood_top_k": 5,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-action-token-neighborhood" in args
    idx = args.index("--pattern-action-token-neighborhood-radius")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-action-token-neighborhood-top-k")
    assert args[idx + 1] == "5"


def test_pattern_eval_args_include_dynamic_tree_width_flags() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "tree_width": 4,
            "tree_branch_width": 3,
            "dynamic_tree_width": True,
            "min_tree_width": 1,
            "tree_width_growth": 2,
            "tree_width_shrink": 1,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-dynamic-tree-width" in args
    idx = args.index("--pattern-min-tree-width")
    assert args[idx + 1] == "1"
    idx = args.index("--pattern-tree-width-growth")
    assert args[idx + 1] == "2"
    idx = args.index("--pattern-tree-width-shrink")
    assert args[idx + 1] == "1"


def test_pattern_eval_args_include_tree_anchor_target_token_flag() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "tree_width": 4,
            "tree_branch_width": 3,
            "tree_anchor_target_token": True,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-tree-anchor-target-token" in args


def test_pattern_eval_args_include_tree_anchor_target_continuation_flag() -> None:
    config = dict(_summary()["top"][0]["config"])
    config.update(
        {
            "tree_width": 4,
            "tree_branch_width": 3,
            "tree_anchor_target_continuation": True,
        }
    )

    args = pattern_eval_args_from_config(config)

    assert "--pattern-tree-anchor-target-continuation" in args


def test_pattern_eval_args_enforce_thresholds() -> None:
    try:
        build_eval_args(
            _summary(),
            rank=2,
            min_modeled_speedup=2.0,
            min_forward_reduction=None,
        )
    except ValueError as exc:
        assert "modeled_speedup" in str(exc)
    else:
        raise AssertionError("expected threshold failure")


def test_pattern_eval_args_enforce_task_robustness_thresholds() -> None:
    try:
        build_eval_args(
            _summary(),
            rank=2,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            min_task_forward_reduction=1.5,
        )
    except ValueError as exc:
        assert "min_task_target_forward_reduction" in str(exc)
    else:
        raise AssertionError("expected task-forward threshold failure")

    try:
        build_eval_args(
            _summary(),
            rank=2,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            min_task_acceptance_rate=0.5,
        )
    except ValueError as exc:
        assert "min_task_acceptance_rate" in str(exc)
    else:
        raise AssertionError("expected task-acceptance threshold failure")

    try:
        build_eval_args(
            _summary(),
            rank=2,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            min_task_count=3,
        )
    except ValueError as exc:
        assert "task_count" in str(exc)
    else:
        raise AssertionError("expected task-count threshold failure")


def test_pattern_eval_args_enforce_heldout_thresholds() -> None:
    args, selected = build_eval_args(
        _summary(),
        rank=1,
        min_modeled_speedup=None,
        min_forward_reduction=None,
        min_heldout_modeled_speedup=2.0,
        min_heldout_forward_reduction=2.0,
        min_heldout_task_forward_reduction=1.8,
        min_heldout_task_acceptance_rate=0.6,
        min_heldout_task_count=3,
    )

    assert selected["heldout_modeled_speedup"] == 2.15
    assert "--pattern-lookahead" in args

    try:
        build_eval_args(
            _summary(),
            rank=2,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            min_heldout_modeled_speedup=1.0,
        )
    except ValueError as exc:
        assert "heldout_modeled_speedup" in str(exc)
    else:
        raise AssertionError("expected missing heldout metric failure")

    try:
        build_eval_args(
            _summary(),
            rank=1,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            min_heldout_task_acceptance_rate=0.8,
        )
    except ValueError as exc:
        assert "heldout_min_task_acceptance_rate" in str(exc)
    else:
        raise AssertionError("expected heldout threshold failure")


def test_pattern_eval_args_enforce_named_metric_thresholds() -> None:
    args, selected = build_eval_args(
        _summary(),
        rank=1,
        min_modeled_speedup=None,
        min_forward_reduction=None,
        min_metric=["ngram_continuation_accepted_tokens=5"],
        min_heldout_metric=["ngram_continuation_accepted_tokens=2"],
    )

    assert selected["ngram_continuation_accepted_tokens"] == 9
    assert "--pattern-ngram-continuation" in args

    args, selected = build_eval_args(
        _summary(),
        rank=1,
        min_modeled_speedup=None,
        min_forward_reduction=None,
        min_metric=["min_task_ngram_continuation_accepted_tokens=1"],
        min_heldout_metric=["min_task_ngram_continuation_accepted_tokens=1"],
    )

    assert selected["min_task_ngram_continuation_accepted_tokens"] == 2
    assert "--pattern-ngram-continuation" in args

    try:
        build_eval_args(
            _summary(),
            rank=1,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            min_metric=["ngram_continuation_accepted_tokens=10"],
        )
    except ValueError as exc:
        assert "ngram_continuation_accepted_tokens" in str(exc)
    else:
        raise AssertionError("expected named metric threshold failure")

    try:
        build_eval_args(
            _summary(),
            rank=1,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            min_heldout_metric=["ngram_continuation_accepted_tokens=4"],
        )
    except ValueError as exc:
        assert "heldout_ngram_continuation_accepted_tokens" in str(exc)
    else:
        raise AssertionError("expected heldout named metric threshold failure")

    try:
        build_eval_args(
            _summary(),
            rank=1,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            min_metric=["min_task_ngram_continuation_accepted_tokens=3"],
        )
    except ValueError as exc:
        assert "min_task_ngram_continuation_accepted_tokens" in str(exc)
    else:
        raise AssertionError("expected min-task named metric threshold failure")


def test_pattern_eval_args_enforce_required_source_coverage() -> None:
    summary = _summary()
    summary["evaluated_source_counts"] = {
        "chunk_delta_template": 2,
        "action_delta_histogram": 3,
    }
    summary["evaluated_source_coverage"] = ["chunk_delta_template", "action_delta_histogram"]

    _args, selected = build_eval_args(
        summary,
        rank=1,
        min_modeled_speedup=None,
        min_forward_reduction=None,
        required_source_coverage=["chunk_delta_template,action_delta_histogram"],
    )

    assert selected["required_source_coverage"] == ["chunk_delta_template", "action_delta_histogram"]
    assert selected["required_source_counts"] == {
        "chunk_delta_template": 2,
        "action_delta_histogram": 3,
    }

    try:
        build_eval_args(
            summary,
            rank=1,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            required_source_coverage=["action_delta_ngram"],
        )
    except ValueError as exc:
        assert "action_delta_ngram" in str(exc)
        assert "required source coverage" in str(exc)
    else:
        raise AssertionError("expected missing required source coverage failure")

    try:
        build_eval_args(
            _summary(),
            rank=1,
            min_modeled_speedup=None,
            min_forward_reduction=None,
            required_source_coverage=["chunk_delta_template"],
        )
    except ValueError as exc:
        assert "evaluated_source_counts" in str(exc)
    else:
        raise AssertionError("expected missing source coverage summary failure")


def test_pattern_eval_args_auto_selects_first_threshold_passing_row() -> None:
    summary = _summary()
    rows = [dict(row) for row in summary["top"]]
    rows[0]["heldout_min_task_acceptance_rate"] = 0.1
    rows[1].update(
        {
            "modeled_speedup": 2.1,
            "target_forward_reduction": 2.2,
            "min_task_target_forward_reduction": 1.6,
            "min_task_acceptance_rate": 0.55,
            "task_count": 3,
            "heldout_modeled_speedup": 1.5,
            "heldout_target_forward_reduction": 1.4,
            "heldout_min_task_target_forward_reduction": 1.2,
            "heldout_min_task_acceptance_rate": 0.5,
            "heldout_task_count": 2,
        }
    )

    args, selected = build_eval_args(
        {"top": rows},
        rank=0,
        min_modeled_speedup=2.0,
        min_forward_reduction=2.0,
        min_task_forward_reduction=1.5,
        min_task_acceptance_rate=0.5,
        min_task_count=3,
        min_heldout_modeled_speedup=1.2,
        min_heldout_forward_reduction=1.2,
        min_heldout_task_forward_reduction=1.1,
        min_heldout_task_acceptance_rate=0.4,
        min_heldout_task_count=2,
    )

    assert selected["requested_sweep_rank"] == 0
    assert selected["selected_sweep_rank"] == 2
    assert selected["config"]["linear_action_extrapolation"] is False
    assert "--no-pattern-linear-action-extrapolation" in args


def test_pattern_eval_args_auto_select_uses_per_row_auto_source_metrics() -> None:
    summary = _summary()
    rows = [dict(row) for row in summary["top"]]
    rows[1].update(
        {
            "modeled_speedup": 2.1,
            "target_forward_reduction": 2.2,
            "min_task_target_forward_reduction": 1.6,
            "min_task_acceptance_rate": 0.55,
            "task_count": 3,
            "heldout_modeled_speedup": 1.5,
            "heldout_target_forward_reduction": 1.4,
            "heldout_min_task_target_forward_reduction": 1.2,
            "heldout_min_task_acceptance_rate": 0.5,
            "heldout_task_count": 2,
            "chunk_length_stop_accepted_tokens": 2,
            "min_task_chunk_length_stop_accepted_tokens": 1,
            "heldout_chunk_length_stop_accepted_tokens": 1,
            "heldout_min_task_chunk_length_stop_accepted_tokens": 1,
        }
    )
    rows[1]["config"] = dict(rows[1]["config"], chunk_length_stop=True)

    args, selected = build_eval_args(
        {"top": rows},
        rank=0,
        min_modeled_speedup=2.0,
        min_forward_reduction=2.0,
        auto_source_min_metrics=True,
    )

    assert selected["selected_sweep_rank"] == 2
    assert selected["auto_source_min_metric"] == [
        "chunk_length_stop_accepted_tokens=1",
        "min_task_chunk_length_stop_accepted_tokens=1",
    ]
    assert selected["auto_source_min_heldout_metric"] == selected["auto_source_min_metric"]
    assert "--pattern-chunk-length-stop" in args
