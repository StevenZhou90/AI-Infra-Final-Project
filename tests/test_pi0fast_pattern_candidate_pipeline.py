from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from scripts.run_pi0fast_pattern_candidate_pipeline import (
    build_manifest,
    parse_args,
    resolve_pattern_task_count_thresholds,
    resolve_required_source_coverage,
)


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "root": tmp_path / "pipeline",
        "trace_dir": None,
        "trace_eval_root": None,
        "sweep_output": None,
        "gate_root": None,
        "python": "python",
        "eval_script": "scripts/run_pi0fast_chunk_eval.py",
        "sweep_script": "scripts/sweep_pi0fast_pattern_offline.py",
        "gate_runner_script": "scripts/run_pi0fast_100_eval_gate.py",
        "suites": "libero_goal,libero_spatial",
        "task_ids": "0,1",
        "trace_episodes": 2,
        "gate_episodes": 4,
        "steps": 300,
        "seed": 42,
        "device": "cuda",
        "dtype": "bfloat16",
        "policy_kind": "pi0fast",
        "policy": None,
        "num_inference_steps": None,
        "smooth_position_delta": 0.06,
        "smooth_rotation_delta": 0.22,
        "trace_max_rows_per_shard": 128,
        "heldout_split": "task",
        "heldout_val_fraction": 0.2,
        "lookaheads": "4,8",
        "action_dims": "7",
        "max_periods": "8",
        "min_period_repeats": "2",
        "repeat_token_min_runs": "2",
        "second_order_action_extrapolation": "both",
        "second_order_max_accels": "4",
        "action_trend_regression": "both",
        "action_trend_histories": "4",
        "action_trend_top_ks": "3",
        "action_trend_max_abs_values": "8",
        "action_prefix_lookup": "both",
        "action_prefix_history_sizes": "4",
        "action_prefix_top_ks": "3",
        "action_prefix_min_prefixes": "1",
        "action_prefix_max_mismatches_values": "0",
        "action_vector_suffix_lookup": "both",
        "action_vector_suffix_history_sizes": "4",
        "action_vector_suffix_top_ks": "3",
        "action_vector_suffix_min_prefixes": "1,2",
        "action_vector_suffix_min_counts": "1,2",
        "action_vector_suffix_max_prefix_delta_values": "0",
        "action_vector_transition": "both",
        "action_vector_transition_history_sizes": "4",
        "action_vector_transition_top_ks": "3",
        "action_vector_transition_min_counts": "1,2",
        "action_vector_transition_max_prev_delta_values": "0",
        "action_vector_transition_max_prefix_delta_values": "0",
        "action_repeat_vector": "both",
        "action_repeat_min_repeats_values": "2",
        "action_repeat_max_delta_values": "0",
        "chunk_length_stop": "both",
        "chunk_length_stop_history_sizes": "4",
        "chunk_length_stop_min_counts": "2",
        "action_context_tree": "both",
        "action_context_tree_history_sizes": "4",
        "action_context_tree_max_contexts": "2",
        "action_context_tree_top_ks": "2",
        "action_context_tree_min_counts": "1",
        "action_transition_histogram": "both",
        "action_transition_history_sizes": "2",
        "action_transition_top_ks": "2",
        "action_delta_histogram": "both",
        "action_delta_histories": "4",
        "action_delta_top_ks": "2",
        "action_delta_min_counts": "1,2",
        "action_delta_max_abs_values": "8",
        "action_delta_ngram": "both",
        "action_delta_ngram_history_sizes": "4",
        "action_delta_ngram_min_contexts": "1",
        "action_delta_ngram_max_contexts": "2",
        "action_delta_ngram_top_ks": "2",
        "action_delta_ngram_min_counts": "1",
        "action_delta_ngram_max_abs_values": "8",
        "chunk_position_delta": "both",
        "chunk_position_delta_history_sizes": "2,4",
        "chunk_position_delta_top_ks": "2,3",
        "chunk_position_delta_min_counts": "1,2",
        "chunk_position_delta_max_abs_values": "8,12",
        "chunk_delta_template": "both",
        "chunk_delta_template_history_sizes": "2,4",
        "chunk_delta_template_top_ks": "2,3",
        "chunk_delta_template_min_prefix_deltas": "1,2",
        "chunk_delta_template_max_delta_mismatch_values": "0,2",
        "chunk_delta_template_max_abs_values": "8,12",
        "previous_chunk_position": "both",
        "previous_chunk_history_sizes": "1,2",
        "chunk_prefix_retrieval": "both",
        "chunk_prefix_history_sizes": "3,4",
        "chunk_prefix_top_ks": "2,3",
        "chunk_prefix_min_matches": "1,2",
        "chunk_prefix_max_mismatches": "1",
        "action_token_neighborhood": "both",
        "action_token_neighborhood_radii": "1,2",
        "action_token_neighborhood_top_ks": "2,3",
        "position_mode_histogram": "false",
        "position_mode_history_sizes": "2",
        "position_mode_top_ks": "2",
        "position_mode_min_counts": "2",
        "global_position_mode": "both",
        "global_position_history_sizes": "16",
        "global_position_top_ks": "3",
        "global_position_min_counts": "3",
        "action_dimension_mode": "both",
        "action_dimension_mode_history_sizes": "2",
        "action_dimension_mode_top_ks": "2",
        "action_dimension_mode_min_counts": "2",
        "hold_action_token": "false",
        "ngram_continuation": "both",
        "ngram_min_contexts": "2",
        "ngram_max_contexts": "8",
        "ngram_history_sizes": "4",
        "min_source_agreements": "1,2",
        "source_priority_modes": "smooth_first",
        "source_cooldown": "both",
        "source_cooldown_afters": "1",
        "source_cooldown_steps": "1,2",
        "source_acceptance_bias": "both",
        "source_acceptance_bias_history_sizes": "32",
        "source_acceptance_bias_min_observations": "1",
        "reuse_full_blocks": "true",
        "emit_bonus_token": "both",
        "defer_correction_token": False,
        "dynamic_lookahead": "both",
        "min_lookaheads": "1,2",
        "lookahead_growths": "1",
        "lookahead_shrinks": "2,4",
        "history_reset": "task_seed",
        "tree_widths": "1,4",
        "tree_branch_widths": "4",
        "dynamic_tree_width": "both",
        "min_tree_widths": "1",
        "tree_width_growths": "1",
        "tree_width_shrinks": "1",
        "tree_anchor_target_token": "both",
        "tree_anchor_target_continuation": "both",
        "stop_token_ids": "",
        "target_forward_ms": 1.0,
        "draft_token_ms": 0.0,
        "top_k": 12,
        "max_enabled_sources": 4,
        "max_sweep_configs": 4096,
        "pattern_sweep_rank": 0,
        "pattern_min_modeled_speedup": 2.0,
        "pattern_min_forward_reduction": 2.0,
        "pattern_min_task_forward_reduction": 1.25,
        "pattern_min_task_acceptance_rate": 0.30,
        "pattern_min_task_count": None,
        "pattern_min_heldout_modeled_speedup": 1.10,
        "pattern_min_heldout_forward_reduction": 1.10,
        "pattern_min_heldout_task_forward_reduction": 1.10,
        "pattern_min_heldout_task_acceptance_rate": 0.20,
        "pattern_min_heldout_task_count": None,
        "pattern_min_metric": ["accepted_tokens=1"],
        "pattern_min_heldout_metric": ["accepted_tokens=1"],
        "pattern_auto_source_min_metrics": True,
        "pattern_required_source_coverage": "auto",
        "min_pairs": 120,
        "min_speedup": 2.0,
        "max_success_drop": 0.0,
        "max_baseline_success_regressions": 0,
        "min_validation_episodes": 120,
        "min_exact_verifies": 1,
        "max_action_diff": 0.0,
        "gate_dry_run": True,
        "skip_trace_collection": False,
        "skip_sweep": False,
        "skip_gate_manifest": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_pattern_candidate_pipeline_manifest_uses_target_eos_reference(tmp_path: Path) -> None:
    manifest = build_manifest(_args(tmp_path))

    assert manifest["early_stop_reference"] == "target_eos"
    assert manifest["matched_gate_eval_count"] == 2 * 2 * 4
    assert len(manifest["trace_commands"]) == 2

    trace_command = manifest["trace_commands"][0]
    assert "--modes" in trace_command
    assert "target_eos" in trace_command
    assert "--enable-fast-token-hooks" in trace_command
    assert "--token-trace-modes" in trace_command
    assert "target_eos" in trace_command
    assert "--policy-kind" in trace_command
    assert trace_command[trace_command.index("--policy-kind") + 1] == "pi0fast"

    sweep_command = manifest["sweep_command"]
    assert sweep_command is not None
    assert "--data-dir" in sweep_command
    assert str(tmp_path / "pipeline" / "target_eos_traces") in sweep_command
    assert "--heldout-split" in sweep_command
    assert sweep_command[sweep_command.index("--heldout-split") + 1] == "task"
    assert "--action-transition-histogram" in sweep_command
    assert "both" in sweep_command
    assert "--action-context-tree" in sweep_command
    assert "--action-context-tree-max-contexts" in sweep_command
    assert sweep_command[sweep_command.index("--action-context-tree-max-contexts") + 1] == "2"
    assert "--action-delta-histogram" in sweep_command
    assert "--action-delta-min-counts" in sweep_command
    assert sweep_command[sweep_command.index("--action-delta-min-counts") + 1] == "1,2"
    assert "--action-delta-ngram" in sweep_command
    assert sweep_command[sweep_command.index("--action-delta-ngram") + 1] == "both"
    assert "--action-delta-ngram-history-sizes" in sweep_command
    assert "--action-delta-ngram-min-contexts" in sweep_command
    assert "--action-delta-ngram-max-contexts" in sweep_command
    assert "--action-delta-ngram-top-ks" in sweep_command
    assert "--action-delta-ngram-min-counts" in sweep_command
    assert "--action-delta-ngram-max-abs-values" in sweep_command
    assert "--action-vector-transition" in sweep_command
    assert sweep_command[sweep_command.index("--action-vector-transition") + 1] == "both"
    assert "--action-vector-suffix-lookup" in sweep_command
    assert sweep_command[sweep_command.index("--action-vector-suffix-lookup") + 1] == "both"
    assert "--action-vector-suffix-history-sizes" in sweep_command
    assert "--action-vector-suffix-top-ks" in sweep_command
    assert "--action-vector-suffix-min-prefixes" in sweep_command
    assert "--action-vector-suffix-min-counts" in sweep_command
    assert "--action-vector-suffix-max-prefix-delta-values" in sweep_command
    assert "--action-vector-transition-history-sizes" in sweep_command
    assert "--action-vector-transition-top-ks" in sweep_command
    assert "--action-vector-transition-min-counts" in sweep_command
    assert "--action-vector-transition-max-prev-delta-values" in sweep_command
    assert "--action-vector-transition-max-prefix-delta-values" in sweep_command
    assert "--chunk-position-delta" in sweep_command
    assert "--chunk-position-delta-history-sizes" in sweep_command
    assert "--chunk-position-delta-top-ks" in sweep_command
    assert "--chunk-position-delta-min-counts" in sweep_command
    assert "--chunk-position-delta-max-abs-values" in sweep_command
    assert "--chunk-delta-template" in sweep_command
    assert "--chunk-delta-template-history-sizes" in sweep_command
    assert "--chunk-delta-template-top-ks" in sweep_command
    assert "--chunk-delta-template-min-prefix-deltas" in sweep_command
    assert "--chunk-delta-template-max-delta-mismatch-values" in sweep_command
    assert "--chunk-delta-template-max-abs-values" in sweep_command
    assert "--previous-chunk-position" in sweep_command
    assert "--chunk-prefix-retrieval" in sweep_command
    assert "3,4" in sweep_command
    assert "--action-token-neighborhood" in sweep_command
    assert "--action-token-neighborhood-radii" in sweep_command
    assert "--action-token-neighborhood-top-ks" in sweep_command
    assert "--position-mode-histogram" in sweep_command
    assert "--global-position-mode" in sweep_command
    assert sweep_command[sweep_command.index("--global-position-mode") + 1] == "both"
    assert "--global-position-history-sizes" in sweep_command
    assert sweep_command[sweep_command.index("--global-position-history-sizes") + 1] == "16"
    assert "--global-position-top-ks" in sweep_command
    assert sweep_command[sweep_command.index("--global-position-top-ks") + 1] == "3"
    assert "--global-position-min-counts" in sweep_command
    assert sweep_command[sweep_command.index("--global-position-min-counts") + 1] == "3"
    assert "--action-dimension-mode" in sweep_command
    assert "--hold-action-token" in sweep_command
    assert "--ngram-continuation" in sweep_command
    assert "--source-cooldown" in sweep_command
    assert "1,2" in sweep_command
    assert "--source-acceptance-bias" in sweep_command
    assert sweep_command[sweep_command.index("--source-acceptance-bias") + 1] == "both"
    assert "--source-acceptance-bias-history-sizes" in sweep_command
    assert "--source-acceptance-bias-min-observations" in sweep_command
    assert "--action-trend-regression" in sweep_command
    assert sweep_command[sweep_command.index("--action-trend-regression") + 1] == "both"
    assert "--action-trend-histories" in sweep_command
    assert sweep_command[sweep_command.index("--action-trend-max-abs-values") + 1] == "8"
    assert "--action-prefix-lookup" in sweep_command
    assert sweep_command[sweep_command.index("--action-prefix-lookup") + 1] == "both"
    assert "--action-prefix-history-sizes" in sweep_command
    assert sweep_command[sweep_command.index("--action-prefix-top-ks") + 1] == "3"
    assert sweep_command[sweep_command.index("--action-prefix-min-prefixes") + 1] == "1"
    assert sweep_command[sweep_command.index("--action-prefix-max-mismatches-values") + 1] == "0"
    assert "--action-repeat-vector" in sweep_command
    assert sweep_command[sweep_command.index("--action-repeat-vector") + 1] == "both"
    assert sweep_command[sweep_command.index("--action-repeat-min-repeats-values") + 1] == "2"
    assert sweep_command[sweep_command.index("--action-repeat-max-delta-values") + 1] == "0"
    assert "--chunk-length-stop" in sweep_command
    assert sweep_command[sweep_command.index("--chunk-length-stop") + 1] == "both"
    assert sweep_command[sweep_command.index("--chunk-length-stop-history-sizes") + 1] == "4"
    assert sweep_command[sweep_command.index("--chunk-length-stop-min-counts") + 1] == "2"
    assert "--tree-widths" in sweep_command
    assert sweep_command[sweep_command.index("--tree-widths") + 1] == "1,4"
    assert "--dynamic-tree-width" in sweep_command
    assert sweep_command[sweep_command.index("--dynamic-tree-width") + 1] == "both"
    assert "--tree-anchor-target-token" in sweep_command
    assert sweep_command[sweep_command.index("--tree-anchor-target-token") + 1] == "both"
    assert "--tree-anchor-target-continuation" in sweep_command
    assert sweep_command[sweep_command.index("--tree-anchor-target-continuation") + 1] == "both"
    assert "--max-enabled-sources" in sweep_command
    assert sweep_command[sweep_command.index("--max-enabled-sources") + 1] == "4"
    assert "--max-configs" in sweep_command
    assert sweep_command[sweep_command.index("--max-configs") + 1] == "4096"
    assert manifest["sweep_budget"] == {"max_enabled_sources": 4, "max_configs": 4096}

    gate_command = manifest["gate_command"]
    assert gate_command is not None
    assert "--speed-modes" in gate_command
    assert "baseline,target_eos,pattern_sd_direct" in gate_command
    assert "--candidate-mode" in gate_command
    assert "pattern_sd_direct" in gate_command
    assert "--reference-mode" in gate_command
    assert "target_eos" in gate_command
    assert "--min-pairs" in gate_command
    assert gate_command[gate_command.index("--min-pairs") + 1] == "120"
    assert "--min-validation-episodes" in gate_command
    assert gate_command[gate_command.index("--min-validation-episodes") + 1] == "120"
    assert "--pattern-sweep-json" in gate_command
    assert "--pattern-sweep-rank" in gate_command
    assert gate_command[gate_command.index("--pattern-sweep-rank") + 1] == "0"
    assert "--pattern-min-modeled-speedup" in gate_command
    assert "2.0" in gate_command
    assert "--pattern-min-heldout-modeled-speedup" in gate_command
    assert "--pattern-min-heldout-forward-reduction" in gate_command
    assert "--pattern-min-heldout-task-forward-reduction" in gate_command
    assert "--pattern-min-task-count" in gate_command
    assert gate_command[gate_command.index("--pattern-min-task-count") + 1] == "2"
    assert "--pattern-min-heldout-task-count" in gate_command
    assert gate_command[gate_command.index("--pattern-min-heldout-task-count") + 1] == "2"
    assert "--pattern-min-metric" in gate_command
    assert "--pattern-min-heldout-metric" in gate_command
    assert "--pattern-auto-source-min-metrics" in gate_command
    assert "--pattern-required-source-coverage" in gate_command
    assert "chunk_delta_template" in gate_command
    assert "action_delta_histogram" in gate_command
    assert "action_vector_suffix_lookup" in gate_command
    assert "action_vector_transition" in gate_command
    assert "action_token_neighborhood" in gate_command
    assert "--dry-run" in gate_command
    assert manifest["thresholds"]["pattern_auto_source_min_metrics"] is True
    assert "chunk_delta_template" in manifest["thresholds"]["pattern_required_source_coverage"]
    assert "action_delta_histogram" in manifest["thresholds"]["pattern_required_source_coverage"]
    assert "action_vector_transition" in manifest["thresholds"]["pattern_required_source_coverage"]
    assert "action_token_neighborhood" in manifest["thresholds"]["pattern_required_source_coverage"]
    assert manifest["thresholds"]["planned_task_count"] == 4
    assert manifest["thresholds"]["pattern_min_task_count"] == 2
    assert manifest["thresholds"]["pattern_min_heldout_modeled_speedup"] == 1.10
    assert manifest["thresholds"]["pattern_min_heldout_task_forward_reduction"] == 1.10
    assert manifest["thresholds"]["pattern_min_heldout_task_count"] == 2
    assert "--policy-kind" not in gate_command


def test_pattern_candidate_pipeline_rejects_pi05_until_token_adapter_exists(tmp_path: Path) -> None:
    try:
        build_manifest(_args(tmp_path, policy_kind="pi05", num_inference_steps=8))
    except ValueError as exc:
        assert "PI0.5 is not currently supported" in str(exc)
        assert "target_eos FAST-token hooks" in str(exc)
    else:
        raise AssertionError("expected PI0.5 pattern pipeline to require a token adapter")


def test_pattern_candidate_pipeline_defaults_include_compact_tree_screen() -> None:
    old_argv = sys.argv
    try:
        sys.argv = ["run_pi0fast_pattern_candidate_pipeline.py"]
        args = parse_args()
    finally:
        sys.argv = old_argv

    assert args.action_delta_ngram == "both"
    assert args.action_delta_min_counts == "1,2"
    assert args.action_trend_regression == "both"
    assert args.action_prefix_lookup == "both"
    assert args.action_vector_suffix_lookup == "both"
    assert args.action_vector_transition == "both"
    assert args.action_repeat_vector == "both"
    assert args.chunk_length_stop == "both"
    assert args.chunk_position_delta == "both"
    assert args.chunk_position_delta_min_counts == "1,2"
    assert args.chunk_delta_template == "both"
    assert args.chunk_delta_template_min_prefix_deltas == "1"
    assert args.chunk_delta_template_max_delta_mismatch_values == "0"
    assert args.pattern_required_source_coverage == "auto"
    default_coverage = resolve_required_source_coverage(args)
    assert "chunk_delta_template" in default_coverage
    assert "action_delta_histogram" in default_coverage
    assert "action_vector_suffix_lookup" in default_coverage
    assert "action_vector_transition" in default_coverage
    assert "action_token_neighborhood" not in default_coverage
    assert args.chunk_prefix_retrieval == "both"
    assert args.global_position_mode == "both"
    assert args.global_position_history_sizes == "16"
    assert args.global_position_top_ks == "3"
    assert args.global_position_min_counts == "3"
    assert args.tree_widths == "1,4"
    assert args.pattern_min_heldout_modeled_speedup == 1.10
    assert args.pattern_min_heldout_task_forward_reduction == 1.10
    assert args.pattern_min_task_count is None
    assert args.pattern_min_heldout_task_count is None
    assert args.pattern_sweep_rank == 0
    assert args.heldout_split == "task"
    assert args.min_pairs == 120
    assert args.min_validation_episodes == 120
    assert args.dynamic_tree_width == "both"
    assert args.tree_anchor_target_token == "both"
    assert args.tree_anchor_target_continuation == "both"
    assert args.source_priority_modes == "smooth_first"
    assert args.source_acceptance_bias == "both"
    assert args.reuse_full_blocks == "true"
    assert args.max_enabled_sources == 4
    assert args.max_sweep_configs == 4096


def test_pattern_candidate_pipeline_derives_full_task_count_thresholds() -> None:
    train_count, heldout_count = resolve_pattern_task_count_thresholds(
        suite_count=3,
        task_id_count=10,
        heldout_split="task",
        heldout_val_fraction=0.2,
        min_task_count=None,
        min_heldout_task_count=None,
    )

    assert train_count == 24
    assert heldout_count == 6


def test_pattern_candidate_pipeline_derives_suite_stratified_small_holdout_thresholds() -> None:
    train_count, heldout_count = resolve_pattern_task_count_thresholds(
        suite_count=3,
        task_id_count=2,
        heldout_split="task",
        heldout_val_fraction=0.2,
        min_task_count=None,
        min_heldout_task_count=None,
    )

    assert train_count == 3
    assert heldout_count == 3


def test_pattern_candidate_pipeline_task_seed_threshold_keeps_all_tasks() -> None:
    train_count, heldout_count = resolve_pattern_task_count_thresholds(
        suite_count=3,
        task_id_count=10,
        heldout_split="task_seed",
        heldout_val_fraction=0.2,
        min_task_count=None,
        min_heldout_task_count=None,
    )

    assert train_count == 30
    assert heldout_count == 6


def test_pattern_candidate_pipeline_keeps_explicit_task_count_thresholds(tmp_path: Path) -> None:
    manifest = build_manifest(_args(tmp_path, pattern_min_task_count=2, pattern_min_heldout_task_count=2))

    gate_command = manifest["gate_command"]
    assert gate_command is not None
    assert gate_command[gate_command.index("--pattern-min-task-count") + 1] == "2"
    assert gate_command[gate_command.index("--pattern-min-heldout-task-count") + 1] == "2"
    assert manifest["thresholds"]["pattern_min_task_count"] == 2
    assert manifest["thresholds"]["pattern_min_heldout_task_count"] == 2
