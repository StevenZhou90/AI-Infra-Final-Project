#!/usr/bin/env python3
"""Run the PI0-FAST 120-eval speed/validation gate layout.

This is an orchestration wrapper around ``scripts/run_pi0fast_chunk_eval.py``.
It intentionally runs every mode in separate subprocesses so baseline,
target-EOS early stop, and any speculative candidate mode get clean timing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pattern_sweep_to_eval_args import (  # noqa: E402
    build_eval_args as build_pattern_eval_args,
    load_sweep,
    source_usage_metric_thresholds,
)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _arg_value(args: list[str], flag: str, default: str | None = None) -> str | None:
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
        prefix = f"{flag}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return default


def eval_metadata(extra_args: list[str]) -> dict[str, str]:
    metadata = {"policy_kind": _arg_value(extra_args, "--policy-kind", "pi0fast") or "pi0fast"}
    policy = _arg_value(extra_args, "--policy")
    if policy is not None:
        metadata["policy"] = policy
    num_inference_steps = _arg_value(extra_args, "--num-inference-steps")
    if num_inference_steps is not None:
        metadata["num_inference_steps"] = num_inference_steps
    return metadata


def default_validation_mode(candidate_mode: str) -> str:
    if candidate_mode.startswith("target_eos_prefix"):
        return "target_eos_prefix_validate"
    if candidate_mode.startswith("target_eos_constrained") and "validate" not in candidate_mode:
        return f"{candidate_mode}_validate"
    if candidate_mode.startswith("target_cutoff") and "validate" not in candidate_mode:
        return f"{candidate_mode}_validate"
    if candidate_mode == "target_eos":
        return "target_eos_validate"
    if candidate_mode.startswith("pattern_sd"):
        return "pattern_sd_validate"
    if candidate_mode.startswith("ngram_sd"):
        return "ngram_sd_validate"
    if candidate_mode.startswith("block_sd"):
        return "block_sd_validate"
    if candidate_mode.startswith("target_eos_adaptive"):
        return "target_eos_adaptive_validate"
    return "target_eos_validate"


def resolve_validation_modes(value: str, *, candidate_mode: str, reference_mode: str | None = None) -> list[str]:
    if value.strip().lower() == "auto":
        modes = [default_validation_mode(candidate_mode)]
        if reference_mode == "target_eos" and "target_eos_validate" not in modes:
            modes.append("target_eos_validate")
        return modes
    modes = parse_csv(value)
    if not modes:
        raise ValueError("--validation-modes must contain at least one mode or be 'auto'")
    return modes


def _task_ids_arg(task_ids: str) -> str:
    values = parse_csv(task_ids)
    if not values:
        raise ValueError("--task-ids must contain at least one id")
    return ",".join(values)


def resolve_min_unique_tasks(value: int | None, *, suites: list[str], task_ids: str) -> int:
    if value is not None:
        return int(value)
    return len(suites) * len(parse_csv(task_ids))


def _parse_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    merged = list(values)
    seen = set(merged)
    for item in additions:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def eval_shard_complete(
    output_dir: Path,
    *,
    suite: str,
    mode: str,
    task_ids: str,
    episodes: int,
    seed: int,
) -> bool:
    """Return true when one suite/mode output dir has every expected row."""

    rows = _parse_metric_rows(output_dir / "metrics.jsonl")
    expected_task_ids = {int(part) for part in parse_csv(task_ids)}
    expected = {
        (task_id, episode, seed + episode)
        for task_id in expected_task_ids
        for episode in range(int(episodes))
    }
    seen: set[tuple[int, int, int]] = set()
    for row in rows:
        if row.get("task") != suite or row.get("mode") != mode:
            continue
        try:
            task_id = int(row["task_id"])
            episode = int(row["episode"])
            row_seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        seen.add((task_id, episode, row_seed))
    return expected <= seen


def build_eval_command(
    *,
    python: str,
    eval_script: str,
    suite: str,
    task_ids: str,
    episodes: int,
    steps: int,
    mode: str,
    output_dir: Path,
    seed: int,
    device: str,
    dtype: str,
    enable_fast_token_hooks: bool,
    smooth_position_delta: float,
    smooth_rotation_delta: float,
    extra_args: list[str],
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
        mode,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--dtype",
        dtype,
        "--smooth-position-delta",
        str(smooth_position_delta),
        "--smooth-rotation-delta",
        str(smooth_rotation_delta),
    ]
    if enable_fast_token_hooks:
        cmd.append("--enable-fast-token-hooks")
    cmd.extend(extra_args)
    return cmd


def build_preflight_command(
    *,
    python: str,
    preflight_script: str,
    device: str,
    eval_script: str,
    gate_script: str,
    modules: str,
    require_hf_token: bool,
) -> list[str]:
    cmd = [
        python,
        preflight_script,
        "--device",
        device,
        "--eval-script",
        eval_script,
        "--gate-script",
        gate_script,
        "--modules",
        modules,
    ]
    if require_hf_token:
        cmd.append("--require-hf-token")
    return cmd


def build_gate_command(
    *,
    python: str,
    gate_script: str,
    speed_root: Path,
    validation_root: Path | None,
    output: Path,
    baseline_mode: str,
    candidate_mode: str,
    reference_mode: str | None,
    validation_mode: str,
    extra_validation_modes: list[str],
    min_pairs: int,
    min_unique_tasks: int,
    min_speedup: float,
    min_baseline_successes: int,
    min_suite_matched_pairs: int,
    min_suite_baseline_successes: int,
    min_suite_speedup: float | None,
    require_matched_steps: bool,
    max_success_drop: float,
    max_baseline_success_regressions: int,
    min_validation_episodes: int,
    min_exact_verifies: int,
    max_action_diff: float,
    min_reference_speedup: float | None,
    max_reference_success_drop: float | None,
    max_reference_success_regressions: int | None,
    min_candidate_trace_stats: list[str] | None = None,
    max_candidate_trace_stats: list[str] | None = None,
    min_candidate_trace_stat_totals: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> list[str]:
    cmd = [
        python,
        gate_script,
        str(speed_root),
        "--baseline-mode",
        baseline_mode,
        "--candidate-mode",
        candidate_mode,
        "--min-pairs",
        str(min_pairs),
        "--min-unique-tasks",
        str(min_unique_tasks),
        "--min-speedup",
        str(min_speedup),
        "--min-baseline-successes",
        str(min_baseline_successes),
        "--min-suite-matched-pairs",
        str(min_suite_matched_pairs),
        "--min-suite-baseline-successes",
        str(min_suite_baseline_successes),
        "--max-success-drop",
        str(max_success_drop),
        "--max-baseline-success-regressions",
        str(max_baseline_success_regressions),
        "--require-matched-steps" if require_matched_steps else "--no-require-matched-steps",
        "--min-validation-episodes",
        str(min_validation_episodes),
        "--min-exact-verifies",
        str(min_exact_verifies),
        "--max-action-diff",
        str(max_action_diff),
        "--output",
        str(output),
        "--markdown",
    ]
    if validation_root is not None:
        cmd.extend(["--validation-root", str(validation_root), "--validation-mode", validation_mode])
        for mode in extra_validation_modes:
            cmd.extend(["--extra-validation-mode", mode])
    else:
        cmd.append("--no-require-exact-validation")
    if reference_mode is not None:
        cmd.extend(["--reference-mode", reference_mode])
    if min_suite_speedup is not None:
        cmd.extend(["--min-suite-speedup", str(min_suite_speedup)])
    if min_reference_speedup is not None:
        cmd.extend(["--min-reference-speedup", str(min_reference_speedup)])
    if max_reference_success_drop is not None:
        cmd.extend(["--max-reference-success-drop", str(max_reference_success_drop)])
    if max_reference_success_regressions is not None:
        cmd.extend(["--max-reference-success-regressions", str(max_reference_success_regressions)])
    for threshold in min_candidate_trace_stats or []:
        cmd.extend(["--min-candidate-trace-stat", threshold])
    for threshold in max_candidate_trace_stats or []:
        cmd.extend(["--max-candidate-trace-stat", threshold])
    for threshold in min_candidate_trace_stat_totals or []:
        cmd.extend(["--min-candidate-trace-stat-total", threshold])
    for key, value in (metadata or {}).items():
        cmd.extend(["--metadata", f"{key}={value}"])
    return cmd


def build_synthetic_command(
    *,
    python: str,
    synthetic_script: str,
    output: Path,
    tasks: int,
    min_speedup: float,
    max_accuracy_drop: float,
) -> list[str]:
    return [
        python,
        synthetic_script,
        "--num-tasks",
        str(tasks),
        "--min-speedup",
        str(min_speedup),
        "--max-accuracy-drop",
        str(max_accuracy_drop),
        "--output",
        str(output),
        "--markdown",
    ]


def build_audit_command(
    *,
    python: str,
    audit_script: str,
    pi0fast_gate: Path,
    pi0fast_manifest: Path | None,
    synthetic_gate: Path,
    output: Path,
    min_pairs: int,
    min_unique_tasks: int,
    min_speedup: float,
    min_baseline_successes: int,
    min_suite_matched_pairs: int,
    min_suite_baseline_successes: int,
    min_suite_speedup: float | None,
    max_success_drop: float,
    max_baseline_success_regressions: int,
    min_validation_episodes: int,
    max_action_diff: float,
    expected_policy_kind: str | None = None,
    require_pattern_source_coverage: bool = False,
    require_pattern_heldout_evidence: bool = False,
) -> list[str]:
    cmd = [
        python,
        audit_script,
        "--pi0fast-gate",
        str(pi0fast_gate),
        "--synthetic-gate",
        str(synthetic_gate),
        "--min-pairs",
        str(min_pairs),
        "--min-unique-tasks",
        str(min_unique_tasks),
        "--min-speedup",
        str(min_speedup),
        "--min-baseline-successes",
        str(min_baseline_successes),
        "--min-suite-matched-pairs",
        str(min_suite_matched_pairs),
        "--min-suite-baseline-successes",
        str(min_suite_baseline_successes),
        "--max-success-drop",
        str(max_success_drop),
        "--max-baseline-success-regressions",
        str(max_baseline_success_regressions),
        "--min-validation-episodes",
        str(min_validation_episodes),
        "--max-action-diff",
        str(max_action_diff),
        "--output",
        str(output),
        "--markdown",
    ]
    if pi0fast_manifest is not None:
        cmd.extend(["--pi0fast-manifest", str(pi0fast_manifest)])
    if min_suite_speedup is not None:
        cmd.extend(["--min-suite-speedup", str(min_suite_speedup)])
    if expected_policy_kind is not None:
        cmd.extend(["--expected-policy-kind", expected_policy_kind])
    if require_pattern_source_coverage:
        cmd.append("--require-pattern-source-coverage")
    if require_pattern_heldout_evidence:
        cmd.append("--require-pattern-heldout-evidence")
    return cmd


def build_result_card_command(
    *,
    python: str,
    result_card_script: str,
    audit_json: Path,
    output: Path,
) -> list[str]:
    return [
        python,
        result_card_script,
        str(audit_json),
        "--output",
        str(output),
    ]


def json_gate_passed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(row.get("gate_passed") or row.get("objective_audit_passed"))


def _effective_int_option(args: list[str], option: str) -> int | None:
    value: int | None = None
    prefix = f"{option}="
    index = 0
    while index < len(args):
        item = args[index]
        raw: str | None = None
        if item == option:
            index += 1
            if index >= len(args):
                raise ValueError(f"{option} requires an integer value")
            raw = args[index]
        elif item.startswith(prefix):
            raw = item[len(prefix) :]
        if raw is not None:
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f"{option} requires an integer value, got {raw!r}") from exc
        index += 1
    return value


def _effective_bool_flag(args: list[str], option: str, *, no_option: str | None = None) -> bool:
    value = False
    no_option = no_option or f"--no-{option.removeprefix('--')}"
    for item in args:
        if item == option:
            value = True
        elif item == no_option:
            value = False
    return value


def tree_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> tuple[list[str], list[str]]:
    """Return gate trace-stat thresholds implied by exact pattern tree verification."""

    if not candidate_mode.startswith("pattern_sd"):
        return [], []
    tree_width = _effective_int_option(eval_extra_args, "--pattern-tree-width")
    if tree_width is None or tree_width <= 1:
        return [], []
    return (
        [
            f"tree_width={tree_width}",
            "tree_verifies=1e-09",
        ],
        [
            "unverified_pattern_tokens=0",
            "unverified_pattern_eos_tokens=0",
        ],
    )


def dynamic_tree_width_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-dynamic-tree-width"):
        return []
    return [
        "dynamic_tree_width=1",
        "mean_tree_width=1e-09",
    ]


def tree_anchor_target_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-tree-anchor-target-token"):
        return []
    return [
        "tree_anchor_target_token=1",
        "tree_anchor_verifies=1e-09",
        "tree_anchor_accepted_tokens=1e-09",
    ]


def tree_anchor_target_continuation_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-tree-anchor-target-continuation"):
        return []
    return [
        "tree_anchor_target_continuation=1",
        "tree_anchor_verifies=1e-09",
        "tree_anchor_accepted_tokens=1e-09",
    ]


def previous_chunk_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-previous-chunk-position"):
        return []
    return [
        "previous_chunk_position_drafted_tokens=0",
        "previous_chunk_position_accepted_tokens=0",
    ]


def second_order_action_extrapolation_trace_stat_thresholds(
    candidate_mode: str,
    eval_extra_args: list[str],
) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-second-order-action-extrapolation"):
        return []
    return [
        "second_order_action_extrapolation_drafted_tokens=0",
        "second_order_action_extrapolation_accepted_tokens=0",
    ]


def chunk_prefix_retrieval_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-chunk-prefix-retrieval"):
        return []
    return [
        "chunk_prefix_retrieval_drafted_tokens=0",
        "chunk_prefix_retrieval_accepted_tokens=0",
    ]


def action_token_neighborhood_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-token-neighborhood"):
        return []
    return [
        "action_token_neighborhood_drafted_tokens=0",
        "action_token_neighborhood_accepted_tokens=0",
    ]


def position_mode_histogram_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-position-mode-histogram"):
        return []
    return [
        "position_mode_histogram_drafted_tokens=0",
        "position_mode_histogram_accepted_tokens=0",
    ]


def global_position_mode_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-global-position-mode"):
        return []
    return [
        "global_position_mode_drafted_tokens=0",
        "global_position_mode_accepted_tokens=0",
    ]


def action_dimension_mode_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-dimension-mode"):
        return []
    return [
        "action_dimension_mode_drafted_tokens=0",
        "action_dimension_mode_accepted_tokens=0",
    ]


def hold_action_token_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-hold-action-token"):
        return []
    return [
        "hold_action_token_drafted_tokens=0",
        "hold_action_token_accepted_tokens=0",
    ]


def ngram_continuation_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-ngram-continuation"):
        return []
    return [
        "ngram_continuation_drafted_tokens=0",
        "ngram_continuation_accepted_tokens=0",
    ]


def source_agreement_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    min_source_agreement = _effective_int_option(eval_extra_args, "--pattern-min-source-agreement")
    if min_source_agreement is None or min_source_agreement <= 1:
        return []
    return [
        "source_agreement_drafted_tokens=0",
        "source_agreement_accepted_tokens=0",
    ]


def action_trend_regression_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-trend-regression"):
        return []
    return [
        "action_trend_regression_drafted_tokens=0",
        "action_trend_regression_accepted_tokens=0",
    ]


def action_prefix_lookup_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-prefix-lookup"):
        return []
    return [
        "action_prefix_lookup_drafted_tokens=0",
        "action_prefix_lookup_accepted_tokens=0",
    ]


def action_vector_suffix_lookup_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-vector-suffix-lookup"):
        return []
    return [
        "action_vector_suffix_lookup_drafted_tokens=0",
        "action_vector_suffix_lookup_accepted_tokens=0",
    ]


def action_vector_transition_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-vector-transition"):
        return []
    return [
        "action_vector_transition_drafted_tokens=0",
        "action_vector_transition_accepted_tokens=0",
    ]


def action_repeat_vector_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-repeat-vector"):
        return []
    return [
        "action_repeat_vector_drafted_tokens=0",
        "action_repeat_vector_accepted_tokens=0",
    ]


def chunk_length_stop_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-chunk-length-stop"):
        return []
    return [
        "chunk_length_stop_drafted_tokens=0",
        "chunk_length_stop_accepted_tokens=0",
    ]


def action_context_tree_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-context-tree"):
        return []
    return [
        "action_context_tree_drafted_tokens=0",
        "action_context_tree_accepted_tokens=0",
    ]


def source_cooldown_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-source-cooldown"):
        return []
    return [
        "source_cooldown_events=0",
        "source_cooldown_skipped_sources=0",
    ]


def source_acceptance_bias_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-source-acceptance-bias"):
        return []
    return [
        "source_acceptance_bias_events=0",
        "source_acceptance_bias_reorders=0",
    ]


def action_delta_histogram_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-delta-histogram"):
        return []
    return [
        "action_delta_histogram_drafted_tokens=0",
        "action_delta_histogram_accepted_tokens=0",
    ]


def action_delta_ngram_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-delta-ngram"):
        return []
    return [
        "action_delta_ngram_drafted_tokens=0",
        "action_delta_ngram_accepted_tokens=0",
    ]


def chunk_position_delta_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-chunk-position-delta"):
        return []
    return [
        "chunk_position_delta_drafted_tokens=0",
        "chunk_position_delta_accepted_tokens=0",
    ]


def chunk_delta_template_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-chunk-delta-template"):
        return []
    return [
        "chunk_delta_template_drafted_tokens=0",
        "chunk_delta_template_accepted_tokens=0",
    ]


def action_transition_histogram_trace_stat_thresholds(candidate_mode: str, eval_extra_args: list[str]) -> list[str]:
    if not candidate_mode.startswith("pattern_sd"):
        return []
    if not _effective_bool_flag(eval_extra_args, "--pattern-action-transition-histogram"):
        return []
    return [
        "action_transition_histogram_drafted_tokens=0",
        "action_transition_histogram_accepted_tokens=0",
    ]


def trace_stat_total_thresholds(thresholds: list[str], *, minimum: str = "1") -> list[str]:
    totals: list[str] = []
    for threshold in thresholds:
        name = threshold.split("=", 1)[0].strip()
        if name:
            totals.append(f"{name}={minimum}")
    return totals


def add_source_trace_stat_thresholds(
    min_candidate_trace_stats: list[str],
    min_candidate_trace_stat_totals: list[str],
    thresholds: list[str],
) -> None:
    min_candidate_trace_stats.extend(thresholds)
    min_candidate_trace_stat_totals.extend(trace_stat_total_thresholds(thresholds))


def sweep_selection_metadata(sweep: dict[str, Any]) -> dict[str, Any]:
    selection = sweep.get("selection")
    if not isinstance(selection, dict):
        return {}
    heldout_trace_indices = selection.get("heldout_trace_indices")
    heldout_trace_count = len(heldout_trace_indices) if isinstance(heldout_trace_indices, list) else 0
    return {
        "heldout_split": selection.get("heldout_split"),
        "heldout_trace_count": heldout_trace_count,
        "rank_task_count": selection.get("rank_task_count"),
        "heldout_selection_task_count": selection.get("heldout_task_count"),
        "heldout_task_disjoint": selection.get("heldout_task_disjoint"),
        "heldout_task_overlap_count": selection.get("heldout_task_overlap_count"),
        "heldout_task_overlap": selection.get("heldout_task_overlap"),
        "rank_suite_count": selection.get("rank_suite_count"),
        "rank_suite_keys": selection.get("rank_suite_keys"),
        "heldout_suite_count": selection.get("heldout_suite_count"),
        "heldout_suite_keys": selection.get("heldout_suite_keys"),
    }


def validate_pattern_sweep_final_audit_evidence(
    selection: dict[str, Any],
    *,
    suites: list[str],
    require_heldout_suite_coverage: bool,
) -> None:
    """Fail early when a pattern sweep manifest cannot satisfy final audit checks."""

    heldout_split = str(selection.get("heldout_split", "")).strip().lower()
    if heldout_split in {"", "none", "null"}:
        raise ValueError(
            "--pattern-sweep-json must include a non-empty heldout split for final audit; "
            "rerun sweep_pi0fast_pattern_offline.py with --heldout-split task."
        )
    try:
        heldout_trace_count = int(selection.get("heldout_trace_count", 0))
    except (TypeError, ValueError):
        heldout_trace_count = 0
    if heldout_trace_count <= 0:
        raise ValueError("--pattern-sweep-json has no heldout traces for final audit evidence")
    if selection.get("heldout_task_disjoint") is not True:
        raise ValueError("--pattern-sweep-json must prove heldout tasks are disjoint from ranking tasks")
    if not isinstance(selection.get("heldout_task_overlap"), list):
        raise ValueError("--pattern-sweep-json is missing heldout task overlap metadata")
    try:
        overlap_count = int(selection.get("heldout_task_overlap_count", 1))
    except (TypeError, ValueError):
        overlap_count = 1
    if overlap_count != 0:
        raise ValueError("--pattern-sweep-json heldout tasks overlap ranking tasks")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("--pattern-sweep-json did not record a selected sweep row")
    try:
        selected_heldout_tasks = int(selected.get("heldout_task_count", 0))
    except (TypeError, ValueError):
        selected_heldout_tasks = 0
    if selected_heldout_tasks <= 0:
        raise ValueError("--pattern-sweep-json selected row is missing heldout task metrics")
    if not require_heldout_suite_coverage:
        return
    heldout_suite_keys = selection.get("heldout_suite_keys")
    if not isinstance(heldout_suite_keys, list):
        raise ValueError(
            "--pattern-sweep-json is missing heldout suite metadata; rerun "
            "sweep_pi0fast_pattern_offline.py so selection.heldout_suite_keys is recorded."
        )
    missing_suites = sorted(set(suites) - {str(item) for item in heldout_suite_keys})
    if missing_suites:
        raise ValueError(
            "--pattern-sweep-json heldout split does not cover requested suites: "
            + ", ".join(missing_suites)
        )


def resolve_extra_eval_args(args: argparse.Namespace, extra_args: list[str]) -> tuple[list[str], dict[str, Any] | None]:
    if args.pattern_sweep_json is None:
        return list(extra_args), None
    if not args.candidate_mode.startswith("pattern_sd"):
        raise ValueError("--pattern-sweep-json is only valid for pattern_sd candidate modes")
    sweep = load_sweep(args.pattern_sweep_json)
    pattern_args, selected = build_pattern_eval_args(
        sweep,
        rank=args.pattern_sweep_rank,
        min_modeled_speedup=args.pattern_min_modeled_speedup,
        min_forward_reduction=args.pattern_min_forward_reduction,
        min_task_forward_reduction=args.pattern_min_task_forward_reduction,
        min_task_acceptance_rate=args.pattern_min_task_acceptance_rate,
        min_task_count=args.pattern_min_task_count,
        min_heldout_modeled_speedup=args.pattern_min_heldout_modeled_speedup,
        min_heldout_forward_reduction=args.pattern_min_heldout_forward_reduction,
        min_heldout_task_forward_reduction=args.pattern_min_heldout_task_forward_reduction,
        min_heldout_task_acceptance_rate=args.pattern_min_heldout_task_acceptance_rate,
        min_heldout_task_count=args.pattern_min_heldout_task_count,
        min_metric=args.pattern_min_metric,
        min_heldout_metric=args.pattern_min_heldout_metric,
        auto_source_min_metrics=getattr(args, "pattern_auto_source_min_metrics", False),
        auto_source_min_task_metrics=getattr(args, "pattern_auto_source_min_task_metrics", True),
        auto_source_heldout_metrics=getattr(args, "pattern_auto_source_heldout_metrics", True),
        required_source_coverage=getattr(args, "pattern_required_source_coverage", []),
    )
    auto_min_metric: list[str] = []
    auto_min_heldout_metric: list[str] = []
    if getattr(args, "pattern_auto_source_min_metrics", False):
        auto_min_metric, auto_min_heldout_metric = source_usage_metric_thresholds(
            selected,
            include_min_task=getattr(args, "pattern_auto_source_min_task_metrics", True),
            include_heldout=getattr(args, "pattern_auto_source_heldout_metrics", True),
        )
    effective_min_metric = _append_unique(args.pattern_min_metric, auto_min_metric)
    effective_min_heldout_metric = _append_unique(
        args.pattern_min_heldout_metric,
        auto_min_heldout_metric,
    )
    selection = {
        "sweep_json": str(args.pattern_sweep_json),
        "requested_rank": args.pattern_sweep_rank,
        "selected_rank": selected.get("selected_sweep_rank", args.pattern_sweep_rank),
        **sweep_selection_metadata(sweep),
        "min_modeled_speedup": args.pattern_min_modeled_speedup,
        "min_forward_reduction": args.pattern_min_forward_reduction,
        "min_task_forward_reduction": args.pattern_min_task_forward_reduction,
        "min_task_acceptance_rate": args.pattern_min_task_acceptance_rate,
        "min_task_count": args.pattern_min_task_count,
        "min_heldout_modeled_speedup": args.pattern_min_heldout_modeled_speedup,
        "min_heldout_forward_reduction": args.pattern_min_heldout_forward_reduction,
        "min_heldout_task_forward_reduction": args.pattern_min_heldout_task_forward_reduction,
        "min_heldout_task_acceptance_rate": args.pattern_min_heldout_task_acceptance_rate,
        "min_heldout_task_count": args.pattern_min_heldout_task_count,
        "min_metric": args.pattern_min_metric,
        "min_heldout_metric": args.pattern_min_heldout_metric,
        "auto_source_min_metrics": getattr(args, "pattern_auto_source_min_metrics", False),
        "auto_source_min_task_metrics": getattr(args, "pattern_auto_source_min_task_metrics", True),
        "auto_source_heldout_metrics": getattr(args, "pattern_auto_source_heldout_metrics", True),
        "auto_source_min_metric": auto_min_metric,
        "auto_source_min_heldout_metric": auto_min_heldout_metric,
        "required_source_coverage": selected.get(
            "required_source_coverage",
            getattr(args, "pattern_required_source_coverage", []),
        ),
        "required_source_counts": selected.get("required_source_counts", {}),
        "effective_min_metric": effective_min_metric,
        "effective_min_heldout_metric": effective_min_heldout_metric,
        "args": pattern_args,
        "selected": selected,
    }
    return [*extra_args, *pattern_args], selection


def build_manifest(args: argparse.Namespace, extra_args: list[str]) -> dict[str, Any]:
    root = args.root
    run_manifest_path = root / "run_manifest.json"
    speed_modes = parse_csv(args.speed_modes)
    suites = parse_csv(args.suites)
    task_ids = _task_ids_arg(args.task_ids)
    if not speed_modes:
        raise ValueError("--speed-modes must contain at least one mode")
    if not suites:
        raise ValueError("--suites must contain at least one suite")
    min_unique_tasks = resolve_min_unique_tasks(args.min_unique_tasks, suites=suites, task_ids=task_ids)
    if args.candidate_mode not in speed_modes:
        raise ValueError(f"--candidate-mode {args.candidate_mode!r} must be present in --speed-modes")
    eval_extra_args, pattern_sweep_selection = resolve_extra_eval_args(args, extra_args)
    if args.run_final_audit and pattern_sweep_selection is not None:
        validate_pattern_sweep_final_audit_evidence(
            pattern_sweep_selection,
            suites=suites,
            require_heldout_suite_coverage=getattr(args, "pattern_require_heldout_suite_coverage", True),
        )
    run_metadata = eval_metadata(eval_extra_args)
    expected_policy_kind = run_metadata["policy_kind"] if run_metadata.get("policy_kind") != "pi0fast" else None

    reference_mode = args.reference_mode
    if args.candidate_mode != "target_eos":
        if "target_eos" not in speed_modes:
            raise ValueError(
                "PI0-FAST speculative candidates must include target_eos in --speed-modes so "
                "candidate comparisons use stop-token early stop, not fixed-budget decode alone."
            )
        if reference_mode is None:
            reference_mode = "target_eos"
        elif reference_mode != "target_eos":
            raise ValueError(
                "PI0-FAST speculative candidates must use --reference-mode target_eos so "
                "comparisons are against stop-token early stop."
            )
    min_reference_speedup = args.min_reference_speedup
    max_reference_success_drop = args.max_reference_success_drop
    max_reference_success_regressions = args.max_reference_success_regressions
    candidate_is_target_eos_variant = args.candidate_mode.startswith("target_eos_")
    if args.candidate_mode != "target_eos" and reference_mode == "target_eos" and not candidate_is_target_eos_variant:
        if min_reference_speedup is None:
            min_reference_speedup = args.min_speedup
        if max_reference_success_drop is None:
            max_reference_success_drop = args.max_success_drop
        if max_reference_success_regressions is None:
            max_reference_success_regressions = args.max_baseline_success_regressions
    validation_modes = resolve_validation_modes(
        args.validation_modes,
        candidate_mode=args.candidate_mode,
        reference_mode=reference_mode,
    )
    gate_validation_mode = validation_modes[0] if args.gate_validation_mode.strip().lower() == "auto" else args.gate_validation_mode
    extra_gate_validation_modes = [mode for mode in validation_modes if mode != gate_validation_mode]

    preflight_command = None
    if args.run_preflight:
        preflight_command = build_preflight_command(
            python=args.python,
            preflight_script=args.preflight_script,
            device=args.device,
            eval_script=args.eval_script,
            gate_script=args.gate_script,
            modules=args.preflight_modules,
            require_hf_token=args.require_hf_token,
        )

    speed_commands: list[list[str]] = []
    skipped_existing: list[str] = []
    if not args.skip_speed:
        for mode in speed_modes:
            for suite in suites:
                output_dir = root / "speed" / mode / suite
                if args.skip_existing and eval_shard_complete(
                    output_dir,
                    suite=suite,
                    mode=mode,
                    task_ids=task_ids,
                    episodes=args.episodes,
                    seed=args.seed,
                ):
                    skipped_existing.append(str(output_dir))
                    continue
                speed_commands.append(
                    build_eval_command(
                        python=args.python,
                        eval_script=args.eval_script,
                        suite=suite,
                        task_ids=task_ids,
                        episodes=args.episodes,
                        steps=args.steps,
                        mode=mode,
                        output_dir=output_dir,
                        seed=args.seed,
                        device=args.device,
                        dtype=args.dtype,
                        enable_fast_token_hooks=args.enable_fast_token_hooks,
                        smooth_position_delta=args.smooth_position_delta,
                        smooth_rotation_delta=args.smooth_rotation_delta,
                        extra_args=eval_extra_args,
                    )
                )

    validation_commands: list[list[str]] = []
    if not args.skip_validation:
        for mode in validation_modes:
            for suite in suites:
                output_dir = root / "validate" / mode / suite
                if args.skip_existing and eval_shard_complete(
                    output_dir,
                    suite=suite,
                    mode=mode,
                    task_ids=task_ids,
                    episodes=args.episodes,
                    seed=args.seed,
                ):
                    skipped_existing.append(str(output_dir))
                    continue
                validation_commands.append(
                    build_eval_command(
                        python=args.python,
                        eval_script=args.eval_script,
                        suite=suite,
                        task_ids=task_ids,
                        episodes=args.episodes,
                        steps=args.steps,
                        mode=mode,
                        output_dir=output_dir,
                        seed=args.seed,
                        device=args.device,
                        dtype=args.dtype,
                        enable_fast_token_hooks=args.enable_fast_token_hooks,
                        smooth_position_delta=args.smooth_position_delta,
                        smooth_rotation_delta=args.smooth_rotation_delta,
                        extra_args=eval_extra_args,
                    )
                )

    validation_root = None if args.skip_validation else root / "validate"
    gate_command = None
    gate_output = root / "gate.json"
    min_candidate_trace_stats = list(args.min_candidate_trace_stat)
    max_candidate_trace_stats = list(args.max_candidate_trace_stat)
    min_candidate_trace_stat_totals = list(args.min_candidate_trace_stat_total)
    auto_min_trace_stats, auto_max_trace_stats = tree_trace_stat_thresholds(args.candidate_mode, eval_extra_args)
    min_candidate_trace_stats.extend(auto_min_trace_stats)
    max_candidate_trace_stats.extend(auto_max_trace_stats)
    min_candidate_trace_stats.extend(dynamic_tree_width_trace_stat_thresholds(args.candidate_mode, eval_extra_args))
    min_candidate_trace_stats.extend(tree_anchor_target_trace_stat_thresholds(args.candidate_mode, eval_extra_args))
    min_candidate_trace_stats.extend(tree_anchor_target_continuation_trace_stat_thresholds(args.candidate_mode, eval_extra_args))
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        second_order_action_extrapolation_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        previous_chunk_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        chunk_prefix_retrieval_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_token_neighborhood_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        position_mode_histogram_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        global_position_mode_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_dimension_mode_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        hold_action_token_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        ngram_continuation_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        source_agreement_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_trend_regression_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_prefix_lookup_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_vector_suffix_lookup_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_vector_transition_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_repeat_vector_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        chunk_length_stop_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_context_tree_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        source_cooldown_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        source_acceptance_bias_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_transition_histogram_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_delta_histogram_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        action_delta_ngram_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        chunk_position_delta_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    add_source_trace_stat_thresholds(
        min_candidate_trace_stats,
        min_candidate_trace_stat_totals,
        chunk_delta_template_trace_stat_thresholds(args.candidate_mode, eval_extra_args),
    )
    if not args.skip_gate:
        gate_command = build_gate_command(
            python=args.python,
            gate_script=args.gate_script,
            speed_root=root / "speed",
            validation_root=validation_root,
            output=gate_output,
            baseline_mode=args.baseline_mode,
            candidate_mode=args.candidate_mode,
            reference_mode=reference_mode,
            validation_mode=gate_validation_mode,
            extra_validation_modes=extra_gate_validation_modes,
            min_pairs=args.min_pairs,
            min_unique_tasks=min_unique_tasks,
            min_speedup=args.min_speedup,
            min_baseline_successes=args.min_baseline_successes,
            min_suite_matched_pairs=args.min_suite_matched_pairs,
            min_suite_baseline_successes=args.min_suite_baseline_successes,
            min_suite_speedup=args.min_suite_speedup,
            require_matched_steps=args.require_matched_steps,
            max_success_drop=args.max_success_drop,
            max_baseline_success_regressions=args.max_baseline_success_regressions,
            min_validation_episodes=args.min_validation_episodes,
            min_exact_verifies=args.min_exact_verifies,
            max_action_diff=args.max_action_diff,
            min_reference_speedup=min_reference_speedup,
            max_reference_success_drop=max_reference_success_drop,
            max_reference_success_regressions=max_reference_success_regressions,
            min_candidate_trace_stats=min_candidate_trace_stats,
            max_candidate_trace_stats=max_candidate_trace_stats,
            min_candidate_trace_stat_totals=min_candidate_trace_stat_totals,
            metadata=run_metadata,
        )

    synthetic_output = args.synthetic_output or (root / "synthetic_gate.json")
    synthetic_command = None
    if args.run_synthetic:
        if args.skip_existing and json_gate_passed(synthetic_output):
            skipped_existing.append(str(synthetic_output))
        else:
            synthetic_command = build_synthetic_command(
                python=args.python,
                synthetic_script=args.synthetic_script,
                output=synthetic_output,
                tasks=args.synthetic_tasks,
                min_speedup=args.synthetic_min_speedup,
                max_accuracy_drop=args.synthetic_max_accuracy_drop,
            )

    audit_output = args.audit_output or (root / "objective_audit.json")
    audit_command = None
    if args.run_final_audit:
        if args.skip_existing and json_gate_passed(audit_output):
            skipped_existing.append(str(audit_output))
        else:
            audit_command = build_audit_command(
                python=args.python,
                audit_script=args.audit_script,
                pi0fast_gate=gate_output,
                pi0fast_manifest=run_manifest_path,
                synthetic_gate=synthetic_output,
                output=audit_output,
                min_pairs=args.min_pairs,
                min_unique_tasks=min_unique_tasks,
                min_speedup=args.min_speedup,
                min_baseline_successes=args.min_baseline_successes,
                min_suite_matched_pairs=args.min_suite_matched_pairs,
                min_suite_baseline_successes=args.min_suite_baseline_successes,
                min_suite_speedup=args.min_suite_speedup,
                max_success_drop=args.max_success_drop,
                max_baseline_success_regressions=args.max_baseline_success_regressions,
                min_validation_episodes=args.min_validation_episodes,
                max_action_diff=args.max_action_diff,
                expected_policy_kind=expected_policy_kind,
                require_pattern_source_coverage=pattern_sweep_selection is not None,
                require_pattern_heldout_evidence=pattern_sweep_selection is not None,
            )

    result_card_output = args.result_card_output or (root / "result_card.md")
    result_card_command = None
    if args.render_result_card:
        if args.skip_existing and result_card_output.exists():
            skipped_existing.append(str(result_card_output))
        else:
            result_card_command = build_result_card_command(
                python=args.python,
                result_card_script=args.result_card_script,
                audit_json=audit_output,
                output=result_card_output,
            )

    return {
        "root": str(root),
        "run_manifest": str(run_manifest_path),
        "speed_modes": speed_modes,
        "validation_modes": validation_modes,
        "gate_validation_mode": gate_validation_mode,
        "extra_gate_validation_modes": extra_gate_validation_modes,
        "suites": suites,
        "task_ids": task_ids,
        "episodes": args.episodes,
        "matched_eval_count": len(suites) * len(parse_csv(task_ids)) * args.episodes,
        "min_unique_tasks": min_unique_tasks,
        "candidate_mode": args.candidate_mode,
        "eval_metadata": run_metadata,
        "expected_policy_kind": expected_policy_kind,
        "reference_mode": reference_mode,
        "min_reference_speedup": min_reference_speedup,
        "min_baseline_successes": args.min_baseline_successes,
        "min_suite_matched_pairs": args.min_suite_matched_pairs,
        "min_suite_baseline_successes": args.min_suite_baseline_successes,
        "min_suite_speedup": args.min_suite_speedup,
        "require_matched_steps": args.require_matched_steps,
        "max_reference_success_drop": max_reference_success_drop,
        "max_reference_success_regressions": max_reference_success_regressions,
        "eval_extra_args": eval_extra_args,
        "pattern_sweep_selection": pattern_sweep_selection,
        "min_candidate_trace_stats": min_candidate_trace_stats,
        "max_candidate_trace_stats": max_candidate_trace_stats,
        "min_candidate_trace_stat_totals": min_candidate_trace_stat_totals,
        "preflight_command": preflight_command,
        "speed_commands": speed_commands,
        "validation_commands": validation_commands,
        "skipped_existing": skipped_existing,
        "gate_command": gate_command,
        "synthetic_command": synthetic_command,
        "audit_command": audit_command,
        "result_card_command": result_card_command,
    }


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PI0-FAST 120-eval speedup gate.")
    parser.add_argument("--root", type=Path, default=Path("outputs/pi0fast_target_eos_120"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--preflight-script", default="scripts/preflight_pi0fast_eval_env.py")
    parser.add_argument("--eval-script", default="scripts/run_pi0fast_chunk_eval.py")
    parser.add_argument("--gate-script", default="scripts/gate_pi0fast_target_eos.py")
    parser.add_argument("--synthetic-script", default="scripts/benchmark_robotics_spec_decode_synthetic.py")
    parser.add_argument("--audit-script", default="scripts/audit_robotics_spec_goal.py")
    parser.add_argument("--result-card-script", default="scripts/render_robotics_spec_result_card.py")
    parser.add_argument("--suites", default="libero_object,libero_spatial,libero_goal")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enable-fast-token-hooks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-position-delta", type=float, default=0.06)
    parser.add_argument("--smooth-rotation-delta", type=float, default=0.22)
    parser.add_argument(
        "--speed-modes",
        default="baseline,target_eos",
        help="Comma-separated speed modes. Add speculative candidates here, e.g. baseline,target_eos,block_sd_direct.",
    )
    parser.add_argument(
        "--validation-modes",
        default="auto",
        help="Comma-separated validation modes, or auto for candidate-specific exact validation.",
    )
    parser.add_argument("--baseline-mode", default="baseline")
    parser.add_argument("--candidate-mode", default="target_eos")
    parser.add_argument(
        "--reference-mode",
        default=None,
        help=(
            "Optional early-stop reference. Speculative PI0-FAST candidates require target_eos; "
            "unset reference thresholds inherit the main gate thresholds."
        ),
    )
    parser.add_argument("--gate-validation-mode", default="auto")
    parser.add_argument("--min-pairs", type=int, default=120)
    parser.add_argument(
        "--min-unique-tasks",
        type=int,
        default=None,
        help="Minimum unique (suite, task_id) coverage; defaults to requested suites times task IDs.",
    )
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--min-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-matched-pairs", type=int, default=1)
    parser.add_argument("--min-suite-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-speedup", type=float, default=1.0)
    parser.add_argument("--max-success-drop", type=float, default=0.0)
    parser.add_argument("--max-baseline-success-regressions", type=int, default=0)
    parser.add_argument("--require-matched-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-validation-episodes", type=int, default=120)
    parser.add_argument("--min-exact-verifies", type=int, default=1)
    parser.add_argument("--max-action-diff", type=float, default=0.0)
    parser.add_argument("--min-reference-speedup", type=float, default=None)
    parser.add_argument("--max-reference-success-drop", type=float, default=None)
    parser.add_argument("--max-reference-success-regressions", type=int, default=None)
    parser.add_argument(
        "--min-candidate-trace-stat",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Forwarded to the PI0-FAST gate; every matched candidate row must report trace_stats[NAME] >= VALUE.",
    )
    parser.add_argument(
        "--max-candidate-trace-stat",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Forwarded to the PI0-FAST gate; matched candidate trace_stats[NAME] must be <= VALUE.",
    )
    parser.add_argument(
        "--min-candidate-trace-stat-total",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Forwarded to the PI0-FAST gate; total matched candidate trace_stats[NAME] must be >= VALUE.",
    )
    parser.add_argument(
        "--pattern-sweep-json",
        type=Path,
        default=None,
        help="Optional sweep_pi0fast_pattern_offline.py JSON used to append selected pattern_sd args.",
    )
    parser.add_argument(
        "--pattern-sweep-rank",
        type=int,
        default=1,
        help="1-based row rank from the pattern sweep top list. Use 0 to auto-select the first row passing thresholds.",
    )
    parser.add_argument("--pattern-min-modeled-speedup", type=float, default=None)
    parser.add_argument("--pattern-min-forward-reduction", type=float, default=None)
    parser.add_argument("--pattern-min-task-forward-reduction", type=float, default=None)
    parser.add_argument("--pattern-min-task-acceptance-rate", type=float, default=None)
    parser.add_argument("--pattern-min-task-count", type=int, default=None)
    parser.add_argument("--pattern-min-heldout-modeled-speedup", type=float, default=None)
    parser.add_argument("--pattern-min-heldout-forward-reduction", type=float, default=None)
    parser.add_argument("--pattern-min-heldout-task-forward-reduction", type=float, default=None)
    parser.add_argument("--pattern-min-heldout-task-acceptance-rate", type=float, default=None)
    parser.add_argument("--pattern-min-heldout-task-count", type=int, default=None)
    parser.add_argument(
        "--pattern-min-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require selected pattern sweep train metric NAME to be at least VALUE.",
    )
    parser.add_argument(
        "--pattern-min-heldout-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require selected pattern sweep heldout metric NAME to be at least VALUE; heldout_ is added automatically.",
    )
    parser.add_argument(
        "--pattern-auto-source-min-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For the selected pattern sweep row, automatically require every enabled optional source prior "
            "to have accepted tokens before launching eval shards."
        ),
    )
    parser.add_argument(
        "--pattern-auto-source-min-task-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When auto source metrics are enabled, also require min_task_* accepted-token metrics.",
    )
    parser.add_argument(
        "--pattern-auto-source-heldout-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When auto source metrics are enabled and the selected row has heldout metrics, require heldout source metrics too.",
    )
    parser.add_argument(
        "--pattern-required-source-coverage",
        action="append",
        default=[],
        metavar="SOURCE",
        help=(
            "Require the pattern sweep JSON to report evaluated source coverage for SOURCE before "
            "selecting a row. May be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--pattern-require-heldout-suite-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For final-audit pattern runs, require sweep heldout traces to cover every requested suite.",
    )
    parser.add_argument("--run-synthetic", action="store_true")
    parser.add_argument("--synthetic-output", type=Path, default=None)
    parser.add_argument("--synthetic-tasks", type=int, default=120)
    parser.add_argument("--synthetic-min-speedup", type=float, default=2.0)
    parser.add_argument("--synthetic-max-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--run-final-audit", action="store_true")
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--render-result-card", action="store_true")
    parser.add_argument("--result-card-output", type=Path, default=None)
    parser.add_argument("--run-preflight", action="store_true")
    parser.add_argument("--preflight-modules", default="torch,lerobot,libero,robosuite,mujoco,networkx,dist:hf_libero")
    parser.add_argument("--require-hf-token", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rerun suite/mode shards whose metrics.jsonl already contains every expected row.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    manifest = build_manifest(args, extra_args)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Matched eval count: {manifest['matched_eval_count']}")
    if manifest["skipped_existing"]:
        print(f"Skipping completed shards: {len(manifest['skipped_existing'])}")

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "osmesa")
    if manifest["preflight_command"] is not None:
        run_command(manifest["preflight_command"], env=env, dry_run=args.dry_run)
    for cmd in manifest["speed_commands"]:
        run_command(cmd, env=env, dry_run=args.dry_run)
    for cmd in manifest["validation_commands"]:
        run_command(cmd, env=env, dry_run=args.dry_run)
    if manifest["gate_command"] is not None:
        run_command(manifest["gate_command"], env=env, dry_run=args.dry_run)
    if manifest["synthetic_command"] is not None:
        run_command(manifest["synthetic_command"], env=env, dry_run=args.dry_run)
    if manifest["audit_command"] is not None:
        run_command(manifest["audit_command"], env=env, dry_run=args.dry_run)
    if manifest["result_card_command"] is not None:
        run_command(manifest["result_card_command"], env=env, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run only; no eval or gate commands were executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
