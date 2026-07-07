from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

from scripts.run_pi0fast_100_eval_gate import build_manifest, eval_metadata, eval_shard_complete, parse_args


def _args(**overrides):
    values = {
        "root": Path("outputs/test_pi0fast_gate"),
        "python": "python",
        "preflight_script": "scripts/preflight_pi0fast_eval_env.py",
        "eval_script": "scripts/run_pi0fast_chunk_eval.py",
        "gate_script": "scripts/gate_pi0fast_target_eos.py",
        "synthetic_script": "scripts/benchmark_robotics_spec_decode_synthetic.py",
        "audit_script": "scripts/audit_robotics_spec_goal.py",
        "result_card_script": "scripts/render_robotics_spec_result_card.py",
        "suites": "libero_object,libero_goal",
        "task_ids": "0,1,2",
        "episodes": 4,
        "steps": 300,
        "seed": 42,
        "device": "cuda",
        "dtype": "bfloat16",
        "enable_fast_token_hooks": True,
        "smooth_position_delta": 0.06,
        "smooth_rotation_delta": 0.22,
        "speed_modes": "baseline,target_eos,block_sd_direct",
        "validation_modes": "auto",
        "baseline_mode": "baseline",
        "candidate_mode": "block_sd_direct",
        "reference_mode": None,
        "gate_validation_mode": "auto",
        "min_pairs": 120,
        "min_unique_tasks": None,
        "min_speedup": 2.0,
        "min_baseline_successes": 1,
        "min_suite_matched_pairs": 1,
        "min_suite_baseline_successes": 1,
        "min_suite_speedup": 1.0,
        "max_success_drop": 0.0,
        "max_baseline_success_regressions": 0,
        "require_matched_steps": True,
        "min_validation_episodes": 120,
        "min_exact_verifies": 1,
        "max_action_diff": 0.0,
        "min_reference_speedup": None,
        "max_reference_success_drop": None,
        "max_reference_success_regressions": None,
        "min_candidate_trace_stat": [],
        "max_candidate_trace_stat": [],
        "min_candidate_trace_stat_total": [],
        "pattern_sweep_json": None,
        "pattern_sweep_rank": 1,
        "pattern_min_modeled_speedup": None,
        "pattern_min_forward_reduction": None,
        "pattern_min_task_forward_reduction": None,
        "pattern_min_task_acceptance_rate": None,
        "pattern_min_task_count": None,
        "pattern_min_heldout_modeled_speedup": None,
        "pattern_min_heldout_forward_reduction": None,
        "pattern_min_heldout_task_forward_reduction": None,
        "pattern_min_heldout_task_acceptance_rate": None,
        "pattern_min_heldout_task_count": None,
        "pattern_min_metric": [],
        "pattern_min_heldout_metric": [],
        "pattern_auto_source_min_metrics": False,
        "pattern_auto_source_min_task_metrics": True,
        "pattern_auto_source_heldout_metrics": True,
        "pattern_required_source_coverage": [],
        "pattern_require_heldout_suite_coverage": True,
        "run_synthetic": False,
        "synthetic_output": None,
        "synthetic_tasks": 120,
        "synthetic_min_speedup": 2.0,
        "synthetic_max_accuracy_drop": 0.0,
        "run_final_audit": False,
        "audit_output": None,
        "render_result_card": False,
        "result_card_output": None,
        "run_preflight": False,
        "preflight_modules": "torch,lerobot,libero,robosuite,mujoco,networkx,dist:hf_libero",
        "require_hf_token": False,
        "skip_speed": False,
        "skip_validation": False,
        "skip_gate": False,
        "skip_existing": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_manifest_auto_compares_spec_candidate_to_target_eos() -> None:
    manifest = build_manifest(_args(), [])

    assert manifest["matched_eval_count"] == 2 * 3 * 4
    assert manifest["reference_mode"] == "target_eos"
    assert len(manifest["speed_commands"]) == 2 * 3
    assert len(manifest["validation_commands"]) == 4
    assert manifest["validation_modes"] == ["block_sd_validate", "target_eos_validate"]
    assert manifest["gate_validation_mode"] == "block_sd_validate"
    assert manifest["extra_gate_validation_modes"] == ["target_eos_validate"]
    assert "block_sd_validate" in manifest["gate_command"]
    assert "--extra-validation-mode" in manifest["gate_command"]
    assert "--reference-mode" in manifest["gate_command"]
    assert "target_eos" in manifest["gate_command"]
    assert manifest["min_reference_speedup"] == 2.0
    assert manifest["max_reference_success_drop"] == 0.0
    assert manifest["max_reference_success_regressions"] == 0
    assert "--min-reference-speedup" in manifest["gate_command"]
    assert "--max-reference-success-drop" in manifest["gate_command"]
    assert "--max-reference-success-regressions" in manifest["gate_command"]
    assert "--min-pairs" in manifest["gate_command"]
    assert manifest["gate_command"][manifest["gate_command"].index("--min-pairs") + 1] == "120"
    assert "--min-validation-episodes" in manifest["gate_command"]
    assert manifest["gate_command"][manifest["gate_command"].index("--min-validation-episodes") + 1] == "120"
    assert "--require-matched-steps" in manifest["gate_command"]


def test_manifest_auto_validates_target_cutoff_candidate() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,target_cutoff96",
            candidate_mode="target_cutoff96",
        ),
        [],
    )

    assert manifest["reference_mode"] == "target_eos"
    assert manifest["validation_modes"] == ["target_cutoff96_validate", "target_eos_validate"]
    assert manifest["gate_validation_mode"] == "target_cutoff96_validate"
    assert manifest["extra_gate_validation_modes"] == ["target_eos_validate"]
    assert "target_cutoff96_validate" in manifest["gate_command"]


def test_manifest_auto_validates_target_eos_charstop_candidate() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,target_eos_charstop",
            candidate_mode="target_eos_charstop",
        ),
        [],
    )

    assert manifest["reference_mode"] == "target_eos"
    assert manifest["validation_modes"] == ["target_eos_charstop_validate", "target_eos_validate"]
    assert manifest["gate_validation_mode"] == "target_eos_charstop_validate"
    assert manifest["extra_gate_validation_modes"] == ["target_eos_validate"]


def test_manifest_auto_validates_target_eos_constrained_candidate_without_reference_speedup() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,target_eos_constrained_noforce",
            candidate_mode="target_eos_constrained_noforce",
        ),
        [
            "--target-eos-constrained-full-head-margin",
            "1.0",
            "--target-eos-constrained-no-force-prefix",
        ],
    )

    assert manifest["reference_mode"] == "target_eos"
    assert manifest["validation_modes"] == ["target_eos_constrained_noforce_validate", "target_eos_validate"]
    assert manifest["gate_validation_mode"] == "target_eos_constrained_noforce_validate"
    assert manifest["extra_gate_validation_modes"] == ["target_eos_validate"]
    assert manifest["min_reference_speedup"] is None
    assert "--min-reference-speedup" not in manifest["gate_command"]
    assert "--target-eos-constrained-full-head-margin" in manifest["eval_extra_args"]
    assert "--target-eos-constrained-no-force-prefix" in manifest["eval_extra_args"]


def test_eval_metadata_extracts_pi05_policy_args() -> None:
    metadata = eval_metadata(["--policy-kind", "pi05", "--num-inference-steps=8", "--policy", "local/pi05"])

    assert metadata == {
        "policy_kind": "pi05",
        "policy": "local/pi05",
        "num_inference_steps": "8",
    }


def test_manifest_records_pi05_metadata_and_audit_expectation(tmp_path: Path) -> None:
    manifest = build_manifest(
        _args(
            root=tmp_path / "run",
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
            run_final_audit=True,
        ),
        ["--policy-kind", "pi05", "--num-inference-steps", "8"],
    )

    assert manifest["eval_metadata"]["policy_kind"] == "pi05"
    assert manifest["eval_metadata"]["num_inference_steps"] == "8"
    assert manifest["expected_policy_kind"] == "pi05"
    assert "--metadata" in manifest["gate_command"]
    assert "policy_kind=pi05" in manifest["gate_command"]
    assert "num_inference_steps=8" in manifest["gate_command"]
    assert "--pi0fast-manifest" in manifest["audit_command"]
    assert str(tmp_path / "run" / "run_manifest.json") in manifest["audit_command"]
    assert "--expected-policy-kind" in manifest["audit_command"]
    assert "pi05" in manifest["audit_command"]
    assert "--require-pattern-source-coverage" not in manifest["audit_command"]
    assert "--reference-mode" in manifest["gate_command"]
    assert "target_eos" in manifest["gate_command"]


def test_pi0fast_gate_parser_defaults_to_120_eval_thresholds() -> None:
    old_argv = sys.argv
    try:
        sys.argv = ["run_pi0fast_100_eval_gate.py"]
        args = parse_args()
    finally:
        sys.argv = old_argv

    assert args.min_pairs == 120
    assert args.min_validation_episodes == 120
    assert args.synthetic_tasks == 120
    assert args.require_matched_steps is True
    assert args.pattern_require_heldout_suite_coverage is True


def test_manifest_allows_explicit_reference_threshold_overrides() -> None:
    manifest = build_manifest(
        _args(
            min_reference_speedup=1.25,
            max_reference_success_drop=0.01,
            max_reference_success_regressions=2,
        ),
        [],
    )

    assert manifest["min_reference_speedup"] == 1.25
    assert manifest["max_reference_success_drop"] == 0.01
    assert manifest["max_reference_success_regressions"] == 2
    assert "1.25" in manifest["gate_command"]
    assert "0.01" in manifest["gate_command"]
    assert "2" in manifest["gate_command"]


def test_manifest_rejects_spec_candidate_without_target_eos_reference() -> None:
    try:
        build_manifest(
            _args(
                speed_modes="baseline,block_sd_direct",
                candidate_mode="block_sd_direct",
            ),
            [],
        )
    except ValueError as exc:
        assert "target_eos" in str(exc)
        assert "stop-token early stop" in str(exc)
    else:
        raise AssertionError("expected missing target_eos reference to fail")


def test_manifest_rejects_non_target_eos_reference_for_spec_candidate() -> None:
    try:
        build_manifest(
            _args(
                speed_modes="baseline,target_eos,block_sd_direct",
                candidate_mode="block_sd_direct",
                reference_mode="baseline",
            ),
            [],
        )
    except ValueError as exc:
        assert "--reference-mode target_eos" in str(exc)
        assert "stop-token early stop" in str(exc)
    else:
        raise AssertionError("expected non-target_eos reference to fail")


def test_manifest_forwards_pattern_candidate_args() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-lookahead", "8", "--pattern-action-dim", "7"],
    )

    pattern_commands = [cmd for cmd in manifest["speed_commands"] if "pattern_sd_direct" in cmd]
    assert len(pattern_commands) == 2
    assert manifest["reference_mode"] == "target_eos"
    assert manifest["min_reference_speedup"] == 2.0
    assert manifest["max_reference_success_drop"] == 0.0
    assert manifest["max_reference_success_regressions"] == 0
    assert manifest["validation_modes"] == ["pattern_sd_validate", "target_eos_validate"]
    assert manifest["extra_gate_validation_modes"] == ["target_eos_validate"]
    assert "pattern_sd_validate" in manifest["gate_command"]
    assert "--extra-validation-mode" in manifest["gate_command"]
    assert "--reference-mode" in manifest["gate_command"]
    assert "pattern_sd_direct" in manifest["gate_command"]
    for cmd in pattern_commands:
        assert "--pattern-lookahead" in cmd
        assert "8" in cmd
        assert "--pattern-action-dim" in cmd
        assert "7" in cmd


def test_manifest_manual_pattern_tree_args_require_tree_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-tree-width", "4", "--pattern-tree-branch-width=3"],
    )

    assert "--pattern-tree-width" in manifest["eval_extra_args"]
    assert manifest["min_candidate_trace_stats"] == [
        "tree_width=4",
        "tree_verifies=1e-09",
    ]
    assert manifest["max_candidate_trace_stats"] == [
        "unverified_pattern_tokens=0",
        "unverified_pattern_eos_tokens=0",
    ]
    assert "tree_width=4" in manifest["gate_command"]
    assert "tree_verifies=1e-09" in manifest["gate_command"]
    assert "unverified_pattern_tokens=0" in manifest["gate_command"]
    assert "unverified_pattern_eos_tokens=0" in manifest["gate_command"]


def test_manifest_dynamic_tree_width_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-tree-width", "4", "--pattern-dynamic-tree-width"],
    )

    assert "dynamic_tree_width=1" in manifest["min_candidate_trace_stats"]
    assert "mean_tree_width=1e-09" in manifest["min_candidate_trace_stats"]
    assert "dynamic_tree_width=1" in manifest["gate_command"]
    assert "mean_tree_width=1e-09" in manifest["gate_command"]


def test_manifest_tree_anchor_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-tree-width", "4", "--pattern-tree-anchor-target-token"],
    )

    assert "tree_anchor_target_token=1" in manifest["min_candidate_trace_stats"]
    assert "tree_anchor_verifies=1e-09" in manifest["min_candidate_trace_stats"]
    assert "tree_anchor_accepted_tokens=1e-09" in manifest["min_candidate_trace_stats"]
    assert "tree_anchor_target_token=1" in manifest["gate_command"]
    assert "tree_anchor_verifies=1e-09" in manifest["gate_command"]
    assert "tree_anchor_accepted_tokens=1e-09" in manifest["gate_command"]


def test_manifest_tree_anchor_continuation_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-tree-width", "4", "--pattern-tree-anchor-target-continuation"],
    )

    assert "tree_anchor_target_continuation=1" in manifest["min_candidate_trace_stats"]
    assert "tree_anchor_verifies=1e-09" in manifest["min_candidate_trace_stats"]
    assert "tree_anchor_accepted_tokens=1e-09" in manifest["min_candidate_trace_stats"]
    assert "tree_anchor_target_continuation=1" in manifest["gate_command"]
    assert "tree_anchor_verifies=1e-09" in manifest["gate_command"]
    assert "tree_anchor_accepted_tokens=1e-09" in manifest["gate_command"]


def test_manifest_previous_chunk_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-previous-chunk-position", "--pattern-previous-chunk-history-size", "2"],
    )

    assert "previous_chunk_position_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "previous_chunk_position_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "previous_chunk_position_drafted_tokens=0" in manifest["gate_command"]
    assert "previous_chunk_position_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_chunk_prefix_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-chunk-prefix-retrieval"],
    )

    assert "chunk_prefix_retrieval_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_prefix_retrieval_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_prefix_retrieval_drafted_tokens=0" in manifest["gate_command"]
    assert "chunk_prefix_retrieval_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_action_token_neighborhood_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-token-neighborhood", "--pattern-action-token-neighborhood-radius", "2"],
    )

    assert "action_token_neighborhood_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_token_neighborhood_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_token_neighborhood_drafted_tokens=0" in manifest["gate_command"]
    assert "action_token_neighborhood_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_previous_chunk_pattern_args_respect_last_boolean_flag() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-previous-chunk-position", "--no-pattern-previous-chunk-position"],
    )

    assert "previous_chunk_position_drafted_tokens=0" not in manifest["min_candidate_trace_stats"]
    assert "previous_chunk_position_drafted_tokens=0" not in manifest["gate_command"]


def test_manifest_ngram_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-ngram-continuation", "--pattern-ngram-min-context", "3"],
    )

    assert "ngram_continuation_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "ngram_continuation_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "ngram_continuation_drafted_tokens=0" in manifest["gate_command"]
    assert "ngram_continuation_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_position_mode_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-position-mode-histogram", "--pattern-position-mode-history-size", "4"],
    )

    assert "position_mode_histogram_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "position_mode_histogram_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "position_mode_histogram_drafted_tokens=0" in manifest["gate_command"]
    assert "position_mode_histogram_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_global_position_mode_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-global-position-mode", "--pattern-global-position-history-size", "16"],
    )

    assert "global_position_mode_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "global_position_mode_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "global_position_mode_drafted_tokens=0" in manifest["gate_command"]
    assert "global_position_mode_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_action_dimension_mode_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-dimension-mode", "--pattern-action-dimension-mode-history-size", "4"],
    )

    assert "action_dimension_mode_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_dimension_mode_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_dimension_mode_drafted_tokens=0" in manifest["gate_command"]
    assert "action_dimension_mode_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_hold_action_token_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-hold-action-token"],
    )

    assert "hold_action_token_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "hold_action_token_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "hold_action_token_drafted_tokens=0" in manifest["gate_command"]
    assert "hold_action_token_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_source_agreement_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-min-source-agreement", "2"],
    )

    assert "source_agreement_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "source_agreement_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "source_agreement_drafted_tokens=0" in manifest["gate_command"]
    assert "source_agreement_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_action_trend_regression_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-trend-regression", "--pattern-action-trend-history", "4"],
    )

    assert "action_trend_regression_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_trend_regression_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_trend_regression_drafted_tokens=0" in manifest["gate_command"]
    assert "action_trend_regression_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_action_prefix_lookup_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-prefix-lookup", "--pattern-action-prefix-history-size", "4"],
    )

    assert "action_prefix_lookup_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_prefix_lookup_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_prefix_lookup_drafted_tokens=0" in manifest["gate_command"]
    assert "action_prefix_lookup_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_action_vector_suffix_lookup_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-vector-suffix-lookup", "--pattern-action-vector-suffix-history-size", "4"],
    )

    assert "action_vector_suffix_lookup_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_vector_suffix_lookup_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_vector_suffix_lookup_drafted_tokens=0" in manifest["gate_command"]
    assert "action_vector_suffix_lookup_accepted_tokens=0" in manifest["gate_command"]
    assert "action_vector_suffix_lookup_drafted_tokens=1" in manifest["min_candidate_trace_stat_totals"]
    assert "action_vector_suffix_lookup_accepted_tokens=1" in manifest["min_candidate_trace_stat_totals"]
    assert "--min-candidate-trace-stat-total" in manifest["gate_command"]
    assert "action_vector_suffix_lookup_accepted_tokens=1" in manifest["gate_command"]


def test_manifest_action_vector_transition_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-vector-transition", "--pattern-action-vector-transition-history-size", "4"],
    )

    assert "action_vector_transition_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_vector_transition_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_vector_transition_drafted_tokens=0" in manifest["gate_command"]
    assert "action_vector_transition_accepted_tokens=0" in manifest["gate_command"]
    assert "action_vector_transition_drafted_tokens=1" in manifest["min_candidate_trace_stat_totals"]
    assert "action_vector_transition_accepted_tokens=1" in manifest["min_candidate_trace_stat_totals"]
    assert "--min-candidate-trace-stat-total" in manifest["gate_command"]
    assert "action_vector_transition_accepted_tokens=1" in manifest["gate_command"]


def test_manifest_action_repeat_vector_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-repeat-vector", "--pattern-action-repeat-min-repeats", "2"],
    )

    assert "action_repeat_vector_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_repeat_vector_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_repeat_vector_drafted_tokens=0" in manifest["gate_command"]
    assert "action_repeat_vector_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_chunk_length_stop_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-chunk-length-stop", "--pattern-chunk-length-stop-min-count", "2"],
    )

    assert "chunk_length_stop_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_length_stop_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_length_stop_drafted_tokens=0" in manifest["gate_command"]
    assert "chunk_length_stop_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_action_context_tree_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-context-tree", "--pattern-action-context-tree-max-context", "3"],
    )

    assert "action_context_tree_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_context_tree_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_context_tree_drafted_tokens=0" in manifest["gate_command"]
    assert "action_context_tree_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_source_cooldown_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-source-cooldown", "--pattern-source-cooldown-after", "1"],
    )

    assert "source_cooldown_events=0" in manifest["min_candidate_trace_stats"]
    assert "source_cooldown_skipped_sources=0" in manifest["min_candidate_trace_stats"]
    assert "source_cooldown_events=0" in manifest["gate_command"]
    assert "source_cooldown_skipped_sources=0" in manifest["gate_command"]


def test_manifest_source_acceptance_bias_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-source-acceptance-bias", "--pattern-source-acceptance-bias-history-size", "8"],
    )

    assert "source_acceptance_bias_events=0" in manifest["min_candidate_trace_stats"]
    assert "source_acceptance_bias_reorders=0" in manifest["min_candidate_trace_stats"]
    assert "source_acceptance_bias_events=0" in manifest["gate_command"]
    assert "source_acceptance_bias_reorders=0" in manifest["gate_command"]


def test_manifest_action_delta_histogram_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        [
            "--pattern-action-delta-histogram",
            "--pattern-action-delta-history",
            "4",
            "--pattern-action-delta-min-count",
            "2",
        ],
    )

    assert "action_delta_histogram_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_delta_histogram_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_delta_histogram_drafted_tokens=0" in manifest["gate_command"]
    assert "action_delta_histogram_accepted_tokens=0" in manifest["gate_command"]
    pattern_commands = [cmd for cmd in manifest["speed_commands"] if "pattern_sd_direct" in cmd]
    assert pattern_commands
    for cmd in pattern_commands:
        assert "--pattern-action-delta-min-count" in cmd
        assert cmd[cmd.index("--pattern-action-delta-min-count") + 1] == "2"


def test_manifest_action_delta_ngram_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-delta-ngram", "--pattern-action-delta-ngram-history-size", "4"],
    )

    assert "action_delta_ngram_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_delta_ngram_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_delta_ngram_drafted_tokens=0" in manifest["gate_command"]
    assert "action_delta_ngram_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_chunk_position_delta_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        [
            "--pattern-chunk-position-delta",
            "--pattern-chunk-position-delta-history-size",
            "4",
            "--pattern-chunk-position-delta-min-count",
            "2",
        ],
    )

    assert "chunk_position_delta_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_position_delta_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_position_delta_drafted_tokens=0" in manifest["gate_command"]
    assert "chunk_position_delta_accepted_tokens=0" in manifest["gate_command"]
    pattern_commands = [cmd for cmd in manifest["speed_commands"] if "pattern_sd_direct" in cmd]
    assert pattern_commands
    for cmd in pattern_commands:
        assert "--pattern-chunk-position-delta-min-count" in cmd
        assert cmd[cmd.index("--pattern-chunk-position-delta-min-count") + 1] == "2"


def test_manifest_chunk_delta_template_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-chunk-delta-template", "--pattern-chunk-delta-template-history-size", "4"],
    )

    assert "chunk_delta_template_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_delta_template_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_delta_template_drafted_tokens=0" in manifest["gate_command"]
    assert "chunk_delta_template_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_action_transition_histogram_pattern_args_request_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-action-transition-histogram", "--pattern-action-transition-history-size", "4"],
    )

    assert "action_transition_histogram_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_transition_histogram_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "action_transition_histogram_drafted_tokens=0" in manifest["gate_command"]
    assert "action_transition_histogram_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_manual_pattern_tree_args_use_effective_last_width() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-tree-width", "1", "--pattern-tree-width=5"],
    )

    assert manifest["min_candidate_trace_stats"] == ["tree_width=5", "tree_verifies=1e-09"]
    assert "tree_width=5" in manifest["gate_command"]


def test_manifest_manual_chain_pattern_args_do_not_require_tree_trace_stats() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
        ),
        ["--pattern-tree-width", "1"],
    )

    assert manifest["min_candidate_trace_stats"] == []
    assert manifest["max_candidate_trace_stats"] == []
    assert "tree_width=" not in manifest["gate_command"]


def test_manifest_can_load_pattern_sweep_args(tmp_path: Path) -> None:
    import json

    sweep = {
        "selection": {
            "heldout_split": "task",
            "heldout_trace_indices": [2, 5, 8],
            "rank_task_count": 10,
            "heldout_task_count": 3,
            "heldout_task_disjoint": True,
            "heldout_task_overlap_count": 0,
            "heldout_task_overlap": [],
            "rank_suite_count": 3,
            "rank_suite_keys": ["libero_goal", "libero_object", "libero_spatial"],
            "heldout_suite_count": 3,
            "heldout_suite_keys": ["libero_goal", "libero_object", "libero_spatial"],
        },
        "top": [
            {
                "modeled_speedup": 2.25,
                "target_forward_reduction": 2.5,
                "min_task_target_forward_reduction": 2.05,
                "min_task_acceptance_rate": 0.72,
                "task_count": 10,
                "heldout_modeled_speedup": 2.1,
                "heldout_target_forward_reduction": 2.2,
                "heldout_min_task_target_forward_reduction": 1.8,
                "heldout_min_task_acceptance_rate": 0.68,
                "heldout_task_count": 3,
                "ngram_continuation_accepted_tokens": 7,
                "heldout_ngram_continuation_accepted_tokens": 2,
                "config": {
                    "lookahead": 8,
                    "action_dim": 7,
                    "max_period": 16,
                    "min_period_repeats": 2,
                    "repeat_token_min_run": 3,
                    "linear_action_extrapolation": True,
                    "second_order_action_extrapolation": True,
                    "second_order_max_accel": 4,
                    "reuse_full_blocks": True,
                    "emit_bonus_token": True,
                    "tree_width": 4,
                    "tree_branch_width": 3,
                    "source_priority": ["ngram_continuation", "linear_action_extrapolation"],
                },
            }
        ]
    }
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(sweep) + "\n")

    manifest = build_manifest(
        _args(
            root=tmp_path / "run",
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
            pattern_sweep_json=sweep_path,
            pattern_min_modeled_speedup=2.0,
            pattern_min_forward_reduction=2.0,
            pattern_min_task_forward_reduction=2.0,
            pattern_min_task_acceptance_rate=0.7,
            pattern_min_task_count=10,
            pattern_min_heldout_modeled_speedup=2.0,
            pattern_min_heldout_forward_reduction=2.0,
            pattern_min_heldout_task_forward_reduction=1.5,
            pattern_min_heldout_task_acceptance_rate=0.6,
            pattern_min_heldout_task_count=3,
            pattern_min_metric=["ngram_continuation_accepted_tokens=5"],
            pattern_min_heldout_metric=["ngram_continuation_accepted_tokens=2"],
            run_final_audit=True,
        ),
        [],
    )

    assert manifest["pattern_sweep_selection"]["sweep_json"] == str(sweep_path)
    assert manifest["pattern_sweep_selection"]["heldout_split"] == "task"
    assert manifest["pattern_sweep_selection"]["heldout_trace_count"] == 3
    assert manifest["pattern_sweep_selection"]["rank_task_count"] == 10
    assert manifest["pattern_sweep_selection"]["heldout_selection_task_count"] == 3
    assert manifest["pattern_sweep_selection"]["heldout_task_disjoint"] is True
    assert manifest["pattern_sweep_selection"]["heldout_task_overlap_count"] == 0
    assert manifest["pattern_sweep_selection"]["heldout_task_overlap"] == []
    assert manifest["pattern_sweep_selection"]["rank_suite_count"] == 3
    assert manifest["pattern_sweep_selection"]["rank_suite_keys"] == [
        "libero_goal",
        "libero_object",
        "libero_spatial",
    ]
    assert manifest["pattern_sweep_selection"]["heldout_suite_count"] == 3
    assert manifest["pattern_sweep_selection"]["heldout_suite_keys"] == [
        "libero_goal",
        "libero_object",
        "libero_spatial",
    ]
    assert manifest["pattern_sweep_selection"]["selected"]["modeled_speedup"] == 2.25
    assert manifest["pattern_sweep_selection"]["min_task_forward_reduction"] == 2.0
    assert manifest["pattern_sweep_selection"]["min_task_acceptance_rate"] == 0.7
    assert manifest["pattern_sweep_selection"]["min_task_count"] == 10
    assert manifest["pattern_sweep_selection"]["min_heldout_modeled_speedup"] == 2.0
    assert manifest["pattern_sweep_selection"]["min_heldout_forward_reduction"] == 2.0
    assert manifest["pattern_sweep_selection"]["min_heldout_task_forward_reduction"] == 1.5
    assert manifest["pattern_sweep_selection"]["min_heldout_task_acceptance_rate"] == 0.6
    assert manifest["pattern_sweep_selection"]["min_heldout_task_count"] == 3
    assert manifest["pattern_sweep_selection"]["min_metric"] == ["ngram_continuation_accepted_tokens=5"]
    assert manifest["pattern_sweep_selection"]["min_heldout_metric"] == ["ngram_continuation_accepted_tokens=2"]
    assert "--pattern-max-period" in manifest["eval_extra_args"]
    assert "16" in manifest["eval_extra_args"]
    assert "--pattern-reuse-full-blocks" in manifest["eval_extra_args"]
    assert "--pattern-emit-bonus-token" in manifest["eval_extra_args"]
    assert "--pattern-second-order-action-extrapolation" in manifest["eval_extra_args"]
    assert "--pattern-second-order-max-accel" in manifest["eval_extra_args"]
    assert "--pattern-tree-width" in manifest["eval_extra_args"]
    assert "4" in manifest["eval_extra_args"]
    assert "--pattern-tree-branch-width" in manifest["eval_extra_args"]
    assert "3" in manifest["eval_extra_args"]
    assert "--pattern-source-priority" in manifest["eval_extra_args"]
    assert "ngram_continuation,linear_action_extrapolation" in manifest["eval_extra_args"]
    pattern_commands = [cmd for cmd in manifest["speed_commands"] if "pattern_sd_direct" in cmd]
    assert "--pattern-repeat-token-min-run" in pattern_commands[0]
    assert "3" in pattern_commands[0]
    assert "--pattern-second-order-action-extrapolation" in pattern_commands[0]
    assert "--pattern-tree-width" in pattern_commands[0]
    assert "--pattern-source-priority" in pattern_commands[0]
    assert "--min-candidate-trace-stat" in manifest["gate_command"]
    assert "tree_width=4" in manifest["gate_command"]
    assert "tree_verifies=1e-09" in manifest["gate_command"]
    assert "--max-candidate-trace-stat" in manifest["gate_command"]
    assert "unverified_pattern_tokens=0" in manifest["gate_command"]
    assert manifest["min_candidate_trace_stats"] == [
        "tree_width=4",
        "tree_verifies=1e-09",
        "second_order_action_extrapolation_drafted_tokens=0",
        "second_order_action_extrapolation_accepted_tokens=0",
    ]
    assert manifest["audit_command"] is not None
    assert "--require-pattern-heldout-evidence" in manifest["audit_command"]


def test_manifest_rejects_pattern_final_audit_without_heldout_metadata(tmp_path: Path) -> None:
    import json

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "top": [
                    {
                        "modeled_speedup": 2.25,
                        "target_forward_reduction": 2.5,
                        "heldout_task_count": 2,
                        "heldout_ngram_continuation_accepted_tokens": 2,
                        "config": {
                            "lookahead": 8,
                            "action_dim": 7,
                            "max_period": 16,
                            "min_period_repeats": 2,
                            "repeat_token_min_run": 3,
                            "linear_action_extrapolation": False,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    try:
        build_manifest(
            _args(
                root=tmp_path / "run",
                speed_modes="baseline,target_eos,pattern_sd_direct",
                candidate_mode="pattern_sd_direct",
                pattern_sweep_json=sweep_path,
                run_final_audit=True,
            ),
            [],
        )
    except ValueError as exc:
        assert "non-empty heldout split" in str(exc)
    else:
        raise AssertionError("expected final-audit pattern run without heldout metadata to fail")


def test_manifest_rejects_pattern_final_audit_missing_heldout_suite(tmp_path: Path) -> None:
    import json

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "selection": {
                    "heldout_split": "task",
                    "heldout_trace_indices": [2],
                    "rank_task_count": 4,
                    "heldout_task_count": 1,
                    "heldout_task_disjoint": True,
                    "heldout_task_overlap_count": 0,
                    "heldout_task_overlap": [],
                    "heldout_suite_count": 1,
                    "heldout_suite_keys": ["libero_goal"],
                },
                "top": [
                    {
                        "modeled_speedup": 2.25,
                        "target_forward_reduction": 2.5,
                        "heldout_task_count": 1,
                        "config": {
                            "lookahead": 8,
                            "action_dim": 7,
                            "max_period": 16,
                            "min_period_repeats": 2,
                            "repeat_token_min_run": 3,
                            "linear_action_extrapolation": False,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    try:
        build_manifest(
            _args(
                root=tmp_path / "run",
                suites="libero_goal,libero_object",
                speed_modes="baseline,target_eos,pattern_sd_direct",
                candidate_mode="pattern_sd_direct",
                pattern_sweep_json=sweep_path,
                run_final_audit=True,
            ),
            [],
        )
    except ValueError as exc:
        assert "heldout split does not cover requested suites" in str(exc)
        assert "libero_object" in str(exc)
    else:
        raise AssertionError("expected final-audit pattern run missing heldout suite to fail")


def test_manifest_auto_selects_pattern_sweep_row_after_thresholds(tmp_path: Path) -> None:
    import json

    base_config = {
        "lookahead": 4,
        "action_dim": 7,
        "max_period": 8,
        "min_period_repeats": 2,
        "repeat_token_min_run": 2,
        "linear_action_extrapolation": False,
        "reuse_full_blocks": True,
        "emit_bonus_token": False,
    }
    sweep = {
        "top": [
            {
                "modeled_speedup": 2.8,
                "target_forward_reduction": 2.7,
                "min_task_target_forward_reduction": 2.0,
                "min_task_acceptance_rate": 0.7,
                "task_count": 4,
                "heldout_modeled_speedup": 2.0,
                "heldout_target_forward_reduction": 2.0,
                "heldout_min_task_target_forward_reduction": 1.5,
                "heldout_min_task_acceptance_rate": 0.5,
                "heldout_task_count": 2,
                "config": dict(base_config, ngram_continuation=True),
            },
            {
                "modeled_speedup": 2.2,
                "target_forward_reduction": 2.1,
                "min_task_target_forward_reduction": 1.8,
                "min_task_acceptance_rate": 0.6,
                "task_count": 4,
                "heldout_modeled_speedup": 1.6,
                "heldout_target_forward_reduction": 1.5,
                "heldout_min_task_target_forward_reduction": 1.2,
                "heldout_min_task_acceptance_rate": 0.4,
                "heldout_task_count": 2,
                "chunk_length_stop_accepted_tokens": 3,
                "min_task_chunk_length_stop_accepted_tokens": 1,
                "heldout_chunk_length_stop_accepted_tokens": 2,
                "heldout_min_task_chunk_length_stop_accepted_tokens": 1,
                "config": dict(base_config, chunk_length_stop=True),
            },
        ]
    }
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(sweep) + "\n")

    manifest = build_manifest(
        _args(
            root=tmp_path / "run",
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
            pattern_sweep_json=sweep_path,
            pattern_sweep_rank=0,
            pattern_min_modeled_speedup=2.0,
            pattern_min_forward_reduction=2.0,
            pattern_auto_source_min_metrics=True,
        ),
        [],
    )

    selection = manifest["pattern_sweep_selection"]
    assert selection["requested_rank"] == 0
    assert selection["selected_rank"] == 2
    assert selection["auto_source_min_metric"] == [
        "chunk_length_stop_accepted_tokens=1",
        "min_task_chunk_length_stop_accepted_tokens=1",
    ]
    assert selection["effective_min_metric"] == selection["auto_source_min_metric"]
    assert "--pattern-chunk-length-stop" in manifest["eval_extra_args"]


def test_manifest_auto_source_metrics_require_enabled_source_usage(tmp_path: Path) -> None:
    import json

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "top": [
                    {
                        "modeled_speedup": 2.5,
                        "target_forward_reduction": 2.6,
                        "second_order_action_extrapolation_accepted_tokens": 3,
                        "min_task_second_order_action_extrapolation_accepted_tokens": 1,
                        "heldout_second_order_action_extrapolation_accepted_tokens": 2,
                        "heldout_min_task_second_order_action_extrapolation_accepted_tokens": 1,
                        "hold_action_token_accepted_tokens": 4,
                        "min_task_hold_action_token_accepted_tokens": 1,
                        "heldout_hold_action_token_accepted_tokens": 2,
                        "heldout_min_task_hold_action_token_accepted_tokens": 1,
                        "config": {
                            "lookahead": 4,
                            "action_dim": 7,
                            "max_period": 8,
                            "min_period_repeats": 2,
                            "repeat_token_min_run": 2,
                            "linear_action_extrapolation": False,
                            "second_order_action_extrapolation": True,
                            "second_order_max_accel": 4,
                            "hold_action_token": True,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    manifest = build_manifest(
        _args(
            root=tmp_path / "run",
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
            pattern_sweep_json=sweep_path,
            pattern_auto_source_min_metrics=True,
        ),
        [],
    )

    selection = manifest["pattern_sweep_selection"]
    assert selection["auto_source_min_metric"] == [
        "second_order_action_extrapolation_accepted_tokens=1",
        "min_task_second_order_action_extrapolation_accepted_tokens=1",
        "hold_action_token_accepted_tokens=1",
        "min_task_hold_action_token_accepted_tokens=1",
    ]
    assert selection["auto_source_min_heldout_metric"] == [
        "second_order_action_extrapolation_accepted_tokens=1",
        "min_task_second_order_action_extrapolation_accepted_tokens=1",
        "hold_action_token_accepted_tokens=1",
        "min_task_hold_action_token_accepted_tokens=1",
    ]
    assert selection["effective_min_metric"] == selection["auto_source_min_metric"]
    assert selection["effective_min_heldout_metric"] == selection["auto_source_min_heldout_metric"]
    assert "--pattern-hold-action-token" in manifest["eval_extra_args"]
    assert "--pattern-second-order-action-extrapolation" in manifest["eval_extra_args"]
    assert "second_order_action_extrapolation_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "hold_action_token_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "second_order_action_extrapolation_accepted_tokens=1" in manifest["min_candidate_trace_stat_totals"]
    assert "hold_action_token_accepted_tokens=1" in manifest["min_candidate_trace_stat_totals"]


def test_manifest_auto_source_metrics_include_chunk_position_delta_min_count(tmp_path: Path) -> None:
    import json

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "selection": {
                    "heldout_split": "task",
                    "heldout_trace_indices": [2, 5],
                    "rank_task_count": 4,
                    "heldout_task_count": 2,
                    "heldout_task_disjoint": True,
                    "heldout_task_overlap_count": 0,
                    "heldout_task_overlap": [],
                    "rank_suite_count": 2,
                    "rank_suite_keys": ["libero_goal", "libero_object"],
                    "heldout_suite_count": 2,
                    "heldout_suite_keys": ["libero_goal", "libero_object"],
                },
                "evaluated_source_counts": {"chunk_delta_template": 3},
                "evaluated_source_coverage": ["chunk_delta_template"],
                "top": [
                    {
                        "modeled_speedup": 2.7,
                        "target_forward_reduction": 2.5,
                        "heldout_task_count": 2,
                        "chunk_position_delta_accepted_tokens": 4,
                        "min_task_chunk_position_delta_accepted_tokens": 1,
                        "heldout_chunk_position_delta_accepted_tokens": 2,
                        "heldout_min_task_chunk_position_delta_accepted_tokens": 1,
                        "config": {
                            "lookahead": 4,
                            "action_dim": 7,
                            "max_period": 8,
                            "min_period_repeats": 2,
                            "repeat_token_min_run": 2,
                            "linear_action_extrapolation": False,
                            "chunk_position_delta": True,
                            "chunk_position_delta_history_size": 4,
                            "chunk_position_delta_top_k": 2,
                            "chunk_position_delta_min_count": 2,
                            "chunk_position_delta_max_abs": None,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    manifest = build_manifest(
        _args(
            root=tmp_path / "run",
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
            pattern_sweep_json=sweep_path,
            pattern_auto_source_min_metrics=True,
            pattern_required_source_coverage=["chunk_delta_template"],
            run_final_audit=True,
        ),
        [],
    )

    selection = manifest["pattern_sweep_selection"]
    assert selection["auto_source_min_metric"] == [
        "chunk_position_delta_accepted_tokens=1",
        "min_task_chunk_position_delta_accepted_tokens=1",
    ]
    assert selection["auto_source_min_heldout_metric"] == [
        "chunk_position_delta_accepted_tokens=1",
        "min_task_chunk_position_delta_accepted_tokens=1",
    ]
    assert selection["effective_min_metric"] == selection["auto_source_min_metric"]
    assert selection["effective_min_heldout_metric"] == selection["auto_source_min_heldout_metric"]
    assert "--pattern-chunk-position-delta" in manifest["eval_extra_args"]
    assert "--pattern-chunk-position-delta-min-count" in manifest["eval_extra_args"]
    min_count_index = manifest["eval_extra_args"].index("--pattern-chunk-position-delta-min-count")
    assert manifest["eval_extra_args"][min_count_index + 1] == "2"
    assert "chunk_position_delta_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_position_delta_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_position_delta_accepted_tokens=0" in manifest["gate_command"]


def test_manifest_auto_source_metrics_include_chunk_delta_template_guard(tmp_path: Path) -> None:
    import json

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "selection": {
                    "heldout_split": "task",
                    "heldout_trace_indices": [2, 5],
                    "rank_task_count": 4,
                    "heldout_task_count": 2,
                    "heldout_task_disjoint": True,
                    "heldout_task_overlap_count": 0,
                    "heldout_task_overlap": [],
                    "rank_suite_count": 2,
                    "rank_suite_keys": ["libero_goal", "libero_object"],
                    "heldout_suite_count": 2,
                    "heldout_suite_keys": ["libero_goal", "libero_object"],
                },
                "evaluated_source_counts": {"chunk_delta_template": 3},
                "evaluated_source_coverage": ["chunk_delta_template"],
                "top": [
                    {
                        "modeled_speedup": 2.7,
                        "target_forward_reduction": 2.5,
                        "heldout_task_count": 2,
                        "chunk_delta_template_accepted_tokens": 4,
                        "min_task_chunk_delta_template_accepted_tokens": 1,
                        "heldout_chunk_delta_template_accepted_tokens": 2,
                        "heldout_min_task_chunk_delta_template_accepted_tokens": 1,
                        "config": {
                            "lookahead": 4,
                            "action_dim": 7,
                            "max_period": 8,
                            "min_period_repeats": 2,
                            "repeat_token_min_run": 2,
                            "linear_action_extrapolation": False,
                            "chunk_delta_template": True,
                            "chunk_delta_template_history_size": 4,
                            "chunk_delta_template_top_k": 2,
                            "chunk_delta_template_min_prefix_deltas": 1,
                            "chunk_delta_template_max_delta_mismatch": 0,
                            "chunk_delta_template_max_abs": 8,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    manifest = build_manifest(
        _args(
            root=tmp_path / "run",
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
            pattern_sweep_json=sweep_path,
            pattern_auto_source_min_metrics=True,
            pattern_required_source_coverage=["chunk_delta_template"],
            run_final_audit=True,
        ),
        [],
    )

    selection = manifest["pattern_sweep_selection"]
    assert selection["auto_source_min_metric"] == [
        "chunk_delta_template_accepted_tokens=1",
        "min_task_chunk_delta_template_accepted_tokens=1",
    ]
    assert selection["auto_source_min_heldout_metric"] == [
        "chunk_delta_template_accepted_tokens=1",
        "min_task_chunk_delta_template_accepted_tokens=1",
    ]
    assert selection["effective_min_metric"] == selection["auto_source_min_metric"]
    assert selection["effective_min_heldout_metric"] == selection["auto_source_min_heldout_metric"]
    assert selection["required_source_coverage"] == ["chunk_delta_template"]
    assert selection["required_source_counts"] == {"chunk_delta_template": 3}
    assert "--pattern-chunk-delta-template" in manifest["eval_extra_args"]
    assert "--pattern-chunk-delta-template-max-delta-mismatch" in manifest["eval_extra_args"]
    mismatch_index = manifest["eval_extra_args"].index("--pattern-chunk-delta-template-max-delta-mismatch")
    assert manifest["eval_extra_args"][mismatch_index + 1] == "0"
    assert "chunk_delta_template_drafted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_delta_template_accepted_tokens=0" in manifest["min_candidate_trace_stats"]
    assert "chunk_delta_template_accepted_tokens=0" in manifest["gate_command"]
    assert "--pi0fast-manifest" in manifest["audit_command"]
    assert "--require-pattern-source-coverage" in manifest["audit_command"]


def test_manifest_auto_source_metrics_require_tree_anchor_usage(tmp_path: Path) -> None:
    import json

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "top": [
                    {
                        "modeled_speedup": 2.5,
                        "target_forward_reduction": 2.6,
                        "tree_anchor_accepted_tokens": 4,
                        "min_task_tree_anchor_accepted_tokens": 1,
                        "heldout_tree_anchor_accepted_tokens": 2,
                        "heldout_min_task_tree_anchor_accepted_tokens": 1,
                        "config": {
                            "lookahead": 4,
                            "action_dim": 7,
                            "max_period": 8,
                            "min_period_repeats": 2,
                            "repeat_token_min_run": 2,
                            "linear_action_extrapolation": False,
                            "tree_width": 4,
                            "tree_branch_width": 4,
                            "tree_anchor_target_continuation": True,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    manifest = build_manifest(
        _args(
            root=tmp_path / "run",
            speed_modes="baseline,target_eos,pattern_sd_direct",
            candidate_mode="pattern_sd_direct",
            pattern_sweep_json=sweep_path,
            pattern_auto_source_min_metrics=True,
        ),
        [],
    )

    selection = manifest["pattern_sweep_selection"]
    assert selection["auto_source_min_metric"] == [
        "tree_anchor_accepted_tokens=1",
        "min_task_tree_anchor_accepted_tokens=1",
    ]
    assert selection["auto_source_min_heldout_metric"] == [
        "tree_anchor_accepted_tokens=1",
        "min_task_tree_anchor_accepted_tokens=1",
    ]
    assert "--pattern-tree-anchor-target-continuation" in manifest["eval_extra_args"]
    assert "tree_anchor_accepted_tokens=1e-09" in manifest["min_candidate_trace_stats"]


def test_manifest_auto_source_metrics_reject_weak_enabled_source(tmp_path: Path) -> None:
    import json

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "top": [
                    {
                        "modeled_speedup": 2.5,
                        "target_forward_reduction": 2.6,
                        "hold_action_token_accepted_tokens": 0,
                        "min_task_hold_action_token_accepted_tokens": 0,
                        "config": {
                            "lookahead": 4,
                            "action_dim": 7,
                            "max_period": 8,
                            "min_period_repeats": 2,
                            "repeat_token_min_run": 2,
                            "linear_action_extrapolation": False,
                            "hold_action_token": True,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    try:
        build_manifest(
            _args(
                root=tmp_path / "run",
                speed_modes="baseline,target_eos,pattern_sd_direct",
                candidate_mode="pattern_sd_direct",
                pattern_sweep_json=sweep_path,
                pattern_auto_source_min_metrics=True,
            ),
            [],
        )
    except ValueError as exc:
        assert "hold_action_token_accepted_tokens" in str(exc)
        assert "below required" in str(exc)
    else:
        raise AssertionError("expected weak enabled source usage to fail")


def test_manifest_rejects_weak_pattern_sweep_row(tmp_path: Path) -> None:
    import json

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "top": [
                    {
                        "modeled_speedup": 1.5,
                        "target_forward_reduction": 1.6,
                        "config": {
                            "lookahead": 4,
                            "action_dim": 7,
                            "max_period": 8,
                            "min_period_repeats": 2,
                            "repeat_token_min_run": 2,
                            "linear_action_extrapolation": False,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    try:
        build_manifest(
            _args(
                speed_modes="baseline,target_eos,pattern_sd_direct",
                candidate_mode="pattern_sd_direct",
                pattern_sweep_json=sweep_path,
                pattern_min_modeled_speedup=2.0,
            ),
            [],
        )
    except ValueError as exc:
        assert "modeled_speedup" in str(exc)
    else:
        raise AssertionError("expected weak sweep row to fail")


def test_manifest_target_eos_candidate_has_no_self_reference() -> None:
    manifest = build_manifest(
        _args(
            speed_modes="baseline,target_eos",
            candidate_mode="target_eos",
        ),
        [],
    )

    assert manifest["reference_mode"] is None
    assert manifest["min_reference_speedup"] is None
    assert manifest["max_reference_success_drop"] is None
    assert manifest["max_reference_success_regressions"] is None
    assert "--reference-mode" not in manifest["gate_command"]
    assert "--min-reference-speedup" not in manifest["gate_command"]


def _write_metrics(path: Path, *, suite: str, mode: str, task_ids: list[int], episodes: int, seed: int) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id in task_ids:
        for episode in range(episodes):
            rows.append(
                {
                    "task": suite,
                    "task_id": task_id,
                    "mode": mode,
                    "episode": episode,
                    "seed": seed + episode,
                    "success": True,
                    "avg_ms_per_control_step": 100.0,
                    "chunk_stats": {},
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_eval_shard_complete_requires_all_task_episode_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "speed" / "baseline" / "libero_object"
    _write_metrics(
        output_dir / "metrics.jsonl",
        suite="libero_object",
        mode="baseline",
        task_ids=[0, 1],
        episodes=2,
        seed=42,
    )

    assert eval_shard_complete(
        output_dir,
        suite="libero_object",
        mode="baseline",
        task_ids="0,1",
        episodes=2,
        seed=42,
    )
    assert not eval_shard_complete(
        output_dir,
        suite="libero_object",
        mode="baseline",
        task_ids="0,1,2",
        episodes=2,
        seed=42,
    )


def test_manifest_skip_existing_omits_completed_shards(tmp_path: Path) -> None:
    root = tmp_path / "run"
    output_dir = root / "speed" / "baseline" / "libero_object"
    _write_metrics(
        output_dir / "metrics.jsonl",
        suite="libero_object",
        mode="baseline",
        task_ids=[0, 1],
        episodes=2,
        seed=42,
    )

    manifest = build_manifest(
        _args(
            root=root,
            suites="libero_object",
            task_ids="0,1",
            episodes=2,
            speed_modes="baseline,target_eos",
            candidate_mode="target_eos",
            skip_validation=True,
            skip_existing=True,
        ),
        [],
    )

    assert str(output_dir) in manifest["skipped_existing"]
    assert len(manifest["speed_commands"]) == 1
    assert "target_eos" in manifest["speed_commands"][0]


def test_manifest_can_run_synthetic_and_final_audit(tmp_path: Path) -> None:
    root = tmp_path / "run"

    manifest = build_manifest(
        _args(
            root=root,
            suites="libero_object",
            task_ids="0,1",
            episodes=2,
            speed_modes="baseline,target_eos",
            candidate_mode="target_eos",
            run_synthetic=True,
            run_final_audit=True,
        ),
        [],
    )

    assert manifest["synthetic_command"] is not None
    assert "scripts/benchmark_robotics_spec_decode_synthetic.py" in manifest["synthetic_command"]
    assert str(root / "synthetic_gate.json") in manifest["synthetic_command"]
    assert manifest["audit_command"] is not None
    assert "scripts/audit_robotics_spec_goal.py" in manifest["audit_command"]
    assert str(root / "gate.json") in manifest["audit_command"]
    assert str(root / "run_manifest.json") in manifest["audit_command"]
    assert str(root / "synthetic_gate.json") in manifest["audit_command"]
    assert manifest["min_unique_tasks"] == 2
    assert "--min-unique-tasks" in manifest["gate_command"]
    assert "--min-unique-tasks" in manifest["audit_command"]
    assert "--min-baseline-successes" in manifest["gate_command"]
    assert "--min-baseline-successes" in manifest["audit_command"]
    assert "--min-suite-matched-pairs" in manifest["gate_command"]
    assert "--min-suite-baseline-successes" in manifest["gate_command"]
    assert "--min-suite-speedup" in manifest["gate_command"]
    assert "--min-suite-matched-pairs" in manifest["audit_command"]
    assert "--min-suite-baseline-successes" in manifest["audit_command"]
    assert "--min-suite-speedup" in manifest["audit_command"]


def test_manifest_can_render_result_card(tmp_path: Path) -> None:
    root = tmp_path / "run"

    manifest = build_manifest(
        _args(
            root=root,
            suites="libero_object",
            task_ids="0,1",
            episodes=2,
            speed_modes="baseline,target_eos",
            candidate_mode="target_eos",
            run_final_audit=True,
            render_result_card=True,
        ),
        [],
    )

    assert manifest["result_card_command"] is not None
    assert "scripts/render_robotics_spec_result_card.py" in manifest["result_card_command"]
    assert str(root / "objective_audit.json") in manifest["result_card_command"]
    assert str(root / "result_card.md") in manifest["result_card_command"]


def test_manifest_can_run_preflight(tmp_path: Path) -> None:
    manifest = build_manifest(
        _args(
            root=tmp_path / "run",
            suites="libero_object",
            task_ids="0,1",
            episodes=2,
            speed_modes="baseline,target_eos",
            candidate_mode="target_eos",
            run_preflight=True,
            require_hf_token=True,
        ),
        [],
    )

    assert manifest["preflight_command"] is not None
    assert "scripts/preflight_pi0fast_eval_env.py" in manifest["preflight_command"]
    assert "--require-hf-token" in manifest["preflight_command"]
    assert "torch,lerobot,libero,robosuite,mujoco,networkx,dist:hf_libero" in manifest["preflight_command"]


def test_manifest_skip_existing_omits_synthetic_and_audit(tmp_path: Path) -> None:
    import json

    root = tmp_path / "run"
    root.mkdir()
    (root / "synthetic_gate.json").write_text(json.dumps({"gate_passed": True}) + "\n")
    (root / "objective_audit.json").write_text(json.dumps({"objective_audit_passed": True}) + "\n")

    manifest = build_manifest(
        _args(
            root=root,
            suites="libero_object",
            task_ids="0,1",
            episodes=2,
            speed_modes="baseline,target_eos",
            candidate_mode="target_eos",
            run_synthetic=True,
            run_final_audit=True,
            skip_existing=True,
        ),
        [],
    )

    assert manifest["synthetic_command"] is None
    assert manifest["audit_command"] is None
    assert str(root / "synthetic_gate.json") in manifest["skipped_existing"]
    assert str(root / "objective_audit.json") in manifest["skipped_existing"]
