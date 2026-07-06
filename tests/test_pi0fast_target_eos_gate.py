from __future__ import annotations

from unittest.mock import patch

from scripts.gate_pi0fast_target_eos import build_gate_summary, parse_args


def _row(
    mode: str,
    idx: int,
    *,
    success: bool = True,
    ms: float = 100.0,
    exact_verifies: int = 0,
    max_action_diff: float = 0.0,
    trace_stats: dict | None = None,
    task: str = "libero_object",
    steps: int | None = 300,
) -> dict:
    row = {
        "mode": mode,
        "episode": idx,
        "seed": 1000 + idx,
        "task": task,
        "task_id": idx % 10,
        "success": success,
        "avg_ms_per_control_step": ms,
        "chunk_stats": {
            "avg_model_call_ms": ms,
            "avg_fast_token_count": 128.0 if mode == "baseline" else 48.0,
            "exact_verifies": exact_verifies,
            "max_action_diff": max_action_diff,
            "mean_action_diff": max_action_diff,
            "trace_stats": trace_stats or {},
        },
    }
    if steps is not None:
        row["steps"] = steps
    return row


def test_direct_gate_parser_defaults_to_120_validation_episodes() -> None:
    with patch("sys.argv", ["gate_pi0fast_target_eos.py", "outputs/pi0fast_target_eos_120"]):
        args = parse_args()

    assert args.min_pairs == 120
    assert args.min_unique_tasks == 0
    assert args.min_validation_episodes == 120
    assert args.min_baseline_successes == 1
    assert args.min_suite_matched_pairs == 0
    assert args.min_suite_baseline_successes == 0
    assert args.min_suite_speedup is None
    assert args.require_matched_steps is True
    assert args.min_candidate_trace_stat_total == []


def test_gate_passes_100_matched_zero_drop_2x_speedup_with_exact_validation() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(_row("target_eos", idx, ms=100.0))
        validation_rows.append(_row("target_eos_validate", idx, ms=200.0, exact_verifies=3, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
    )

    assert summary["gate_passed"] is True
    assert summary["speed"]["matched_pairs"] == 100
    assert summary["speed"]["success_drop_abs"] == 0.0
    assert summary["speed"]["baseline_success_regressions"] == 0
    assert summary["speed"]["speedup"] == 2.4
    assert summary["exact_validation"]["max_action_diff"] == 0.0


def test_gate_fails_baseline_success_regression_even_when_total_success_rate_matches() -> None:
    speed_rows = []
    for idx in range(100):
        baseline_success = idx != 99
        candidate_success = idx != 0
        speed_rows.append(_row("baseline", idx, success=baseline_success, ms=240.0))
        speed_rows.append(_row("target_eos", idx, success=candidate_success, ms=100.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=None,
        require_exact_validation=False,
    )

    assert summary["gate_passed"] is False
    assert summary["speed"]["success_drop_abs"] == 0.0
    assert summary["speed"]["baseline_success_regressions"] == 1
    assert summary["checks"]["baseline_success_regressions"] is False


def test_gate_fails_mismatched_control_steps() -> None:
    speed_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0, steps=300))
        speed_rows.append(_row("target_eos", idx, ms=100.0, steps=300 if idx != 42 else 250))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=None,
        require_exact_validation=False,
    )

    assert summary["gate_passed"] is False
    assert summary["speed"]["step_mismatch_count"] == 1
    assert summary["speed"]["missing_step_count"] == 0
    assert summary["checks"]["matched_steps"] is False


def test_gate_fails_missing_control_step_evidence() -> None:
    speed_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0, steps=300))
        speed_rows.append(_row("target_eos", idx, ms=100.0, steps=None if idx == 7 else 300))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=None,
        require_exact_validation=False,
    )

    assert summary["gate_passed"] is False
    assert summary["speed"]["step_mismatch_count"] == 0
    assert summary["speed"]["missing_step_count"] == 1
    assert summary["checks"]["matched_steps"] is False


def test_gate_fails_all_fail_success_artifact() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, success=False, ms=240.0))
        speed_rows.append(_row("target_eos", idx, success=False, ms=100.0))
        validation_rows.append(_row("target_eos_validate", idx, success=False, exact_verifies=1, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
    )

    assert summary["gate_passed"] is False
    assert summary["checks"]["baseline_successes"] is False
    assert summary["speed"]["baseline_successes"] == 0
    assert summary["speed"]["success_drop_abs"] == 0.0


def test_gate_fails_weak_suite_speedup_when_required() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        task = "suite_fast" if idx < 50 else "suite_slow"
        candidate_ms = 80.0 if task == "suite_fast" else 260.0
        speed_rows.append(_row("baseline", idx, ms=240.0, task=task))
        speed_rows.append(_row("target_eos", idx, ms=candidate_ms, task=task))
        validation_rows.append(_row("target_eos_validate", idx, exact_verifies=1, max_action_diff=0.0, task=task))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=1.2,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        min_suite_matched_pairs=1,
        min_suite_baseline_successes=1,
        min_suite_speedup=1.0,
        validation_rows=validation_rows,
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
    )

    assert summary["speed"]["speedup"] > 1.2
    assert summary["speed"]["suites"]["suite_slow"]["speedup"] < 1.0
    assert summary["gate_passed"] is False
    assert summary["checks"]["suite_speedup"] is False


def test_gate_fails_low_unique_task_coverage_when_required() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(_row("target_eos", idx, ms=100.0))
        validation_rows.append(_row("target_eos_validate", idx, exact_verifies=1, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_unique_tasks=20,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
    )

    assert summary["speed"]["unique_tasks"] == 10
    assert summary["gate_passed"] is False
    assert summary["checks"]["unique_tasks"] is False


def test_gate_fails_nonzero_exact_action_diff() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(_row("target_eos", idx, ms=100.0))
        validation_rows.append(_row("target_eos_validate", idx, exact_verifies=1, max_action_diff=0.0))
    validation_rows[-1]["chunk_stats"]["max_action_diff"] = 1e-6

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
    )

    assert summary["gate_passed"] is False
    assert summary["checks"]["exact_action_diff"] is False


def test_gate_fails_when_validation_rows_do_not_cover_speed_keys() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(_row("target_eos", idx, ms=100.0))
        validation_rows.append(_row("target_eos_validate", idx + 1000, exact_verifies=1, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
    )

    assert summary["gate_passed"] is False
    assert summary["exact_validation"]["episodes"] == 0
    assert summary["exact_validation"]["missing_validation_rows"] == 100
    assert summary["checks"]["validation_episode_count"] is False
    assert summary["checks"]["validation_key_matching"] is False


def test_gate_fails_validation_row_without_exact_verify() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(_row("target_eos", idx, ms=100.0))
        exact_verifies = 0 if idx == 42 else 1
        validation_rows.append(_row("target_eos_validate", idx, exact_verifies=exact_verifies, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="target_eos",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
    )

    assert summary["gate_passed"] is False
    assert summary["exact_validation"]["rows_missing_exact_verifies"] == 1
    assert summary["checks"]["per_row_exact_verify"] is False


def test_gate_can_require_candidate_to_match_early_stop_reference() -> None:
    speed_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(_row("target_eos", idx, ms=100.0))
        speed_rows.append(_row("block_sd_direct", idx, ms=120.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="block_sd_direct",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=None,
        require_exact_validation=False,
        reference_mode="target_eos",
        min_reference_speedup=1.0,
        max_reference_success_drop=0.0,
        max_reference_success_regressions=0,
    )

    assert summary["gate_passed"] is False
    assert summary["speed"]["speedup"] == 2.0
    assert summary["reference"]["speedup"] < 1.0
    assert summary["checks"]["reference_speedup"] is False


def test_gate_fails_reference_mismatched_control_steps() -> None:
    speed_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0, steps=300))
        speed_rows.append(_row("target_eos", idx, ms=120.0, steps=250 if idx == 9 else 300))
        speed_rows.append(_row("pattern_sd_direct", idx, ms=90.0, steps=300))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="pattern_sd_direct",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=None,
        require_exact_validation=False,
        reference_mode="target_eos",
        max_reference_success_drop=0.0,
        max_reference_success_regressions=0,
    )

    assert summary["gate_passed"] is False
    assert summary["speed"]["step_mismatch_count"] == 0
    assert summary["reference"]["step_mismatch_count"] == 1
    assert summary["checks"]["reference_matched_steps"] is False


def test_gate_requires_extra_exact_validation_modes() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(_row("target_eos", idx, ms=100.0))
        speed_rows.append(_row("pattern_sd_direct", idx, ms=90.0))
        validation_rows.append(_row("pattern_sd_validate", idx, exact_verifies=1, max_action_diff=0.0))
        validation_rows.append(_row("target_eos_validate", idx, exact_verifies=1, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="pattern_sd_direct",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        validation_mode="pattern_sd_validate",
        extra_validation_modes=["target_eos_validate"],
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
        reference_mode="target_eos",
        max_reference_success_drop=0.0,
        max_reference_success_regressions=0,
    )

    assert summary["gate_passed"] is True
    assert summary["extra_exact_validations"]["target_eos_validate"]["episodes"] == 100
    assert summary["checks"]["extra_validation_target_eos_validate_exact_action_diff"] is True

    validation_rows = [row for row in validation_rows if row["mode"] != "target_eos_validate"]
    failed = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="pattern_sd_direct",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        validation_mode="pattern_sd_validate",
        extra_validation_modes=["target_eos_validate"],
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
    )

    assert failed["gate_passed"] is False
    assert failed["checks"]["extra_validation_target_eos_validate_key_matching"] is False


def test_gate_can_require_candidate_trace_tree_stats() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(
            _row(
                "pattern_sd_direct",
                idx,
                ms=90.0,
                trace_stats={
                    "tree_width": 4.0,
                    "tree_verifies": 1.5,
                    "unverified_pattern_tokens": 0.0,
                },
            )
        )
        validation_rows.append(_row("pattern_sd_validate", idx, exact_verifies=1, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="pattern_sd_direct",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        validation_mode="pattern_sd_validate",
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
        min_candidate_trace_stats={"tree_width": 4.0, "tree_verifies": 1e-9},
        max_candidate_trace_stats={"unverified_pattern_tokens": 0.0},
    )

    assert summary["gate_passed"] is True
    assert summary["checks"]["candidate_trace_min_tree_width"] is True
    assert summary["checks"]["candidate_trace_min_tree_verifies"] is True
    assert summary["checks"]["candidate_trace_max_unverified_pattern_tokens"] is True
    assert summary["candidate_trace_stats"]["tree_width"]["min"] == 4.0
    assert summary["candidate_trace_stats"]["tree_width"]["total"] == 400.0


def test_gate_can_require_candidate_trace_stat_total_without_per_row_use() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        accepted_tokens = 3.0 if idx == 7 else 0.0
        speed_rows.append(
            _row(
                "pattern_sd_direct",
                idx,
                ms=90.0,
                trace_stats={"source_accepted_tokens": accepted_tokens},
            )
        )
        validation_rows.append(_row("pattern_sd_validate", idx, exact_verifies=1, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="pattern_sd_direct",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        validation_mode="pattern_sd_validate",
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
        min_candidate_trace_stats={"source_accepted_tokens": 0.0},
        min_candidate_trace_stat_totals={"source_accepted_tokens": 1.0},
    )

    assert summary["gate_passed"] is True
    assert summary["checks"]["candidate_trace_min_source_accepted_tokens"] is True
    assert summary["checks"]["candidate_trace_total_source_accepted_tokens"] is True
    assert summary["candidate_trace_stats"]["source_accepted_tokens"]["min"] == 0.0
    assert summary["candidate_trace_stats"]["source_accepted_tokens"]["total"] == 3.0


def test_gate_fails_when_candidate_trace_stat_total_is_zero() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        speed_rows.append(
            _row(
                "pattern_sd_direct",
                idx,
                ms=90.0,
                trace_stats={"source_accepted_tokens": 0.0},
            )
        )
        validation_rows.append(_row("pattern_sd_validate", idx, exact_verifies=1, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="pattern_sd_direct",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        validation_mode="pattern_sd_validate",
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
        min_candidate_trace_stats={"source_accepted_tokens": 0.0},
        min_candidate_trace_stat_totals={"source_accepted_tokens": 1.0},
    )

    assert summary["gate_passed"] is False
    assert summary["checks"]["candidate_trace_min_source_accepted_tokens"] is True
    assert summary["checks"]["candidate_trace_total_source_accepted_tokens"] is False


def test_gate_fails_missing_or_weak_candidate_trace_tree_stats() -> None:
    speed_rows = []
    validation_rows = []
    for idx in range(100):
        speed_rows.append(_row("baseline", idx, ms=240.0))
        trace_stats = {"tree_width": 4.0, "tree_verifies": 1.0}
        if idx == 0:
            trace_stats = {"tree_width": 1.0, "tree_verifies": 0.0}
        if idx == 1:
            trace_stats = {}
        speed_rows.append(_row("pattern_sd_direct", idx, ms=90.0, trace_stats=trace_stats))
        validation_rows.append(_row("pattern_sd_validate", idx, exact_verifies=1, max_action_diff=0.0))

    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode="baseline",
        candidate_mode="pattern_sd_direct",
        min_pairs=100,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        validation_rows=validation_rows,
        validation_mode="pattern_sd_validate",
        min_validation_episodes=100,
        min_exact_verifies=1,
        max_action_diff=0.0,
        min_candidate_trace_stats={"tree_width": 4.0, "tree_verifies": 1e-9},
    )

    assert summary["gate_passed"] is False
    assert summary["candidate_trace_stats"]["tree_width"]["missing_rows"] == 1
    assert summary["checks"]["candidate_trace_min_tree_width"] is False
    assert summary["checks"]["candidate_trace_min_tree_verifies"] is False
