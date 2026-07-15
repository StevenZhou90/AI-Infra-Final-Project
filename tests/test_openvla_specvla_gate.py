from __future__ import annotations

from scripts.gate_openvla_specvla import build_gate_summary


def _row(
    mode: str,
    idx: int,
    *,
    success: bool = True,
    ms: float = 100.0,
    steps: int = 10,
    spec_stats: dict | None = None,
    suite: str = "libero_goal",
) -> dict:
    row = {
        "suite": suite,
        "task_id": idx % 10,
        "trial": idx,
        "mode": mode,
        "success": success,
        "steps": steps,
        "inference_ms_total": ms * steps,
    }
    if spec_stats is not None:
        row["spec_stats"] = spec_stats
    return row


def test_openvla_gate_passes_120_matched_zero_drop_2x_speedup() -> None:
    baseline = [_row("ar", idx, ms=240.0) for idx in range(120)]
    candidate = [_row("spec", idx, ms=100.0) for idx in range(120)]

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
    )

    assert summary["gate_passed"] is True
    assert summary["platform"] == "openvla_specvla"
    assert summary["speed"]["matched_pairs"] == 120
    assert summary["speed"]["success_drop_abs"] == 0.0
    assert summary["speed"]["baseline_success_regressions"] == 0
    assert summary["speed"]["speedup"] == 2.4


def test_openvla_gate_fails_regression_even_when_aggregate_success_matches() -> None:
    baseline = []
    candidate = []
    for idx in range(120):
        baseline_success = idx != 119
        candidate_success = idx != 0
        baseline.append(_row("ar", idx, success=baseline_success, ms=240.0))
        candidate.append(_row("spec", idx, success=candidate_success, ms=100.0))

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
    )

    assert summary["gate_passed"] is False
    assert summary["speed"]["success_drop_abs"] == 0.0
    assert summary["speed"]["baseline_success_regressions"] == 1
    assert summary["checks"]["baseline_success_regressions"] is False


def test_openvla_gate_fails_all_fail_success_artifact() -> None:
    baseline = [_row("ar", idx, success=False, ms=240.0) for idx in range(120)]
    candidate = [_row("spec", idx, success=False, ms=100.0) for idx in range(120)]

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
    )

    assert summary["gate_passed"] is False
    assert summary["checks"]["baseline_successes"] is False
    assert summary["speed"]["baseline_successes"] == 0
    assert summary["speed"]["success_drop_abs"] == 0.0


def test_openvla_gate_fails_weak_suite_speedup_when_required() -> None:
    baseline = []
    candidate = []
    for idx in range(120):
        suite = "suite_fast" if idx < 60 else "suite_slow"
        candidate_ms = 80.0 if suite == "suite_fast" else 260.0
        baseline.append(_row("ar", idx, ms=240.0, suite=suite))
        candidate.append(_row("spec", idx, ms=candidate_ms, suite=suite))

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=1.2,
        min_suite_matched_pairs=1,
        min_suite_baseline_successes=1,
        min_suite_speedup=1.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
    )

    assert summary["speed"]["speedup"] > 1.2
    assert summary["speed"]["suites"]["suite_slow"]["speedup"] < 1.0
    assert summary["gate_passed"] is False
    assert summary["checks"]["suite_speedup"] is False


def test_openvla_gate_fails_low_unique_task_coverage_when_required() -> None:
    baseline = [_row("ar", idx, ms=240.0) for idx in range(120)]
    candidate = [_row("spec", idx, ms=100.0) for idx in range(120)]

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_unique_tasks=20,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
    )

    assert summary["speed"]["unique_tasks"] == 10
    assert summary["gate_passed"] is False
    assert summary["checks"]["unique_tasks"] is False


def test_openvla_gate_fails_unmatched_episode_keys() -> None:
    baseline = [_row("ar", idx, ms=240.0) for idx in range(120)]
    candidate = [_row("spec", idx + 1_000, ms=100.0) for idx in range(120)]

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
    )

    assert summary["gate_passed"] is False
    assert summary["speed"]["matched_pairs"] == 0
    assert summary["checks"]["key_matching"] is False


def test_openvla_gate_fails_mismatched_control_steps_by_default() -> None:
    baseline = [_row("ar", idx, ms=240.0, steps=10) for idx in range(120)]
    candidate = [_row("spec", idx, ms=100.0, steps=10) for idx in range(120)]
    candidate[0]["steps"] = 7
    candidate[0]["inference_ms_total"] = 700.0

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
    )

    assert summary["gate_passed"] is False
    assert summary["checks"]["matched_steps"] is False
    assert summary["speed"]["step_mismatch_count"] == 1
    assert summary["per_episode"][0]["baseline_steps"] == 10
    assert summary["per_episode"][0]["candidate_steps"] == 7


def test_openvla_gate_strict_spec_stats_passes_with_verified_depth_one_path() -> None:
    baseline = [_row("ar", idx, ms=240.0) for idx in range(120)]
    candidate = [
        _row(
            "spec",
            idx,
            ms=100.0,
            spec_stats={
                "fast_draft_calls": 0,
                "fast_draft_tokens": 0,
                "chunk_buffer_hits": 0,
                "relaxed_group_accepts": 0,
                "max_tree_depth_used": 1,
                "unverified_action_shortcuts": 0,
            },
        )
        for idx in range(120)
    ]

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        require_spec_stats=True,
        max_unverified_action_shortcuts=0,
        max_fast_draft_calls=0,
        max_chunk_buffer_hits=0,
        max_relaxed_group_accepts=0,
        max_tree_depth_used=1,
    )

    assert summary["gate_passed"] is True
    assert summary["checks"]["spec_stats_present"] is True
    assert summary["quality"]["unverified_action_shortcuts"] == 0
    assert summary["quality"]["max_tree_depth_used"] == 1


def test_openvla_gate_strict_spec_stats_fails_unverified_shortcuts() -> None:
    baseline = [_row("ar", idx, ms=240.0) for idx in range(120)]
    candidate = [
        _row(
            "spec",
            idx,
            ms=100.0,
            spec_stats={
                "fast_draft_calls": 1 if idx == 0 else 0,
                "fast_draft_tokens": 7 if idx == 0 else 0,
                "chunk_buffer_hits": 1 if idx == 1 else 0,
                "relaxed_group_accepts": 1 if idx == 0 else 0,
                "max_tree_depth_used": 2 if idx == 2 else 1,
            },
        )
        for idx in range(120)
    ]

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        require_spec_stats=True,
        max_unverified_action_shortcuts=0,
        max_fast_draft_calls=0,
        max_chunk_buffer_hits=0,
        max_relaxed_group_accepts=0,
        max_tree_depth_used=1,
    )

    assert summary["gate_passed"] is False
    assert summary["quality"]["unverified_action_shortcuts"] == 2
    assert summary["checks"]["unverified_action_shortcuts"] is False
    assert summary["checks"]["fast_draft_calls"] is False
    assert summary["checks"]["chunk_buffer_hits"] is False
    assert summary["checks"]["relaxed_group_accepts"] is False
    assert summary["checks"]["max_tree_depth_used"] is False


def test_openvla_gate_strict_spec_stats_fails_missing_spec_stats() -> None:
    baseline = [_row("ar", idx, ms=240.0) for idx in range(120)]
    candidate = [_row("spec", idx, ms=100.0) for idx in range(120)]

    summary = build_gate_summary(
        baseline_rows=baseline,
        candidate_rows=candidate,
        min_pairs=120,
        min_speedup=2.0,
        max_success_drop=0.0,
        max_baseline_success_regressions=0,
        require_spec_stats=True,
    )

    assert summary["gate_passed"] is False
    assert summary["checks"]["spec_stats_present"] is False
    assert summary["quality"]["candidate_rows_missing_spec_stats"] == 120
