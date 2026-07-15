from __future__ import annotations

from scripts.render_robotics_spec_result_card import render_result_card


def _audit(passed: bool = True) -> dict:
    return {
        "objective_audit_passed": passed,
        "pi0fast": {
            "candidate_mode": "target_eos",
            "speed": {
                "candidate_mode": "target_eos",
                "matched_pairs": 100,
                "baseline_successes": 81,
                "candidate_successes": 81,
                "success_drop_abs": 0.0,
                "speedup": 2.44,
                "baseline_success_regressions": 0,
            },
            "exact_validation": {
                "validation_mode": "target_eos_validate",
                "episodes": 100,
                "max_action_diff": 0.0,
            },
            "extra_exact_validations": {},
            "reference": None,
        },
        "synthetic": {
            "summary": {
                "tasks": 100,
                "all_exact": True,
                "speedup": 5.44,
            }
        },
    }


def test_render_result_card_summarizes_passed_audit() -> None:
    card = render_result_card(_audit())

    assert "# Robotics Speculative Decoding Result: PASS" in card
    assert "`100` matched PI0-FAST/LIBERO evals" in card
    assert "`0.00%`" in card
    assert "`2.44x`" in card
    assert "exact=`True`" in card


def test_render_result_card_summarizes_extra_exact_validations() -> None:
    audit = _audit()
    audit["pi0fast"]["candidate_mode"] = "pattern_sd_direct"
    audit["pi0fast"]["speed"]["candidate_mode"] = "pattern_sd_direct"
    audit["pi0fast"]["exact_validation"] = {
        "validation_mode": "pattern_sd_validate",
        "episodes": 100,
        "max_action_diff": 0.0,
    }
    audit["pi0fast"]["extra_exact_validations"] = {
        "target_eos_validate": {
            "validation_mode": "target_eos_validate",
            "episodes": 100,
            "max_action_diff": 0.0,
        }
    }
    audit["pi0fast"]["reference"] = {
        "baseline_mode": "target_eos",
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    audit["pi0fast"]["metadata"] = {"policy_kind": "pi05"}
    audit["pi0fast"]["manifest"] = {
        "eval_metadata": {"policy_kind": "pi05"},
        "pattern_sweep_selection": {
            "required_source_coverage": ["chunk_delta_template", "action_delta_histogram"],
            "required_source_counts": {"chunk_delta_template": 3, "action_delta_histogram": 2},
            "effective_min_heldout_metric": ["chunk_delta_template_accepted_tokens=2"],
            "selected": {
                "heldout_task_count": 3,
                "heldout_modeled_speedup": 2.1,
                "heldout_target_forward_reduction": 2.2,
            },
        },
    }

    card = render_result_card(audit)

    assert "`pattern_sd_validate`" in card
    assert "`target_eos_validate`" in card
    assert "Early-stop reference" in card
    assert "Speedup vs early-stop reference" in card
    assert "speedup=`2.05x`" in card
    assert "Policy kind" in card
    assert "`pi05`" in card
    assert "Pattern source coverage" in card
    assert "`chunk_delta_template`=`3`" in card
    assert "`action_delta_histogram`=`2`" in card
    assert "Pattern heldout evidence" in card
    assert "tasks=`3`" in card
    assert "metric_thresholds=`1`" in card


def test_render_result_card_summarizes_candidate_trace_stats() -> None:
    audit = _audit()
    audit["pi0fast"]["candidate_trace_stats"] = {
        "tree_width": {"rows": 100, "rows_with_stat": 100, "missing_rows": 0, "avg": 4.0, "min": 4.0, "max": 4.0},
        "tree_verifies": {"rows": 100, "rows_with_stat": 100, "missing_rows": 0, "avg": 3.25, "min": 1.0, "max": 8.0},
        "unverified_pattern_tokens": {
            "rows": 100,
            "rows_with_stat": 100,
            "missing_rows": 0,
            "total": 0.0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
    }

    card = render_result_card(audit)

    assert "Candidate trace stat" in card
    assert "`tree_width`: min=`4`, avg=`4`, max=`4`, missing=`0/100`" in card
    assert "`tree_verifies`: min=`1`, avg=`3.25`, max=`8`, missing=`0/100`" in card
    assert "`unverified_pattern_tokens`: min=`0`, avg=`0`, max=`0`, missing=`0/100`" in card
    assert "total=`0`" in card


def test_render_result_card_can_summarize_openvla_path() -> None:
    audit = _audit()
    audit["checks"] = {"real_robotics": True, "synthetic": True}
    audit["pi0fast"]["passed"] = False
    audit["pi0fast"]["candidate_trace_stats"] = {
        "tree_width": {"rows": 100, "rows_with_stat": 100, "missing_rows": 0, "avg": 4.0, "min": 4.0, "max": 4.0}
    }
    audit["openvla"] = {
        "passed": True,
        "candidate_mode": "spec",
        "speed": {
            "candidate_mode": "spec",
            "matched_pairs": 120,
            "baseline_successes": 90,
            "candidate_successes": 90,
            "success_drop_abs": 0.0,
            "speedup": 2.2,
            "baseline_success_regressions": 0,
        },
        "quality": {
            "candidate_rows_missing_spec_stats": 0,
            "unverified_action_shortcuts": 0,
            "unverified_draft_tokens": 0,
            "fast_draft_calls": 0,
            "chunk_buffer_hits": 0,
            "relaxed_group_accepts": 0,
            "max_tree_depth_used": 1,
        },
    }

    card = render_result_card(audit)

    assert "`120` matched OpenVLA/SpecVLA evals" in card
    assert "`spec`" in card
    assert "`2.20x`" in card
    assert "OpenVLA strict spec quality" in card
    assert "unverified shortcuts=`0`" in card
    assert "max tree depth=`1`" in card
    assert "Candidate trace stat" not in card


def test_render_result_card_marks_failed_audit() -> None:
    audit = _audit(passed=False)
    audit["missing_evidence"] = ["real PI0-FAST matched eval count is below threshold"]
    card = render_result_card(audit)

    assert "# Robotics Speculative Decoding Result: FAIL" in card
    assert "Only use this result card" in card
    assert "Missing evidence:" in card
    assert "real PI0-FAST matched eval count is below threshold" in card
