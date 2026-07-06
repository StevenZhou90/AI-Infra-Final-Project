from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.audit_robotics_spec_goal import build_audit, parse_args


def _args(**overrides):
    values = {
        "pi0fast_gate": Path("/missing/pi0fast_gate.json"),
        "pi0fast_manifest": None,
        "openvla_gate": None,
        "synthetic_gate": Path("/missing/synthetic_gate.json"),
        "min_pairs": 120,
        "min_unique_tasks": 0,
        "min_speedup": 2.0,
        "min_baseline_successes": 1,
        "min_suite_matched_pairs": 0,
        "min_suite_baseline_successes": 0,
        "min_suite_speedup": None,
        "max_success_drop": 0.0,
        "max_baseline_success_regressions": 0,
        "min_validation_episodes": 120,
        "max_action_diff": 0.0,
        "early_stop_mode": "target_eos",
        "require_early_stop_reference": True,
        "expected_policy_kind": None,
        "require_pattern_source_coverage": False,
        "require_pattern_heldout_evidence": False,
        "require_openvla_strict_spec": True,
        "require_synthetic": True,
        "synthetic_min_tasks": 120,
        "synthetic_min_speedup": 2.0,
        "synthetic_max_accuracy_drop": 0.0,
    }
    values.update(overrides)
    return Namespace(**values)


def _pi0fast_gate(
    candidate_mode: str = "target_eos",
    reference: dict | None = None,
    metadata: dict | None = None,
    thresholds: dict | None = None,
) -> dict:
    validation_mode = "target_eos_validate"
    if candidate_mode.startswith("block_sd"):
        validation_mode = "block_sd_validate"
    elif candidate_mode.startswith("pattern_sd"):
        validation_mode = "pattern_sd_validate"
    elif candidate_mode.startswith("ngram_sd"):
        validation_mode = "ngram_sd_validate"
    exact_validation = {
        "validation_mode": validation_mode,
        "episodes": 120,
        "expected_rows": 120,
        "missing_validation_rows": 0,
        "rows_missing_exact_verifies": 0,
        "max_action_diff": 0.0,
    }
    extra_exact_validations = {}
    if candidate_mode != "target_eos":
        extra_exact_validations["target_eos_validate"] = {
            "validation_mode": "target_eos_validate",
            "episodes": 120,
            "expected_rows": 120,
            "missing_validation_rows": 0,
            "rows_missing_exact_verifies": 0,
            "max_action_diff": 0.0,
        }
    if reference is not None:
        reference_defaults = {
            "step_mismatch_count": 0,
            "missing_step_count": 0,
        }
        reference_defaults.update(reference)
        reference = reference_defaults
    return {
        "gate_passed": True,
        "speed": {
            "baseline_mode": "baseline",
            "candidate_mode": candidate_mode,
            "matched_pairs": 120,
            "unique_tasks": 30,
            "baseline_successes": 81,
            "candidate_successes": 81,
            "suites": {
                "libero_goal": {
                    "matched_pairs": 120,
                    "baseline_successes": 81,
                    "candidate_successes": 81,
                    "speedup": 2.4,
                }
            },
            "speedup": 2.4,
            "success_drop_abs": 0.0,
            "baseline_success_regressions": 0,
            "step_mismatch_count": 0,
            "missing_step_count": 0,
        },
        "exact_validation": exact_validation,
        "extra_exact_validations": extra_exact_validations,
        "reference": reference,
        "candidate_trace_stats": {},
        "metadata": metadata or {},
        "thresholds": thresholds or {"require_matched_steps": True},
    }


def _synthetic_gate() -> dict:
    return {
        "gate_passed": True,
        "tasks": 120,
        "all_exact": True,
        "speedup": 5.44,
        "accuracy_drop_abs": 0.0,
    }


def _pi0fast_manifest(
    *,
    candidate_mode: str = "pattern_sd_direct",
    reference_mode: str | None = "target_eos",
    eval_metadata: dict | None = None,
    pattern_sweep_selection: dict | None = None,
    suites: list[str] | None = None,
) -> dict:
    row = {
        "candidate_mode": candidate_mode,
        "reference_mode": reference_mode,
        "eval_metadata": eval_metadata or {"policy_kind": "pi0fast"},
        "pattern_sweep_selection": pattern_sweep_selection,
        "suites": suites or ["libero_goal"],
    }
    return row


def _openvla_gate(
    speedup: float = 2.4,
    regressions: int = 0,
    matched_pairs: int = 120,
    quality: dict | None = None,
    thresholds: dict | None = None,
) -> dict:
    return {
        "gate_passed": speedup >= 2.0 and regressions == 0 and matched_pairs >= 120,
        "platform": "openvla_specvla",
        "speed": {
            "baseline_mode": "ar",
            "candidate_mode": "spec",
            "matched_pairs": matched_pairs,
            "unique_tasks": 40,
            "baseline_successes": 81,
            "candidate_successes": 81,
            "suites": {
                "libero_goal": {
                    "matched_pairs": matched_pairs,
                    "baseline_successes": 81,
                    "candidate_successes": 81,
                    "speedup": speedup,
                }
            },
            "success_drop_abs": 0.0,
            "baseline_success_regressions": regressions,
            "step_mismatch_count": 0,
            "speedup": speedup,
        },
        "quality": quality
        or {
            "candidate_rows_missing_spec_stats": 0,
            "fast_draft_calls": 0,
            "fast_draft_tokens": 0,
            "chunk_buffer_hits": 0,
            "chunk_buffered_actions": 0,
            "relaxed_group_accepts": 0,
            "max_tree_depth_used": 1,
            "unverified_action_shortcuts": 0,
            "unverified_draft_tokens": 0,
        },
        "thresholds": thresholds
        or {
            "require_spec_stats": True,
            "require_matched_steps": True,
            "max_unverified_action_shortcuts": 0,
            "max_fast_draft_calls": 0,
            "max_chunk_buffer_hits": 0,
            "max_relaxed_group_accepts": 0,
            "max_tree_depth_used": 1,
        },
    }


def _write_json(path: Path, row: dict) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n")
    return path


def test_objective_audit_cli_defaults_match_120_eval_objective() -> None:
    with patch("sys.argv", ["audit_robotics_spec_goal.py"]):
        args = parse_args()

    assert args.min_pairs == 120
    assert args.min_unique_tasks == 0
    assert args.min_validation_episodes == 120
    assert args.min_baseline_successes == 1
    assert args.min_suite_matched_pairs == 0
    assert args.min_suite_baseline_successes == 0
    assert args.min_suite_speedup is None
    assert args.synthetic_min_tasks == 120
    assert args.require_pattern_heldout_evidence is False


def test_objective_audit_passes_with_real_and_synthetic_artifacts(tmp_path: Path) -> None:
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", _pi0fast_gate()),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is True
    assert audit["checks"] == {"pi0fast": True, "synthetic": True}


def test_objective_audit_accepts_openvla_gate_as_real_path(tmp_path: Path) -> None:
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(tmp_path / "openvla_gate.json", _openvla_gate()),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is True
    assert audit["checks"] == {"real_robotics": True, "synthetic": True}
    assert audit["openvla"]["checks"]["openvla_gate_passed"] is True


def test_objective_audit_rejects_weak_openvla_gate(tmp_path: Path) -> None:
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(tmp_path / "openvla_gate.json", _openvla_gate(speedup=1.5)),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["checks"]["real_robotics"] is False
    assert audit["openvla"]["checks"]["openvla_speedup"] is False


def test_objective_audit_rejects_openvla_all_fail_success_artifact(tmp_path: Path) -> None:
    gate = _openvla_gate()
    gate["speed"]["baseline_successes"] = 0
    gate["speed"]["candidate_successes"] = 0
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(tmp_path / "openvla_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["checks"]["real_robotics"] is False
    assert audit["openvla"]["checks"]["openvla_baseline_successes"] is False
    assert "real OpenVLA/SpecVLA baseline success count is below threshold" in audit["missing_evidence"]


def test_objective_audit_rejects_openvla_low_unique_task_coverage(tmp_path: Path) -> None:
    gate = _openvla_gate()
    gate["speed"]["unique_tasks"] = 1
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(tmp_path / "openvla_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        min_unique_tasks=40,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["openvla"]["checks"]["openvla_unique_tasks"] is False
    assert "real OpenVLA/SpecVLA unique task coverage is below threshold" in audit["missing_evidence"]


def test_objective_audit_rejects_openvla_weak_suite_artifact(tmp_path: Path) -> None:
    gate = _openvla_gate()
    gate["speed"]["suites"]["libero_goal"]["speedup"] = 0.95
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(tmp_path / "openvla_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        min_suite_matched_pairs=1,
        min_suite_baseline_successes=1,
        min_suite_speedup=1.0,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["openvla"]["checks"]["openvla_suite_speedup"] is False
    assert "real OpenVLA/SpecVLA per-suite speedup is below threshold" in audit["missing_evidence"]


def test_objective_audit_rejects_openvla_gate_without_strict_spec_thresholds(tmp_path: Path) -> None:
    gate = _openvla_gate(thresholds={"require_spec_stats": False})
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(tmp_path / "openvla_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["checks"]["real_robotics"] is False
    assert audit["openvla"]["checks"]["openvla_strict_spec_thresholds"] is False
    assert "OpenVLA/SpecVLA gate did not enforce strict zero-shortcut thresholds" in audit["missing_evidence"]


def test_objective_audit_rejects_openvla_gate_without_matched_step_enforcement(tmp_path: Path) -> None:
    gate = _openvla_gate()
    gate["thresholds"]["require_matched_steps"] = False
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(tmp_path / "openvla_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["checks"]["real_robotics"] is False
    assert audit["openvla"]["checks"]["openvla_matched_steps"] is False
    assert "OpenVLA/SpecVLA gate did not enforce matched control-step counts" in audit["missing_evidence"]


def test_objective_audit_rejects_openvla_unverified_shortcuts(tmp_path: Path) -> None:
    quality = {
        "candidate_rows_missing_spec_stats": 0,
        "fast_draft_calls": 1,
        "fast_draft_tokens": 4,
        "chunk_buffer_hits": 1,
        "chunk_buffered_actions": 2,
        "relaxed_group_accepts": 1,
        "max_tree_depth_used": 2,
        "unverified_action_shortcuts": 3,
        "unverified_draft_tokens": 4,
    }
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(tmp_path / "openvla_gate.json", _openvla_gate(quality=quality)),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["openvla"]["checks"]["openvla_unverified_action_shortcuts"] is False
    assert audit["openvla"]["checks"]["openvla_unverified_draft_tokens"] is False
    assert audit["openvla"]["checks"]["openvla_fast_draft_calls"] is False
    assert audit["openvla"]["checks"]["openvla_chunk_buffer_hits"] is False
    assert audit["openvla"]["checks"]["openvla_chunk_buffered_actions"] is False
    assert audit["openvla"]["checks"]["openvla_relaxed_group_accepts"] is False
    assert audit["openvla"]["checks"]["openvla_tree_depth"] is False


def test_objective_audit_can_disable_openvla_strict_spec_for_research_artifacts(tmp_path: Path) -> None:
    args = _args(
        pi0fast_gate=tmp_path / "missing_pi0fast_gate.json",
        openvla_gate=_write_json(
            tmp_path / "openvla_gate.json",
            _openvla_gate(quality={}, thresholds={}),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        require_openvla_strict_spec=False,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is True
    assert audit["openvla"]["checks"]["openvla_strict_spec_thresholds"] is True


def test_objective_audit_fails_without_real_pi0fast_gate(tmp_path: Path) -> None:
    args = _args(
        pi0fast_gate=tmp_path / "missing_gate.json",
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_gate_present"] is False
    assert "missing PI0-FAST gate JSON artifact" in audit["missing_evidence"]


def test_objective_audit_rejects_pi0fast_all_fail_success_artifact(tmp_path: Path) -> None:
    gate = _pi0fast_gate()
    gate["speed"]["baseline_successes"] = 0
    gate["speed"]["candidate_successes"] = 0
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_baseline_successes"] is False
    assert "real PI0-FAST baseline success count is below threshold" in audit["missing_evidence"]


def test_objective_audit_rejects_pi0fast_low_unique_task_coverage(tmp_path: Path) -> None:
    gate = _pi0fast_gate()
    gate["speed"]["unique_tasks"] = 1
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        min_unique_tasks=30,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_unique_tasks"] is False
    assert "real PI0-FAST unique task coverage is below threshold" in audit["missing_evidence"]


def test_objective_audit_rejects_pi0fast_weak_suite_artifact(tmp_path: Path) -> None:
    gate = _pi0fast_gate()
    gate["speed"]["suites"]["libero_goal"]["baseline_successes"] = 0
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        min_suite_matched_pairs=1,
        min_suite_baseline_successes=1,
        min_suite_speedup=1.0,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_suite_baseline_successes"] is False
    assert "real PI0-FAST per-suite baseline success count is below threshold" in audit["missing_evidence"]


def test_objective_audit_rejects_pi0fast_without_matched_step_threshold(tmp_path: Path) -> None:
    gate = _pi0fast_gate(thresholds={"require_matched_steps": False})
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_matched_steps"] is False
    assert "PI0-FAST gate did not enforce matched control-step counts" in audit["missing_evidence"]


def test_objective_audit_rejects_pi0fast_step_mismatch_artifact(tmp_path: Path) -> None:
    gate = _pi0fast_gate()
    gate["speed"]["step_mismatch_count"] = 1
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_matched_steps"] is False


def test_objective_audit_requires_target_eos_reference_for_spec_candidate(tmp_path: Path) -> None:
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", _pi0fast_gate(candidate_mode="block_sd_direct")),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_early_stop_reference"] is False


def test_objective_audit_accepts_target_eos_reference_for_spec_candidate(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "block_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="block_sd_direct", reference=reference),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_early_stop_reference"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_reference_speedup"] is True


def test_objective_audit_rejects_slow_target_eos_reference_speedup(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "block_sd_direct",
        "matched_pairs": 120,
        "speedup": 1.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="block_sd_direct", reference=reference),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_reference_speedup"] is False
    assert "candidate speedup versus target_eos is below threshold" in audit["missing_evidence"]


def test_objective_audit_preserves_pi0fast_candidate_trace_stats(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    gate = _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference)
    gate["candidate_trace_stats"] = {
        "tree_width": {"rows": 120, "rows_with_stat": 120, "missing_rows": 0, "avg": 4.0, "min": 4.0, "max": 4.0},
        "tree_verifies": {"rows": 120, "rows_with_stat": 120, "missing_rows": 0, "avg": 3.2, "min": 1.0, "max": 8.0},
        "unverified_pattern_tokens": {
            "rows": 120,
            "rows_with_stat": 120,
            "missing_rows": 0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
    }
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is True
    assert audit["pi0fast"]["candidate_trace_stats"]["tree_width"]["min"] == 4.0
    assert audit["pi0fast"]["candidate_trace_stats"]["unverified_pattern_tokens"]["max"] == 0.0


def test_objective_audit_rejects_target_eos_validation_for_spec_candidate(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    gate = _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference)
    gate["exact_validation"]["validation_mode"] = "target_eos_validate"
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_candidate_validation_mode"] is False


def test_objective_audit_rejects_missing_early_stop_exact_validation(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    gate = _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference)
    gate["extra_exact_validations"] = {}
    args = _args(
        pi0fast_gate=_write_json(tmp_path / "pi0fast_gate.json", gate),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_early_stop_exact_validation"] is False
    assert "missing passing target_eos_validate exactness rows for the early-stop reference" in audit["missing_evidence"]


def test_objective_audit_rejects_drop_against_target_eos_reference(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "block_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.01,
        "baseline_success_regressions": 1,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="block_sd_direct", reference=reference),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_reference_success_drop"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_reference_success_regressions"] is False


def test_objective_audit_rejects_pi0fast_reference_step_mismatch(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "block_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
        "step_mismatch_count": 1,
        "missing_step_count": 0,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="block_sd_direct", reference=reference),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_reference_matched_steps"] is False


def test_objective_audit_rejects_wrong_reference_candidate(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "block_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_reference_candidate_mode"] is False


def test_objective_audit_rejects_too_few_reference_pairs(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 119,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_reference_matched_pairs"] is False


def test_objective_audit_requires_expected_pi05_policy_kind(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(
                candidate_mode="pattern_sd_direct",
                reference=reference,
                metadata={"policy_kind": "pi0fast"},
            ),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        expected_policy_kind="pi05",
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_policy_kind"] is False
    assert "PI0-FAST/PI0.5 gate policy kind does not match expected policy" in audit["missing_evidence"]


def test_objective_audit_accepts_expected_pi05_policy_kind(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(
                candidate_mode="pattern_sd_direct",
                reference=reference,
                metadata={"policy_kind": "pi05", "num_inference_steps": "8"},
            ),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        expected_policy_kind="pi05",
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_policy_kind"] is True
    assert audit["pi0fast"]["metadata"]["policy_kind"] == "pi05"


def test_objective_audit_rejects_wrong_manifest_reference_for_spec_candidate(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference),
        ),
        pi0fast_manifest=_write_json(
            tmp_path / "run_manifest.json",
            _pi0fast_manifest(candidate_mode="pattern_sd_direct", reference_mode="baseline"),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_manifest_reference_mode"] is False
    assert "PI0-FAST run manifest does not record target_eos reference mode" in audit["missing_evidence"]


def test_objective_audit_accepts_manifest_pattern_source_coverage(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    selection = {
        "required_source_coverage": ["chunk_delta_template", "action_delta_histogram"],
        "required_source_counts": {"chunk_delta_template": 3, "action_delta_histogram": 2},
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference, metadata={"policy_kind": "pi05"}),
        ),
        pi0fast_manifest=_write_json(
            tmp_path / "run_manifest.json",
            _pi0fast_manifest(
                candidate_mode="pattern_sd_direct",
                reference_mode="target_eos",
                eval_metadata={"policy_kind": "pi05"},
                pattern_sweep_selection=selection,
            ),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        expected_policy_kind="pi05",
        require_pattern_source_coverage=True,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_manifest_policy_kind"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_source_coverage"] is True


def test_objective_audit_accepts_manifest_pattern_heldout_evidence(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    selection = {
        "required_source_coverage": ["chunk_delta_template"],
        "required_source_counts": {"chunk_delta_template": 3},
        "heldout_split": "task",
        "heldout_trace_count": 12,
        "heldout_task_disjoint": True,
        "heldout_task_overlap_count": 0,
        "heldout_task_overlap": [],
        "heldout_suite_keys": ["libero_goal"],
        "heldout_suite_count": 1,
        "min_heldout_modeled_speedup": 2.0,
        "min_heldout_task_count": 3,
        "effective_min_heldout_metric": ["chunk_delta_template_accepted_tokens=2"],
        "selected": {
            "heldout_modeled_speedup": 2.1,
            "heldout_task_count": 3,
            "heldout_chunk_delta_template_accepted_tokens": 2,
        },
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference),
        ),
        pi0fast_manifest=_write_json(
            tmp_path / "run_manifest.json",
            _pi0fast_manifest(pattern_sweep_selection=selection),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        require_pattern_source_coverage=True,
        require_pattern_heldout_evidence=True,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_source_coverage"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_heldout_evidence"] is True


def test_objective_audit_rejects_overlapping_pattern_heldout_tasks(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    selection = {
        "required_source_coverage": ["chunk_delta_template"],
        "required_source_counts": {"chunk_delta_template": 3},
        "heldout_split": "task_seed",
        "heldout_trace_count": 12,
        "heldout_task_disjoint": False,
        "heldout_task_overlap_count": 1,
        "heldout_task_overlap": ["libero_goal:0"],
        "heldout_suite_keys": ["libero_goal"],
        "heldout_suite_count": 1,
        "min_heldout_modeled_speedup": 2.0,
        "min_heldout_task_count": 3,
        "effective_min_heldout_metric": ["chunk_delta_template_accepted_tokens=2"],
        "selected": {
            "heldout_modeled_speedup": 2.1,
            "heldout_task_count": 3,
            "heldout_chunk_delta_template_accepted_tokens": 2,
        },
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference),
        ),
        pi0fast_manifest=_write_json(
            tmp_path / "run_manifest.json",
            _pi0fast_manifest(pattern_sweep_selection=selection),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        require_pattern_source_coverage=True,
        require_pattern_heldout_evidence=True,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_source_coverage"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_heldout_evidence"] is False
    assert "pattern sweep selection lacks required heldout selection evidence" in audit["missing_evidence"]


def test_objective_audit_rejects_pattern_heldout_missing_manifest_suite(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    selection = {
        "required_source_coverage": ["chunk_delta_template"],
        "required_source_counts": {"chunk_delta_template": 3},
        "heldout_split": "task",
        "heldout_trace_count": 12,
        "heldout_task_disjoint": True,
        "heldout_task_overlap_count": 0,
        "heldout_task_overlap": [],
        "heldout_suite_keys": ["libero_goal"],
        "heldout_suite_count": 1,
        "min_heldout_modeled_speedup": 2.0,
        "min_heldout_task_count": 3,
        "effective_min_heldout_metric": ["chunk_delta_template_accepted_tokens=2"],
        "selected": {
            "heldout_modeled_speedup": 2.1,
            "heldout_task_count": 3,
            "heldout_chunk_delta_template_accepted_tokens": 2,
        },
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference),
        ),
        pi0fast_manifest=_write_json(
            tmp_path / "run_manifest.json",
            _pi0fast_manifest(
                pattern_sweep_selection=selection,
                suites=["libero_goal", "libero_spatial"],
            ),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        require_pattern_source_coverage=True,
        require_pattern_heldout_evidence=True,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_source_coverage"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_heldout_evidence"] is False
    assert "pattern sweep selection lacks required heldout selection evidence" in audit["missing_evidence"]


def test_objective_audit_rejects_manifest_without_pattern_heldout_requirement(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    selection = {
        "required_source_coverage": ["chunk_delta_template"],
        "required_source_counts": {"chunk_delta_template": 3},
        "selected": {
            "heldout_modeled_speedup": 2.1,
            "heldout_task_count": 3,
        },
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference),
        ),
        pi0fast_manifest=_write_json(
            tmp_path / "run_manifest.json",
            _pi0fast_manifest(pattern_sweep_selection=selection),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        require_pattern_source_coverage=True,
        require_pattern_heldout_evidence=True,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_source_coverage"] is True
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_heldout_evidence"] is False
    assert "pattern sweep selection lacks required heldout selection evidence" in audit["missing_evidence"]


def test_objective_audit_rejects_manifest_missing_required_source_count(tmp_path: Path) -> None:
    reference = {
        "baseline_mode": "target_eos",
        "candidate_mode": "pattern_sd_direct",
        "matched_pairs": 120,
        "speedup": 2.05,
        "success_drop_abs": 0.0,
        "baseline_success_regressions": 0,
    }
    selection = {
        "required_source_coverage": ["chunk_delta_template", "action_delta_histogram"],
        "required_source_counts": {"chunk_delta_template": 3, "action_delta_histogram": 0},
    }
    args = _args(
        pi0fast_gate=_write_json(
            tmp_path / "pi0fast_gate.json",
            _pi0fast_gate(candidate_mode="pattern_sd_direct", reference=reference),
        ),
        pi0fast_manifest=_write_json(
            tmp_path / "run_manifest.json",
            _pi0fast_manifest(pattern_sweep_selection=selection),
        ),
        synthetic_gate=_write_json(tmp_path / "synthetic_gate.json", _synthetic_gate()),
        require_pattern_source_coverage=True,
    )

    audit = build_audit(args)

    assert audit["objective_audit_passed"] is False
    assert audit["pi0fast"]["checks"]["pi0fast_pattern_source_coverage"] is False
    assert "pattern sweep selection lacks required evaluated source coverage" in audit["missing_evidence"]
