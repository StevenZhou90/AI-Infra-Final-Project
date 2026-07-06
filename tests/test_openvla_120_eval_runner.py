from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.run_openvla_120_eval_gate import build_manifest, scheduled_command_keys


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                f"output_root: {tmp_path / 'outputs'}",
                "benchmark:",
                "  suites: [libero_goal]",
                "  tasks_per_suite: 10",
                "  trials_per_task: 12",
            ]
        )
        + "\n"
    )
    return path


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "config": _write_config(tmp_path),
        "python": "python",
        "runner_script": "scripts/run_libero_specvla_distributed.py",
        "gate_script": "scripts/gate_openvla_specvla.py",
        "synthetic_script": "scripts/benchmark_robotics_spec_decode_synthetic.py",
        "audit_script": "scripts/audit_robotics_spec_goal.py",
        "result_card_script": "scripts/render_robotics_spec_result_card.py",
        "run_id": "test_openvla_120",
        "mode": "full",
        "baseline_mode": "ar",
        "candidate_mode": "spec",
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
        "allow_unverified_spec_shortcuts": False,
        "max_unverified_action_shortcuts": 0,
        "max_fast_draft_calls": 0,
        "max_chunk_buffer_hits": 0,
        "max_relaxed_group_accepts": 0,
        "max_tree_depth_used": 1,
        "pi0fast_gate": None,
        "gate_output": None,
        "synthetic_output": None,
        "audit_output": None,
        "result_card_output": None,
        "synthetic_tasks": 120,
        "synthetic_min_speedup": 2.0,
        "synthetic_max_accuracy_drop": 0.0,
        "skip_runner": False,
        "skip_gate": False,
        "skip_synthetic": False,
        "skip_audit": False,
        "skip_result_card": False,
        "runner_dry_run": False,
        "dry_run": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_openvla_manifest_builds_strict_gate_and_audit_commands(tmp_path: Path) -> None:
    manifest = build_manifest(_args(tmp_path))

    mode_dir = tmp_path / "outputs" / "test_openvla_120" / "full"
    assert manifest["mode_dir"] == str(mode_dir)
    assert manifest["baseline_jsonl"] == str(mode_dir / "ar_episodes.jsonl")
    assert manifest["candidate_jsonl"] == str(mode_dir / "spec_episodes.jsonl")
    assert manifest["gate_output"] == str(mode_dir / "openvla_gate.json")
    assert "--min-pairs" in manifest["gate_command"]
    assert "120" in manifest["gate_command"]
    assert "--min-unique-tasks" in manifest["gate_command"]
    assert "--min-baseline-successes" in manifest["gate_command"]
    assert "--min-suite-matched-pairs" in manifest["gate_command"]
    assert "--min-suite-baseline-successes" in manifest["gate_command"]
    assert "--min-suite-speedup" in manifest["gate_command"]
    assert "--max-baseline-success-regressions" in manifest["gate_command"]
    assert "--require-matched-steps" in manifest["gate_command"]
    assert "--require-spec-stats" in manifest["gate_command"]
    assert "--max-unverified-action-shortcuts" in manifest["gate_command"]
    assert "--max-tree-depth-used" in manifest["gate_command"]
    assert manifest["thresholds"]["strict_spec_stats"] is True
    assert manifest["thresholds"]["min_unique_tasks"] == 10
    assert manifest["thresholds"]["min_baseline_successes"] == 1
    assert manifest["thresholds"]["min_suite_matched_pairs"] == 1
    assert manifest["thresholds"]["min_suite_baseline_successes"] == 1
    assert manifest["thresholds"]["min_suite_speedup"] == 1.0
    assert manifest["thresholds"]["require_matched_steps"] is True
    assert "--openvla-gate" in manifest["audit_command"]
    assert "--min-unique-tasks" in manifest["audit_command"]
    assert "--min-baseline-successes" in manifest["audit_command"]
    assert "--min-suite-matched-pairs" in manifest["audit_command"]
    assert "--min-suite-baseline-successes" in manifest["audit_command"]
    assert "--min-suite-speedup" in manifest["audit_command"]
    assert "--require-openvla-strict-spec" in manifest["audit_command"]
    assert str(mode_dir / "openvla_gate.json") in manifest["audit_command"]
    assert str(mode_dir / "unused_pi0fast_gate.json") in manifest["audit_command"]


def test_openvla_manifest_prefers_existing_global_artifacts(tmp_path: Path) -> None:
    args = _args(tmp_path)
    mode_dir = tmp_path / "outputs" / "test_openvla_120" / "full"
    mode_dir.mkdir(parents=True)
    (mode_dir / "ar_episodes.global.jsonl").write_text("")
    (mode_dir / "spec_episodes.global.jsonl").write_text("")

    manifest = build_manifest(args)

    assert manifest["baseline_jsonl"] == str(mode_dir / "ar_episodes.global.jsonl")
    assert manifest["candidate_jsonl"] == str(mode_dir / "spec_episodes.global.jsonl")


def test_openvla_manifest_can_pass_runner_dry_run(tmp_path: Path) -> None:
    manifest = build_manifest(_args(tmp_path, runner_dry_run=True))

    assert "--dry-run" in manifest["runner_command"]


def test_openvla_manifest_requires_research_shortcut_mode_to_skip_final_audit(tmp_path: Path) -> None:
    try:
        build_manifest(_args(tmp_path, allow_unverified_spec_shortcuts=True))
    except ValueError as exc:
        assert "--allow-unverified-spec-shortcuts is research-only" in str(exc)
        assert "--skip-audit" in str(exc)
        assert "--skip-result-card" in str(exc)
    else:
        raise AssertionError("expected relaxed OpenVLA final-audit run to fail")


def test_openvla_manifest_can_allow_unverified_shortcut_research_mode(tmp_path: Path) -> None:
    manifest = build_manifest(
        _args(
            tmp_path,
            allow_unverified_spec_shortcuts=True,
            skip_audit=True,
            skip_result_card=True,
        )
    )

    assert "--require-spec-stats" not in manifest["gate_command"]
    assert "--max-unverified-action-shortcuts" not in manifest["gate_command"]
    assert "--require-openvla-strict-spec" not in manifest["audit_command"]
    assert manifest["thresholds"]["strict_spec_stats"] is False


def test_openvla_dry_run_prints_only_scheduled_commands(tmp_path: Path) -> None:
    args = _args(tmp_path, skip_audit=True, skip_result_card=True)

    assert scheduled_command_keys(args) == ("runner_command", "gate_command", "synthetic_command")
