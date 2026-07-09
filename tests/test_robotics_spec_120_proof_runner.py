from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from scripts.run_robotics_spec_120_proof import build_manifest


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "path": "pi0fast-target-eos",
        "root": tmp_path / "proof",
        "python": "python",
        "pi0_runner_script": "scripts/run_pi0fast_100_eval_gate.py",
        "openvla_runner_script": "scripts/run_openvla_120_eval_gate.py",
        "openvla_config": Path("configs/libero_specvla_distributed.yaml"),
        "openvla_run_id": "openvla_specvla_120",
        "openvla_mode": "full",
        "suites": "libero_object,libero_spatial,libero_goal",
        "task_ids": "0,1,2,3,4,5,6,7,8,9",
        "episodes": 4,
        "steps": 300,
        "seed": 42,
        "device": "cuda",
        "dtype": "bfloat16",
        "min_pairs": 120,
        "min_unique_tasks": None,
        "min_speedup": 2.0,
        "min_baseline_successes": 1,
        "min_suite_matched_pairs": 1,
        "min_suite_baseline_successes": 1,
        "min_suite_speedup": 1.0,
        "max_success_drop": 0.0,
        "max_baseline_success_regressions": 0,
        "min_validation_episodes": 120,
        "min_exact_verifies": 1,
        "max_action_diff": 0.0,
        "synthetic_tasks": 120,
        "synthetic_min_speedup": 2.0,
        "synthetic_max_accuracy_drop": 0.0,
        "run_preflight": True,
        "require_hf_token": True,
        "run_synthetic": True,
        "run_final_audit": True,
        "render_result_card": True,
        "skip_existing": True,
        "pattern_sweep_json": None,
        "allow_manual_pattern_args": False,
        "pattern_sweep_rank": 0,
        "pattern_min_modeled_speedup": 2.0,
        "pattern_min_forward_reduction": 2.0,
        "pattern_min_task_forward_reduction": 1.25,
        "pattern_min_task_acceptance_rate": 0.30,
        "pattern_min_task_count": None,
        "pattern_min_heldout_modeled_speedup": 1.25,
        "pattern_min_heldout_forward_reduction": 1.25,
        "pattern_min_heldout_task_forward_reduction": 1.10,
        "pattern_min_heldout_task_acceptance_rate": 0.25,
        "pattern_min_heldout_task_count": None,
        "pattern_heldout_fraction": 0.2,
        "pattern_require_disjoint_heldout_tasks": True,
        "pattern_require_heldout_suite_coverage": True,
        "pattern_min_metric": [],
        "pattern_min_heldout_metric": [],
        "pattern_auto_source_min_metrics": True,
        "pattern_required_source_coverage": [],
        "pi05_policy": None,
        "pi05_num_inference_steps": None,
        "dry_run": True,
        "extra_args": [],
    }
    values.update(overrides)
    return Namespace(**values)


def test_target_eos_proof_runs_strict_pi0_gate_steps(tmp_path: Path) -> None:
    manifest = build_manifest(_args(tmp_path))
    command = manifest["command"]

    assert manifest["path"] == "pi0fast-target-eos"
    assert manifest["min_unique_tasks"] == 30
    assert "--speed-modes" in command
    assert command[command.index("--speed-modes") + 1] == "baseline,target_eos"
    assert "--candidate-mode" in command
    assert command[command.index("--candidate-mode") + 1] == "target_eos"
    assert "--run-preflight" in command
    assert "--require-hf-token" in command
    assert "--run-synthetic" in command
    assert "--run-final-audit" in command
    assert "--render-result-card" in command
    assert "--min-pairs" in command
    assert command[command.index("--min-pairs") + 1] == "120"
    assert "--min-unique-tasks" in command
    assert command[command.index("--min-unique-tasks") + 1] == "30"
    assert "--max-success-drop" in command
    assert command[command.index("--max-success-drop") + 1] == "0.0"


def test_adaptive_proof_runs_against_target_eos_reference(tmp_path: Path) -> None:
    manifest = build_manifest(_args(tmp_path, path="pi0fast-adaptive"))
    command = manifest["command"]

    assert manifest["path"] == "pi0fast-adaptive"
    assert manifest["early_stop_reference"] == "target_eos"
    assert "--speed-modes" in command
    assert command[command.index("--speed-modes") + 1] == "baseline,target_eos,target_eos_adaptive"
    assert "--candidate-mode" in command
    assert command[command.index("--candidate-mode") + 1] == "target_eos_adaptive"
    assert "--reference-mode" in command
    assert command[command.index("--reference-mode") + 1] == "target_eos"
    assert "--adaptive-prefix-checkpoints" in command
    assert command[command.index("--adaptive-prefix-checkpoints") + 1] == "32,64,96,128,160,192,224"
    assert "--adaptive-stable-checks" in command
    assert command[command.index("--adaptive-stable-checks") + 1] == "1"


def test_pattern_proof_requires_sweep_json_by_default(tmp_path: Path) -> None:
    try:
        build_manifest(_args(tmp_path, path="pi0fast-pattern"))
    except ValueError as exc:
        assert "--pattern-sweep-json" in str(exc)
        assert "heldout" in str(exc)
    else:
        raise AssertionError("expected pattern proof without sweep JSON to fail")


def test_manual_pattern_args_are_dry_run_only(tmp_path: Path) -> None:
    try:
        build_manifest(
            _args(
                tmp_path,
                path="pi0fast-pattern",
                allow_manual_pattern_args=True,
                dry_run=False,
            )
        )
    except ValueError as exc:
        assert "--dry-run" in str(exc)
    else:
        raise AssertionError("expected real manual pattern proof to fail")


def test_real_pattern_proof_requires_existing_sweep_json(tmp_path: Path) -> None:
    try:
        build_manifest(
            _args(
                tmp_path,
                path="pi0fast-pattern",
                pattern_sweep_json=tmp_path / "missing_sweep.json",
                dry_run=False,
            )
        )
    except ValueError as exc:
        assert "--pattern-sweep-json does not exist" in str(exc)
    else:
        raise AssertionError("expected missing sweep JSON to fail for real pattern proof")


def test_pattern_proof_rejects_sweep_without_heldout_split(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep_no_heldout.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {"heldout_split": "none", "heldout_trace_indices": []},
                "top": [
                    {
                        "heldout_modeled_speedup": 2.0,
                        "heldout_target_forward_reduction": 2.0,
                        "heldout_min_task_target_forward_reduction": 1.5,
                        "heldout_min_task_acceptance_rate": 0.5,
                        "heldout_task_count": 1,
                    }
                ],
            }
        )
        + "\n"
    )

    try:
        build_manifest(
            _args(
                tmp_path,
                path="pi0fast-pattern",
                pattern_sweep_json=sweep,
                pattern_min_heldout_task_count=1,
            )
        )
    except ValueError as exc:
        assert "non-empty heldout split" in str(exc)
    else:
        raise AssertionError("expected no-heldout sweep to fail")


def test_pattern_proof_rejects_overlapping_heldout_tasks(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep_overlap.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {
                    "heldout_split": "task_seed",
                    "heldout_trace_indices": [2],
                    "heldout_task_disjoint": False,
                    "heldout_task_overlap": ["libero_goal:0"],
                    "heldout_task_overlap_count": 1,
                },
                "top": [
                    {
                        "heldout_modeled_speedup": 2.0,
                        "heldout_target_forward_reduction": 2.0,
                        "heldout_min_task_target_forward_reduction": 1.5,
                        "heldout_min_task_acceptance_rate": 0.5,
                        "heldout_task_count": 1,
                    }
                ],
            }
        )
        + "\n"
    )

    try:
        build_manifest(
            _args(
                tmp_path,
                path="pi0fast-pattern",
                pattern_sweep_json=sweep,
                pattern_min_heldout_task_count=1,
            )
        )
    except ValueError as exc:
        assert "heldout tasks overlap ranking tasks" in str(exc)
        assert "libero_goal:0" in str(exc)
    else:
        raise AssertionError("expected overlapping heldout tasks to fail")


def test_pattern_proof_records_heldout_sweep_precheck(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep_with_heldout.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {
                    "heldout_split": "task",
                    "heldout_trace_indices": [2, 5],
                    "heldout_task_disjoint": True,
                    "heldout_task_overlap": [],
                    "heldout_task_overlap_count": 0,
                    "heldout_suite_keys": ["libero_goal", "libero_object", "libero_spatial"],
                    "heldout_suite_count": 3,
                },
                "top": [
                    {
                        "heldout_modeled_speedup": 2.0,
                        "heldout_target_forward_reduction": 2.0,
                        "heldout_min_task_target_forward_reduction": 1.5,
                        "heldout_min_task_acceptance_rate": 0.5,
                        "heldout_task_count": 2,
                    }
                ],
            }
        )
        + "\n"
    )

    manifest = build_manifest(
        _args(
            tmp_path,
            path="pi0fast-pattern",
            pattern_sweep_json=sweep,
            pattern_min_heldout_task_count=2,
        )
    )

    assert manifest["pattern_sweep_precheck"] == {
        "path": str(sweep),
        "exists": True,
        "requested_rank": 0,
        "min_heldout_task_count": 2,
        "require_disjoint_heldout_tasks": True,
        "required_heldout_suites": ["libero_object", "libero_spatial", "libero_goal"],
        "require_heldout_suite_coverage": True,
        "status": "ok",
        "heldout_split": "task",
        "heldout_trace_count": 2,
        "heldout_task_disjoint": True,
        "heldout_task_overlap_count": 0,
        "heldout_suite_count": 3,
        "heldout_suite_keys": ["libero_goal", "libero_object", "libero_spatial"],
        "eligible_ranks": [1],
    }


def test_pattern_proof_rejects_missing_heldout_suite_coverage(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep_undercovered.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {
                    "heldout_split": "task",
                    "heldout_trace_indices": [2, 5],
                    "heldout_task_disjoint": True,
                    "heldout_task_overlap": [],
                    "heldout_task_overlap_count": 0,
                    "heldout_suite_keys": ["libero_goal"],
                    "heldout_suite_count": 1,
                },
                "top": [
                    {
                        "heldout_modeled_speedup": 2.0,
                        "heldout_target_forward_reduction": 2.0,
                        "heldout_min_task_target_forward_reduction": 1.5,
                        "heldout_min_task_acceptance_rate": 0.5,
                        "heldout_task_count": 2,
                    }
                ],
            }
        )
        + "\n"
    )

    try:
        build_manifest(
            _args(
                tmp_path,
                path="pi0fast-pattern",
                pattern_sweep_json=sweep,
                pattern_min_heldout_task_count=2,
            )
        )
    except ValueError as exc:
        assert "heldout split does not cover requested suites" in str(exc)
        assert "libero_object" in str(exc)
        assert "libero_spatial" in str(exc)
    else:
        raise AssertionError("expected under-covered heldout suites to fail")


def test_pi05_pattern_proof_requires_pi05_token_adapter(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep.json"
    try:
        build_manifest(
            _args(
                tmp_path,
                path="pi05-pattern",
                pattern_sweep_json=sweep,
                pi05_policy="local/pi05",
                pi05_num_inference_steps=8,
            )
        )
    except ValueError as exc:
        assert "pi05-pattern is not currently runnable" in str(exc)
        assert "FAST-token hooks" in str(exc)
    else:
        raise AssertionError("expected pi05-pattern proof to require a PI0.5 token adapter")


def test_openvla_proof_keeps_strict_spec_stats_thresholds(tmp_path: Path) -> None:
    manifest = build_manifest(_args(tmp_path, path="openvla"))
    command = manifest["command"]

    assert manifest["runner"] == "openvla"
    assert manifest["strict_spec_stats"] is True
    assert "--allow-unverified-spec-shortcuts" not in command
    assert "--max-unverified-action-shortcuts" in command
    assert command[command.index("--max-unverified-action-shortcuts") + 1] == "0"
    assert "--max-fast-draft-calls" in command
    assert command[command.index("--max-fast-draft-calls") + 1] == "0"
    assert "--max-chunk-buffer-hits" in command
    assert command[command.index("--max-chunk-buffer-hits") + 1] == "0"
    assert "--max-relaxed-group-accepts" in command
    assert command[command.index("--max-relaxed-group-accepts") + 1] == "0"
    assert "--max-tree-depth-used" in command
    assert command[command.index("--max-tree-depth-used") + 1] == "1"
    assert "--synthetic-tasks" in command
    assert command[command.index("--synthetic-tasks") + 1] == "120"
