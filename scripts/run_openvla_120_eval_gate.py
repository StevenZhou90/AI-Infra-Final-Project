#!/usr/bin/env python3
"""Orchestrate an OpenVLA/SpecVLA 120 matched-eval gate.

This wrapper does not implement a new benchmark. It runs the existing
``run_libero_specvla_distributed.py`` artifact layout, then applies the strict
matched AR-vs-spec gate used for the robotics speculative decoding objective.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mode_dir_from_config(config: dict[str, Any], *, run_id: str, mode: str) -> Path:
    return Path(config["output_root"]) / run_id / mode


def artifact_paths(mode_dir: Path) -> tuple[Path, Path]:
    global_ar = mode_dir / "ar_episodes.global.jsonl"
    global_spec = mode_dir / "spec_episodes.global.jsonl"
    if global_ar.exists() or global_spec.exists():
        return global_ar, global_spec
    return mode_dir / "ar_episodes.jsonl", mode_dir / "spec_episodes.jsonl"


def _suite_value(raw: Any, suite_name: str, default: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get(suite_name, default)
    if raw is None:
        return default
    return raw


def _task_ids_for_suite(config: dict[str, Any], suite_name: str, task_limit: int) -> list[int]:
    raw = _suite_value(config["benchmark"].get("task_ids_by_suite"), suite_name, None)
    if raw is None:
        return list(range(task_limit))
    if isinstance(raw, str):
        task_ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        task_ids = [int(value) for value in raw]
    return task_ids[:task_limit]


def expected_unique_tasks(config: dict[str, Any], *, mode: str) -> int:
    task_limit = int(config["smoke"]["tasks_per_suite"] if mode == "smoke" else config["benchmark"]["tasks_per_suite"])
    return sum(len(_task_ids_for_suite(config, str(suite), task_limit)) for suite in config["benchmark"]["suites"])


def build_runner_command(
    *,
    python: str,
    runner_script: str,
    config: Path,
    mode: str,
    run_id: str,
    runner_dry_run: bool,
) -> list[str]:
    cmd = [
        python,
        runner_script,
        "--config",
        str(config),
        "--mode",
        mode,
        "--run-id",
        run_id,
    ]
    if runner_dry_run:
        cmd.append("--dry-run")
    return cmd


def build_gate_command(
    *,
    python: str,
    gate_script: str,
    baseline_jsonl: Path,
    candidate_jsonl: Path,
    output: Path,
    baseline_mode: str,
    candidate_mode: str,
    min_pairs: int,
    min_unique_tasks: int,
    min_speedup: float,
    min_baseline_successes: int,
    min_suite_matched_pairs: int,
    min_suite_baseline_successes: int,
    min_suite_speedup: float | None,
    max_success_drop: float,
    max_baseline_success_regressions: int,
    require_matched_steps: bool,
    require_spec_stats: bool,
    max_unverified_action_shortcuts: int | None,
    max_fast_draft_calls: int | None,
    max_chunk_buffer_hits: int | None,
    max_relaxed_group_accepts: int | None,
    max_tree_depth_used: int | None,
) -> list[str]:
    cmd = [
        python,
        gate_script,
        "--baseline-jsonl",
        str(baseline_jsonl),
        "--candidate-jsonl",
        str(candidate_jsonl),
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
        "--output",
        str(output),
        "--markdown",
    ]
    cmd.append("--require-matched-steps" if require_matched_steps else "--no-require-matched-steps")
    if require_spec_stats:
        cmd.append("--require-spec-stats")
    if max_unverified_action_shortcuts is not None:
        cmd.extend(["--max-unverified-action-shortcuts", str(max_unverified_action_shortcuts)])
    if min_suite_speedup is not None:
        cmd.extend(["--min-suite-speedup", str(min_suite_speedup)])
    if max_fast_draft_calls is not None:
        cmd.extend(["--max-fast-draft-calls", str(max_fast_draft_calls)])
    if max_chunk_buffer_hits is not None:
        cmd.extend(["--max-chunk-buffer-hits", str(max_chunk_buffer_hits)])
    if max_relaxed_group_accepts is not None:
        cmd.extend(["--max-relaxed-group-accepts", str(max_relaxed_group_accepts)])
    if max_tree_depth_used is not None:
        cmd.extend(["--max-tree-depth-used", str(max_tree_depth_used)])
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
    openvla_gate: Path,
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
    synthetic_min_tasks: int,
    synthetic_min_speedup: float,
    synthetic_max_accuracy_drop: float,
    require_openvla_strict_spec: bool,
) -> list[str]:
    cmd = [
        python,
        audit_script,
        "--pi0fast-gate",
        str(pi0fast_gate),
        "--openvla-gate",
        str(openvla_gate),
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
        "--synthetic-min-tasks",
        str(synthetic_min_tasks),
        "--synthetic-min-speedup",
        str(synthetic_min_speedup),
        "--synthetic-max-accuracy-drop",
        str(synthetic_max_accuracy_drop),
        "--output",
        str(output),
        "--markdown",
    ]
    if require_openvla_strict_spec:
        cmd.append("--require-openvla-strict-spec")
    if min_suite_speedup is not None:
        cmd.extend(["--min-suite-speedup", str(min_suite_speedup)])
    return cmd


def build_result_card_command(
    *,
    python: str,
    result_card_script: str,
    audit_json: Path,
    output: Path,
) -> list[str]:
    return [python, result_card_script, str(audit_json), "--output", str(output)]


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    mode_dir = mode_dir_from_config(config, run_id=args.run_id, mode=args.mode)
    baseline_jsonl, candidate_jsonl = artifact_paths(mode_dir)
    gate_output = args.gate_output or (mode_dir / "openvla_gate.json")
    synthetic_output = args.synthetic_output or (mode_dir / "synthetic_gate.json")
    audit_output = args.audit_output or (mode_dir / "objective_audit.json")
    result_card_output = args.result_card_output or (mode_dir / "result_card.md")
    pi0fast_gate = args.pi0fast_gate or (mode_dir / "unused_pi0fast_gate.json")
    strict_spec_stats = not bool(getattr(args, "allow_unverified_spec_shortcuts", False))
    if not strict_spec_stats and (not args.skip_audit or not args.skip_result_card):
        raise ValueError(
            "--allow-unverified-spec-shortcuts is research-only; pass --skip-audit and "
            "--skip-result-card so relaxed OpenVLA artifacts cannot be treated as final objective evidence."
        )
    min_unique_tasks = int(args.min_unique_tasks) if args.min_unique_tasks is not None else expected_unique_tasks(
        config,
        mode=args.mode,
    )

    runner_command = build_runner_command(
        python=args.python,
        runner_script=args.runner_script,
        config=args.config,
        mode=args.mode,
        run_id=args.run_id,
        runner_dry_run=args.runner_dry_run,
    )
    gate_command = build_gate_command(
        python=args.python,
        gate_script=args.gate_script,
        baseline_jsonl=baseline_jsonl,
        candidate_jsonl=candidate_jsonl,
        output=gate_output,
        baseline_mode=args.baseline_mode,
        candidate_mode=args.candidate_mode,
        min_pairs=args.min_pairs,
        min_unique_tasks=min_unique_tasks,
        min_speedup=args.min_speedup,
        min_baseline_successes=args.min_baseline_successes,
        min_suite_matched_pairs=args.min_suite_matched_pairs,
        min_suite_baseline_successes=args.min_suite_baseline_successes,
        min_suite_speedup=args.min_suite_speedup,
        max_success_drop=args.max_success_drop,
        max_baseline_success_regressions=args.max_baseline_success_regressions,
        require_matched_steps=args.require_matched_steps,
        require_spec_stats=strict_spec_stats,
        max_unverified_action_shortcuts=args.max_unverified_action_shortcuts if strict_spec_stats else None,
        max_fast_draft_calls=args.max_fast_draft_calls if strict_spec_stats else None,
        max_chunk_buffer_hits=args.max_chunk_buffer_hits if strict_spec_stats else None,
        max_relaxed_group_accepts=args.max_relaxed_group_accepts if strict_spec_stats else None,
        max_tree_depth_used=args.max_tree_depth_used if strict_spec_stats else None,
    )
    synthetic_command = build_synthetic_command(
        python=args.python,
        synthetic_script=args.synthetic_script,
        output=synthetic_output,
        tasks=args.synthetic_tasks,
        min_speedup=args.synthetic_min_speedup,
        max_accuracy_drop=args.synthetic_max_accuracy_drop,
    )
    audit_command = build_audit_command(
        python=args.python,
        audit_script=args.audit_script,
        pi0fast_gate=pi0fast_gate,
        openvla_gate=gate_output,
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
        synthetic_min_tasks=args.synthetic_tasks,
        synthetic_min_speedup=args.synthetic_min_speedup,
        synthetic_max_accuracy_drop=args.synthetic_max_accuracy_drop,
        require_openvla_strict_spec=strict_spec_stats,
    )
    result_card_command = build_result_card_command(
        python=args.python,
        result_card_script=args.result_card_script,
        audit_json=audit_output,
        output=result_card_output,
    )
    return {
        "config": str(args.config),
        "run_id": args.run_id,
        "mode": args.mode,
        "mode_dir": str(mode_dir),
        "baseline_jsonl": str(baseline_jsonl),
        "candidate_jsonl": str(candidate_jsonl),
        "gate_output": str(gate_output),
        "synthetic_output": str(synthetic_output),
        "audit_output": str(audit_output),
        "result_card_output": str(result_card_output),
        "thresholds": {
            "min_pairs": args.min_pairs,
            "min_unique_tasks": min_unique_tasks,
            "min_speedup": args.min_speedup,
            "min_baseline_successes": args.min_baseline_successes,
            "min_suite_matched_pairs": args.min_suite_matched_pairs,
            "min_suite_baseline_successes": args.min_suite_baseline_successes,
            "min_suite_speedup": args.min_suite_speedup,
            "max_success_drop": args.max_success_drop,
            "max_baseline_success_regressions": args.max_baseline_success_regressions,
            "require_matched_steps": args.require_matched_steps,
            "synthetic_tasks": args.synthetic_tasks,
            "synthetic_min_speedup": args.synthetic_min_speedup,
            "synthetic_max_accuracy_drop": args.synthetic_max_accuracy_drop,
            "strict_spec_stats": strict_spec_stats,
            "max_unverified_action_shortcuts": args.max_unverified_action_shortcuts if strict_spec_stats else None,
            "max_fast_draft_calls": args.max_fast_draft_calls if strict_spec_stats else None,
            "max_chunk_buffer_hits": args.max_chunk_buffer_hits if strict_spec_stats else None,
            "max_relaxed_group_accepts": args.max_relaxed_group_accepts if strict_spec_stats else None,
            "max_tree_depth_used": args.max_tree_depth_used if strict_spec_stats else None,
        },
        "runner_command": runner_command,
        "gate_command": gate_command,
        "synthetic_command": synthetic_command,
        "audit_command": audit_command,
        "result_card_command": result_card_command,
    }


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def scheduled_command_keys(args: argparse.Namespace) -> tuple[str, ...]:
    keys: list[str] = []
    if not args.skip_runner:
        keys.append("runner_command")
    if not args.skip_gate:
        keys.append("gate_command")
    if not args.skip_synthetic:
        keys.append("synthetic_command")
    if not args.skip_audit:
        keys.append("audit_command")
    if not args.skip_result_card:
        keys.append("result_card_command")
    return tuple(keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/gate a 120 OpenVLA AR-vs-SpecVLA eval.")
    parser.add_argument("--config", type=Path, default=Path("configs/libero_specvla_distributed.yaml"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner-script", default="scripts/run_libero_specvla_distributed.py")
    parser.add_argument("--gate-script", default="scripts/gate_openvla_specvla.py")
    parser.add_argument("--synthetic-script", default="scripts/benchmark_robotics_spec_decode_synthetic.py")
    parser.add_argument("--audit-script", default="scripts/audit_robotics_spec_goal.py")
    parser.add_argument("--result-card-script", default="scripts/render_robotics_spec_result_card.py")
    parser.add_argument("--run-id", default="openvla_specvla_120")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--baseline-mode", default="ar")
    parser.add_argument("--candidate-mode", default="spec")
    parser.add_argument("--min-pairs", type=int, default=120)
    parser.add_argument(
        "--min-unique-tasks",
        type=int,
        default=None,
        help="Minimum unique (suite, task_id) coverage; defaults to suites times configured tasks per suite.",
    )
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--min-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-matched-pairs", type=int, default=1)
    parser.add_argument("--min-suite-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-speedup", type=float, default=1.0)
    parser.add_argument("--max-success-drop", type=float, default=0.0)
    parser.add_argument("--max-baseline-success-regressions", type=int, default=0)
    parser.add_argument(
        "--require-matched-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require paired AR/spec rows to report the same number of control steps.",
    )
    parser.add_argument(
        "--allow-unverified-spec-shortcuts",
        action="store_true",
        help="Do not require strict zero-risk spec_stats counters in the OpenVLA gate.",
    )
    parser.add_argument("--max-unverified-action-shortcuts", type=int, default=0)
    parser.add_argument("--max-fast-draft-calls", type=int, default=0)
    parser.add_argument("--max-chunk-buffer-hits", type=int, default=0)
    parser.add_argument("--max-relaxed-group-accepts", type=int, default=0)
    parser.add_argument("--max-tree-depth-used", type=int, default=1)
    parser.add_argument("--pi0fast-gate", type=Path, default=None)
    parser.add_argument("--gate-output", type=Path, default=None)
    parser.add_argument("--synthetic-output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--result-card-output", type=Path, default=None)
    parser.add_argument("--synthetic-tasks", type=int, default=120)
    parser.add_argument("--synthetic-min-speedup", type=float, default=2.0)
    parser.add_argument("--synthetic-max-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--skip-runner", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-result-card", action="store_true")
    parser.add_argument("--runner-dry-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write and print manifest commands without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    mode_dir = Path(manifest["mode_dir"])
    mode_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = mode_dir / "openvla_120_gate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")
    if args.dry_run:
        for key in scheduled_command_keys(args):
            print("+ " + " ".join(manifest[key]))
        print("Dry run only; no commands were executed.")
        return 0
    if not args.skip_runner:
        run_command(manifest["runner_command"])
    if not args.skip_gate:
        run_command(manifest["gate_command"])
    if not args.skip_synthetic:
        run_command(manifest["synthetic_command"])
    if not args.skip_audit:
        run_command(manifest["audit_command"])
    if not args.skip_result_card:
        run_command(manifest["result_card_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
