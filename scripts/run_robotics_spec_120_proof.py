#!/usr/bin/env python3
"""Launch the canonical 120-task robotics speculative decoding proof path.

This script is intentionally thin: it builds one strict command around the
existing PI0-FAST/PI0.5 or OpenVLA wrappers and records the command in a
manifest. The expensive simulator work and pass/fail logic remain in the
underlying gate runners.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


PI0_PATTERN_PATHS = {"pi0fast-pattern", "pi05-pattern"}
PI0_ADAPTIVE_PATHS = {"pi0fast-adaptive"}
PI0_CONSTRAINED_PATHS = {"pi0fast-constrained"}
PI0_PATHS = {"pi0fast-target-eos", *PI0_ADAPTIVE_PATHS, *PI0_CONSTRAINED_PATHS, *PI0_PATTERN_PATHS}
OPENVLA_PATHS = {"openvla"}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def planned_unique_tasks(*, suites: str, task_ids: str) -> int:
    return len(parse_csv(suites)) * len(parse_csv(task_ids))


def proof_subdir(path: str) -> str:
    return path.replace("-", "_")


def append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _pattern_task_count_thresholds(args: argparse.Namespace, *, min_unique_tasks: int) -> tuple[int, int]:
    task_count = args.pattern_min_task_count
    if task_count is None:
        task_count = min_unique_tasks
    heldout_task_count = args.pattern_min_heldout_task_count
    if heldout_task_count is None:
        heldout_task_count = max(1, math.ceil(min_unique_tasks * args.pattern_heldout_fraction))
    return int(task_count), int(heldout_task_count)


def _row_has_heldout_evidence(row: dict[str, Any], *, min_heldout_task_count: int) -> bool:
    required_metrics = (
        "heldout_modeled_speedup",
        "heldout_target_forward_reduction",
        "heldout_min_task_target_forward_reduction",
        "heldout_min_task_acceptance_rate",
    )
    if any(metric not in row for metric in required_metrics):
        return False
    return int(row.get("heldout_task_count", 0)) >= int(min_heldout_task_count)


def inspect_pattern_sweep_json(
    path: Path,
    *,
    rank: int,
    min_heldout_task_count: int,
    dry_run: bool,
    require_disjoint_heldout_tasks: bool,
    required_heldout_suites: list[str] | None,
    require_heldout_suite_coverage: bool,
) -> dict[str, Any]:
    """Precheck heldout evidence before a pattern proof command is launched."""

    details: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "requested_rank": int(rank),
        "min_heldout_task_count": int(min_heldout_task_count),
        "require_disjoint_heldout_tasks": bool(require_disjoint_heldout_tasks),
        "required_heldout_suites": list(required_heldout_suites or []),
        "require_heldout_suite_coverage": bool(require_heldout_suite_coverage),
    }
    if not path.exists():
        if dry_run:
            details["status"] = "not_inspected_missing_dry_run"
            return details
        raise ValueError(f"--pattern-sweep-json does not exist: {path}")

    summary = json.loads(path.read_text())
    rows = summary.get("top")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not contain a non-empty top sweep list")

    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    heldout_split = str(selection.get("heldout_split", "")).strip().lower()
    heldout_trace_indices = selection.get("heldout_trace_indices")
    if heldout_split in {"", "none", "null"} or not isinstance(heldout_trace_indices, list) or not heldout_trace_indices:
        raise ValueError(
            "--pattern-sweep-json must come from a non-empty heldout split; "
            "rerun sweep_pi0fast_pattern_offline.py with --heldout-split task."
        )
    overlap = selection.get("heldout_task_overlap")
    if bool(require_disjoint_heldout_tasks):
        disjoint = selection.get("heldout_task_disjoint")
        if not isinstance(overlap, list) or not isinstance(disjoint, bool):
            raise ValueError(
                "--pattern-sweep-json is missing task-disjoint heldout metadata; rerun "
                "sweep_pi0fast_pattern_offline.py so selection.heldout_task_disjoint is recorded."
            )
        if disjoint is not True or overlap:
            raise ValueError(
                "--pattern-sweep-json heldout tasks overlap ranking tasks: "
                + ", ".join(str(item) for item in overlap)
                + ". Rerun sweep_pi0fast_pattern_offline.py with --heldout-split task."
            )
    required_suites = [suite for suite in (required_heldout_suites or []) if suite]
    if bool(require_heldout_suite_coverage) and required_suites:
        heldout_suite_keys = selection.get("heldout_suite_keys")
        if not isinstance(heldout_suite_keys, list):
            raise ValueError(
                "--pattern-sweep-json is missing heldout suite metadata; rerun "
                "sweep_pi0fast_pattern_offline.py so selection.heldout_suite_keys is recorded."
            )
        heldout_suite_set = {str(item) for item in heldout_suite_keys}
        missing_suites = sorted(set(required_suites) - heldout_suite_set)
        if missing_suites:
            raise ValueError(
                "--pattern-sweep-json heldout split does not cover requested suites: "
                + ", ".join(missing_suites)
                + ". Rerun sweep_pi0fast_pattern_offline.py with traces from every requested suite."
            )

    if rank == 0:
        inspected_rows = [
            (idx, row)
            for idx, row in enumerate(rows, start=1)
            if isinstance(row, dict)
        ]
    elif rank > 0 and rank <= len(rows) and isinstance(rows[rank - 1], dict):
        inspected_rows = [(rank, rows[rank - 1])]
    else:
        raise ValueError(f"--pattern-sweep-rank {rank} exceeds available sweep rows ({len(rows)})")

    eligible = [
        idx
        for idx, row in inspected_rows
        if _row_has_heldout_evidence(row, min_heldout_task_count=min_heldout_task_count)
    ]
    if not eligible:
        raise ValueError(
            "--pattern-sweep-json selected rows do not contain enough heldout evidence: "
            f"required heldout_task_count>={int(min_heldout_task_count)} and heldout speed/acceptance metrics"
        )

    details.update(
        {
            "status": "ok",
            "heldout_split": heldout_split,
            "heldout_trace_count": len(heldout_trace_indices),
            "heldout_task_disjoint": bool(selection.get("heldout_task_disjoint", False)),
            "heldout_task_overlap_count": int(selection.get("heldout_task_overlap_count", 0)),
            "heldout_suite_count": int(selection.get("heldout_suite_count", 0)),
            "heldout_suite_keys": selection.get("heldout_suite_keys", []),
            "eligible_ranks": eligible,
        }
    )
    return details


def _append_common_pi0_thresholds(cmd: list[str], args: argparse.Namespace, *, min_unique_tasks: int) -> None:
    cmd.extend(
        [
            "--suites",
            args.suites,
            "--task-ids",
            args.task_ids,
            "--episodes",
            str(args.episodes),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--min-pairs",
            str(args.min_pairs),
            "--min-unique-tasks",
            str(min_unique_tasks),
            "--min-speedup",
            str(args.min_speedup),
            "--min-baseline-successes",
            str(args.min_baseline_successes),
            "--min-suite-matched-pairs",
            str(args.min_suite_matched_pairs),
            "--min-suite-baseline-successes",
            str(args.min_suite_baseline_successes),
            "--min-suite-speedup",
            str(args.min_suite_speedup),
            "--max-success-drop",
            str(args.max_success_drop),
            "--max-baseline-success-regressions",
            str(args.max_baseline_success_regressions),
            "--min-validation-episodes",
            str(args.min_validation_episodes),
            "--min-exact-verifies",
            str(args.min_exact_verifies),
            "--max-action-diff",
            str(args.max_action_diff),
        ]
    )
    append_flag(cmd, "--run-preflight", args.run_preflight)
    append_flag(cmd, "--require-hf-token", args.require_hf_token)
    append_flag(cmd, "--run-synthetic", args.run_synthetic)
    append_flag(cmd, "--run-final-audit", args.run_final_audit)
    append_flag(cmd, "--render-result-card", args.render_result_card)
    append_flag(cmd, "--skip-existing", args.skip_existing)
    if args.dry_run:
        cmd.append("--dry-run")


def _append_pattern_sweep_args(cmd: list[str], args: argparse.Namespace, *, min_unique_tasks: int) -> None:
    if args.pattern_sweep_json is None:
        if not args.allow_manual_pattern_args:
            raise ValueError(
                "pattern proof paths require --pattern-sweep-json by default so the final audit has heldout "
                "selection evidence; pass --allow-manual-pattern-args only for research dry runs."
            )
        if not args.dry_run:
            raise ValueError("--allow-manual-pattern-args is only permitted with --dry-run.")
    if args.pattern_sweep_json is not None:
        task_count, heldout_task_count = _pattern_task_count_thresholds(
            args,
            min_unique_tasks=min_unique_tasks,
        )
        cmd.extend(
            [
                "--pattern-sweep-json",
                str(args.pattern_sweep_json),
                "--pattern-sweep-rank",
                str(args.pattern_sweep_rank),
                "--pattern-min-modeled-speedup",
                str(args.pattern_min_modeled_speedup),
                "--pattern-min-forward-reduction",
                str(args.pattern_min_forward_reduction),
                "--pattern-min-task-forward-reduction",
                str(args.pattern_min_task_forward_reduction),
                "--pattern-min-task-acceptance-rate",
                str(args.pattern_min_task_acceptance_rate),
                "--pattern-min-task-count",
                str(task_count),
                "--pattern-min-heldout-modeled-speedup",
                str(args.pattern_min_heldout_modeled_speedup),
                "--pattern-min-heldout-forward-reduction",
                str(args.pattern_min_heldout_forward_reduction),
                "--pattern-min-heldout-task-forward-reduction",
                str(args.pattern_min_heldout_task_forward_reduction),
                "--pattern-min-heldout-task-acceptance-rate",
                str(args.pattern_min_heldout_task_acceptance_rate),
                "--pattern-min-heldout-task-count",
                str(heldout_task_count),
            ]
        )
        append_flag(cmd, "--pattern-auto-source-min-metrics", args.pattern_auto_source_min_metrics)
        for threshold in args.pattern_min_metric:
            cmd.extend(["--pattern-min-metric", threshold])
        for threshold in args.pattern_min_heldout_metric:
            cmd.extend(["--pattern-min-heldout-metric", threshold])
        for source in args.pattern_required_source_coverage:
            for item in parse_csv(source):
                cmd.extend(["--pattern-required-source-coverage", item])


def build_pi0_command(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    min_unique_tasks = args.min_unique_tasks
    if min_unique_tasks is None:
        min_unique_tasks = planned_unique_tasks(suites=args.suites, task_ids=args.task_ids)
    if args.path == "pi05-pattern":
        raise ValueError(
            "pi05-pattern is not currently runnable: LeRobot PI0.5 uses flow-action sampling "
            "and does not expose the PI0-FAST FAST-token hooks required for target_eos/pattern_sd. "
            "Add a PI0.5-specific stop-token/token-adapter path before using this proof path."
        )
    root = args.root / proof_subdir(args.path)
    cmd = [
        args.python,
        args.pi0_runner_script,
        "--root",
        str(root),
    ]
    if args.path == "pi0fast-target-eos":
        cmd.extend(["--speed-modes", "baseline,target_eos", "--candidate-mode", "target_eos"])
    elif args.path in PI0_ADAPTIVE_PATHS:
        cmd.extend(
            [
                "--speed-modes",
                "baseline,target_eos,target_eos_adaptive",
                "--candidate-mode",
                "target_eos_adaptive",
                "--reference-mode",
                "target_eos",
            ]
        )
    elif args.path in PI0_CONSTRAINED_PATHS:
        cmd.extend(
            [
                "--speed-modes",
                "baseline,target_eos,target_eos_constrained_noforce",
                "--candidate-mode",
                "target_eos_constrained_noforce",
                "--reference-mode",
                "target_eos",
            ]
        )
    else:
        cmd.extend(
            [
                "--speed-modes",
                "baseline,target_eos,pattern_sd_direct",
                "--candidate-mode",
                "pattern_sd_direct",
                "--reference-mode",
                "target_eos",
                "--min-reference-speedup",
                str(args.min_speedup),
                "--max-reference-success-drop",
                str(args.max_success_drop),
                "--max-reference-success-regressions",
                str(args.max_baseline_success_regressions),
            ]
        )
    _append_common_pi0_thresholds(cmd, args, min_unique_tasks=min_unique_tasks)
    if args.path in PI0_PATTERN_PATHS:
        pattern_sweep_precheck = None
        if args.pattern_sweep_json is not None:
            _task_count, heldout_task_count = _pattern_task_count_thresholds(
                args,
                min_unique_tasks=min_unique_tasks,
            )
            pattern_sweep_precheck = inspect_pattern_sweep_json(
                args.pattern_sweep_json,
                rank=args.pattern_sweep_rank,
                min_heldout_task_count=heldout_task_count,
                dry_run=args.dry_run,
                require_disjoint_heldout_tasks=args.pattern_require_disjoint_heldout_tasks,
                required_heldout_suites=parse_csv(args.suites),
                require_heldout_suite_coverage=args.pattern_require_heldout_suite_coverage,
            )
        _append_pattern_sweep_args(cmd, args, min_unique_tasks=min_unique_tasks)
    else:
        pattern_sweep_precheck = None

    extra_args: list[str] = []
    if args.path in PI0_CONSTRAINED_PATHS:
        extra_args.extend(
            [
                "--target-eos-constrained-full-head-margin",
                "1.0",
                "--target-eos-constrained-no-force-prefix",
            ]
        )
    if args.path in PI0_ADAPTIVE_PATHS:
        extra_args.extend(
            [
                "--adaptive-prefix-checkpoints",
                "32,64,96,128,160,192,224",
                "--adaptive-stable-checks",
                "3",
                "--adaptive-stable-tolerance",
                "0.0",
            ]
        )
    if args.path == "pi05-pattern":
        extra_args.extend(["--policy-kind", "pi05"])
        if args.pi05_policy is not None:
            extra_args.extend(["--policy", args.pi05_policy])
        if args.pi05_num_inference_steps is not None:
            extra_args.extend(["--num-inference-steps", str(args.pi05_num_inference_steps)])
    extra_args.extend(args.extra_args)
    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)
    return cmd, {
        "proof_root": str(root),
        "runner": "pi0fast",
        "min_unique_tasks": min_unique_tasks,
        "early_stop_reference": "target_eos"
        if args.path in PI0_PATTERN_PATHS | PI0_ADAPTIVE_PATHS | PI0_CONSTRAINED_PATHS
        else None,
        "expected_policy_kind": "pi05" if args.path == "pi05-pattern" else "pi0fast",
        "pattern_sweep_precheck": pattern_sweep_precheck,
    }


def build_openvla_command(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    cmd = [
        args.python,
        args.openvla_runner_script,
        "--config",
        str(args.openvla_config),
        "--run-id",
        args.openvla_run_id,
        "--mode",
        args.openvla_mode,
        "--min-pairs",
        str(args.min_pairs),
        "--min-speedup",
        str(args.min_speedup),
        "--min-baseline-successes",
        str(args.min_baseline_successes),
        "--min-suite-matched-pairs",
        str(args.min_suite_matched_pairs),
        "--min-suite-baseline-successes",
        str(args.min_suite_baseline_successes),
        "--min-suite-speedup",
        str(args.min_suite_speedup),
        "--max-success-drop",
        str(args.max_success_drop),
        "--max-baseline-success-regressions",
        str(args.max_baseline_success_regressions),
        "--synthetic-tasks",
        str(args.synthetic_tasks),
        "--synthetic-min-speedup",
        str(args.synthetic_min_speedup),
        "--synthetic-max-accuracy-drop",
        str(args.synthetic_max_accuracy_drop),
        "--max-unverified-action-shortcuts",
        "0",
        "--max-fast-draft-calls",
        "0",
        "--max-chunk-buffer-hits",
        "0",
        "--max-relaxed-group-accepts",
        "0",
        "--max-tree-depth-used",
        "1",
    ]
    if args.min_unique_tasks is not None:
        cmd.extend(["--min-unique-tasks", str(args.min_unique_tasks)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd, {
        "proof_root": None,
        "runner": "openvla",
        "strict_spec_stats": True,
        "expected_policy_kind": None,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.path in PI0_PATHS:
        command, details = build_pi0_command(args)
    elif args.path in OPENVLA_PATHS:
        command, details = build_openvla_command(args)
    else:
        raise ValueError(f"unknown proof path: {args.path}")
    return {
        "path": args.path,
        "command": command,
        "dry_run": args.dry_run,
        "thresholds": {
            "min_pairs": args.min_pairs,
            "min_speedup": args.min_speedup,
            "max_success_drop": args.max_success_drop,
            "max_baseline_success_regressions": args.max_baseline_success_regressions,
            "min_validation_episodes": args.min_validation_episodes if args.path in PI0_PATHS else None,
            "synthetic_tasks": args.synthetic_tasks,
            "synthetic_min_speedup": args.synthetic_min_speedup,
            "synthetic_max_accuracy_drop": args.synthetic_max_accuracy_drop,
        },
        **details,
    }


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strict 120-task robotics spec-decoding proof wrapper.")
    parser.add_argument(
        "--path",
        choices=sorted(PI0_PATHS | OPENVLA_PATHS),
        default="pi0fast-target-eos",
        help="Proof path to launch. Pattern paths require a heldout sweep JSON unless explicitly overridden.",
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/robotics_spec_120_proof"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pi0-runner-script", default="scripts/run_pi0fast_100_eval_gate.py")
    parser.add_argument("--openvla-runner-script", default="scripts/run_openvla_120_eval_gate.py")
    parser.add_argument("--openvla-config", type=Path, default=Path("configs/libero_specvla_distributed.yaml"))
    parser.add_argument("--openvla-run-id", default="openvla_specvla_120")
    parser.add_argument("--openvla-mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--suites", default="libero_object,libero_spatial,libero_goal")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--min-pairs", type=int, default=120)
    parser.add_argument("--min-unique-tasks", type=int, default=None)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--min-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-matched-pairs", type=int, default=1)
    parser.add_argument("--min-suite-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-speedup", type=float, default=1.0)
    parser.add_argument("--max-success-drop", type=float, default=0.0)
    parser.add_argument("--max-baseline-success-regressions", type=int, default=0)
    parser.add_argument("--min-validation-episodes", type=int, default=120)
    parser.add_argument("--min-exact-verifies", type=int, default=1)
    parser.add_argument("--max-action-diff", type=float, default=0.0)
    parser.add_argument("--synthetic-tasks", type=int, default=120)
    parser.add_argument("--synthetic-min-speedup", type=float, default=2.0)
    parser.add_argument("--synthetic-max-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--run-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-hf-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-synthetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-final-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-result-card", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pattern-sweep-json", type=Path, default=None)
    parser.add_argument("--allow-manual-pattern-args", action="store_true")
    parser.add_argument("--pattern-sweep-rank", type=int, default=0)
    parser.add_argument("--pattern-min-modeled-speedup", type=float, default=2.0)
    parser.add_argument("--pattern-min-forward-reduction", type=float, default=2.0)
    parser.add_argument("--pattern-min-task-forward-reduction", type=float, default=1.25)
    parser.add_argument("--pattern-min-task-acceptance-rate", type=float, default=0.30)
    parser.add_argument("--pattern-min-task-count", type=int, default=None)
    parser.add_argument("--pattern-min-heldout-modeled-speedup", type=float, default=1.25)
    parser.add_argument("--pattern-min-heldout-forward-reduction", type=float, default=1.25)
    parser.add_argument("--pattern-min-heldout-task-forward-reduction", type=float, default=1.10)
    parser.add_argument("--pattern-min-heldout-task-acceptance-rate", type=float, default=0.25)
    parser.add_argument("--pattern-min-heldout-task-count", type=int, default=None)
    parser.add_argument("--pattern-heldout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--pattern-require-disjoint-heldout-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require pattern sweep artifacts to prove heldout tasks are disjoint from ranking tasks.",
    )
    parser.add_argument(
        "--pattern-require-heldout-suite-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require pattern sweep heldout traces to cover every suite requested by the proof wrapper.",
    )
    parser.add_argument("--pattern-min-metric", action="append", default=[])
    parser.add_argument("--pattern-min-heldout-metric", action="append", default=[])
    parser.add_argument("--pattern-auto-source-min-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pattern-required-source-coverage", action="append", default=[])
    parser.add_argument("--pi05-policy", default=None)
    parser.add_argument("--pi05-num-inference-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    return args


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / f"{proof_subdir(args.path)}_proof_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote proof manifest: {manifest_path}")
    run_command(manifest["command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
