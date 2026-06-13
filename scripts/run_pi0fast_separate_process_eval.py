#!/usr/bin/env python3
"""Run PI0-FAST eval modes in separate processes and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strs(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _load_metric(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one metric row in {path}, got {len(rows)}")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PI0-FAST task/mode evals in isolated subprocesses")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0,1,2,3,4")
    parser.add_argument("--modes", default="target_eos,target_eos_adaptive")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval-script", default="scripts/run_pi0fast_chunk_eval.py")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.episodes != 1:
        raise ValueError("This runner expects --episodes 1 so task/mode aggregation is unambiguous")

    task_ids = _parse_csv_ints(args.task_ids)
    modes = _parse_csv_strs(args.modes)
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    for task_id in task_ids:
        for mode in modes:
            run_dir = output_dir / f"task{task_id}_{mode}"
            cmd = [
                args.python,
                args.eval_script,
                "--task",
                args.task,
                "--task-id",
                str(task_id),
                "--episodes",
                str(args.episodes),
                "--steps",
                str(args.steps),
                "--modes",
                mode,
                "--seed",
                str(args.seed),
                "--output-dir",
                str(run_dir),
                *extra_args,
            ]
            env = dict(os.environ)
            env.setdefault("MUJOCO_GL", "osmesa")
            print(f"RUN task={task_id} mode={mode}", flush=True)
            subprocess.run(cmd, check=True, env=env)
            metrics.append(_load_metric(run_dir / "metrics.jsonl"))

    aggregate: dict[str, dict] = {}
    for row in metrics:
        aggregate.setdefault(row["mode"], {"rows": []})["rows"].append(row)

    summary = {"rows": metrics, "by_mode": {}}
    for mode, bucket in aggregate.items():
        rows = bucket["rows"]
        avg_ms = sum(row["avg_ms_per_control_step"] for row in rows) / len(rows)
        success_rate = sum(1 for row in rows if row["success"]) / len(rows)
        summary["by_mode"][mode] = {
            "episodes": len(rows),
            "successes": sum(1 for row in rows if row["success"]),
            "success_rate": success_rate,
            "avg_ms_per_control_step": avg_ms,
            "avg_fast_token_count": sum(row["chunk_stats"]["avg_fast_token_count"] for row in rows) / len(rows),
        }

    if len(modes) >= 2 and modes[0] in summary["by_mode"]:
        baseline = summary["by_mode"][modes[0]]
        baseline_ms = baseline["avg_ms_per_control_step"]
        baseline_success = baseline["success_rate"]
        for mode in modes[1:]:
            if mode in summary["by_mode"]:
                row = summary["by_mode"][mode]
                row["speedup_vs_baseline"] = baseline_ms / row["avg_ms_per_control_step"]
                row["success_drop_abs_vs_baseline"] = baseline_success - row["success_rate"]

    (output_dir / "separate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["by_mode"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
