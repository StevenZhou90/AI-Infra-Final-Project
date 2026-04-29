#!/usr/bin/env python3
"""Run published SimplerEnv/OpenVLA sweep benchmarks.

Examples:
    uv run python scripts/run_published_sweep.py --sweep mini --decoder baseline
    uv run python scripts/run_published_sweep.py --sweep mini --decoder both
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ORIENTATION_TO_TASK = {
    "horizontal": "google_robot_pick_horizontal_coke_can",
    "vertical": "google_robot_pick_vertical_coke_can",
    "standing": "google_robot_pick_standing_coke_can",
}

MINI_XS = [-0.35, -0.2925, -0.235, -0.1775, -0.12]
FULL_XS = MINI_XS
FULL_YS = [-0.02, 0.09, 0.20, 0.31, 0.42]


def sweep_points(sweep: str) -> list[tuple[str, float, float]]:
    if sweep == "mini":
        return [(orientation, x, -0.02) for orientation in ORIENTATION_TO_TASK for x in MINI_XS]
    if sweep == "full":
        return [
            (orientation, x, y)
            for orientation in ORIENTATION_TO_TASK
            for y in FULL_YS
            for x in FULL_XS
        ]
    raise ValueError(f"Unknown sweep: {sweep}")


def decoders(choice: str) -> list[str]:
    return ["baseline", "trajectory-spec"] if choice == "both" else [choice]


def parse_result(stdout: str) -> dict:
    result: dict = {}
    for line in stdout.splitlines():
        if " avg:" in line and "success=" in line and "reward=" in line and "steps=" in line:
            parts = line.split(" avg:", 1)
            result["decoder_label"] = parts[0].split()[-1]
            result["avg_ms"] = float(parts[1].split("ms/step", 1)[0].strip())
            result["success"] = "success=True" in line
            result["reward"] = float(line.split("reward=", 1)[1].split()[0])
            result["steps"] = int(line.split("steps=", 1)[1].split()[0])
        elif "Spec stats:" in line:
            stats_text = line.split("Spec stats:", 1)[1].strip()
            result["spec_stats_text"] = stats_text
    return result


def summarize(rows: list[dict]) -> dict:
    summary: dict = {"total_runs": len(rows), "by_decoder": {}}
    for decoder in sorted({r["decoder"] for r in rows}):
        ds = [r for r in rows if r["decoder"] == decoder]
        summary["by_decoder"][decoder] = {
            "runs": len(ds),
            "successes": sum(1 for r in ds if r.get("success")),
            "success_rate": sum(1 for r in ds if r.get("success")) / max(len(ds), 1),
            "avg_ms": sum(r.get("avg_ms", 0.0) for r in ds) / max(len(ds), 1),
            "avg_steps": sum(r.get("steps", 0) for r in ds) / max(len(ds), 1),
        }
        summary["by_decoder"][decoder]["by_orientation"] = {}
        for orientation in ORIENTATION_TO_TASK:
            os = [r for r in ds if r["orientation"] == orientation]
            if not os:
                continue
            summary["by_decoder"][decoder]["by_orientation"][orientation] = {
                "runs": len(os),
                "successes": sum(1 for r in os if r.get("success")),
                "success_rate": sum(1 for r in os if r.get("success")) / max(len(os), 1),
                "avg_ms": sum(r.get("avg_ms", 0.0) for r in os) / max(len(os), 1),
                "avg_steps": sum(r.get("steps", 0) for r in os) / max(len(os), 1),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run published SimplerEnv OpenVLA sweep")
    parser.add_argument("--sweep", choices=["mini", "full"], default="mini")
    parser.add_argument("--decoder", choices=["baseline", "trajectory-spec", "both"], default="baseline")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectory-head-checkpoint", default=None)
    parser.add_argument("--trajectory-head-threshold", type=float, default=0.5)
    parser.add_argument("--trajectory-head-top-k", type=int, default=3)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"outputs/benchmarks/{args.sweep}_{args.decoder}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    rows: list[dict] = []
    for decoder in decoders(args.decoder):
        for idx, (orientation, x, y) in enumerate(sweep_points(args.sweep)):
            task = ORIENTATION_TO_TASK[orientation]
            run_dir = output_dir / decoder / f"{orientation}_{idx:02d}_x{x}_y{y}"
            cmd = [
                sys.executable,
                "scripts/run_openvla_sim.py",
                "--task",
                task,
                "--episodes",
                "1",
                "--steps",
                str(args.steps),
                "--published-eval-setup",
                "--obj-init-x",
                str(x),
                "--obj-init-y",
                str(y),
                "--decoder",
                decoder,
                "--output_dir",
                str(run_dir),
                "--device",
                args.device,
            ]
            if args.trajectory_head_checkpoint:
                cmd.extend(
                    [
                        "--trajectory-head-checkpoint",
                        args.trajectory_head_checkpoint,
                        "--trajectory-head-threshold",
                        str(args.trajectory_head_threshold),
                        "--trajectory-head-top-k",
                        str(args.trajectory_head_top_k),
                    ]
                )
            print(f"=== {decoder} {task} x={x} y={y} ===", flush=True)
            start = time.perf_counter()
            proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
            elapsed_s = time.perf_counter() - start
            (run_dir / "stdout.log").parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "stdout.log").write_text(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                raise SystemExit(proc.returncode)

            row = {
                "decoder": decoder,
                "orientation": orientation,
                "task": task,
                "obj_init_x": x,
                "obj_init_y": y,
                "wall_s": elapsed_s,
                "output_dir": str(run_dir),
            }
            row.update(parse_result(proc.stdout + proc.stderr))
            rows.append(row)
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"result success={row.get('success')} avg_ms={row.get('avg_ms')} steps={row.get('steps')}",
                flush=True,
            )

    summary = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
