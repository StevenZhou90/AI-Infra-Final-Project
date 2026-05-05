#!/usr/bin/env python3
"""Run the README benchmark matrix for baseline vs adaptive fast trajectory policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


TASKS = {
    "vertical": "google_robot_pick_vertical_coke_can",
    "horizontal": "google_robot_pick_horizontal_coke_can",
    "standing": "google_robot_pick_standing_coke_can",
}
DEFAULT_MIN_CONFIDENT = {
    "vertical": 5,
    "horizontal": 6,
    "standing": 6,
}
XS = [-0.3500, -0.2925, -0.2350]
Y = -0.02


def parse_result(text: str) -> dict:
    result: dict = {}
    for line in text.splitlines():
        if " avg:" in line and "ms/step" in line and "success=" in line:
            label = line.split(" avg:", 1)[0].split()[-1]
            result["decoder_label"] = label
            result["avg_ms"] = float(line.split(" avg:", 1)[1].split("ms/step", 1)[0].strip())
            result["successes"] = int(line.split("Success:", 1)[1].split("/", 1)[0].strip()) if "Success:" in line else None
            result["success"] = "success=True" in line
            if "reward=" in line:
                result["reward"] = float(line.split("reward=", 1)[1].split()[0])
            if "steps=" in line:
                result["steps"] = int(line.split("steps=", 1)[1].split()[0])
        elif "  Success:" in line and "/" in line:
            result["successes"] = int(line.split("Success:", 1)[1].split("/", 1)[0].strip())
            result["episodes"] = int(line.split("/", 1)[1].split()[0].strip())
        elif "  Baseline:" in line and "ms/step" in line:
            result["avg_ms"] = float(line.split("Baseline:", 1)[1].split("ms/step", 1)[0].strip())
            result["decoder_label"] = "baseline"
    return result


def run(cmd: list[str], log_path: Path) -> tuple[dict, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return parse_result(proc.stdout + proc.stderr), elapsed


def summarize(rows: list[dict]) -> dict:
    out = {"total_runs": len(rows), "by_decoder": {}}
    for decoder in ("baseline", "trajectory-spec"):
        ds = [r for r in rows if r["decoder"] == decoder]
        if not ds:
            continue
        successes = sum(int(r.get("successes", 0)) for r in ds)
        episodes = sum(int(r.get("episodes", 0) or r.get("requested_episodes", 0)) for r in ds)
        avg_ms = sum(float(r.get("avg_ms", 0.0)) * int(r.get("episodes", 0) or r.get("requested_episodes", 0)) for r in ds)
        out["by_decoder"][decoder] = {
            "runs": len(ds),
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / max(episodes, 1),
            "avg_ms": avg_ms / max(episodes, 1),
        }
    if "baseline" in out["by_decoder"] and "trajectory-spec" in out["by_decoder"]:
        baseline_ms = out["by_decoder"]["baseline"]["avg_ms"]
        spec_ms = out["by_decoder"]["trajectory-spec"]["avg_ms"]
        out["speedup"] = baseline_ms / spec_ms if spec_ms else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vertical-gate", type=int, default=DEFAULT_MIN_CONFIDENT["vertical"])
    parser.add_argument("--horizontal-gate", type=int, default=DEFAULT_MIN_CONFIDENT["horizontal"])
    parser.add_argument("--standing-gate", type=int, default=DEFAULT_MIN_CONFIDENT["standing"])
    parser.add_argument("--head-threshold", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--decoder", choices=["both", "baseline", "trajectory-spec"], default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    rows: list[dict] = []
    min_confident = {
        "vertical": args.vertical_gate,
        "horizontal": args.horizontal_gate,
        "standing": args.standing_gate,
    }

    for orientation, task in TASKS.items():
        for x in XS:
            common = [
                sys.executable,
                "scripts/run_openvla_sim.py",
                "--task",
                task,
                "--published-eval-setup",
                "--obj-init-x",
                str(x),
                "--obj-init-y",
                str(Y),
                "--episodes",
                str(args.episodes),
                "--steps",
                str(args.steps),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
            ]

            decoders = ("baseline", "trajectory-spec") if args.decoder == "both" else (args.decoder,)
            for decoder in decoders:
                run_dir = output_dir / decoder / f"{orientation}_x{x:.4f}_y{Y:.2f}"
                cmd = [*common, "--decoder", decoder, "--output_dir", str(run_dir)]
                if decoder == "trajectory-spec":
                    cmd.extend(
                        [
                            "--trajectory-head-checkpoint",
                            args.checkpoint,
                            "--trajectory-fast-draft-only",
                            "--trajectory-head-threshold",
                            str(args.head_threshold),
                            "--trajectory-fast-min-confident-tokens",
                            str(min_confident[orientation]),
                        ]
                    )
                print(f"=== {decoder} {orientation} x={x:.4f} ===", flush=True)
                result, wall_s = run(cmd, run_dir / "stdout.log")
                row = {
                    "decoder": decoder,
                    "orientation": orientation,
                    "task": task,
                    "obj_init_x": x,
                    "obj_init_y": Y,
                    "requested_episodes": args.episodes,
                    "head_threshold": args.head_threshold if decoder == "trajectory-spec" else None,
                    "fast_min_confident_tokens": min_confident[orientation] if decoder == "trajectory-spec" else None,
                    "steps": args.steps,
                    "wall_s": wall_s,
                    "output_dir": str(run_dir),
                    **result,
                }
                rows.append(row)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)

    summary = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
