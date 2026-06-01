#!/usr/bin/env python
"""Run PI0.5 real serving load sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_pi05_real_serving_runtime import run_serving_smoke  # noqa: E402


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load-test PI0.5 serving runtime.")
    parser.add_argument("--robots", default="1,4,8")
    parser.add_argument("--request-period-ms", default="1000,1500")
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--soak-seconds", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-delay-ms", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--max-active-sessions", type=int, default=None)
    parser.add_argument("--max-admission-utilization", type=float, default=None)
    parser.add_argument("--action-buffer-mode", action="store_true")
    parser.add_argument("--stagger-arrivals", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/pi05_real_serving_runtime/load_sweep.json"))
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough
    return args


def child_args(args: argparse.Namespace, robots: int, request_period_ms: float, output: Path) -> argparse.Namespace:
    raw: dict[str, Any] = {
        "policy": "lerobot/pi05_libero_finetuned_v044",
        "task": "libero_object",
        "task_id": 0,
        "control_mode": "relative",
        "robots": robots,
        "steps": args.steps,
        "soak_seconds": args.soak_seconds,
        "warmup": args.warmup,
        "control_period_ms": 20.0,
        "request_period_ms": request_period_ms,
        "deadline_ms": args.deadline_ms,
        "max_batch_size": args.max_batch_size,
        "max_batch_delay_ms": args.max_batch_delay_ms,
        "deadline_slack_ms": 8.0,
        "num_inference_steps": args.num_inference_steps,
        "max_active_sessions": args.max_active_sessions,
        "max_admission_utilization": args.max_admission_utilization,
        "estimated_base_ms": 160.0,
        "estimated_per_request_ms": 35.0,
        "action_buffer_mode": args.action_buffer_mode,
        "action_horizon": 50,
        "buffer_low_watermark": 5,
        "seed": 123,
        "device": "cuda",
        "dtype": "bfloat16",
        "amp_dtype": "bfloat16",
        "stagger_arrivals": args.stagger_arrivals,
        "distinct_observations": False,
        "libero_config_path": None,
        "output": output,
    }
    return argparse.Namespace(**raw)


def main() -> None:
    args = parse_args()
    rows = []
    for robots in parse_csv_ints(args.robots):
        for request_period_ms in parse_csv_floats(args.request_period_ms):
            output = args.output.parent / f"load_r{robots}_req{int(request_period_ms)}_d{int(args.deadline_ms)}.json"
            scenario = child_args(args, robots, request_period_ms, output)
            if args.dry_run:
                rows.append({"robots": robots, "request_period_ms": request_period_ms, "output": str(output)})
                continue
            rows.append(run_serving_smoke(scenario))

    summary = {"rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
