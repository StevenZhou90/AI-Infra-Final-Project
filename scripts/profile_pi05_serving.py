#!/usr/bin/env python
"""Profile a short PI0.5 serving run with PyTorch profiler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_pi05_real_serving_runtime import run_serving_smoke  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile PI0.5 serving runtime.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pi05_profile"))
    parser.add_argument("--robots", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--request-period-ms", type=float, default=1000.0)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--with-stack", action="store_true")
    return parser.parse_args()


def serving_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        policy="lerobot/pi05_libero_finetuned_v044",
        task="libero_object",
        task_id=0,
        control_mode="relative",
        robots=args.robots,
        steps=args.steps,
        soak_seconds=None,
        warmup=args.warmup,
        control_period_ms=20.0,
        request_period_ms=args.request_period_ms,
        deadline_ms=args.deadline_ms,
        max_batch_size=8,
        max_batch_delay_ms=5.0,
        deadline_slack_ms=8.0,
        num_inference_steps=args.num_inference_steps,
        max_active_sessions=None,
        max_admission_utilization=None,
        estimated_base_ms=160.0,
        estimated_per_request_ms=35.0,
        action_buffer_mode=False,
        action_horizon=50,
        buffer_low_watermark=5,
        seed=123,
        device=args.device,
        dtype="bfloat16",
        amp_dtype="bfloat16",
        stagger_arrivals=True,
        distinct_observations=False,
        libero_config_path=None,
        output=args.output_dir / "summary.json",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
    ) as prof:
        summary = run_serving_smoke(serving_args(args))
    trace_path = args.output_dir / "trace.json"
    table_path = args.output_dir / "key_averages.txt"
    prof.export_chrome_trace(str(trace_path))
    table_path.write_text(prof.key_averages().table(sort_by="cuda_time_total", row_limit=80) + "\n")
    print(json.dumps({"summary": summary, "trace": str(trace_path), "table": str(table_path)}, indent=2))


if __name__ == "__main__":
    main()
