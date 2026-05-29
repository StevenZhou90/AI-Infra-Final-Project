#!/usr/bin/env python
"""Synthetic PI0-FAST multi-robot serving benchmark.

This does not load PI0-FAST.  It evaluates the system-level runtime policy with
latency parameters calibrated from PI0-FAST experiments, so we can quickly scan
continuous-batching and prefix-gating regimes before running expensive LIBERO
rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_serving_runtime import (
    NS_PER_MS,
    PI0FastRequest,
    PI0FastServingConfig,
    PI0FastServingRuntime,
    SyntheticPI0FastBackend,
    deadline_ns_from_period,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark synthetic PI0-FAST serving runtime.")
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--control-period-ms", type=float, default=200.0)
    parser.add_argument("--mode", choices=["full_eos", "prefix_gate", "cutoff16"], default="prefix_gate")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-delay-ms", type=float, default=5.0)
    parser.add_argument("--deadline-slack-ms", type=float, default=8.0)
    parser.add_argument("--prefill-ms", type=float, default=70.0)
    parser.add_argument("--decode-ms-per-token", type=float, default=2.0)
    parser.add_argument("--batch-efficiency", type=float, default=0.72)
    parser.add_argument("--baseline-tokens", type=int, default=96)
    parser.add_argument("--prefix-gate-tokens", type=int, default=28)
    parser.add_argument("--cutoff16-tokens", type=int, default=16)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/pi0fast_serving_runtime/synthetic_summary.json"))
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def simulate(args: argparse.Namespace) -> dict:
    config = PI0FastServingConfig(
        max_batch_size=args.max_batch_size,
        max_batch_delay_ms=args.max_batch_delay_ms,
        deadline_slack_ms=args.deadline_slack_ms,
        estimated_prefill_ms=args.prefill_ms,
        estimated_decode_ms_per_token=args.decode_ms_per_token,
        default_control_period_ms=args.control_period_ms,
    )
    backend = SyntheticPI0FastBackend(
        baseline_tokens=args.baseline_tokens,
        prefix_gate_tokens=args.prefix_gate_tokens,
        cutoff16_tokens=args.cutoff16_tokens,
        prefill_ms=args.prefill_ms,
        decode_ms_per_token=args.decode_ms_per_token,
        batch_efficiency=args.batch_efficiency,
        sleep=False,
    )
    runtime = PI0FastServingRuntime(backend, config)

    current_ns = 0
    request_idx = 0
    rng = np.random.default_rng(123)

    for step in range(args.steps):
        for robot in range(args.robots):
            jitter_ns = int(rng.normal(0.0, args.jitter_ms) * NS_PER_MS) if args.jitter_ms else 0
            enqueued_ns = max(0, current_ns + jitter_ns)
            req = PI0FastRequest(
                request_id=f"r{robot}-s{step}",
                session_id=f"robot-{robot}",
                robot_id=f"robot-{robot}",
                enqueued_ns=enqueued_ns,
                deadline_ns=deadline_ns_from_period(enqueued_ns, args.control_period_ms),
                control_period_ms=args.control_period_ms,
                decode_mode=args.mode,
                prompt="pick up the object",
            )
            runtime.submit(req)
            request_idx += 1

        drain_ns = current_ns + int(args.max_batch_delay_ms * NS_PER_MS)
        runtime.drain_ready(at_ns=drain_ns, force=False)
        current_ns += int(args.control_period_ms * NS_PER_MS)

    runtime.drain_ready(at_ns=current_ns, force=True)

    latencies = [resp.telemetry.queue_ms + resp.telemetry.runtime_ms for resp in runtime.responses]
    runtimes = [resp.telemetry.runtime_ms for resp in runtime.responses]
    queues = [resp.telemetry.queue_ms for resp in runtime.responses]
    slacks = [resp.telemetry.deadline_slack_ms for resp in runtime.responses]
    batch_sizes = [resp.telemetry.batch_size for resp in runtime.responses]
    tokens = [resp.telemetry.action_tokens for resp in runtime.responses]

    summary = runtime.stats()
    summary.update(
        {
            "mode": args.mode,
            "robots": args.robots,
            "steps": args.steps,
            "control_period_ms": args.control_period_ms,
            "max_batch_size": args.max_batch_size,
            "max_batch_delay_ms": args.max_batch_delay_ms,
            "prefill_ms": args.prefill_ms,
            "decode_ms_per_token": args.decode_ms_per_token,
            "batch_efficiency": args.batch_efficiency,
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "p99_latency_ms": percentile(latencies, 99),
            "p50_runtime_ms": percentile(runtimes, 50),
            "p95_runtime_ms": percentile(runtimes, 95),
            "p50_queue_ms": percentile(queues, 50),
            "p95_queue_ms": percentile(queues, 95),
            "min_deadline_slack_ms": float(min(slacks)) if slacks else 0.0,
            "p05_deadline_slack_ms": percentile(slacks, 5),
            "avg_action_tokens": float(np.mean(tokens)) if tokens else 0.0,
            "max_observed_batch_size": int(max(batch_sizes)) if batch_sizes else 0,
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = simulate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
