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
    SyntheticPI05Backend,
    SyntheticPI0FastBackend,
    deadline_ns_from_period,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark synthetic PI0-FAST serving runtime.")
    parser.add_argument("--backend", choices=["pi0fast", "pi05"], default="pi0fast")
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--control-period-ms", type=float, default=200.0)
    parser.add_argument(
        "--request-period-ms",
        type=float,
        default=None,
        help="Inter-arrival period for each robot's model requests. Defaults to --control-period-ms.",
    )
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=None,
        help="Per-request serving deadline. Defaults to --control-period-ms.",
    )
    parser.add_argument("--mode", choices=["full_eos", "prefix_gate", "cutoff16", "flow"], default="prefix_gate")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-delay-ms", type=float, default=5.0)
    parser.add_argument("--deadline-slack-ms", type=float, default=8.0)
    parser.add_argument("--prefill-ms", type=float, default=70.0)
    parser.add_argument("--decode-ms-per-token", type=float, default=2.0)
    parser.add_argument("--batch-efficiency", type=float, default=0.72)
    parser.add_argument("--pi05-base-ms", type=float, default=158.0)
    parser.add_argument("--pi05-per-request-ms", type=float, default=31.0)
    parser.add_argument("--pi05-num-inference-steps", type=int, default=4)
    parser.add_argument("--baseline-tokens", type=int, default=96)
    parser.add_argument("--prefix-gate-tokens", type=int, default=28)
    parser.add_argument("--cutoff16-tokens", type=int, default=16)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument(
        "--stagger-arrivals",
        action="store_true",
        help="Evenly phase robot request arrivals across the request period.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/pi0fast_serving_runtime/synthetic_summary.json"))
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def simulate(args: argparse.Namespace) -> dict:
    request_period_ms = args.request_period_ms if args.request_period_ms is not None else args.control_period_ms
    deadline_ms = args.deadline_ms if args.deadline_ms is not None else args.control_period_ms
    config = PI0FastServingConfig(
        max_batch_size=args.max_batch_size,
        max_batch_delay_ms=args.max_batch_delay_ms,
        deadline_slack_ms=args.deadline_slack_ms,
        estimated_prefill_ms=args.prefill_ms,
        estimated_decode_ms_per_token=args.decode_ms_per_token,
        default_control_period_ms=deadline_ms,
    )
    if args.backend == "pi05":
        backend = SyntheticPI05Backend(
            base_ms=args.pi05_base_ms,
            per_request_ms=args.pi05_per_request_ms,
            num_inference_steps=args.pi05_num_inference_steps,
            sleep=False,
        )
    else:
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

    events: list[tuple[int, int, int]] = []
    rng = np.random.default_rng(123)

    for step in range(args.steps):
        for robot in range(args.robots):
            phase_ns = int((robot / max(args.robots, 1)) * request_period_ms * NS_PER_MS) if args.stagger_arrivals else 0
            jitter_ns = int(rng.normal(0.0, args.jitter_ms) * NS_PER_MS) if args.jitter_ms else 0
            enqueued_ns = max(0, int(step * request_period_ms * NS_PER_MS) + phase_ns + jitter_ns)
            events.append((enqueued_ns, robot, step))
    events.sort()

    service_available_ns = 0

    def drain_once(at_ns: int, *, force: bool = False) -> None:
        nonlocal service_available_ns
        start_ns = max(at_ns, service_available_ns)
        before = len(runtime.responses)
        runtime.drain_ready(at_ns=start_ns, force=force)
        if len(runtime.responses) > before:
            runtime_ms = runtime.responses[-1].telemetry.runtime_ms
            service_available_ns = start_ns + int(runtime_ms * NS_PER_MS)

    for enqueued_ns, robot, step in events:
        while runtime.scheduler.queue_depth() and service_available_ns <= enqueued_ns:
            before = len(runtime.responses)
            drain_once(service_available_ns, force=False)
            if len(runtime.responses) == before:
                break
        req = PI0FastRequest(
            request_id=f"r{robot}-s{step}",
            session_id=f"robot-{robot}",
            robot_id=f"robot-{robot}",
            enqueued_ns=enqueued_ns,
            deadline_ns=deadline_ns_from_period(enqueued_ns, deadline_ms),
            control_period_ms=deadline_ms,
            decode_mode=args.mode,
            model_id="lerobot/pi05_libero_finetuned_v044" if args.backend == "pi05" else "lerobot/pi0fast-libero",
            prompt="pick up the object",
        )
        runtime.submit(req)
        drain_once(enqueued_ns + int(args.max_batch_delay_ms * NS_PER_MS), force=False)

    while runtime.scheduler.queue_depth():
        drain_once(service_available_ns, force=True)

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
            "backend": args.backend,
            "robots": args.robots,
            "steps": args.steps,
            "control_period_ms": args.control_period_ms,
            "request_period_ms": request_period_ms,
            "deadline_ms": deadline_ms,
            "stagger_arrivals": args.stagger_arrivals,
            "max_batch_size": args.max_batch_size,
            "max_batch_delay_ms": args.max_batch_delay_ms,
            "prefill_ms": args.prefill_ms,
            "decode_ms_per_token": args.decode_ms_per_token,
            "batch_efficiency": args.batch_efficiency,
            "pi05_base_ms": args.pi05_base_ms,
            "pi05_per_request_ms": args.pi05_per_request_ms,
            "pi05_num_inference_steps": args.pi05_num_inference_steps,
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
