#!/usr/bin/env python
"""Smoke-test real PI0.5 through the serving runtime.

This loads the public PI0.5 LIBERO policy, prepares LIBERO observations, and
executes deadline-aware serving batches with the real model backend.  It is the
expensive validation companion to ``benchmark_pi0fast_serving_runtime.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# PI0.5's first TorchDynamo/Inductor compile can take minutes and is not part of
# serving steady-state latency. Leave this overridable for explicit compile tests.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _ensure_libero_config,
    _import_lerobot,
    _prepare_observation,
)
from serving.pi0fast_serving_runtime import (  # noqa: E402
    NS_PER_MS,
    PI0FastRequest,
    PI0FastServingConfig,
    PI0FastServingRuntime,
    RealPIBatchBackend,
    deadline_ns_from_period,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real PI0.5 serving runtime.")
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--soak-seconds", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--control-period-ms", type=float, default=20.0)
    parser.add_argument("--request-period-ms", type=float, default=1000.0)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-delay-ms", type=float, default=5.0)
    parser.add_argument("--deadline-slack-ms", type=float, default=8.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--max-active-sessions", type=int, default=None)
    parser.add_argument("--estimated-base-ms", type=float, default=160.0)
    parser.add_argument("--estimated-per-request-ms", type=float, default=35.0)
    parser.add_argument("--action-buffer-mode", action="store_true")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--buffer-low-watermark", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "none"])
    parser.add_argument("--stagger-arrivals", action="store_true")
    parser.add_argument("--distinct-observations", action="store_true")
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pi05_real_serving_runtime/summary.json"),
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def prepare_robot_observations(
    *,
    env,
    env_preprocessor,
    policy_preprocessor,
    preprocess_observation,
    robots: int,
    seed: int,
    distinct: bool,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for robot in range(robots):
        robot_seed = seed + robot if distinct else seed
        observation, _info = env.reset(seed=[robot_seed])
        prepared.append(_prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation))
    return prepared


def effective_request_period_ms(args: argparse.Namespace) -> float:
    if not args.action_buffer_mode:
        return args.request_period_ms
    chunk_ms = max(args.action_horizon - args.buffer_low_watermark, 1) * args.control_period_ms
    return max(args.request_period_ms, chunk_ms)


def build_events(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    events: list[tuple[int, int, int]] = []
    request_period_ms = effective_request_period_ms(args)
    steps = args.steps
    if args.soak_seconds is not None:
        steps = max(1, int(np.ceil((args.soak_seconds * 1000.0) / request_period_ms)))
    for step in range(steps):
        for robot in range(args.robots):
            phase_ns = (
                int((robot / max(args.robots, 1)) * request_period_ms * NS_PER_MS)
                if args.stagger_arrivals
                else 0
            )
            enqueued_ns = int(step * request_period_ms * NS_PER_MS) + phase_ns
            events.append((enqueued_ns, robot, step))
    events.sort()
    return events


def run_serving_smoke(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, _PI0FastPolicy = (
        _import_lerobot()
    )
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    amp_dtype = None if args.amp_dtype == "none" else getattr(torch, args.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True

    policy = PI05Policy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    if hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = int(args.num_inference_steps)
    policy_preprocessor, _policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_cfg = LiberoEnv(task=args.task, task_ids=[args.task_id], control_mode=args.control_mode)
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = env_map[args.task][args.task_id]

    try:
        observations = prepare_robot_observations(
            env=env,
            env_preprocessor=env_preprocessor,
            policy_preprocessor=policy_preprocessor,
            preprocess_observation=preprocess_observation,
            robots=args.robots,
            seed=args.seed,
            distinct=args.distinct_observations,
        )

        backend = RealPIBatchBackend(
            policy,
            postprocessor=None,
            accelerator="real_pi05_batch",
            autocast_dtype=amp_dtype if device.type == "cuda" else None,
        )
        config = PI0FastServingConfig(
            max_batch_size=args.max_batch_size,
            max_batch_delay_ms=args.max_batch_delay_ms,
            deadline_slack_ms=args.deadline_slack_ms,
            estimated_batch_base_ms=args.estimated_base_ms,
            estimated_batch_per_request_ms=args.estimated_per_request_ms,
            default_control_period_ms=args.deadline_ms,
            default_decode_mode="flow",
            max_active_sessions=args.max_active_sessions,
        )
        runtime = PI0FastServingRuntime(backend, config)

        warmup_batch = observations[: max(1, min(args.max_batch_size, args.robots))]
        for idx in range(args.warmup):
            warmup_runtime = PI0FastServingRuntime(backend, config)
            for robot, observation in enumerate(warmup_batch):
                warmup_runtime.submit(
                    PI0FastRequest(
                        request_id=f"warmup-{idx}-{robot}",
                        session_id=f"warmup-{robot}",
                        robot_id=f"warmup-{robot}",
                        model_id=args.policy,
                        enqueued_ns=0,
                        deadline_ns=deadline_ns_from_period(0, args.deadline_ms),
                        control_period_ms=args.deadline_ms,
                        decode_mode="flow",
                        observation=observation,
                    )
                )
            warmup_runtime.drain_ready(at_ns=0, force=True)
        sync_if_cuda(device)

        service_available_ns = 0

        def drain_once(at_ns: int, *, force: bool = False) -> None:
            nonlocal service_available_ns
            start_ns = max(at_ns, service_available_ns)
            before = len(runtime.responses)
            runtime.drain_ready(at_ns=start_ns, force=force)
            if len(runtime.responses) > before:
                runtime_ms = runtime.responses[-1].telemetry.runtime_ms
                service_available_ns = start_ns + int(runtime_ms * NS_PER_MS)

        for enqueued_ns, robot, step in build_events(args):
            while runtime.scheduler.queue_depth() and service_available_ns <= enqueued_ns:
                before = len(runtime.responses)
                drain_once(service_available_ns, force=False)
                if len(runtime.responses) == before:
                    break
            admitted = runtime.try_submit(
                PI0FastRequest(
                    request_id=f"r{robot}-s{step}",
                    session_id=f"robot-{robot}",
                    robot_id=f"robot-{robot}",
                    model_id=args.policy,
                    enqueued_ns=enqueued_ns,
                    deadline_ns=deadline_ns_from_period(enqueued_ns, args.deadline_ms),
                    control_period_ms=args.deadline_ms,
                    decode_mode="flow",
                    observation=observations[robot],
                )
            )
            if admitted:
                drain_once(enqueued_ns + int(args.max_batch_delay_ms * NS_PER_MS), force=False)

        while runtime.scheduler.queue_depth():
            drain_once(service_available_ns, force=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    latencies = [resp.telemetry.queue_ms + resp.telemetry.runtime_ms for resp in runtime.responses]
    runtimes = [resp.telemetry.runtime_ms for resp in runtime.responses]
    queues = [resp.telemetry.queue_ms for resp in runtime.responses]
    slacks = [resp.telemetry.deadline_slack_ms for resp in runtime.responses]
    batch_sizes = [resp.telemetry.batch_size for resp in runtime.responses]
    summary = runtime.stats()
    summary.update(
        {
            "config": jsonable_config(args),
            "device": str(device),
            "dtype": args.dtype,
            "amp_dtype": None if args.amp_dtype == "none" else args.amp_dtype,
            "policy_num_inference_steps": getattr(policy.config, "num_inference_steps", None),
            "effective_request_period_ms": effective_request_period_ms(args),
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "p99_latency_ms": percentile(latencies, 99),
            "p50_runtime_ms": percentile(runtimes, 50),
            "p95_runtime_ms": percentile(runtimes, 95),
            "p50_queue_ms": percentile(queues, 50),
            "p95_queue_ms": percentile(queues, 95),
            "min_deadline_slack_ms": float(min(slacks)) if slacks else 0.0,
            "p05_deadline_slack_ms": percentile(slacks, 5),
            "max_observed_batch_size": int(max(batch_sizes)) if batch_sizes else 0,
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_serving_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
