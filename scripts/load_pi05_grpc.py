#!/usr/bin/env python
"""Multi-robot gRPC load test for the PI0.5 server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import grpc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proto import inference_pb2, inference_pb2_grpc  # noqa: E402
from scripts.run_pi0fast_chunk_eval import _ensure_libero_config, _import_lerobot, _prepare_observation  # noqa: E402
from serving.pi05_grpc_codec import encode_prepared_observation  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI0.5 gRPC load test")
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--request-period-ms", type=float, default=1000.0)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--stagger-arrivals", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    parser.add_argument("--save-warmup-observation", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/pi05_grpc_load/summary.json"))
    return parser.parse_args()


def prepare_observation_payloads(args: argparse.Namespace) -> list[bytes]:
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, _PI0FastPolicy = (
        _import_lerobot()
    )
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import torch

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    policy = PI05Policy.from_pretrained(args.policy).to(device=device, dtype=torch.bfloat16).eval()
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
        payloads: list[bytes] = []
        for robot in range(args.robots):
            observation, _info = env.reset(seed=[args.seed + robot])
            prepared = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
            payloads.append(encode_prepared_observation(prepared))
        return payloads
    finally:
        try:
            env.close()
        except Exception:
            pass


def worker(
    *,
    args: argparse.Namespace,
    robot_idx: int,
    payload: bytes,
    start_at: float,
    rows: list[dict[str, Any]],
    lock: threading.Lock,
) -> None:
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    for warmup_idx in range(args.warmup_requests):
        stub.Predict(
            inference_pb2.PredictRequest(
                request_id=f"robot-{robot_idx}-warmup-{warmup_idx}",
                model_id=args.policy,
                robot_id=f"robot-{robot_idx}",
                session_id=f"robot-{robot_idx}",
                timestamp_ns=time.time_ns(),
                deadline_ms=args.deadline_ms,
                request_period_ms=args.request_period_ms,
                prepared_observation=payload,
                observation_format="torch",
            )
        )
    period_s = args.request_period_ms / 1000.0
    phase_s = (robot_idx / max(args.robots, 1)) * period_s if args.stagger_arrivals else 0.0
    end_at = start_at + args.duration_seconds
    step = 0
    next_at = start_at + phase_s
    while next_at < end_at:
        sleep_s = next_at - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        request_id = f"robot-{robot_idx}-s{step}"
        wall_start = time.perf_counter()
        response = stub.Predict(
            inference_pb2.PredictRequest(
                request_id=request_id,
                model_id=args.policy,
                robot_id=f"robot-{robot_idx}",
                session_id=f"robot-{robot_idx}",
                timestamp_ns=time.time_ns(),
                deadline_ms=args.deadline_ms,
                request_period_ms=args.request_period_ms,
                prepared_observation=payload,
                observation_format="torch",
            )
        )
        client_latency_ms = (time.perf_counter() - wall_start) * 1000.0
        with lock:
            rows.append(
                {
                    "request_id": response.request_id,
                    "robot_id": response.robot_id,
                    "session_id": response.session_id,
                    "admitted": bool(response.admitted),
                    "rejection_reason": response.rejection_reason,
                    "client_latency_ms": client_latency_ms,
                    "queue_ms": float(response.queue_ms),
                    "runtime_ms": float(response.runtime_ms),
                    "latency_ms": float(response.latency_ms),
                    "deadline_missed": bool(response.deadline_missed),
                    "deadline_slack_ms": float(response.deadline_slack_ms),
                    "batch_size": int(response.batch_size),
                    "batch_reason": response.batch_reason,
                    "actions_returned": int(response.actions_returned),
                }
            )
        step += 1
        next_at += period_s
    channel.close()


def summarize(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    admitted = [row for row in rows if row["admitted"]]
    rejected = [row for row in rows if not row["admitted"]]
    deadline_misses = [row for row in admitted if row["deadline_missed"]]
    latencies = [row["latency_ms"] for row in admitted]
    client_latencies = [row["client_latency_ms"] for row in rows]
    queues = [row["queue_ms"] for row in admitted]
    runtimes = [row["runtime_ms"] for row in admitted]
    slacks = [row["deadline_slack_ms"] for row in admitted]
    return {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "requests": len(rows),
        "admitted": len(admitted),
        "rejected": len(rejected),
        "rejection_rate": len(rejected) / max(len(rows), 1),
        "deadline_misses": len(deadline_misses),
        "deadline_miss_rate": len(deadline_misses) / max(len(admitted), 1),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "p95_client_latency_ms": percentile(client_latencies, 95),
        "p95_queue_ms": percentile(queues, 95),
        "p95_runtime_ms": percentile(runtimes, 95),
        "min_deadline_slack_ms": float(min(slacks)) if slacks else 0.0,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    payloads = prepare_observation_payloads(args)
    if args.save_warmup_observation is not None:
        args.save_warmup_observation.parent.mkdir(parents=True, exist_ok=True)
        args.save_warmup_observation.write_bytes(payloads[0])
    if args.duration_seconds <= 0:
        summary = summarize(args, [])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
        return
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    start_at = time.monotonic() + 2.0
    threads = [
        threading.Thread(
            target=worker,
            kwargs={
                "args": args,
                "robot_idx": idx,
                "payload": payloads[idx],
                "start_at": start_at,
                "rows": rows,
                "lock": lock,
            },
            daemon=True,
        )
        for idx in range(args.robots)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = summarize(args, sorted(rows, key=lambda row: row["request_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
