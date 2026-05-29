#!/usr/bin/env python
"""Measure real PI0-FAST serving components one at a time.

This is the systems benchmark path, separate from speculative decoding.  It
loads a public PI0-FAST policy, prepares LIBERO observations, and times:

- environment/policy preprocessing
- single-request policy inference
- optional config-level KV cache on/off
- replicated batched inference when the underlying LeRobot policy supports it

The goal is to identify which systems layer is worth optimizing before mixing
changes together.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _ensure_libero_config,
    _env_step,
    _import_lerobot,
    _prepare_observation,
    _to_numpy_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real PI0-FAST system components.")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--kv-modes", default="default,on,off", help="Comma-separated: default,on,off")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pi0fast_system_components/summary.json"))
    return parser.parse_args()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_ms(device: torch.device, fn):
    sync_if_cuda(device)
    t0 = time.perf_counter()
    out = fn()
    sync_if_cuda(device)
    return out, (time.perf_counter() - t0) * 1000.0


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def clone_batch_for_replicas(batch: Any, replicas: int) -> Any:
    """Replicate a processed LeRobot batch along its batch dimension."""
    if replicas == 1:
        return batch
    if torch.is_tensor(batch):
        if batch.ndim == 0:
            return batch
        return batch.repeat((replicas,) + (1,) * (batch.ndim - 1))
    if isinstance(batch, np.ndarray):
        if batch.ndim == 0:
            return batch
        return np.repeat(batch, replicas, axis=0)
    if isinstance(batch, list):
        if len(batch) == 1:
            return batch * replicas
        return [copy.deepcopy(item) for _ in range(replicas) for item in batch]
    if isinstance(batch, tuple):
        return tuple(clone_batch_for_replicas(item, replicas) for item in batch)
    if isinstance(batch, dict):
        return {key: clone_batch_for_replicas(value, replicas) for key, value in batch.items()}
    return copy.deepcopy(batch)


def set_kv_mode(policy, mode: str) -> bool | None:
    """Best-effort toggle for LeRobot PI0-FAST config-level KV cache."""
    if mode == "default":
        return getattr(policy.config, "use_kv_cache", None)
    desired = mode == "on"
    if not hasattr(policy.config, "use_kv_cache"):
        return None
    policy.config.use_kv_cache = desired
    return desired


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    args = parse_args()
    _ensure_libero_config(args.libero_config_path)
    (
        make_env,
        make_env_pre_post_processors,
        preprocess_observation,
        LiberoEnv,
        make_pre_post_processors,
        PI0FastPolicy,
    ) = _import_lerobot()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    kv_modes = parse_csv_strings(args.kv_modes)

    load_start = time.perf_counter()
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    sync_if_cuda(device)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_cfg = LiberoEnv(task=args.task, task_ids=[args.task_id], control_mode=args.control_mode)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = env_map[args.task][args.task_id]

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "config": jsonable_config(args),
        "device": str(device),
        "dtype": args.dtype,
        "load_ms": load_ms,
        "policy_use_kv_cache_initial": getattr(policy.config, "use_kv_cache", None),
        "rows": rows,
    }

    try:
        observation, _info = env.reset(seed=[args.seed])

        for _ in range(args.warmup):
            batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
            with torch.inference_mode():
                raw_action = policy.predict_action_chunk(batch) if hasattr(policy, "predict_action_chunk") else policy.select_action(batch)
            try:
                action = _to_numpy_action(policy_postprocessor(raw_action))[0]
            except Exception:
                action = _to_numpy_action(raw_action)[0]
            observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
            if terminated or truncated:
                observation, _info = env.reset(seed=[args.seed + 1])

        for mode in kv_modes:
            active_kv = set_kv_mode(policy, mode)
            preprocess_ms: list[float] = []
            single_ms: list[float] = []
            postprocess_ms: list[float] = []
            batch_ms: dict[int, list[float]] = {size: [] for size in batch_sizes if size > 1}
            errors: list[str] = []

            for step in range(args.steps):
                batch, prep_ms = timed_ms(
                    device,
                    lambda: _prepare_observation(
                        observation,
                        env,
                        env_preprocessor,
                        policy_preprocessor,
                        preprocess_observation,
                    ),
                )
                preprocess_ms.append(prep_ms)

                with torch.inference_mode():
                    raw_action, infer_ms = timed_ms(
                        device,
                        lambda: policy.predict_action_chunk(batch)
                        if hasattr(policy, "predict_action_chunk")
                        else policy.select_action(batch),
                    )
                single_ms.append(infer_ms)

                processed, post_ms = timed_ms(device, lambda: policy_postprocessor(raw_action))
                postprocess_ms.append(post_ms)
                action = _to_numpy_action(processed)[0]

                for batch_size in batch_sizes:
                    if batch_size <= 1:
                        continue
                    replica_batch = clone_batch_for_replicas(batch, batch_size)
                    try:
                        with torch.inference_mode():
                            _raw, batched_ms = timed_ms(
                                device,
                                lambda: policy.predict_action_chunk(replica_batch)
                                if hasattr(policy, "predict_action_chunk")
                                else policy.select_action(replica_batch),
                            )
                        batch_ms[batch_size].append(batched_ms)
                    except Exception as exc:
                        message = f"batch_size={batch_size}: {type(exc).__name__}: {exc}"
                        if message not in errors:
                            errors.append(message)

                observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
                if terminated or truncated:
                    observation, _info = env.reset(seed=[args.seed + step + 2])

            row: dict[str, Any] = {
                "kv_mode": mode,
                "active_use_kv_cache": active_kv,
                "preprocess": summarize(preprocess_ms),
                "single_inference": summarize(single_ms),
                "postprocess": summarize(postprocess_ms),
                "batch_inference": {},
                "errors": errors,
            }
            for batch_size, values in batch_ms.items():
                stats = summarize(values)
                single_mean = row["single_inference"]["mean_ms"]
                stats["per_request_mean_ms"] = stats["mean_ms"] / batch_size if batch_size else 0.0
                stats["throughput_speedup_vs_serial"] = (
                    (single_mean * batch_size) / stats["mean_ms"] if stats["mean_ms"] > 0 else 0.0
                )
                row["batch_inference"][str(batch_size)] = stats
            rows.append(row)
    finally:
        try:
            env.close()
        except Exception:
            pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
