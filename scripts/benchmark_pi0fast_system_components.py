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
import contextlib
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
from serving.pi0fast_serving_runtime import merge_prepared_pi0fast_batches  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real PI0-FAST system components.")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--policy-kind", choices=["pi0fast", "pi05"], default="pi0fast")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument(
        "--decode-path",
        choices=["public", "action_end"],
        default="public",
        help="public uses policy.predict_action_chunk; action_end uses greedy FAST-token decode that stops on '|'.",
    )
    parser.add_argument(
        "--max-decoding-steps",
        default="default",
        help="Comma-separated max_decoding_steps values to test, or 'default'. This is a plain decode cap, not speculative decoding.",
    )
    parser.add_argument(
        "--num-inference-steps",
        default="default",
        help="Comma-separated flow inference-step values for PI05, or 'default'.",
    )
    parser.add_argument(
        "--batch-source",
        choices=["replicated", "distinct-reset"],
        default="replicated",
        help="Use duplicated observations or distinct reset observations for batched inference.",
    )
    parser.add_argument(
        "--action-token-warn-threshold",
        type=float,
        default=96.0,
        help="PI0-FAST action-end token-count threshold for straggler warnings.",
    )
    parser.add_argument("--kv-modes", default="default,on,off", help="Comma-separated: default,on,off")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--target-latency-ms", type=float, default=250.0)
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pi0fast_system_components/summary.json"))
    args = parser.parse_args()
    if args.policy is None:
        args.policy = "lerobot/pi05_libero_finetuned_v044" if args.policy_kind == "pi05" else "lerobot/pi0fast-libero"
    return args


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


def parse_max_decoding_steps(raw: str, default_value: int) -> list[tuple[str, int]]:
    values: list[tuple[str, int]] = []
    for item in parse_csv_strings(raw):
        if item == "default":
            values.append(("default", default_value))
        else:
            value = int(item)
            values.append((str(value), value))
    return values


def load_policy_class(policy_kind: str):
    if policy_kind == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    return PI0FastPolicy


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


def collect_distinct_reset_batch(
    *,
    env,
    env_preprocessor,
    policy_preprocessor,
    preprocess_observation,
    batch_size: int,
    seed_base: int,
) -> dict[str, Any]:
    prepared = []
    for offset in range(batch_size):
        observation, _info = env.reset(seed=[seed_base + offset])
        prepared.append(_prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation))
    return merge_prepared_pi0fast_batches(prepared)


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


def summarize_token_counts(values: list[float], warn_threshold: float) -> dict[str, Any] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "warn_threshold": float(warn_threshold),
        "straggler_rate": float(np.mean(arr > warn_threshold)),
        "has_straggler": bool(np.any(arr > warn_threshold)),
    }


def extend_token_counts(target: list[float], token_info: dict[str, Any] | None) -> None:
    if not token_info:
        return
    if "row_counts" in token_info:
        target.extend(float(value) for value in token_info["row_counts"])
    elif "mean" in token_info:
        target.append(float(token_info["mean"]))


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def pi05_recommendation(rows: list[dict[str, Any]], target_latency_ms: float) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("decode_path") == "public"
        and not row.get("errors")
        and row.get("single_inference", {}).get("mean_ms", float("inf")) <= target_latency_ms
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: row["single_inference"]["mean_ms"])
    return {
        "num_inference_steps": best.get("num_inference_steps"),
        "single_mean_ms": best["single_inference"]["mean_ms"],
        "target_latency_ms": float(target_latency_ms),
        "meets_target": True,
    }


def predict_chunk(
    *,
    policy,
    token_adapter: PI0FastTokenLogitAdapter | None,
    batch: dict[str, Any],
    decode_path: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    if autocast_dtype is not None and device.type == "cuda":
        cast_context = torch.autocast(device_type=device.type, dtype=autocast_dtype)
    else:
        cast_context = contextlib.nullcontext()
    with cast_context:
        if decode_path == "action_end":
            if token_adapter is None:
                raise ValueError("action_end decode path requires PI0FastTokenLogitAdapter")
            trace = token_adapter.predict_action_chunk_action_end(batch)
            row_counts = (trace.stats or {}).get("row_token_counts")
            if row_counts:
                return trace.actions, {"row_counts": [float(value) for value in row_counts]}
            return trace.actions, {"row_counts": [float(trace.token_count)]}
        raw = policy.predict_action_chunk(batch) if hasattr(policy, "predict_action_chunk") else policy.select_action(batch)
    return raw, None


def main() -> None:
    args = parse_args()
    _ensure_libero_config(args.libero_config_path)
    (
        make_env,
        make_env_pre_post_processors,
        preprocess_observation,
        LiberoEnv,
        make_pre_post_processors,
        _PI0FastPolicy,
    ) = _import_lerobot()
    PolicyClass = load_policy_class(args.policy_kind)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    kv_modes = parse_csv_strings(args.kv_modes)

    load_start = time.perf_counter()
    policy = PolicyClass.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    token_adapter = PI0FastTokenLogitAdapter(policy) if args.decode_path == "action_end" else None
    sync_if_cuda(device)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    inference_autocast_dtype = dtype if args.policy_kind == "pi05" and dtype in (torch.bfloat16, torch.float16) else None

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
        "policy_max_decoding_steps_initial": getattr(policy.config, "max_decoding_steps", None),
        "policy_num_inference_steps_initial": getattr(policy.config, "num_inference_steps", None),
        "recommendation": None,
        "rows": rows,
    }

    try:
        observation, _info = env.reset(seed=[args.seed])

        for _ in range(args.warmup):
            batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
            with torch.inference_mode():
                raw_action, _token_count = predict_chunk(
                    policy=policy,
                    token_adapter=token_adapter,
                    batch=batch,
                    decode_path=args.decode_path,
                    device=device,
                    autocast_dtype=inference_autocast_dtype,
                )
            try:
                action = _to_numpy_action(policy_postprocessor(raw_action))[0]
            except Exception:
                action = _to_numpy_action(raw_action)[0]
            observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
            if terminated or truncated:
                observation, _info = env.reset(seed=[args.seed + 1])

        if args.decode_path == "action_end" and args.policy_kind != "pi0fast":
            raise ValueError("--decode-path action_end is only supported for --policy-kind pi0fast")

        initial_max_decoding_steps = int(getattr(policy.config, "max_decoding_steps", 0) or 0)
        max_decode_modes = parse_max_decoding_steps(args.max_decoding_steps, initial_max_decoding_steps)
        initial_num_inference_steps = int(getattr(policy.config, "num_inference_steps", 0) or 0)
        num_inference_modes = parse_max_decoding_steps(args.num_inference_steps, initial_num_inference_steps)

        for max_decode_label, max_decode_steps in max_decode_modes:
            if hasattr(policy.config, "max_decoding_steps"):
                policy.config.max_decoding_steps = int(max_decode_steps)
            for inference_steps_label, inference_steps in num_inference_modes:
                if hasattr(policy.config, "num_inference_steps"):
                    policy.config.num_inference_steps = int(inference_steps)
                for mode in kv_modes:
                    active_kv = set_kv_mode(policy, mode)
                    preprocess_ms: list[float] = []
                    single_ms: list[float] = []
                    single_token_counts: list[float] = []
                    postprocess_ms: list[float] = []
                    batch_ms: dict[int, list[float]] = {size: [] for size in batch_sizes if size > 1}
                    batch_token_counts: dict[int, list[float]] = {size: [] for size in batch_sizes if size > 1}
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
                            (raw_action, token_info), infer_ms = timed_ms(
                                device,
                                lambda: predict_chunk(
                                    policy=policy,
                                    token_adapter=token_adapter,
                                    batch=batch,
                                    decode_path=args.decode_path,
                                    device=device,
                                    autocast_dtype=inference_autocast_dtype,
                                ),
                        )
                        single_ms.append(infer_ms)
                        extend_token_counts(single_token_counts, token_info)

                        processed, post_ms = timed_ms(device, lambda: policy_postprocessor(raw_action))
                        postprocess_ms.append(post_ms)
                        action = _to_numpy_action(processed)[0]

                        for batch_size in batch_sizes:
                            if batch_size <= 1:
                                continue
                            if args.batch_source == "distinct-reset":
                                replica_batch = collect_distinct_reset_batch(
                                    env=env,
                                    env_preprocessor=env_preprocessor,
                                    policy_preprocessor=policy_preprocessor,
                                    preprocess_observation=preprocess_observation,
                                    batch_size=batch_size,
                                    seed_base=args.seed + 10_000 + step * 100 + batch_size * 10,
                                )
                            else:
                                replica_batch = clone_batch_for_replicas(batch, batch_size)
                            try:
                                with torch.inference_mode():
                                    (_raw, batch_token_info), batched_ms = timed_ms(
                                        device,
                                        lambda: predict_chunk(
                                            policy=policy,
                                            token_adapter=token_adapter,
                                            batch=replica_batch,
                                            decode_path=args.decode_path,
                                            device=device,
                                            autocast_dtype=inference_autocast_dtype,
                                        ),
                                    )
                                batch_ms[batch_size].append(batched_ms)
                                extend_token_counts(batch_token_counts[batch_size], batch_token_info)
                            except Exception as exc:
                                message = f"batch_size={batch_size}: {type(exc).__name__}: {exc}"
                                if message not in errors:
                                    errors.append(message)

                        observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
                        if terminated or truncated:
                            observation, _info = env.reset(seed=[args.seed + step + 2])

                    row: dict[str, Any] = {
                        "max_decoding_steps_label": max_decode_label,
                        "max_decoding_steps": int(max_decode_steps),
                        "num_inference_steps_label": inference_steps_label,
                        "num_inference_steps": int(inference_steps),
                        "decode_path": args.decode_path,
                        "kv_mode": mode,
                        "active_use_kv_cache": active_kv,
                        "preprocess": summarize(preprocess_ms),
                        "single_inference": summarize(single_ms),
                        "single_token_count_mean": float(np.mean(single_token_counts)) if single_token_counts else None,
                        "single_token_counts": summarize_token_counts(
                            single_token_counts,
                            args.action_token_warn_threshold,
                        ),
                        "postprocess": summarize(postprocess_ms),
                        "batch_inference": {},
                        "batch_source": args.batch_source,
                        "errors": errors,
                    }
                    for batch_size, values in batch_ms.items():
                        stats = summarize(values)
                        single_mean = row["single_inference"]["mean_ms"]
                        stats["per_request_mean_ms"] = stats["mean_ms"] / batch_size if batch_size else 0.0
                        stats["throughput_speedup_vs_serial"] = (
                            (single_mean * batch_size) / stats["mean_ms"] if stats["mean_ms"] > 0 else 0.0
                        )
                        stats["token_count_mean"] = (
                            float(np.mean(batch_token_counts[batch_size])) if batch_token_counts[batch_size] else None
                        )
                        stats["token_counts"] = summarize_token_counts(
                            batch_token_counts[batch_size],
                            args.action_token_warn_threshold,
                        )
                        row["batch_inference"][str(batch_size)] = stats
                    rows.append(row)
                    if args.policy_kind == "pi05":
                        summary["recommendation"] = pi05_recommendation(rows, args.target_latency_ms)
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
