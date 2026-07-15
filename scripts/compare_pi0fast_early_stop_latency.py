#!/usr/bin/env python
"""Sweep approximate PI0-FAST FAST-character early stops on identical observations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"
if os.environ.get("MUJOCO_GL") == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _ensure_libero_config,
    _env_step,
    _import_lerobot,
    _prepare_observation,
    _to_numpy_action,
)
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--target-chars",
        default="4,8,12,16,24,32,48,70",
        help="Comma-separated decoded FAST-character stop targets to compare against exact action-end.",
    )
    parser.add_argument("--action-vocab-size", type=int, default=None)
    parser.add_argument("--text-vocab-size", type=int, default=None)
    parser.add_argument("--structural-token-radius", type=int, default=512)
    parser.add_argument(
        "--prefill-action-prefix",
        action="store_true",
        help="Experimentally prefill the fixed 'Action: ' target prefix as causal FAST tokens.",
    )
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pi0fast_system_components/pi0fast_early_stop_compare.json"),
    )
    return parser.parse_args()


def load_dotenv_token() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"} and value:
            os.environ.setdefault("HF_TOKEN", value)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", value)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", value)


def parse_int_csv(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_ms(device: torch.device, fn):
    sync_if_cuda(device)
    start = time.perf_counter()
    out = fn()
    sync_if_cuda(device)
    return out, (time.perf_counter() - start) * 1000.0


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def action_diff(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = torch.abs(reference.detach().float() - candidate.detach().float())
    first_step = diff[:, :1]
    return {
        "max_abs": float(torch.max(diff).item()),
        "mean_abs": float(torch.mean(diff).item()),
        "first_step_max_abs": float(torch.max(first_step).item()),
        "first_step_mean_abs": float(torch.mean(first_step).item()),
    }


def main() -> None:
    args = parse_args()
    load_dotenv_token()
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = (
        _import_lerobot()
    )

    target_chars = parse_int_csv(args.target_chars)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    policy_config = PreTrainedConfig.from_pretrained(args.policy)
    if hasattr(policy_config, "device"):
        policy_config.device = str(device)
    if hasattr(policy_config, "dtype"):
        policy_config.dtype = args.dtype
    if hasattr(policy_config, "use_kv_cache"):
        policy_config.use_kv_cache = True
    if args.disable_gradient_checkpointing and hasattr(policy_config, "gradient_checkpointing"):
        policy_config.gradient_checkpointing = False

    policy = PI0FastPolicy.from_pretrained(args.policy, config=policy_config).to(device=device, dtype=dtype).eval()
    if args.disable_gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
        policy.eval()

    adapter = PI0FastTokenLogitAdapter(policy)
    base_action_kwargs = {
        "constrained_action_vocab": True,
        "constrained_action_vocab_size": args.action_vocab_size,
        "constrained_text_vocab_size": args.text_vocab_size,
        "constrained_structural_token_radius": args.structural_token_radius,
    }
    candidate_action_kwargs = {
        **base_action_kwargs,
        "constrained_prefill_action_prefix": bool(args.prefill_action_prefix),
    }

    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_cfg = LiberoEnv(task=args.task, task_ids=[args.task_id], control_mode=args.control_mode)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = env_map[args.task][args.task_id]
    observation, _info = env.reset(seed=[args.seed])

    def exact_decode(batch):
        return adapter.predict_action_chunk_action_end(batch, **base_action_kwargs)

    def early_decode(batch, target: int):
        return adapter.predict_action_chunk_action_end(
            batch,
            **candidate_action_kwargs,
            stop_on_action_chars=True,
            action_char_target_chars=int(target),
        )

    warm_batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
    for _ in range(args.warmup):
        with torch.inference_mode():
            _ = exact_decode(warm_batch)
            for target in target_chars:
                _ = early_decode(warm_batch, target)

    rows: list[dict[str, Any]] = []
    exact_latencies: list[float] = []
    exact_token_counts: list[float] = []
    by_target: dict[int, dict[str, list[float]]] = {
        target: {
            "latency_ms": [],
            "token_count": [],
            "action_max_abs": [],
            "action_mean_abs": [],
            "first_step_max_abs": [],
            "first_step_mean_abs": [],
        }
        for target in target_chars
    }

    try:
        for step in range(args.steps):
            batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
            with torch.inference_mode():
                exact_trace, exact_ms = timed_ms(device, lambda: exact_decode(batch))
                exact_latencies.append(float(exact_ms))
                exact_token_counts.append(float(exact_trace.token_count))

                early_rows = []
                for target in target_chars:
                    early_trace, early_ms = timed_ms(device, lambda target=target: early_decode(batch, target))
                    diff = action_diff(exact_trace.actions, early_trace.actions)
                    stats = early_trace.stats or {}
                    token_count = float((stats.get("row_token_counts") or [early_trace.token_count])[0])
                    by_target[target]["latency_ms"].append(float(early_ms))
                    by_target[target]["token_count"].append(token_count)
                    by_target[target]["action_max_abs"].append(diff["max_abs"])
                    by_target[target]["action_mean_abs"].append(diff["mean_abs"])
                    by_target[target]["first_step_max_abs"].append(diff["first_step_max_abs"])
                    by_target[target]["first_step_mean_abs"].append(diff["first_step_mean_abs"])
                    early_rows.append(
                        {
                            "target_chars": int(target),
                            "latency_ms": float(early_ms),
                            "token_count": token_count,
                            "action_diff": diff,
                            "stats": stats,
                        }
                    )

            rows.append(
                {
                    "step": int(step),
                    "exact_latency_ms": float(exact_ms),
                    "exact_token_count": float(exact_trace.token_count),
                    "exact_stats": exact_trace.stats or {},
                    "early": early_rows,
                }
            )

            try:
                processed = policy_postprocessor(exact_trace.actions)
            except Exception:
                processed = exact_trace.actions
            action = _to_numpy_action(processed)[0]
            observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
            if terminated or truncated:
                observation, _info = env.reset(seed=[args.seed + step + 1])
    finally:
        try:
            env.close()
        except Exception:
            pass

    target_summary = {}
    for target, metrics in by_target.items():
        target_summary[str(target)] = {
            "latency_ms": summarize(metrics["latency_ms"]),
            "token_count": summarize(metrics["token_count"]),
            "action_max_abs": summarize(metrics["action_max_abs"]),
            "action_mean_abs": summarize(metrics["action_mean_abs"]),
            "first_step_max_abs": summarize(metrics["first_step_max_abs"]),
            "first_step_mean_abs": summarize(metrics["first_step_mean_abs"]),
            "speedup_vs_exact_mean": (
                float(np.mean(exact_latencies) / np.mean(metrics["latency_ms"]))
                if exact_latencies and metrics["latency_ms"] and np.mean(metrics["latency_ms"]) > 0
                else 0.0
            ),
        }

    output = {
        "config": {
            "policy": args.policy,
            "task": args.task,
            "task_id": args.task_id,
            "seed": args.seed,
            "warmup": args.warmup,
            "steps": args.steps,
            "device": str(device),
            "dtype": args.dtype,
            "target_chars": target_chars,
            "prefill_action_prefix": bool(args.prefill_action_prefix),
            "disable_gradient_checkpointing": bool(args.disable_gradient_checkpointing),
        },
        "exact": {
            "latency_ms": summarize(exact_latencies),
            "token_count": summarize(exact_token_counts),
        },
        "early_stop": target_summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2) + "\n")
    print(json.dumps(jsonable(output), indent=2))


if __name__ == "__main__":
    main()
