#!/usr/bin/env python3
"""Collect PI0-FAST action chunks for trajectory-tail draft training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_pi0fast_eagle_data import _ensure_libero_config, _parse_task_ids  # noqa: E402
from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _env_step,
    _import_lerobot,
    _prepare_observation,
    _to_numpy_action,
)

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger("generate_pi0fast_trajectory_data")


def _call_policy_chunk(policy, batch: dict[str, Any], postprocessor) -> tuple[np.ndarray, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if hasattr(policy, "predict_action_chunk"):
        raw = policy.predict_action_chunk(batch)
    else:
        raw = policy.select_action(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    try:
        raw = postprocessor(raw)
    except Exception:
        pass
    return _to_numpy_action(raw), elapsed_ms


@torch.inference_mode()
def collect_for_task(
    *,
    args: argparse.Namespace,
    task_id: int,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    policy_preprocessor,
    policy_postprocessor,
    preprocess_observation,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observation, _info = env.reset(seed=[args.seed + task_id * 1000])
    chunks: list[np.ndarray] = []
    decode_ms: list[float] = []
    steps = 0
    resets = 0

    while steps < args.steps_per_task and len(chunks) < args.chunks_per_task:
        batch = _prepare_observation(
            observation,
            env,
            env_preprocessor,
            policy_preprocessor,
            preprocess_observation,
        )
        chunk, elapsed_ms = _call_policy_chunk(policy, batch, policy_postprocessor)
        chunks.append(chunk.astype(np.float32))
        decode_ms.append(elapsed_ms)
        logger.info("task=%d chunk=%d horizon=%d decode_ms=%.1f", task_id, len(chunks), chunk.shape[0], elapsed_ms)

        for action in chunk[: args.execute_actions_per_chunk]:
            observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
            steps += 1
            if terminated or truncated or steps >= args.steps_per_task:
                resets += 1
                observation, _info = env.reset(seed=[args.seed + task_id * 1000 + resets])
                break

    for idx in range(len(chunks) - 1):
        cur = chunks[idx]
        nxt = chunks[idx + 1]
        if cur.shape[0] < args.input_horizon or nxt.shape[0] < args.tail_horizon:
            continue
        rows.append(
            {
                "input_chunk": torch.from_numpy(cur[: args.input_horizon]).to(dtype=torch.float32),
                "target_tail": torch.from_numpy(nxt[: args.tail_horizon]).to(dtype=torch.float32),
                "task_id": int(task_id),
                "seed": int(args.seed + task_id * 1000),
                "trace_id": f"task{task_id}_pair{idx:05d}",
                "decode_ms": float(decode_ms[idx]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PI0-FAST trajectory-tail training data")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0-4")
    parser.add_argument("--chunks-per-task", type=int, default=16)
    parser.add_argument("--steps-per-task", type=int, default=250)
    parser.add_argument("--execute-actions-per-chunk", type=int, default=10)
    parser.add_argument("--input-horizon", type=int, default=10)
    parser.add_argument("--tail-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--output-dir", default="outputs/pi0fast_trajectory_data")
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = _parse_task_ids(args.task_ids)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)

    env_cfg = LiberoEnv(task=args.task, task_ids=task_ids, control_mode=args.control_mode)
    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)

    summary: dict[str, Any] = {"config": vars(args), "tasks": {}}
    total_rows = 0
    try:
        for task_id in task_ids:
            env = env_map[args.task][task_id]
            rows = collect_for_task(
                args=args,
                task_id=task_id,
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                preprocess_observation=preprocess_observation,
            )
            shard_path = out_dir / f"shard_task{task_id:02d}.pt"
            torch.save(rows, shard_path)
            total_rows += len(rows)
            summary["tasks"][str(task_id)] = {"rows": len(rows), "shard": str(shard_path)}
    finally:
        for task_envs in env_map.values():
            for env in task_envs.values():
                try:
                    env.close()
                except Exception:
                    pass

    summary["total_rows"] = total_rows
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
