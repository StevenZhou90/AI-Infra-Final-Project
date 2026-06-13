#!/usr/bin/env python3
"""Benchmark exact n-gram speculative PI0-FAST decode latency in LIBERO."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_pi0fast_eagle_data import _ensure_libero_config  # noqa: E402
from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _env_step,
    _import_lerobot,
    _prepare_observation,
    _to_numpy_action,
)
from serving.pi0fast_eagle import load_trace_records  # noqa: E402
from serving.pi0fast_ngram import NgramDraftConfig, NgramFastTokenDrafter  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger("benchmark_pi0fast_ngram_latency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PI0-FAST n-gram speculative decode latency")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--data-dir", default="outputs/pi0fast_eagle_tasks0_4_20trace_data")
    parser.add_argument("--train-task-ids", default="0,1,2,3")
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=4)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument(
        "--reuse-full-blocks",
        action="store_true",
        help="Reuse verifier KV on fully accepted draft blocks. Faster but currently approximate.",
    )
    parser.add_argument(
        "--verify-from-scratch",
        action="store_true",
        help="Recompute prefix+generated+draft for each verify block. Slower than cached verify but exact.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--output", default="outputs/pi0fast_ngram_latency_task4.json")
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


def _parse_ids(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _postprocess_action(trace, postprocessor) -> np.ndarray:
    try:
        processed = postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return _to_numpy_action(processed)


def _action_diff_summary(baseline_actions: np.ndarray, spec_actions: np.ndarray) -> dict[str, float | int | bool]:
    baseline = np.asarray(baseline_actions, dtype=np.float32)
    spec = np.asarray(spec_actions, dtype=np.float32)
    horizon = min(baseline.shape[0], spec.shape[0])
    dim = min(baseline.shape[1], spec.shape[1]) if baseline.ndim == 2 and spec.ndim == 2 else 0
    if horizon == 0 or dim == 0:
        return {
            "comparable": False,
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "max_pos_l2": float("nan"),
            "max_rot_l2": float("nan"),
            "gripper_mismatches": 0,
        }
    delta = baseline[:horizon, :dim] - spec[:horizon, :dim]
    pos_dim = min(3, dim)
    rot_start = min(3, dim)
    rot_end = min(6, dim)
    gripper_index = 6
    gripper_mismatches = 0
    if dim > gripper_index:
        gripper_mismatches = int(np.sum(np.sign(baseline[:horizon, gripper_index]) != np.sign(spec[:horizon, gripper_index])))
    return {
        "comparable": True,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "max_pos_l2": float(np.max(np.linalg.norm(delta[:, :pos_dim], axis=1))) if pos_dim else 0.0,
        "max_rot_l2": (
            float(np.max(np.linalg.norm(delta[:, rot_start:rot_end], axis=1))) if rot_end > rot_start else 0.0
        ),
        "gripper_mismatches": gripper_mismatches,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    train_task_ids = _parse_ids(args.train_task_ids)
    records = load_trace_records(args.data_dir)
    drafter = NgramFastTokenDrafter(
        NgramDraftConfig(max_context=args.max_context, min_count=args.min_count, lookahead=args.lookahead)
    )
    drafter.fit(record for record in records if record.task_id in train_task_ids)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    env_cfg = LiberoEnv(task=args.task, task_ids=[args.task_id], control_mode=args.control_mode)
    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = env_map[args.task][args.task_id]

    rows = []
    observation, _info = env.reset(seed=[args.seed])
    try:
        for idx in range(args.chunks):
            batch = _prepare_observation(
                observation,
                env,
                env_preprocessor,
                policy_preprocessor,
                preprocess_observation,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            baseline = adapter.predict_action_chunk_with_trace(batch, temperature=0.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            baseline_ms = (time.perf_counter() - t0) * 1000

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            spec = adapter.predict_action_chunk_ngram_speculative(
                batch,
                drafter=drafter,
                lookahead=args.lookahead,
                reuse_full_blocks=args.reuse_full_blocks,
                verify_from_scratch=args.verify_from_scratch,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            spec_ms = (time.perf_counter() - t1) * 1000

            tokens_equal = bool(torch.equal(baseline.token_ids, spec.token_ids))
            first_token_diff = None
            if not tokens_equal:
                diff = torch.nonzero(baseline.token_ids[0] != spec.token_ids[0], as_tuple=False)
                if diff.numel():
                    pos = int(diff[0].item())
                    first_token_diff = {
                        "pos": pos,
                        "baseline": int(baseline.token_ids[0, pos].item()),
                        "spec": int(spec.token_ids[0, pos].item()),
                    }
            baseline_actions = _postprocess_action(baseline, policy_postprocessor)
            spec_actions = _postprocess_action(spec, policy_postprocessor)
            action_diff = _action_diff_summary(baseline_actions, spec_actions)
            rows.append(
                {
                    "chunk": idx,
                    "baseline_ms": baseline_ms,
                    "spec_ms": spec_ms,
                    "speedup": baseline_ms / spec_ms if spec_ms else None,
                    "tokens_equal": tokens_equal,
                    "first_token_diff": first_token_diff,
                    "action_diff": action_diff,
                    "spec_stats": spec.stats,
                }
            )
            logger.info(
                "chunk=%d baseline=%.1fms spec=%.1fms speedup=%.3fx equal=%s action_diff=%s stats=%s",
                idx,
                baseline_ms,
                spec_ms,
                baseline_ms / spec_ms if spec_ms else 0.0,
                tokens_equal,
                action_diff,
                spec.stats,
            )

            for action in baseline_actions[: policy.config.n_action_steps]:
                observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
                if terminated or truncated:
                    observation, _info = env.reset(seed=[args.seed + idx + 1])
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass

    summary = {
        "config": vars(args),
        "chunks": rows,
        "avg_baseline_ms": float(np.mean([row["baseline_ms"] for row in rows])) if rows else None,
        "avg_spec_ms": float(np.mean([row["spec_ms"] for row in rows])) if rows else None,
        "avg_speedup": float(np.mean([row["speedup"] for row in rows if row["speedup"]])) if rows else None,
        "all_tokens_equal": all(row["tokens_equal"] for row in rows),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
