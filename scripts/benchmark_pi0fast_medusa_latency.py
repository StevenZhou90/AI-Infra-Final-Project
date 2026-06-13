#!/usr/bin/env python3
"""Benchmark exact Medusa speculative PI0-FAST decode latency in LIBERO."""

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

from scripts.benchmark_pi0fast_ngram_latency import _action_diff_summary, _postprocess_action  # noqa: E402
from scripts.generate_pi0fast_eagle_data import _ensure_libero_config  # noqa: E402
from scripts.run_pi0fast_chunk_eval import _env_step, _import_lerobot, _prepare_observation  # noqa: E402
from serving.pi0fast_medusa import load_medusa_checkpoint  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger("benchmark_pi0fast_medusa_latency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PI0-FAST Medusa speculative decode latency")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--checkpoint", default="outputs/pi0fast_real_sd_medusa_30task_20trace_fixed_v1/pi0fast_medusa.pt")
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--min-draft-confidence", type=float, default=0.0)
    parser.add_argument("--min-verify-confidence", type=float, default=0.0)
    parser.add_argument("--min-spec-position", type=int, default=0)
    parser.add_argument("--accept-partial-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-accepted-cache", action="store_true")
    parser.add_argument("--resync-accepted-cache", action="store_true")
    parser.add_argument("--verify-from-scratch", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--output", default="outputs/pi0fast_medusa_latency_task4.json")
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    medusa_head, token_map, medusa_summary = load_medusa_checkpoint(args.checkpoint, device=device)

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
            baseline = adapter.predict_action_chunk_with_trace(batch, temperature=0.0, early_stop_action_end=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            baseline_ms = (time.perf_counter() - t0) * 1000

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            spec = adapter.predict_action_chunk_medusa_speculative(
                batch,
                medusa_head=medusa_head,
                token_map=token_map,
                lookahead=args.lookahead,
                min_draft_confidence=args.min_draft_confidence,
                min_verify_confidence=args.min_verify_confidence,
                min_spec_position=args.min_spec_position,
                accept_partial_blocks=args.accept_partial_blocks,
                replay_accepted_cache=args.replay_accepted_cache,
                resync_accepted_cache=args.resync_accepted_cache,
                verify_from_scratch=args.verify_from_scratch,
                early_stop_action_end=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            spec_ms = (time.perf_counter() - t1) * 1000

            tokens_equal = bool(torch.equal(baseline.token_ids, spec.token_ids))
            first_token_diff = None
            if not tokens_equal:
                horizon = min(baseline.token_ids.shape[1], spec.token_ids.shape[1])
                diff = torch.nonzero(baseline.token_ids[0, :horizon] != spec.token_ids[0, :horizon], as_tuple=False)
                if diff.numel():
                    pos = int(diff[0].item())
                    first_token_diff = {
                        "pos": pos,
                        "baseline": int(baseline.token_ids[0, pos].item()),
                        "spec": int(spec.token_ids[0, pos].item()),
                    }
                else:
                    first_token_diff = {"lengths": [int(baseline.token_ids.shape[1]), int(spec.token_ids.shape[1])]}
            baseline_actions = _postprocess_action(baseline, policy_postprocessor)
            spec_actions = _postprocess_action(spec, policy_postprocessor)
            action_diff = _action_diff_summary(baseline_actions, spec_actions)
            row = {
                "chunk": idx,
                "baseline_ms": baseline_ms,
                "spec_ms": spec_ms,
                "speedup": baseline_ms / spec_ms if spec_ms else None,
                "baseline_tokens": int(baseline.token_ids.shape[1]),
                "spec_tokens": int(spec.token_ids.shape[1]),
                "tokens_equal": tokens_equal,
                "first_token_diff": first_token_diff,
                "action_diff": action_diff,
                "spec_stats": spec.stats,
            }
            rows.append(row)
            logger.info(
                "chunk=%d target_eos=%.1fms medusa=%.1fms speedup=%.3fx equal=%s stats=%s",
                idx,
                baseline_ms,
                spec_ms,
                baseline_ms / spec_ms if spec_ms else 0.0,
                tokens_equal,
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
        "medusa_best_epoch": medusa_summary.get("best_epoch"),
        "chunks": rows,
        "avg_baseline_ms": float(np.mean([row["baseline_ms"] for row in rows])) if rows else None,
        "avg_spec_ms": float(np.mean([row["spec_ms"] for row in rows])) if rows else None,
        "avg_speedup": float(np.mean([row["speedup"] for row in rows if row["speedup"]])) if rows else None,
        "all_tokens_equal": all(row["tokens_equal"] for row in rows),
        "avg_tokens_per_target_forward": float(
            np.mean([row["spec_stats"]["tokens_per_target_forward"] for row in rows])
        )
        if rows
        else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
