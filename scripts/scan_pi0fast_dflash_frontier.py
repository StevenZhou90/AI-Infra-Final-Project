#!/usr/bin/env python3
"""Scan the action-equivalence frontier for PI0-FAST DFlash candidates."""

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
from serving.pi0fast_block_drafter import load_block_drafter_checkpoint  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

logger = logging.getLogger("scan_pi0fast_dflash_frontier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan DFlash action acceptance and oracle speed frontier")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0,1,2,3,4")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--min-spec-position", type=int, default=24)
    parser.add_argument("--max-future-accept", type=int, default=4)
    parser.add_argument("--min-future-accept", type=int, default=0)
    parser.add_argument("--min-draft-confidence", type=float, default=0.0)
    parser.add_argument("--min-verify-confidence", type=float, default=0.0)
    parser.add_argument("--min-verify-margin", type=float, default=0.0)
    parser.add_argument("--allow-unknown-context", action="store_true")
    parser.add_argument("--full-block-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--accept-partial-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-steps", type=int, default=2)
    parser.add_argument("--verify-from-scratch", action="store_true")
    parser.add_argument("--resync-accepted-cache", action="store_true")
    parser.add_argument("--draft-after-known-token", action="store_true")
    parser.add_argument("--windows", default="1,2,3,5,10")
    parser.add_argument("--label-max-abs", type=float, default=0.08)
    parser.add_argument("--label-mean-abs", type=float, default=0.03)
    parser.add_argument("--label-max-pos-l2", type=float, default=0.05)
    parser.add_argument("--label-max-rot-l2", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _safe(diff: dict, args: argparse.Namespace) -> bool:
    return bool(
        diff.get("comparable", False)
        and float(diff.get("max_abs", float("inf"))) <= args.label_max_abs
        and float(diff.get("mean_abs", float("inf"))) <= args.label_mean_abs
        and float(diff.get("max_pos_l2", float("inf"))) <= args.label_max_pos_l2
        and float(diff.get("max_rot_l2", float("inf"))) <= args.label_max_rot_l2
        and int(diff.get("gripper_mismatches", 1)) == 0
    )


def _diff_window(baseline: np.ndarray, spec: np.ndarray, window: int) -> dict:
    horizon = min(int(window), int(baseline.shape[0]), int(spec.shape[0]))
    return _action_diff_summary(baseline[:horizon], spec[:horizon])


def _summarize(rows: list[dict], windows: list[int]) -> dict:
    if not rows:
        return {}

    baseline_ms = np.asarray([row["baseline_ms"] for row in rows], dtype=np.float64)
    spec_ms = np.asarray([row["spec_ms"] for row in rows], dtype=np.float64)
    baseline_horizons = np.asarray([row["baseline_horizon"] for row in rows], dtype=np.float64)
    baseline_ms_per_action = baseline_ms / np.maximum(baseline_horizons, 1.0)

    by_window = {}
    for window in windows:
        accepted = [row for row in rows if row["windows"][str(window)]["safe"]]
        accept_mask = np.asarray([row["windows"][str(window)]["safe"] for row in rows], dtype=bool)
        if accepted:
            accepted_spec_ms = np.asarray([row["spec_ms"] for row in accepted], dtype=np.float64)
            accepted_baseline_ms_per_action = np.asarray(
                [row["baseline_ms"] / max(row["baseline_horizon"], 1) for row in accepted],
                dtype=np.float64,
            )
            oracle_speedups = accepted_baseline_ms_per_action / (accepted_spec_ms / float(window))
        else:
            oracle_speedups = np.asarray([], dtype=np.float64)

        effective_window = np.where(accept_mask, float(window), baseline_horizons)
        mixed_ms = np.where(accept_mask, spec_ms, baseline_ms)
        mixed_ms_per_action = mixed_ms / np.maximum(effective_window, 1.0)
        mixed_speedup = float(np.sum(baseline_ms_per_action) / np.sum(mixed_ms_per_action))

        by_window[str(window)] = {
            "rows": len(rows),
            "accepted": len(accepted),
            "accept_rate": len(accepted) / len(rows),
            "mean_oracle_speedup_on_accepted": float(np.mean(oracle_speedups)) if accepted else 0.0,
            "mixed_oracle_speedup_no_extra_fallback_cost": mixed_speedup,
            "mean_max_abs": float(np.mean([row["windows"][str(window)]["diff"]["max_abs"] for row in rows])),
            "mean_max_pos_l2": float(np.mean([row["windows"][str(window)]["diff"]["max_pos_l2"] for row in rows])),
            "gripper_clean_rate": float(
                np.mean([row["windows"][str(window)]["diff"]["gripper_mismatches"] == 0 for row in rows])
            ),
        }

    return {
        "rows": len(rows),
        "avg_baseline_ms": float(np.mean(baseline_ms)),
        "avg_spec_ms": float(np.mean(spec_ms)),
        "decode_speedup_per_chunk": float(np.sum(baseline_ms) / np.sum(spec_ms)),
        "avg_baseline_horizon": float(np.mean(baseline_horizons)),
        "avg_baseline_ms_per_action": float(np.mean(baseline_ms_per_action)),
        "avg_spec_future_acceptance_rate": float(
            np.mean([row["spec_stats"].get("future_acceptance_rate", 0.0) for row in rows])
        ),
        "avg_spec_tokens_per_target_forward": float(
            np.mean([row["spec_stats"].get("tokens_per_target_forward", 0.0) for row in rows])
        ),
        "by_window": by_window,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    task_ids = _parse_ints(args.task_ids)
    windows = _parse_ints(args.windows)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)

    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    block_drafter, token_map, block_summary = load_block_drafter_checkpoint(args.checkpoint, device=device)

    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_cfg = LiberoEnv(task=args.task, task_ids=task_ids, control_mode=args.control_mode)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)

    rows = []
    try:
        for task_id in task_ids:
            env = env_map[args.task][task_id]
            observation, _info = env.reset(seed=[args.seed])
            for chunk_idx in range(args.chunks):
                batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                baseline = adapter.predict_action_chunk_with_trace(batch, temperature=0.0, early_stop_action_end=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                baseline_ms = (time.perf_counter() - t0) * 1000.0
                baseline_actions = _postprocess_action(baseline, policy_postprocessor)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                spec = adapter.predict_action_chunk_block_speculative(
                    batch,
                    block_drafter=block_drafter,
                    token_map=token_map,
                    lookahead=args.lookahead,
                    min_draft_confidence=args.min_draft_confidence,
                    min_verify_confidence=args.min_verify_confidence,
                    min_verify_margin=args.min_verify_margin,
                    max_future_accept=None if args.max_future_accept < 0 else args.max_future_accept,
                    min_future_accept=args.min_future_accept,
                    min_spec_position=args.min_spec_position,
                    allow_unknown_context=args.allow_unknown_context,
                    full_block_only=args.full_block_only,
                    accept_partial_blocks=args.accept_partial_blocks,
                    refine_steps=args.refine_steps,
                    verify_from_scratch=args.verify_from_scratch,
                    resync_accepted_cache=args.resync_accepted_cache,
                    draft_after_known_token=args.draft_after_known_token,
                    early_stop_action_end=True,
                    debug_event_limit=0,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                spec_ms = (time.perf_counter() - t1) * 1000.0
                spec_actions = _postprocess_action(spec, policy_postprocessor)

                window_rows = {}
                for window in windows:
                    diff = _diff_window(baseline_actions, spec_actions, window)
                    window_rows[str(window)] = {"diff": diff, "safe": _safe(diff, args)}

                row = {
                    "task": args.task,
                    "task_id": task_id,
                    "chunk": chunk_idx,
                    "seed": args.seed,
                    "baseline_ms": baseline_ms,
                    "spec_ms": spec_ms,
                    "baseline_horizon": int(baseline_actions.shape[0]),
                    "spec_horizon": int(spec_actions.shape[0]),
                    "baseline_tokens": int(baseline.token_ids.shape[1]),
                    "spec_tokens": int(spec.token_ids.shape[1]),
                    "spec_stats": spec.stats or {},
                    "windows": window_rows,
                }
                rows.append(row)
                safe_bits = ", ".join(f"m{w}={int(window_rows[str(w)]['safe'])}" for w in windows)
                logger.info(
                    "task=%d chunk=%d target=%.1fms dflash=%.1fms %s",
                    task_id,
                    chunk_idx,
                    baseline_ms,
                    spec_ms,
                    safe_bits,
                )

                for action in baseline_actions[: policy.config.n_action_steps]:
                    observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
                    if terminated or truncated:
                        observation, _info = env.reset(seed=[args.seed + chunk_idx + 1])
                        break
    finally:
        for envs in env_map.values():
            for env in envs.values():
                try:
                    env.close()
                except Exception:
                    pass

    summary = {
        "config": vars(args),
        "block_best_epoch": block_summary.get("best_epoch") if isinstance(block_summary, dict) else None,
        "summary": _summarize(rows, windows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
