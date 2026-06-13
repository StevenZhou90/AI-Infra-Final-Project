#!/usr/bin/env python3
"""Scan action-space DFlash chunk frontier against PI0-FAST target_eos."""

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
from serving.pi0fast_action_dflash import load_action_dflash_checkpoint  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

logger = logging.getLogger("scan_pi0fast_action_dflash_frontier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan action-DFlash oracle frontier")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument(
        "--execute-actions-per-chunk",
        type=int,
        default=3,
        help="Baseline receding-horizon execution window before refreshing from PI0-FAST.",
    )
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--refine-steps", default="1,2,3")
    parser.add_argument("--windows", default="1,2,3,5,10")
    parser.add_argument("--label-max-abs", type=float, default=0.08)
    parser.add_argument("--label-mean-abs", type=float, default=0.03)
    parser.add_argument("--label-max-pos-l2", type=float, default=0.05)
    parser.add_argument("--label-max-rot-l2", type=float, default=0.25)
    parser.add_argument(
        "--freeze-gripper-from-init",
        action="store_true",
        help="Keep previous gripper phase in drafted postprocessed chunks. Diagnostic robot-phase prior.",
    )
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


def _shift_action_prior(actions: np.ndarray, shift: int) -> np.ndarray:
    shift = max(0, int(shift))
    prior = np.empty_like(actions)
    if shift <= 0:
        return actions.copy()
    if shift < actions.shape[0]:
        keep = actions.shape[0] - shift
        prior[:keep] = actions[shift:]
        prior[keep:] = actions[-1:]
    else:
        prior[:] = actions[-1:]
    return prior


def _summarize(rows: list[dict], windows: list[int], refine_steps: list[int]) -> dict:
    summary: dict[str, dict] = {}
    for refine in refine_steps:
        group = [row for row in rows if row["refine_steps"] == refine]
        if not group:
            continue
        baseline_ms = np.asarray([row["baseline_ms"] for row in group], dtype=np.float64)
        draft_ms = np.asarray([row["draft_ms"] for row in group], dtype=np.float64)
        baseline_horizons = np.asarray(
            [row.get("baseline_execute_actions", row["baseline_horizon"]) for row in group],
            dtype=np.float64,
        )
        baseline_ms_per_action = baseline_ms / np.maximum(baseline_horizons, 1.0)
        by_window = {}
        for window in windows:
            accept_mask = np.asarray([row["windows"][str(window)]["safe"] for row in group], dtype=bool)
            safe_rows = int(accept_mask.sum())
            effective_window = np.where(accept_mask, float(window), baseline_horizons)
            mixed_ms = np.where(accept_mask, draft_ms, baseline_ms)
            mixed_cost = mixed_ms / np.maximum(effective_window, 1.0)
            faster_cost = []
            chosen = 0
            for row, base_cost in zip(group, baseline_ms_per_action):
                spec_cost = row["draft_ms"] / float(window)
                if row["windows"][str(window)]["safe"] and spec_cost < base_cost:
                    faster_cost.append(spec_cost)
                    chosen += 1
                else:
                    faster_cost.append(float(base_cost))
            by_window[str(window)] = {
                "accepted": safe_rows,
                "accept_rate": safe_rows / len(group),
                "mixed_oracle_speedup_no_extra_fallback_cost": float(np.sum(baseline_ms_per_action) / np.sum(mixed_cost)),
                "safe_and_faster_oracle_speedup": float(np.sum(baseline_ms_per_action) / np.sum(faster_cost)),
                "safe_and_faster_rows": chosen,
                "mean_max_abs": float(np.mean([row["windows"][str(window)]["diff"]["max_abs"] for row in group])),
                "mean_max_pos_l2": float(np.mean([row["windows"][str(window)]["diff"]["max_pos_l2"] for row in group])),
            }
        summary[str(refine)] = {
            "rows": len(group),
            "avg_baseline_ms": float(np.mean(baseline_ms)),
            "avg_draft_ms": float(np.mean(draft_ms)),
            "chunk_decode_speedup": float(np.sum(baseline_ms) / np.sum(draft_ms)),
            "avg_prefix_ms": float(np.mean([row["prefix_ms"] for row in group])),
            "avg_head_ms": float(np.mean([row["head_ms"] for row in group])),
            "avg_confidence": float(np.mean([row["draft_stats"].get("mean_confidence", 0.0) for row in group])),
            "avg_last_refine_delta": float(np.mean([row["draft_stats"].get("last_refine_delta", 0.0) for row in group])),
            "by_window": by_window,
        }
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    task_ids = _parse_ints(args.task_ids)
    windows = _parse_ints(args.windows)
    refine_steps = _parse_ints(args.refine_steps)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)

    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    action_head, checkpoint_extra = load_action_dflash_checkpoint(args.checkpoint, map_location=device)
    action_head.to(device=device).eval()
    target_space = str(checkpoint_extra.get("config", {}).get("target_space", "raw"))

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
            previous_baseline_actions: np.ndarray | None = None
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

                hidden_cache = None
                prefix_ms = None
                for refine in refine_steps:
                    init = None
                    init_actions = None
                    if target_space == "postprocessed" and previous_baseline_actions is not None:
                        init_actions = _shift_action_prior(previous_baseline_actions, args.execute_actions_per_chunk)
                        init = torch.as_tensor(init_actions, dtype=torch.float32, device=device).unsqueeze(0)
                    if refine <= 0:
                        draft_actions = init_actions.copy() if init_actions is not None else np.zeros_like(baseline_actions)
                        head_ms = 0.0
                        draft_stats = {
                            "refine_steps": 0,
                            "mean_confidence": 1.0 if init_actions is not None else 0.0,
                            "min_confidence": 1.0 if init_actions is not None else 0.0,
                            "mean_refine_delta": 0.0,
                            "last_refine_delta": 0.0,
                        }
                    else:
                        if hidden_cache is None:
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            t1 = time.perf_counter()
                            hidden_cache = adapter.fast_prefix_hidden(batch)
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            prefix_ms = (time.perf_counter() - t1) * 1000.0
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        draft_actions_t, _conf, draft_stats = action_head.draft(
                            hidden_cache.to(device),
                            refine_steps=refine,
                            init=init,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        head_ms = (time.perf_counter() - t2) * 1000.0
                        if target_space == "postprocessed":
                            draft_processed = draft_actions_t
                        else:
                            try:
                                draft_processed = policy_postprocessor(draft_actions_t)
                            except Exception:
                                draft_processed = draft_actions_t
                        draft_actions = np.asarray(draft_processed[0].detach().cpu(), dtype=np.float32)
                    if (
                        args.freeze_gripper_from_init
                        and target_space == "postprocessed"
                        and init_actions is not None
                        and draft_actions.shape[-1] >= 7
                    ):
                        draft_actions[:, 6] = init_actions[: draft_actions.shape[0], 6]
                    window_rows = {}
                    for window in windows:
                        horizon = min(window, baseline_actions.shape[0], draft_actions.shape[0])
                        diff = _action_diff_summary(baseline_actions[:horizon], draft_actions[:horizon])
                        window_rows[str(window)] = {"diff": diff, "safe": _safe(diff, args)}
                    row = {
                        "task": args.task,
                        "task_id": task_id,
                        "chunk": chunk_idx,
                        "seed": args.seed,
                        "refine_steps": refine,
                        "baseline_ms": baseline_ms,
                        "baseline_execute_actions": int(min(args.execute_actions_per_chunk, baseline_actions.shape[0])),
                        "prefix_ms": float(prefix_ms or 0.0),
                        "head_ms": head_ms,
                        "draft_ms": float(prefix_ms or 0.0) + head_ms,
                        "baseline_horizon": int(baseline_actions.shape[0]),
                        "draft_horizon": int(draft_actions.shape[0]),
                        "baseline_tokens": int(baseline.token_ids.shape[1]),
                        "draft_stats": draft_stats,
                        "windows": window_rows,
                    }
                    rows.append(row)
                    safe_bits = ", ".join(f"m{w}={int(window_rows[str(w)]['safe'])}" for w in windows)
                    logger.info(
                        "task=%d chunk=%d refine=%d target=%.1fms draft=%.1fms %s",
                        task_id,
                        chunk_idx,
                        refine,
                        baseline_ms,
                        row["draft_ms"],
                        safe_bits,
                    )
                for action in baseline_actions[: args.execute_actions_per_chunk]:
                    observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
                    if terminated or truncated:
                        observation, _info = env.reset(seed=[args.seed + chunk_idx + 1])
                        previous_baseline_actions = None
                        break
                else:
                    previous_baseline_actions = baseline_actions.copy()
    finally:
        for envs in env_map.values():
            for env in envs.values():
                try:
                    env.close()
                except Exception:
                    pass

    summary = {
        "config": vars(args),
        "checkpoint_extra": checkpoint_extra,
        "summary": _summarize(rows, windows, refine_steps),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
