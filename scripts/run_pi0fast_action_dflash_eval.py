#!/usr/bin/env python3
"""Run PI0-FAST action-DFlash receding-horizon rollouts in LIBERO."""

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

from scripts.generate_pi0fast_eagle_data import _ensure_libero_config  # noqa: E402
from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _env_step,
    _extract_success,
    _import_lerobot,
    _prepare_observation,
    _predict_action_chunk,
)
from serving.pi0fast_action_dflash import load_action_dflash_checkpoint  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

logger = logging.getLogger("run_pi0fast_action_dflash_eval")


def _parse_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


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


def _smooth_guard(actions: np.ndarray, args: argparse.Namespace) -> bool:
    if actions.shape[0] < 2:
        return True
    pos_delta = float(np.max(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)))
    rot_delta = float(np.max(np.linalg.norm(np.diff(actions[:, 3:6], axis=0), axis=1))) if actions.shape[1] >= 6 else 0.0
    if pos_delta > args.max_pos_delta or rot_delta > args.max_rot_delta:
        return False
    if actions.shape[1] >= 7:
        if float(np.max(np.abs(np.diff(actions[:, 6], axis=0)))) > args.max_gripper_delta:
            return False
    return True


@torch.inference_mode()
def _draft_action_chunk(
    *,
    adapter: PI0FastTokenLogitAdapter,
    action_head,
    batch: dict[str, Any],
    init_actions: np.ndarray,
    refine_steps: int,
    device: torch.device,
    freeze_gripper: bool,
    target_space: str,
    policy_postprocessor,
    verify_fast_tokens: bool,
    min_accepted_fast_prefix_ratio: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    hidden = adapter.fast_prefix_hidden(batch)
    init = torch.as_tensor(init_actions, dtype=torch.float32, device=device).unsqueeze(0)
    draft, _conf, stats = action_head.draft(hidden.to(device), refine_steps=refine_steps, init=init)
    if verify_fast_tokens:
        draft_tokens = adapter.tokenize_action_chunk(draft[0])
        verify = adapter.verify_draft_tokens(batch, draft_tokens)
        draft_len = int(draft_tokens.shape[-1])
        accepted = int(verify.accepted_prefix)
        stats = {
            **stats,
            "fast_draft_tokens": draft_len,
            "fast_accepted_prefix": accepted,
            "fast_prefix_ratio": accepted / max(draft_len, 1),
            "fast_verified": accepted >= int(np.ceil(min_accepted_fast_prefix_ratio * draft_len)),
        }
    else:
        stats = {**stats, "fast_verified": True}
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if target_space == "raw":
        try:
            processed = policy_postprocessor(draft)
        except Exception:
            processed = draft
        actions = processed[0].detach().cpu().numpy().astype(np.float32)
    else:
        actions = draft[0].detach().cpu().numpy().astype(np.float32)
    if freeze_gripper and actions.shape[-1] >= 7:
        actions[:, 6] = init_actions[: actions.shape[0], 6]
    return actions, elapsed_ms, stats


def run_one(
    *,
    mode: str,
    task_id: int,
    episode: int,
    seed: int,
    env,
    args: argparse.Namespace,
    policy,
    adapter: PI0FastTokenLogitAdapter,
    action_head,
    env_preprocessor,
    env_postprocessor,
    policy_preprocessor,
    policy_postprocessor,
    preprocess_observation,
    device: torch.device,
    target_space: str,
) -> dict[str, Any]:
    policy.reset()
    observation, _info = env.reset(seed=[seed])
    queue: list[np.ndarray] = []
    previous_chunk: np.ndarray | None = None
    drafts_since_target = 0
    steps = 0
    reward_sum = 0.0
    success = False
    target_calls = 0
    draft_calls = 0
    accepted_drafts = 0
    rejected_drafts = 0
    compute_ms = 0.0
    control_ms: list[float] = []
    wall_start = time.perf_counter()
    done = False

    while not done and steps < args.steps:
        step_start = time.perf_counter()
        if not queue:
            use_target = mode == "baseline" or previous_chunk is None or drafts_since_target >= args.max_drafts_per_target
            batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
            if not use_target:
                init_actions = _shift_action_prior(previous_chunk, args.execute_window)
                draft_actions, draft_ms, stats = _draft_action_chunk(
                    adapter=adapter,
                    action_head=action_head,
                    batch=batch,
                    init_actions=init_actions,
                    refine_steps=args.refine_steps,
                    device=device,
                    freeze_gripper=args.freeze_gripper_from_init,
                    target_space=target_space,
                    policy_postprocessor=policy_postprocessor,
                    verify_fast_tokens=args.verify_fast_tokens,
                    min_accepted_fast_prefix_ratio=args.min_accepted_fast_prefix_ratio,
                )
                draft_calls += 1
                compute_ms += draft_ms
                confidence = float(stats.get("mean_confidence", 0.0))
                if (
                    bool(stats.get("fast_verified", False))
                    and confidence >= args.min_confidence
                    and _smooth_guard(draft_actions[: args.execute_window], args)
                ):
                    accepted_drafts += 1
                    previous_chunk = draft_actions.copy()
                    drafts_since_target += 1
                    queue.extend(row.copy() for row in draft_actions[: args.execute_window])
                else:
                    rejected_drafts += 1
                    use_target = True
            if use_target:
                pred = _predict_action_chunk(
                    policy,
                    batch,
                    policy_postprocessor,
                    str(device),
                    token_adapter=adapter,
                    early_stop_action_end=True,
                )
                target_calls += 1
                compute_ms += pred.elapsed_ms
                previous_chunk = pred.actions.copy()
                drafts_since_target = 0
                queue.extend(row.copy() for row in pred.actions[: args.execute_window])
        action = queue.pop(0)
        observation, reward, terminated, truncated, info = _env_step(env, action, env_postprocessor)
        reward_sum += reward
        success = success or _extract_success(info)
        done = terminated or truncated
        steps += 1
        control_ms.append((time.perf_counter() - step_start) * 1000.0)

    wall_s = time.perf_counter() - wall_start
    return {
        "mode": mode,
        "task_id": task_id,
        "episode": episode,
        "seed": seed,
        "success": bool(success),
        "reward_sum": reward_sum,
        "steps": steps,
        "wall_s": wall_s,
        "avg_ms_per_control_step": float(np.mean(control_ms)) if control_ms else 0.0,
        "compute_ms_per_control_step": compute_ms / max(steps, 1),
        "target_calls": target_calls,
        "draft_calls": draft_calls,
        "accepted_drafts": accepted_drafts,
        "rejected_drafts": rejected_drafts,
        "target_calls_per_step": target_calls / max(steps, 1),
    }


def _summarize(rows: list[dict[str, Any]], baseline_mode: str) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    baseline_compute = None
    baseline_success = None
    if baseline_mode in by_mode:
        base = by_mode[baseline_mode]
        baseline_compute = float(np.mean([row["compute_ms_per_control_step"] for row in base]))
        baseline_success = float(np.mean([row["success"] for row in base]))
    out = {"by_mode": {}, "rows": rows}
    for mode, group in sorted(by_mode.items()):
        compute = float(np.mean([row["compute_ms_per_control_step"] for row in group]))
        success = float(np.mean([row["success"] for row in group]))
        out["by_mode"][mode] = {
            "episodes": len(group),
            "successes": int(sum(bool(row["success"]) for row in group)),
            "success_rate": success,
            "compute_ms_per_control_step": compute,
            "speedup_vs_baseline_compute": (baseline_compute / compute) if baseline_compute and compute else None,
            "success_drop_abs_vs_baseline": (baseline_success - success) if baseline_success is not None else None,
            "target_calls_per_step": float(np.mean([row["target_calls_per_step"] for row in group])),
            "accepted_drafts": int(sum(row["accepted_drafts"] for row in group)),
            "rejected_drafts": int(sum(row["rejected_drafts"] for row in group)),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PI0-FAST action-DFlash rollouts")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0-5")
    parser.add_argument("--modes", default="baseline,action_dflash")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--execute-window", type=int, default=3)
    parser.add_argument("--max-drafts-per-target", type=int, default=1)
    parser.add_argument("--refine-steps", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-pos-delta", type=float, default=0.35)
    parser.add_argument("--max-rot-delta", type=float, default=0.50)
    parser.add_argument("--max-gripper-delta", type=float, default=0.1)
    parser.add_argument("--freeze-gripper-from-init", action="store_true")
    parser.add_argument("--verify-fast-tokens", action="store_true")
    parser.add_argument("--min-accepted-fast-prefix-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    task_ids = _parse_ints(args.task_ids)
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
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

    rows: list[dict[str, Any]] = []
    try:
        for task_id in task_ids:
            env = env_map[args.task][task_id]
            for mode in modes:
                for ep in range(args.episodes):
                    seed = args.seed + ep
                    logger.info("task=%d mode=%s episode=%d seed=%d", task_id, mode, ep, seed)
                    row = run_one(
                        mode=mode,
                        task_id=task_id,
                        episode=ep,
                        seed=seed,
                        env=env,
                        args=args,
                        policy=policy,
                        adapter=adapter,
                        action_head=action_head,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        policy_preprocessor=policy_preprocessor,
                        policy_postprocessor=policy_postprocessor,
                        preprocess_observation=preprocess_observation,
                        device=device,
                        target_space=target_space,
                    )
                    rows.append(row)
                    logger.info(
                        "result task=%d mode=%s success=%s compute_ms=%.1f target_calls=%d accepted=%d rejected=%d",
                        task_id,
                        mode,
                        row["success"],
                        row["compute_ms_per_control_step"],
                        row["target_calls"],
                        row["accepted_drafts"],
                        row["rejected_drafts"],
                    )
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
        **_summarize(rows, baseline_mode=modes[0] if modes else "baseline"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["by_mode"], indent=2))


if __name__ == "__main__":
    main()
