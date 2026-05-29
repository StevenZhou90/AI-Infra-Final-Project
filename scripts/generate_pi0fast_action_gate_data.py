#!/usr/bin/env python3
"""Generate online DFlash candidate rows for an action-space safety gate."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_pi0fast_ngram_latency import _action_diff_summary, _postprocess_action  # noqa: E402
from scripts.generate_pi0fast_eagle_data import _ensure_libero_config  # noqa: E402
from scripts.run_pi0fast_chunk_eval import _env_step, _import_lerobot, _prepare_observation  # noqa: E402
from serving.pi0fast_action_gate import action_gate_feature_values  # noqa: E402
from serving.pi0fast_block_drafter import load_block_drafter_checkpoint  # noqa: E402
from serving.pi0fast_block_gate import load_block_gate  # noqa: E402
from serving.pi0fast_chunking import ChunkGuard, ChunkGuardConfig  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

logger = logging.getLogger("generate_pi0fast_action_gate_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PI0-FAST DFlash action gate training data")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0,1,2,3,4")
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--block-checkpoint", required=True)
    parser.add_argument("--block-gate-checkpoint", default="")
    parser.add_argument("--block-gate-threshold", type=float, default=None)
    parser.add_argument("--block-lookahead", type=int, default=8)
    parser.add_argument("--block-min-draft-confidence", type=float, default=0.0)
    parser.add_argument("--block-min-verify-confidence", type=float, default=0.0)
    parser.add_argument("--block-min-verify-margins", default="0.0")
    parser.add_argument(
        "--block-cutoffs",
        default="",
        help="Optional comma-separated max FAST-token counts. Each candidate is force-ended at that cutoff.",
    )
    parser.add_argument("--block-max-future-accept", type=int, default=4)
    parser.add_argument("--block-min-future-accept", type=int, default=0)
    parser.add_argument("--block-min-spec-position", type=int, default=24)
    parser.add_argument("--block-reject-cooldown-steps", type=int, default=4)
    parser.add_argument("--block-reject-cooldown-after", type=int, default=2)
    parser.add_argument("--block-spec-fallback-cooldown-steps", type=int, default=0)
    parser.add_argument("--block-spec-fallback-cooldown-after", type=int, default=0)
    parser.add_argument("--block-allow-unknown-context", action="store_true")
    parser.add_argument("--block-full-block-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--block-accept-partial-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-refine-steps", type=int, default=2)
    parser.add_argument("--block-verify-from-scratch", action="store_true")
    parser.add_argument("--block-resync-accepted-cache", action="store_true")
    parser.add_argument("--block-draft-after-known-token", action="store_true")
    parser.add_argument("--label-max-abs", type=float, default=0.08)
    parser.add_argument("--label-mean-abs", type=float, default=0.03)
    parser.add_argument("--label-max-pos-l2", type=float, default=0.05)
    parser.add_argument("--label-max-rot-l2", type=float, default=0.25)
    parser.add_argument(
        "--require-guard-accepted",
        action="store_true",
        help="Also require the handcrafted chunk guard to accept the speculative chunk when labeling positives.",
    )
    parser.add_argument("--action-bound", type=float, default=1.05)
    parser.add_argument("--action-jump-threshold", type=float, default=0.18)
    parser.add_argument("--jerk-threshold", type=float, default=0.22)
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _safe_label(diff: dict, guard_accepted: bool, args: argparse.Namespace) -> bool:
    return bool(
        (guard_accepted or not args.require_guard_accepted)
        and diff.get("comparable", False)
        and float(diff.get("max_abs", float("inf"))) <= args.label_max_abs
        and float(diff.get("mean_abs", float("inf"))) <= args.label_mean_abs
        and float(diff.get("max_pos_l2", float("inf"))) <= args.label_max_pos_l2
        and float(diff.get("max_rot_l2", float("inf"))) <= args.label_max_rot_l2
        and int(diff.get("gripper_mismatches", 1)) == 0
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    task_ids = _parse_ints(args.task_ids)
    verify_margins = _parse_floats(args.block_min_verify_margins)
    block_cutoffs = _parse_ints(args.block_cutoffs) if args.block_cutoffs.strip() else [0]
    env_cfg = LiberoEnv(task=args.task, task_ids=task_ids, control_mode=args.control_mode)
    guard = ChunkGuard(
        ChunkGuardConfig(
            action_bound=args.action_bound,
            action_jump_threshold=args.action_jump_threshold,
            jerk_threshold=args.jerk_threshold,
        )
    )

    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    block_drafter, token_map, block_summary = load_block_drafter_checkpoint(args.block_checkpoint, device=device)
    block_gate = None
    block_gate_threshold = 0.0
    block_gate_summary = None
    if args.block_gate_checkpoint:
        block_gate, loaded_threshold, block_gate_summary = load_block_gate(args.block_gate_checkpoint, device=device)
        block_gate_threshold = loaded_threshold if args.block_gate_threshold is None else float(args.block_gate_threshold)
    elif args.block_gate_threshold is not None:
        block_gate_threshold = float(args.block_gate_threshold)

    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with output.open("w") as f:
        try:
            for task_id in task_ids:
                env = env_map[args.task][task_id]
                observation, _info = env.reset(seed=[args.seed])
                guard.reset()
                for chunk_idx in range(args.chunks):
                    batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
                    baseline = adapter.predict_action_chunk_with_trace(batch, temperature=0.0, early_stop_action_end=True)
                    baseline_actions = _postprocess_action(baseline, policy_postprocessor)

                    for verify_margin in verify_margins:
                        for block_cutoff in block_cutoffs:
                            spec_error = None
                            try:
                                spec = adapter.predict_action_chunk_block_speculative(
                                    batch,
                                    block_drafter=block_drafter,
                                    token_map=token_map,
                                    block_gate=block_gate,
                                    lookahead=args.block_lookahead,
                                    min_draft_confidence=args.block_min_draft_confidence,
                                    min_verify_confidence=args.block_min_verify_confidence,
                                    min_verify_margin=verify_margin,
                                    block_gate_threshold=block_gate_threshold,
                                    max_future_accept=None
                                    if args.block_max_future_accept < 0
                                    else args.block_max_future_accept,
                                    min_future_accept=args.block_min_future_accept,
                                    min_spec_position=args.block_min_spec_position,
                                    reject_cooldown_steps=args.block_reject_cooldown_steps,
                                    reject_cooldown_after=args.block_reject_cooldown_after,
                                    spec_fallback_cooldown_steps=args.block_spec_fallback_cooldown_steps,
                                    spec_fallback_cooldown_after=args.block_spec_fallback_cooldown_after,
                                    allow_unknown_context=args.block_allow_unknown_context,
                                    full_block_only=args.block_full_block_only,
                                    accept_partial_blocks=args.block_accept_partial_blocks,
                                    refine_steps=args.block_refine_steps,
                                    verify_from_scratch=args.block_verify_from_scratch,
                                    resync_accepted_cache=args.block_resync_accepted_cache,
                                    draft_after_known_token=args.block_draft_after_known_token,
                                    max_decoding_steps=block_cutoff or None,
                                    force_action_end=block_cutoff > 0,
                                    early_stop_action_end=True,
                                    debug_event_limit=0,
                                )
                                spec_actions = _postprocess_action(spec, policy_postprocessor)
                                diff = _action_diff_summary(baseline_actions, spec_actions)
                                decision = guard.decide(spec_actions, confidence=1.0)
                                label = _safe_label(diff, decision.accepted, args)
                                features = action_gate_feature_values(
                                    spec_actions,
                                    token_count=spec.token_count,
                                    stats=spec.stats,
                                    max_decoding_steps=policy.config.max_decoding_steps,
                                )
                                token_count = spec.token_count
                                stats = spec.stats or {}
                            except Exception as exc:  # noqa: BLE001
                                spec_error = repr(exc)
                                diff = {"comparable": False, "error": spec_error}
                                decision = None
                                label = False
                                features = action_gate_feature_values(
                                    baseline_actions,
                                    token_count=0,
                                    stats={},
                                    max_decoding_steps=policy.config.max_decoding_steps,
                                )
                                token_count = 0
                                stats = {}

                            row = {
                                "task": args.task,
                                "task_id": task_id,
                                "chunk": chunk_idx,
                                "seed": args.seed,
                                "verify_margin": verify_margin,
                                "block_cutoff": int(block_cutoff),
                                "token_count": int(token_count),
                                "baseline_token_count": int(baseline.token_count),
                                "guard_accepted": bool(decision.accepted) if decision is not None else False,
                                "guard_reasons": list(decision.reasons) if decision is not None else ["spec_error"],
                                "label": int(label),
                                "diff": diff,
                                "spec_error": spec_error,
                                "spec_stats": stats,
                                "block_summary_epoch": block_summary.get("best_epoch") if isinstance(block_summary, dict) else None,
                                "block_gate_summary": block_gate_summary,
                                **features,
                            }
                            f.write(json.dumps(row) + "\n")
                            rows_written += 1

                    logger.info("task=%d chunk=%d rows=%d", task_id, chunk_idx, rows_written)
                    for action in baseline_actions[: policy.config.n_action_steps]:
                        observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
                        guard.mark_executed(action)
                        if terminated or truncated:
                            observation, _info = env.reset(seed=[args.seed + chunk_idx + 1])
                            guard.reset()
                            break
        finally:
            for envs in env_map.values():
                for env in envs.values():
                    try:
                        env.close()
                    except Exception:
                        pass
    logger.info("Wrote %d rows to %s", rows_written, output)


if __name__ == "__main__":
    main()
