#!/usr/bin/env python3
"""Benchmark masked-block speculative PI0-FAST decode latency."""

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
from serving.pi0fast_block_gate import load_block_gate  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger("benchmark_pi0fast_block_drafter_latency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PI0-FAST masked-block speculative decode latency")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-checkpoint", default="")
    parser.add_argument("--gate-threshold", type=float, default=None)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--lookahead", type=int, default=7, help="Future tokens drafted after the known target token.")
    parser.add_argument("--min-draft-confidence", type=float, default=0.0)
    parser.add_argument("--min-verify-confidence", type=float, default=0.0)
    parser.add_argument("--min-verify-margin", type=float, default=0.0)
    parser.add_argument(
        "--max-future-accept",
        type=int,
        default=-1,
        help="Maximum future tokens accepted after the known target token; -1 leaves uncapped.",
    )
    parser.add_argument(
        "--min-future-accept",
        type=int,
        default=0,
        help="Reject a speculative block unless at least this many future tokens match.",
    )
    parser.add_argument("--min-spec-position", type=int, default=0)
    parser.add_argument(
        "--min-spec-positions",
        default="",
        help="Optional comma-separated sweep. Overrides --min-spec-position and benchmarks each value per chunk.",
    )
    parser.add_argument("--reject-cooldown-steps", type=int, default=0)
    parser.add_argument("--reject-cooldown-after", type=int, default=1)
    parser.add_argument("--spec-fallback-cooldown-steps", type=int, default=0)
    parser.add_argument("--spec-fallback-cooldown-after", type=int, default=0)
    parser.add_argument(
        "--allow-unknown-context",
        action="store_true",
        help="Pad unknown compact-vocab context tokens instead of disabling a speculative draft attempt.",
    )
    parser.add_argument(
        "--repeat-token-draft",
        action="store_true",
        help="Use a zero-cost repeated-token drafter when the target is already in a same-token run.",
    )
    parser.add_argument("--repeat-token-min-run", type=int, default=2)
    parser.add_argument(
        "--repeat-pattern-draft",
        action="store_true",
        help="Use a zero-cost periodic-pattern drafter for repeated FAST-token motifs.",
    )
    parser.add_argument("--repeat-pattern-max-period", type=int, default=8)
    parser.add_argument("--repeat-pattern-min-position", type=int, default=0)
    parser.add_argument(
        "--pattern-only",
        action="store_true",
        help="Only use zero-cost repeat/pattern drafts; skip learned block drafts elsewhere.",
    )
    parser.add_argument(
        "--unverified-pattern-tail",
        action="store_true",
        help="When a periodic FAST-token tail is detected, synthesize the rest of the tail without verifier calls.",
    )
    parser.add_argument(
        "--unverified-pattern-eos",
        action="store_true",
        help="When a periodic FAST-token tail is detected, force action-end immediately without verifier calls.",
    )
    parser.add_argument(
        "--full-block-only",
        action="store_true",
        help="Accept only complete verified speculative blocks; partial rejects fall back to a normal target step.",
    )
    parser.add_argument("--accept-partial-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-steps", type=int, default=1)
    parser.add_argument("--verify-from-scratch", action="store_true")
    parser.add_argument("--resync-accepted-cache", action="store_true")
    parser.add_argument(
        "--draft-after-known-token",
        action="store_true",
        help="Advance the target's known next token before drafting futures, matching block-drafter training state.",
    )
    parser.add_argument("--compile-draft", action="store_true")
    parser.add_argument("--debug-event-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
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

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    block_drafter, token_map, block_summary = load_block_drafter_checkpoint(args.checkpoint, device=device)
    block_gate = None
    gate_threshold = 0.0
    gate_summary = None
    if args.gate_checkpoint:
        block_gate, gate_threshold, gate_summary = load_block_gate(args.gate_checkpoint, device=device)
        if args.gate_threshold is not None:
            gate_threshold = float(args.gate_threshold)
    if args.compile_draft:
        block_drafter = torch.compile(block_drafter, mode="reduce-overhead")
        hidden = torch.zeros((1, block_summary["model_config"]["hidden_dim"]), device=device)
        context = torch.full((1, block_summary["model_config"]["context_len"]), len(token_map), dtype=torch.long, device=device)
        with torch.inference_mode():
            block_drafter.draft(hidden, context, steps=args.lookahead, refine_steps=args.refine_steps)

    min_spec_positions = (
        [int(part.strip()) for part in args.min_spec_positions.split(",") if part.strip()]
        if args.min_spec_positions
        else [int(args.min_spec_position)]
    )

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
            batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            baseline = adapter.predict_action_chunk_with_trace(batch, temperature=0.0, early_stop_action_end=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            baseline_ms = (time.perf_counter() - t0) * 1000

            baseline_actions = _postprocess_action(baseline, policy_postprocessor)
            for min_spec_position in min_spec_positions:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                spec = adapter.predict_action_chunk_block_speculative(
                    batch,
                    block_drafter=block_drafter,
                    token_map=token_map,
                    block_gate=block_gate,
                    lookahead=args.lookahead,
                    min_draft_confidence=args.min_draft_confidence,
                    min_verify_confidence=args.min_verify_confidence,
                    min_verify_margin=args.min_verify_margin,
                    block_gate_threshold=gate_threshold,
                    max_future_accept=None if args.max_future_accept < 0 else args.max_future_accept,
                    min_future_accept=args.min_future_accept,
                    min_spec_position=min_spec_position,
                    reject_cooldown_steps=args.reject_cooldown_steps,
                    reject_cooldown_after=args.reject_cooldown_after,
                    spec_fallback_cooldown_steps=args.spec_fallback_cooldown_steps,
                    spec_fallback_cooldown_after=args.spec_fallback_cooldown_after,
                    allow_unknown_context=args.allow_unknown_context,
                    repeat_token_draft=args.repeat_token_draft,
                    repeat_token_min_run=args.repeat_token_min_run,
                    repeat_pattern_draft=args.repeat_pattern_draft,
                    repeat_pattern_max_period=args.repeat_pattern_max_period,
                    repeat_pattern_min_position=args.repeat_pattern_min_position,
                    pattern_only=args.pattern_only,
                    unverified_pattern_tail=args.unverified_pattern_tail,
                    unverified_pattern_eos=args.unverified_pattern_eos,
                    full_block_only=args.full_block_only,
                    accept_partial_blocks=args.accept_partial_blocks,
                    refine_steps=args.refine_steps,
                    verify_from_scratch=args.verify_from_scratch,
                    resync_accepted_cache=args.resync_accepted_cache,
                    draft_after_known_token=args.draft_after_known_token,
                    debug_event_limit=args.debug_event_limit,
                    early_stop_action_end=True,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                spec_ms = (time.perf_counter() - t1) * 1000

                tokens_equal = bool(torch.equal(baseline.token_ids, spec.token_ids))
                first_token_diff = None
                if not tokens_equal:
                    horizon = min(baseline.token_ids.shape[1], spec.token_ids.shape[1])
                    diff = torch.nonzero(
                        baseline.token_ids[0, :horizon] != spec.token_ids[0, :horizon],
                        as_tuple=False,
                    )
                    if diff.numel():
                        pos = int(diff[0].item())
                        first_token_diff = {
                            "pos": pos,
                            "baseline": int(baseline.token_ids[0, pos].item()),
                            "spec": int(spec.token_ids[0, pos].item()),
                        }
                    else:
                        first_token_diff = {
                            "lengths": [int(baseline.token_ids.shape[1]), int(spec.token_ids.shape[1])]
                        }
                spec_actions = _postprocess_action(spec, policy_postprocessor)
                action_diff = _action_diff_summary(baseline_actions, spec_actions)
                row = {
                    "chunk": idx,
                    "min_spec_position": int(min_spec_position),
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
                    "chunk=%d min_pos=%d target_eos=%.1fms block_sd=%.1fms speedup=%.3fx equal=%s stats=%s",
                    idx,
                    min_spec_position,
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

    total_baseline_ms = float(np.sum([row["baseline_ms"] for row in rows])) if rows else None
    total_spec_ms = float(np.sum([row["spec_ms"] for row in rows])) if rows else None
    by_min_spec_position = {}
    for min_spec_position in min_spec_positions:
        group = [row for row in rows if row.get("min_spec_position") == int(min_spec_position)]
        group_baseline_ms = float(np.sum([row["baseline_ms"] for row in group])) if group else None
        group_spec_ms = float(np.sum([row["spec_ms"] for row in group])) if group else None
        by_min_spec_position[str(int(min_spec_position))] = {
            "chunks": len(group),
            "avg_baseline_ms": float(np.mean([row["baseline_ms"] for row in group])) if group else None,
            "avg_spec_ms": float(np.mean([row["spec_ms"] for row in group])) if group else None,
            "aggregate_speedup": group_baseline_ms / group_spec_ms if group_baseline_ms and group_spec_ms else None,
            "all_tokens_equal": all(row["tokens_equal"] for row in group),
            "all_actions_equal": all(
                row["action_diff"]["comparable"]
                and row["action_diff"]["max_abs"] == 0.0
                and row["action_diff"]["gripper_mismatches"] == 0
                for row in group
            ),
        }
    summary = {
        "config": vars(args),
        "block_best_epoch": block_summary.get("best_epoch"),
        "gate_summary": gate_summary,
        "gate_threshold": gate_threshold,
        "by_min_spec_position": by_min_spec_position,
        "chunks": rows,
        "avg_baseline_ms": float(np.mean([row["baseline_ms"] for row in rows])) if rows else None,
        "avg_spec_ms": float(np.mean([row["spec_ms"] for row in rows])) if rows else None,
        "avg_speedup": float(np.mean([row["speedup"] for row in rows if row["speedup"]])) if rows else None,
        "total_baseline_ms": total_baseline_ms,
        "total_spec_ms": total_spec_ms,
        "aggregate_speedup": total_baseline_ms / total_spec_ms if total_baseline_ms and total_spec_ms else None,
        "all_tokens_equal": all(row["tokens_equal"] for row in rows),
        "all_actions_equal": all(
            row["action_diff"]["comparable"]
            and row["action_diff"]["max_abs"] == 0.0
            and row["action_diff"]["gripper_mismatches"] == 0
            for row in rows
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
