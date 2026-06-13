#!/usr/bin/env python3
"""Benchmark exact small-transformer speculative PI0-FAST decode latency."""

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
from serving.pi0fast_draft_transformer import load_draft_transformer_checkpoint  # noqa: E402
from serving.pi0fast_prefix_gate import load_prefix_gate  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger("benchmark_pi0fast_draft_transformer_latency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PI0-FAST draft-transformer speculative decode latency")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--prefix-cutoff", type=int, default=0)
    parser.add_argument("--adaptive-prefix-checkpoints", default="")
    parser.add_argument("--adaptive-stable-tolerance", type=float, default=0.0)
    parser.add_argument("--adaptive-stable-checks", type=int, default=1)
    parser.add_argument("--adaptive-early-stable-checks", type=int, default=0)
    parser.add_argument("--adaptive-early-max-stable-checkpoint", type=int, default=0)
    parser.add_argument("--adaptive-max-stable-checkpoint", type=int, default=0)
    parser.add_argument("--adaptive-skip-unproductive-checks", action="store_true")
    parser.add_argument("--adaptive-skip-unproductive-after-checkpoint", type=int, default=0)
    parser.add_argument("--adaptive-continue-to-action-end-on-unstable", action="store_true")
    parser.add_argument("--adaptive-prefix-gate-checkpoint", default=None)
    parser.add_argument("--adaptive-prefix-gate-threshold", type=float, default=0.98)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--min-draft-confidence", type=float, default=0.0)
    parser.add_argument("--min-spec-position", type=int, default=0)
    parser.add_argument("--accept-partial-blocks", action="store_true")
    parser.add_argument("--compile-draft", action="store_true")
    parser.add_argument("--debug-event-limit", type=int, default=0)
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
    draft_model, token_map, draft_summary = load_draft_transformer_checkpoint(args.checkpoint, device=device)
    prefix_gate = None
    prefix_gate_summary = None
    if args.adaptive_prefix_gate_checkpoint:
        prefix_gate, gate_threshold, prefix_gate_summary = load_prefix_gate(args.adaptive_prefix_gate_checkpoint, device=device)
        if args.adaptive_prefix_gate_threshold is None:
            args.adaptive_prefix_gate_threshold = gate_threshold
    if args.compile_draft:
        draft_model = torch.compile(draft_model, mode="reduce-overhead")
        hidden = torch.zeros((1, draft_model.config.hidden_dim), device=device)
        context = torch.full((1, draft_model.config.context_len), len(token_map), dtype=torch.long, device=device)
        with torch.inference_mode():
            draft_model.draft(hidden, context, steps=args.lookahead)

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

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if args.adaptive_prefix_checkpoints:
                checkpoints = [int(part.strip()) for part in args.adaptive_prefix_checkpoints.split(",") if part.strip()]
                spec = adapter.predict_action_chunk_adaptive_prefix_cutoff(
                    batch,
                    checkpoints=checkpoints,
                    stable_tolerance=args.adaptive_stable_tolerance,
                    stable_checks=args.adaptive_stable_checks,
                    early_stable_checks=args.adaptive_early_stable_checks or None,
                    early_max_stable_checkpoint=args.adaptive_early_max_stable_checkpoint or None,
                    max_stable_checkpoint=args.adaptive_max_stable_checkpoint or None,
                    skip_unproductive_checks=args.adaptive_skip_unproductive_checks,
                    skip_unproductive_after_checkpoint=args.adaptive_skip_unproductive_after_checkpoint,
                    continue_to_action_end_on_unstable=args.adaptive_continue_to_action_end_on_unstable,
                    prefix_gate=prefix_gate,
                    prefix_gate_threshold=args.adaptive_prefix_gate_threshold,
                    early_stop_action_end=True,
                )
            elif args.prefix_cutoff > 0:
                spec = adapter.predict_action_chunk_prefix_cutoff(
                    batch,
                    cutoff_tokens=args.prefix_cutoff,
                    early_stop_action_end=True,
                )
            else:
                spec = adapter.predict_action_chunk_draft_transformer_speculative(
                    batch,
                    draft_model=draft_model,
                    token_map=token_map,
                    lookahead=args.lookahead,
                    min_draft_confidence=args.min_draft_confidence,
                    min_spec_position=args.min_spec_position,
                    accept_partial_blocks=args.accept_partial_blocks,
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
                "chunk=%d target_eos=%.1fms draft_tf=%.1fms speedup=%.3fx equal=%s stats=%s",
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

    total_baseline_ms = float(np.sum([row["baseline_ms"] for row in rows])) if rows else None
    total_spec_ms = float(np.sum([row["spec_ms"] for row in rows])) if rows else None
    summary = {
        "config": vars(args),
        "prefix_gate_summary": prefix_gate_summary,
        "draft_best_epoch": draft_summary.get("best_epoch"),
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
