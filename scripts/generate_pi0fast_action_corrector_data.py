#!/usr/bin/env python3
"""Generate PI0-FAST action-corrector rows from DFlash cutoff candidates."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_pi0fast_ngram_latency import _action_diff_summary, _postprocess_action  # noqa: E402
from scripts.generate_pi0fast_eagle_data import _ensure_libero_config  # noqa: E402
from scripts.run_pi0fast_chunk_eval import _env_step, _import_lerobot, _prepare_observation  # noqa: E402
from serving.pi0fast_block_drafter import load_block_drafter_checkpoint  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

logger = logging.getLogger("generate_pi0fast_action_corrector_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DFlash action-corrector rows")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--block-checkpoint", required=True)
    parser.add_argument("--block-cutoffs", default="48,52,56,60,64")
    parser.add_argument("--block-lookahead", type=int, default=8)
    parser.add_argument("--block-min-spec-position", type=int, default=24)
    parser.add_argument("--block-refine-steps", type=int, default=1)
    parser.add_argument("--block-max-future-accept", type=int, default=-1)
    parser.add_argument("--block-full-block-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--block-accept-partial-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-draft-after-known-token", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _tensor_action(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach().cpu()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    task_ids = _parse_ints(args.task_ids)
    cutoffs = _parse_ints(args.block_cutoffs)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)

    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    block_drafter, token_map, block_summary = load_block_drafter_checkpoint(args.block_checkpoint, device=device)
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
            observation, _info = env.reset(seed=[args.seed])
            for chunk_idx in range(args.chunks):
                batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
                baseline = adapter.predict_action_chunk_with_trace(batch, temperature=0.0, early_stop_action_end=True)
                baseline_actions = _postprocess_action(baseline, policy_postprocessor)
                baseline_actions_t = _tensor_action(baseline_actions)

                for cutoff in cutoffs:
                    spec = adapter.predict_action_chunk_block_speculative(
                        batch,
                        block_drafter=block_drafter,
                        token_map=token_map,
                        lookahead=args.block_lookahead,
                        min_spec_position=args.block_min_spec_position,
                        max_future_accept=None if args.block_max_future_accept < 0 else args.block_max_future_accept,
                        full_block_only=args.block_full_block_only,
                        accept_partial_blocks=args.block_accept_partial_blocks,
                        refine_steps=args.block_refine_steps,
                        draft_after_known_token=args.block_draft_after_known_token,
                        max_decoding_steps=cutoff,
                        force_action_end=True,
                        early_stop_action_end=True,
                        debug_event_limit=0,
                    )
                    spec_actions = _postprocess_action(spec, policy_postprocessor)
                    diff = _action_diff_summary(baseline_actions, spec_actions)
                    hidden = None if spec.stats is None else spec.stats.get("prefix_hidden")
                    if hidden is None:
                        hidden = adapter.fast_prefix_hidden(batch)
                    rows.append(
                        {
                            "task": args.task,
                            "task_id": int(task_id),
                            "chunk": int(chunk_idx),
                            "seed": int(args.seed),
                            "cutoff": int(cutoff),
                            "hidden": hidden.detach().float().cpu().reshape(-1),
                            "init_actions": _tensor_action(spec_actions),
                            "target_actions": baseline_actions_t,
                            "baseline_token_count": int(baseline.token_count),
                            "candidate_token_count": int(spec.token_count),
                            "diff": diff,
                            "candidate_stats": {
                                key: float(value)
                                for key, value in (spec.stats or {}).items()
                                if isinstance(value, (int, float, bool))
                            },
                        }
                    )

                logger.info("task=%d chunk=%d rows=%d", task_id, chunk_idx, len(rows))
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(args),
            "rows": rows,
            "block_summary": block_summary,
            "policy_n_action_steps": int(policy.config.n_action_steps),
        },
        output,
    )
    logger.info("Wrote %d rows to %s", len(rows), output)


if __name__ == "__main__":
    main()
