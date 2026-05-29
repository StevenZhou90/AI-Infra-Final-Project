#!/usr/bin/env python3
"""Scan how many PI0-FAST generated tokens are needed to preserve actions."""

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
from serving.pi0fast_token_hooks import PI0FastGenerationTrace, PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

logger = logging.getLogger("scan_pi0fast_action_prefix_cutoff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan action-equivalent PI0-FAST prefix cutoffs")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--cutoffs", default="16,24,32,40,48,56,64,72,80,96,112,128")
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
    cutoffs = [int(part.strip()) for part in args.cutoffs.split(",") if part.strip()]

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
    action_end = adapter.action_end_token_id
    try:
        for chunk_idx in range(args.chunks):
            batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
            baseline = adapter.predict_action_chunk_with_trace(batch, temperature=0.0, early_stop_action_end=True)
            baseline_actions = _postprocess_action(baseline, policy_postprocessor)
            chunk_rows = []
            for cutoff in cutoffs:
                keep = min(cutoff, int(baseline.token_ids.shape[1]))
                prefix = baseline.token_ids[:, :keep]
                if int(prefix[0, -1].item()) != action_end:
                    eos = torch.tensor([[action_end]], dtype=prefix.dtype, device=prefix.device)
                    prefix = torch.cat([prefix, eos], dim=1)
                try:
                    decoded = adapter._detokenize_generated_actions(prefix)
                    trace = PI0FastGenerationTrace(actions=decoded, token_ids=prefix, logits=torch.empty(0, device=device))
                    actions = _postprocess_action(trace, policy_postprocessor)
                    diff = _action_diff_summary(baseline_actions, actions)
                except Exception as exc:  # noqa: BLE001
                    diff = {"comparable": False, "error": repr(exc)}
                chunk_rows.append({"cutoff": cutoff, "tokens": int(prefix.shape[1]), "diff": diff})
            rows.append(
                {
                    "chunk": chunk_idx,
                    "baseline_tokens": int(baseline.token_ids.shape[1]),
                    "cutoffs": chunk_rows,
                }
            )
            logger.info("chunk=%d tokens=%d", chunk_idx, int(baseline.token_ids.shape[1]))
            for action in baseline_actions[: policy.config.n_action_steps]:
                observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
                if terminated or truncated:
                    observation, _info = env.reset(seed=[args.seed + chunk_idx + 1])
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass

    summary = {"config": vars(args), "chunks": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
