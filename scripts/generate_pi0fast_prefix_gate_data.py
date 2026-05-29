#!/usr/bin/env python3
"""Generate labeled PI0-FAST prefix-cutoff safety rows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_pi0fast_ngram_latency import _action_diff_summary, _postprocess_action  # noqa: E402
from scripts.generate_pi0fast_eagle_data import _ensure_libero_config  # noqa: E402
from scripts.run_pi0fast_chunk_eval import _env_step, _import_lerobot, _prepare_observation  # noqa: E402
from serving.pi0fast_prefix_gate import action_feature_values, load_prefix_gate  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastGenerationTrace, PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

logger = logging.getLogger("generate_pi0fast_prefix_gate_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PI0-FAST prefix gate training data")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0,1,2,3,4")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--cutoffs", default="16,24,32,40,48,56,64,72,80,96,112,128,160,192,224")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument(
        "--execute-policy",
        choices=["baseline", "adaptive_prefix"],
        default="baseline",
        help="Policy used to advance the simulator after labeling the current state.",
    )
    parser.add_argument(
        "--adaptive-prefix-checkpoints",
        default="24,32,40,48,56,64",
        help="Checkpoints for adaptive_prefix execution policy.",
    )
    parser.add_argument("--adaptive-stable-tolerance", type=float, default=0.0)
    parser.add_argument("--adaptive-stable-checks", type=int, default=1)
    parser.add_argument("--adaptive-prefix-gate-checkpoint", default="")
    parser.add_argument("--adaptive-prefix-gate-threshold", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


def _parse_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _token_stats(logits: torch.Tensor, token_ids: torch.Tensor) -> dict[str, float]:
    steps = min(logits.shape[1], token_ids.shape[1])
    if steps == 0:
        return {"logprob_mean": 0.0, "logprob_min": 0.0, "entropy_mean": 0.0, "entropy_max": 0.0}
    logits = logits[:, :steps, :].float()
    token_ids = token_ids[:, :steps].to(logits.device)
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return {
        "logprob_mean": float(log_probs.mean().item()),
        "logprob_min": float(log_probs.min().item()),
        "entropy_mean": float(entropy.mean().item()),
        "entropy_max": float(entropy.max().item()),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    task_ids = _parse_ids(args.task_ids)
    cutoffs = _parse_ids(args.cutoffs)
    adaptive_checkpoints = _parse_ids(args.adaptive_prefix_checkpoints)
    env_cfg = LiberoEnv(task=args.task, task_ids=task_ids, control_mode=args.control_mode)

    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    prefix_gate = None
    prefix_gate_threshold = args.adaptive_prefix_gate_threshold
    if args.adaptive_prefix_gate_checkpoint:
        prefix_gate, checkpoint_threshold, _summary = load_prefix_gate(args.adaptive_prefix_gate_checkpoint, device=device)
        if prefix_gate_threshold is None:
            prefix_gate_threshold = checkpoint_threshold
        logger.info(
            "Loaded execution prefix gate %s threshold=%.3f",
            args.adaptive_prefix_gate_checkpoint,
            prefix_gate_threshold,
        )
    elif args.execute_policy == "adaptive_prefix" and prefix_gate_threshold is None:
        prefix_gate_threshold = 0.98
    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    action_end = adapter.action_end_token_id
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with output.open("w") as f:
        try:
            for task_id in task_ids:
                env = env_map[args.task][task_id]
                observation, _info = env.reset(seed=[args.seed])
                for chunk_idx in range(args.chunks):
                    batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
                    baseline = adapter.predict_action_chunk_with_trace(batch, temperature=0.0, early_stop_action_end=True)
                    baseline_actions = _postprocess_action(baseline, policy_postprocessor)
                    baseline_tokens = int(baseline.token_ids.shape[1])
                    exec_trace = baseline
                    exec_actions = baseline_actions
                    exec_error = None
                    if args.execute_policy == "adaptive_prefix":
                        try:
                            exec_trace = adapter.predict_action_chunk_adaptive_prefix_cutoff(
                                batch,
                                checkpoints=adaptive_checkpoints,
                                stable_tolerance=args.adaptive_stable_tolerance,
                                stable_checks=args.adaptive_stable_checks,
                                prefix_gate=prefix_gate,
                                prefix_gate_threshold=float(prefix_gate_threshold),
                                early_stop_action_end=True,
                            )
                            exec_actions = _postprocess_action(exec_trace, policy_postprocessor)
                        except Exception as exc:  # noqa: BLE001
                            exec_error = repr(exc)
                            exec_trace = baseline
                            exec_actions = baseline_actions

                    for cutoff in cutoffs:
                        keep = min(cutoff, baseline_tokens)
                        prefix = baseline.token_ids[:, :keep]
                        forced_eos = int(prefix.shape[1] == 0 or int(prefix[0, -1].item()) != action_end)
                        if forced_eos:
                            eos = torch.tensor([[action_end]], dtype=prefix.dtype, device=prefix.device)
                            prefix = torch.cat([prefix, eos], dim=1)
                        try:
                            decoded = adapter._detokenize_generated_actions(prefix)
                            trace = PI0FastGenerationTrace(actions=decoded, token_ids=prefix, logits=torch.empty(0, device=device))
                            actions = _postprocess_action(trace, policy_postprocessor)
                            diff = _action_diff_summary(baseline_actions, actions)
                            label = bool(diff["comparable"] and diff["max_abs"] == 0.0 and diff["gripper_mismatches"] == 0)
                        except Exception as exc:  # noqa: BLE001
                            actions = np.zeros_like(baseline_actions)
                            diff = {"comparable": False, "error": repr(exc)}
                            label = False

                        row = {
                            "task": args.task,
                            "task_id": task_id,
                            "chunk": chunk_idx,
                            "execute_policy": args.execute_policy,
                            "cutoff": cutoff,
                            "tokens": int(prefix.shape[1]),
                            "baseline_tokens": baseline_tokens,
                            "executed_tokens": int(exec_trace.token_ids.shape[1]),
                            "execution_error": exec_error,
                            "cutoff_norm": cutoff / 256.0,
                            "token_count_norm": int(prefix.shape[1]) / 256.0,
                            "forced_eos": forced_eos,
                            **_token_stats(baseline.logits, prefix),
                            **action_feature_values(actions),
                            "label": int(label),
                            "diff": diff,
                        }
                        f.write(json.dumps(row) + "\n")
                        rows_written += 1

                    logger.info(
                        "task=%d chunk=%d tokens=%d executed_tokens=%d rows=%d",
                        task_id,
                        chunk_idx,
                        baseline_tokens,
                        int(exec_trace.token_ids.shape[1]),
                        rows_written,
                    )
                    for action in exec_actions[: policy.config.n_action_steps]:
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
    logger.info("Wrote %d rows to %s", rows_written, output)


if __name__ == "__main__":
    main()
