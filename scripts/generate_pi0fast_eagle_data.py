#!/usr/bin/env python3
"""Collect PI0-FAST hidden-state/FAST-token traces for EAGLE training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _env_step,
    _import_lerobot,
    _prepare_observation,
    _to_numpy_action,
)
from serving.pi0fast_eagle import PI0FastTraceRecord, save_trace_shard  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger("generate_pi0fast_eagle_data")


def _ensure_libero_config(config_path: str | None) -> None:
    """Avoid LIBERO's interactive first-run dataset-path prompt."""

    if config_path:
        os.environ["LIBERO_CONFIG_PATH"] = config_path
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")).expanduser()
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        return

    libero_root = None
    for entry in sys.path:
        candidate = Path(entry) / "libero" / "libero"
        if (candidate / "bddl_files").exists() and (candidate / "init_files").exists():
            libero_root = candidate.resolve()
            break
    if libero_root is None:
        raise RuntimeError("Could not locate installed LIBERO package root before env creation")

    cfg_dir.mkdir(parents=True, exist_ok=True)
    datasets = libero_root.parent / "datasets"
    cfg_file.write_text(
        "\n".join(
            [
                f"assets: {libero_root / 'assets'}",
                f"bddl_files: {libero_root / 'bddl_files'}",
                f"benchmark_root: {libero_root}",
                f"datasets: {datasets}",
                f"init_states: {libero_root / 'init_files'}",
                "",
            ]
        )
    )
    logger.info("Wrote noninteractive LIBERO config to %s", cfg_file)


def _parse_task_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(dict.fromkeys(ids))


def _action_from_trace(trace, postprocessor) -> torch.Tensor:
    try:
        processed = postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return _to_numpy_action(processed)


@torch.inference_mode()
def collect_for_task(
    *,
    args: argparse.Namespace,
    task_id: int,
    env,
    policy,
    adapter: PI0FastTokenLogitAdapter,
    env_preprocessor,
    env_postprocessor,
    policy_preprocessor,
    policy_postprocessor,
    preprocess_observation,
    device: torch.device,
) -> list[PI0FastTraceRecord]:
    records: list[PI0FastTraceRecord] = []
    observation, _info = env.reset(seed=[args.seed + task_id * 1000])
    steps = 0
    resets = 0

    while steps < args.steps_per_task and len(records) < args.traces_per_task:
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
        trace = adapter.predict_action_chunk_with_trace(batch, return_hidden_states=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000
        if trace.hidden_states is None:
            raise RuntimeError("PI0FastTokenLogitAdapter did not return hidden states")

        record = PI0FastTraceRecord(
            hidden_states=trace.hidden_states[0].detach().cpu().to(dtype=torch.float32),
            token_ids=trace.token_ids[0].detach().cpu().to(dtype=torch.long),
            task_id=task_id,
            seed=args.seed + task_id * 1000 + resets,
            trace_id=f"task{task_id}_trace{len(records):05d}",
            decode_ms=decode_ms,
            success=None,
        )
        records.append(record)
        logger.info(
            "task=%d trace=%d tokens=%d decode_ms=%.1f",
            task_id,
            len(records),
            int(record.token_ids.numel()),
            decode_ms,
        )

        actions = _action_from_trace(trace, policy_postprocessor)
        for action in actions[: args.execute_actions_per_trace]:
            observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
            steps += 1
            if terminated or truncated or steps >= args.steps_per_task:
                resets += 1
                observation, _info = env.reset(seed=[args.seed + task_id * 1000 + resets])
                break

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PI0-FAST EAGLE trace shards")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", default="0-4")
    parser.add_argument("--traces-per-task", type=int, default=20)
    parser.add_argument("--steps-per-task", type=int, default=250)
    parser.add_argument("--execute-actions-per-trace", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--output-dir", default="data/pi0fast_eagle")
    parser.add_argument(
        "--libero-config-path",
        default=os.environ.get("LIBERO_CONFIG_PATH"),
        help="Directory for LIBERO config.yaml. Created noninteractively if missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = _parse_task_ids(args.task_ids)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)

    env_cfg = LiberoEnv(task=args.task, task_ids=task_ids, control_mode=args.control_mode)
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

    summary: dict[str, Any] = {"config": vars(args), "tasks": {}}
    total_records = 0
    try:
        for task_id in task_ids:
            env = env_map[args.task][task_id]
            records = collect_for_task(
                args=args,
                task_id=task_id,
                env=env,
                policy=policy,
                adapter=adapter,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                preprocess_observation=preprocess_observation,
                device=device,
            )
            shard_path = out_dir / f"shard_task{task_id:02d}.pt"
            save_trace_shard(records, shard_path)
            total_records += len(records)
            summary["tasks"][str(task_id)] = {
                "records": len(records),
                "shard": str(shard_path),
                "avg_decode_ms": sum(r.decode_ms or 0.0 for r in records) / max(len(records), 1),
            }
    finally:
        for task_envs in env_map.values():
            for env in task_envs.values():
                try:
                    env.close()
                except Exception:
                    pass

    summary["total_records"] = total_records
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
