#!/usr/bin/env python
"""Compare exact PI0-FAST decode with torch.compile on identical LIBERO observations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"
if os.environ.get("MUJOCO_GL") == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _ensure_libero_config,
    _env_step,
    _import_lerobot,
    _prepare_observation,
    _to_numpy_action,
)
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--action-vocab-size", type=int, default=None)
    parser.add_argument("--text-vocab-size", type=int, default=None)
    parser.add_argument("--structural-token-radius", type=int, default=512)
    parser.add_argument(
        "--full-head-margin",
        type=float,
        default=None,
        help=(
            "Forwarded to the constrained decoder. Use 0.0 to record top-2 "
            "restricted-head margins without triggering full-head fallback."
        ),
    )
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pi0fast_system_components/pi0fast_compile_compare.json"),
    )
    return parser.parse_args()


def load_dotenv_token() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"} and value:
            os.environ.setdefault("HF_TOKEN", value)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", value)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", value)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_ms(device: torch.device, fn):
    sync_if_cuda(device)
    start = time.perf_counter()
    out = fn()
    sync_if_cuda(device)
    return out, (time.perf_counter() - start) * 1000.0


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def pi0fast_language_model(policy):
    return policy.model.paligemma_with_expert.paligemma.model.language_model


def compare_tokens(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_flat = a.detach().flatten().cpu()
    b_flat = b.detach().flatten().cpu()
    common = min(int(a_flat.numel()), int(b_flat.numel()))
    first_diff = None
    for idx in range(common):
        if int(a_flat[idx]) != int(b_flat[idx]):
            first_diff = {
                "index": idx,
                "baseline": int(a_flat[idx]),
                "compiled": int(b_flat[idx]),
            }
            break
    if first_diff is None and int(a_flat.numel()) != int(b_flat.numel()):
        first_diff = {
            "index": common,
            "baseline": None if common >= int(a_flat.numel()) else int(a_flat[common]),
            "compiled": None if common >= int(b_flat.numel()) else int(b_flat[common]),
        }
    return {
        "equal": bool(torch.equal(a_flat, b_flat)),
        "baseline_len": int(a_flat.numel()),
        "compiled_len": int(b_flat.numel()),
        "first_diff": first_diff,
    }


def main() -> None:
    args = parse_args()
    load_dotenv_token()
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = (
        _import_lerobot()
    )

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    policy_config = PreTrainedConfig.from_pretrained(args.policy)
    if hasattr(policy_config, "device"):
        policy_config.device = str(device)
    if hasattr(policy_config, "dtype"):
        policy_config.dtype = args.dtype
    if args.disable_gradient_checkpointing and hasattr(policy_config, "gradient_checkpointing"):
        policy_config.gradient_checkpointing = False

    policy = PI0FastPolicy.from_pretrained(args.policy, config=policy_config).to(device=device, dtype=dtype).eval()
    if args.disable_gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
        policy.eval()

    adapter = PI0FastTokenLogitAdapter(policy)
    action_kwargs = {
        "constrained_action_vocab": True,
        "constrained_action_vocab_size": args.action_vocab_size,
        "constrained_text_vocab_size": args.text_vocab_size,
        "constrained_structural_token_radius": args.structural_token_radius,
        "constrained_full_head_margin": args.full_head_margin,
    }

    language_model = pi0fast_language_model(policy)
    original_forward = language_model.forward
    compiled_forward = torch.compile(original_forward, mode=args.compile_mode)

    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_cfg = LiberoEnv(task=args.task, task_ids=[args.task_id], control_mode=args.control_mode)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = env_map[args.task][args.task_id]
    observation, _info = env.reset(seed=[args.seed])

    def run_decode(batch, forward):
        language_model.forward = forward
        return adapter.predict_action_chunk_action_end(batch, **action_kwargs)

    # Compile and warm the candidate on the initial observation, then restore baseline forward.
    warm_batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
    for _ in range(args.warmup):
        with torch.inference_mode():
            _ = run_decode(warm_batch, compiled_forward)
    language_model.forward = original_forward

    rows: list[dict[str, Any]] = []
    baseline_ms: list[float] = []
    compiled_ms: list[float] = []
    token_equal_count = 0
    action_max_diffs: list[float] = []

    for step in range(args.steps):
        batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)
        with torch.inference_mode():
            baseline_trace, baseline_elapsed = timed_ms(device, lambda: run_decode(batch, original_forward))
            compiled_trace, compiled_elapsed = timed_ms(device, lambda: run_decode(batch, compiled_forward))
        language_model.forward = original_forward

        token_cmp = compare_tokens(baseline_trace.token_ids, compiled_trace.token_ids)
        action_diff = float(torch.max(torch.abs(baseline_trace.actions - compiled_trace.actions)).item())
        token_equal_count += int(token_cmp["equal"])
        action_max_diffs.append(action_diff)
        baseline_ms.append(float(baseline_elapsed))
        compiled_ms.append(float(compiled_elapsed))

        rows.append(
            {
                "step": step,
                "baseline_ms": float(baseline_elapsed),
                "compiled_ms": float(compiled_elapsed),
                "speedup": float(baseline_elapsed / compiled_elapsed) if compiled_elapsed > 0 else 0.0,
                "token_compare": token_cmp,
                "action_max_abs_diff": action_diff,
                "baseline_stats": baseline_trace.stats or {},
                "compiled_stats": compiled_trace.stats or {},
            }
        )

        try:
            processed = policy_postprocessor(baseline_trace.actions)
        except Exception:
            processed = baseline_trace.actions
        action = _to_numpy_action(processed)[0]
        observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
        if terminated or truncated:
            observation, _info = env.reset(seed=[args.seed + step + 1])

    summary = {
        "config": {
            "policy": args.policy,
            "task": args.task,
            "task_id": args.task_id,
            "seed": args.seed,
            "warmup": args.warmup,
            "steps": args.steps,
            "device": str(device),
            "dtype": args.dtype,
            "compile_mode": args.compile_mode,
            "disable_gradient_checkpointing": bool(args.disable_gradient_checkpointing),
            "full_head_margin": args.full_head_margin,
        },
        "baseline_latency": summarize(baseline_ms),
        "compiled_latency": summarize(compiled_ms),
        "aggregate_speedup": float(sum(baseline_ms) / sum(compiled_ms)) if compiled_ms else 0.0,
        "token_equal_rate": float(token_equal_count / args.steps) if args.steps else 0.0,
        "action_max_abs_diff": float(max(action_max_diffs)) if action_max_diffs else 0.0,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(json.dumps(jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
