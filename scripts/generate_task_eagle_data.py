#!/usr/bin/env python3
"""Generate task-specific EAGLE training data from SimplerEnv rollouts.

This is the task-conditioned version of ``generate_eagle_data.py``.  Instead of
random images and generic prompts, it starts a SimplerEnv task, uses the real
task observation image + language instruction, and saves the same shard format
expected by ``scripts/train_eagle.py``.

Per sample saves:
  hidden_states: [7, 4096]  shifted layer[-2] hidden states
  token_ids:     [7]        OpenVLA action tokens
  target_ids:    [7]        next-token targets

Usage:
    python scripts/generate_task_eagle_data.py \
        --task google_robot_pick_coke_can \
        --episodes 10 --steps 50 \
        --out_dir data/eagle_google_robot_pick_coke_can
"""

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
from PIL import Image

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.simpler_env import SimplerEnv, SimplerEnvConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ACTION_DIM = 7


def load_model(pretrained: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    logger.info("Loading OpenVLA from %s ...", pretrained)
    processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        pretrained,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Model loaded: %.2fB params on %s", n_params, device)
    return processor, model


def get_llm(model):
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def _dynamic_cache_cls():
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache
    return DynamicCache


def normalize_kv_cache(kv):
    """Convert legacy tuple caches to DynamicCache when this Transformers needs it."""
    if kv is None or hasattr(kv, "update"):
        return kv

    DynamicCache = _dynamic_cache_cls()
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(kv)
    return kv


def clone_kv_cache(kv):
    if kv is None:
        return None

    if hasattr(kv, "to_legacy_cache"):
        return normalize_kv_cache(kv.to_legacy_cache())

    if hasattr(kv, "key_cache") and hasattr(kv, "value_cache"):
        DynamicCache = _dynamic_cache_cls()
        cloned = DynamicCache()
        cloned.key_cache = [k.clone() for k in kv.key_cache]
        cloned.value_cache = [v.clone() for v in kv.value_cache]
        return cloned

    return tuple((k.clone(), v.clone()) for k, v in kv)


def default_unnorm_key(task: str) -> str:
    if task.startswith("google_robot"):
        return "fractal20220817_data"
    if task.startswith("widowx"):
        return "bridge_orig"
    raise ValueError(f"Cannot infer OpenVLA unnorm_key for task: {task}")


def obs_to_pil(obs: dict) -> Image.Image:
    for key, value in obs.items():
        if "image" not in key:
            continue
        tensor = value
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        arr = (tensor.cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f"No image key in observation: {list(obs.keys())}")


def decode_action_tokens(processor, model, action_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    strs = processor.batch_decode(
        action_ids.unsqueeze(1) if action_ids.dim() == 1 else action_ids,
        skip_special_tokens=True,
    )
    discrete = []
    for s in strs:
        text = s.strip()
        if text.isdigit():
            discrete.append(min(int(text), 255))
        else:
            nums = [c for c in text if c.isdigit()]
            discrete.append(int("".join(nums)) if nums else 128)
    while len(discrete) < ACTION_DIM:
        discrete.append(128)

    raw = np.array(discrete[:ACTION_DIM], dtype=np.float32)
    if hasattr(model, "norm_stats") and unnorm_key in model.norm_stats:
        stats = model.norm_stats[unnorm_key]["action"]
        lo = np.array(stats.get("q01", stats.get("min", [0] * ACTION_DIM)), dtype=np.float32)
        hi = np.array(stats.get("q99", stats.get("max", [1] * ACTION_DIM)), dtype=np.float32)
        return lo + (raw / 255.0) * (hi - lo)

    logger.warning("Missing norm_stats[%s]; falling back to [-1, 1] action bins", unnorm_key)
    return (raw / 255.0) * 2 - 1


def build_eagle_sample(
    processor,
    model,
    llm,
    captured_hidden: dict,
    obs: dict,
    instruction: str,
    unnorm_key: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[dict, np.ndarray]:
    pil_image = obs_to_pil(obs)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    attention_mask = inputs.get("attention_mask")

    with torch.no_grad():
        prefill_out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
        )

    prefill_kv = normalize_kv_cache(prefill_out.past_key_values)
    prefill_hidden = captured_hidden["h"][:, -1:, :].clone()

    kv_cache = clone_kv_cache(prefill_kv)
    action_tokens = []
    current_logits = prefill_out.logits[:, -1:, :]

    for _ in range(ACTION_DIM):
        tok = current_logits.argmax(dim=-1)
        action_tokens.append(tok)
        with torch.no_grad():
            out = llm(tok, past_key_values=kv_cache, use_cache=True)
        kv_cache = out.past_key_values
        current_logits = out.logits[:, -1:, :]

    action_seq = torch.cat(action_tokens, dim=1)

    with torch.no_grad():
        batch_out = llm(
            action_seq,
            past_key_values=clone_kv_cache(prefill_kv),
            use_cache=False,
        )

    batch_hidden = captured_hidden["h"][0].clone().cpu()
    batch_logits = batch_out.logits[0]

    shifted_hidden = torch.cat(
        [
            prefill_hidden[0].cpu(),
            batch_hidden[: ACTION_DIM - 1],
        ],
        dim=0,
    )
    token_ids = action_seq[0].cpu()
    target_ids = torch.cat(
        [
            action_seq[0, 1:].cpu(),
            batch_logits[-1:].argmax(dim=-1).cpu(),
        ],
        dim=0,
    )

    sample = {
        "hidden_states": shifted_hidden.to(torch.bfloat16),
        "token_ids": token_ids,
        "target_ids": target_ids,
    }
    action = decode_action_tokens(processor, model, token_ids, unnorm_key)
    return sample, action


def save_shard(samples: list[dict], out_dir: Path, shard_idx: int) -> None:
    shard_path = out_dir / f"shard_{shard_idx:04d}.pt"
    torch.save(samples, shard_path)
    logger.info("Saved shard %d (%d samples) -> %s", shard_idx, len(samples), shard_path)


def generate_task_samples(
    processor,
    model,
    task: str,
    episodes: int,
    steps: int,
    out_dir: Path,
    shard_size: int,
    seed: int,
    instruction_override: str | None,
    unnorm_key: str,
    device: str,
    dtype: torch.dtype,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": task,
        "episodes": episodes,
        "steps": steps,
        "seed": seed,
        "instruction_override": instruction_override,
        "unnorm_key": unnorm_key,
        "sample_format": {
            "hidden_states": [ACTION_DIM, "hidden_size"],
            "token_ids": [ACTION_DIM],
            "target_ids": [ACTION_DIM],
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    llm = get_llm(model)
    hook_layer = llm.model.layers[-2]
    captured_hidden = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_hidden["h"] = output[0].detach()
        else:
            captured_hidden["h"] = output.detach()

    handle = hook_layer.register_forward_hook(hook_fn)

    env = SimplerEnv(SimplerEnvConfig(task=task, max_episode_steps=steps), device="cpu")
    shard_idx = 0
    shard_samples = []
    total_samples = 0
    t_start = time.perf_counter()

    try:
        for ep in range(episodes):
            obs = env.reset(seed=seed + ep)
            logger.info("Episode %d/%d seed=%d", ep + 1, episodes, seed + ep)

            for step in range(steps):
                instruction = instruction_override or env.get_language_instruction()
                sample, action = build_eagle_sample(
                    processor=processor,
                    model=model,
                    llm=llm,
                    captured_hidden=captured_hidden,
                    obs=obs,
                    instruction=instruction,
                    unnorm_key=unnorm_key,
                    device=device,
                    dtype=dtype,
                )
                shard_samples.append(sample)
                total_samples += 1

                obs, reward, terminated, truncated, info = env.step(action)

                if total_samples % 10 == 0:
                    elapsed = time.perf_counter() - t_start
                    logger.info(
                        "Progress: %d samples | ep=%d step=%d | reward=%.3f | %.2f samples/s",
                        total_samples,
                        ep + 1,
                        step + 1,
                        reward,
                        total_samples / max(elapsed, 1e-6),
                    )

                if len(shard_samples) >= shard_size:
                    save_shard(shard_samples, out_dir, shard_idx)
                    shard_idx += 1
                    shard_samples = []

                if terminated or truncated:
                    break

        if shard_samples:
            save_shard(shard_samples, out_dir, shard_idx)

    finally:
        handle.remove()
        env.close()

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Task data generation complete: %d samples in %.1fs (%.2f samples/s)",
        total_samples,
        elapsed,
        total_samples / max(elapsed, 1e-6),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate task-specific EAGLE data from SimplerEnv")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shard_size", type=int, default=500)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default=None, help="Override SimplerEnv's task instruction")
    parser.add_argument("--unnorm_key", default=None, help="OpenVLA action unnormalization key")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out_dir or f"data/eagle_{args.task}")
    unnorm_key = args.unnorm_key or default_unnorm_key(args.task)

    processor, model = load_model(args.pretrained, args.device, dtype)
    generate_task_samples(
        processor=processor,
        model=model,
        task=args.task,
        episodes=args.episodes,
        steps=args.steps,
        out_dir=out_dir,
        shard_size=args.shard_size,
        seed=args.seed,
        instruction_override=args.instruction,
        unnorm_key=unnorm_key,
        device=args.device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
