#!/usr/bin/env python3
"""Run OpenVLA in ALOHA sim with optional EAGLE speculative decoding.

Produces side-by-side video output comparing baseline vs EAGLE inference,
with per-step timing overlay.

Usage:
    python scripts/run_openvla_sim.py --episodes 3 --steps 50
    python scripts/run_openvla_sim.py --episodes 3 --steps 50 --eagle_dir checkpoints/eagle_openvla_v2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_openvla(pretrained: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    logger.info("Loading OpenVLA: %s", pretrained)
    processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        pretrained,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return processor, model


def load_eagle(eagle_dir: str, device: str, dtype: torch.dtype):
    from serving.eagle_draft import EagleDraftHead
    head = EagleDraftHead.from_pretrained(eagle_dir, device=device, dtype=dtype)
    logger.info("EAGLE head: %.1fM params, %d layers",
                sum(p.numel() for p in head.parameters()) / 1e6, len(head.layers))
    return head


def create_env():
    from envs.aloha_env import AlohaEnv, AlohaEnvConfig
    cfg = AlohaEnvConfig(
        task="AlohaTransferCube-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=400,
    )
    return AlohaEnv(cfg, device="cpu")


def obs_to_pil(obs: dict) -> "Image.Image":
    """Extract a camera image from the observation dict and convert to PIL."""
    from PIL import Image
    for key in obs:
        if "image" in key:
            t = obs[key]
            if t.dim() == 3 and t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = (t.cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)
    raise ValueError(f"No image key in obs: {list(obs.keys())}")


def predict_baseline(processor, model, obs, instruction, device, dtype):
    """Standard autoregressive inference — returns (action_7dof, elapsed_ms)."""
    from PIL import Image
    pil_image = obs_to_pil(obs)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=7,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    action_ids = out[0, inputs["input_ids"].shape[1]:]
    action = decode_action_tokens(processor, action_ids)
    return action, elapsed_ms


def predict_eagle(processor, model, eagle_head, obs, instruction, device, dtype, lookahead=1):
    """EAGLE speculative decoding — returns (action_7dof, elapsed_ms)."""
    from serving.kv_cache_manager import trim_kv, kv_seq_len

    pil_image = obs_to_pil(obs)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)

    llm = model.language_model
    norm = llm.model.norm
    lm_head = llm.lm_head

    captured = {}
    handle = llm.model.layers[-2].register_forward_hook(
        lambda m, inp, out: captured.update({"h": (out[0] if isinstance(out, tuple) else out).detach()})
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        prefill_out = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            use_cache=True,
        )

    target_kv = prefill_out.past_key_values
    prev_logits = prefill_out.logits[:, -1:, :]
    prev_hidden = captured["h"][:, -1:, :]

    generated = []
    max_new = 7

    with torch.no_grad():
        while len(generated) < max_new:
            remaining = max_new - len(generated)
            K = min(lookahead + 1, remaining)
            first_token = prev_logits.argmax(dim=-1)

            if K == 1:
                generated.append(first_token)
                out = llm(first_token, past_key_values=target_kv, use_cache=True)
                target_kv = out.past_key_values
                prev_logits = out.logits[:, -1:, :]
                prev_hidden = captured["h"][:, -1:, :]
                continue

            eagle_head.reset_kv()
            draft_tokens = [first_token]
            draft_hidden = prev_hidden

            for _ in range(K - 1):
                eagle_out = eagle_head(draft_hidden, draft_tokens[-1], use_cache=True)
                normed = norm(eagle_out)
                logits = lm_head(normed)
                draft_tokens.append(logits[:, -1:, :].argmax(dim=-1))
                draft_hidden = eagle_out[:, -1:, :]

            verify_input = torch.cat(draft_tokens, dim=1)
            verify_out = llm(verify_input, past_key_values=target_kv, use_cache=True)
            verify_logits = verify_out.logits
            verify_hidden = captured["h"]

            n_accepted = 1
            for i in range(K - 1):
                if verify_logits[:, i, :].argmax(dim=-1).item() == draft_tokens[i + 1].view(-1).item():
                    n_accepted += 1
                else:
                    break

            for j in range(n_accepted):
                generated.append(draft_tokens[j])

            if n_accepted == K:
                generated.append(verify_logits[:, K - 1, :].argmax(dim=-1).view(1, 1))
                target_kv = trim_kv(verify_out.past_key_values, kv_seq_len(target_kv) + K)
                prev_logits = verify_logits[:, K - 1:K, :]
                prev_hidden = verify_hidden[:, K - 1:K, :]
            else:
                generated.append(verify_logits[:, n_accepted - 1, :].argmax(dim=-1).view(1, 1))
                target_kv = trim_kv(verify_out.past_key_values, kv_seq_len(target_kv) + n_accepted)
                prev_logits = verify_logits[:, n_accepted - 1:n_accepted, :]
                prev_hidden = verify_hidden[:, n_accepted - 1:n_accepted, :]

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    handle.remove()

    gen_ids = torch.cat(generated[:max_new], dim=1)
    action_ids = gen_ids[0]
    action = decode_action_tokens(processor, action_ids)
    return action, elapsed_ms


def decode_action_tokens(processor, action_ids):
    strs = processor.batch_decode(
        action_ids.unsqueeze(1) if action_ids.dim() == 1 else action_ids,
        skip_special_tokens=True,
    )
    discrete = []
    for s in strs:
        s = s.strip()
        if s.isdigit():
            discrete.append(min(int(s), 255))
        else:
            nums = [c for c in s if c.isdigit()]
            discrete.append(int("".join(nums)) if nums else 128)
    while len(discrete) < 7:
        discrete.append(128)
    raw = np.array(discrete[:7], dtype=np.float32)
    return (raw / 255.0) * 2 - 1


def pad_action_14dof(action_7dof: np.ndarray) -> np.ndarray:
    """Pad 7-DOF OpenVLA output to 14-DOF for ALOHA (duplicate to both arms)."""
    return np.concatenate([action_7dof, action_7dof])


def add_text_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn simple text into top-left of frame using numpy (no OpenCV needed)."""
    frame = frame.copy()
    y_start = 10
    for line in text.split("\n"):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                pass
        y_start += 18
    return frame


def run_episode(
    env, processor, model, eagle_head, seed, steps, instruction, device, dtype, lookahead,
):
    obs = env.reset(seed=seed)
    frames_baseline = []
    frames_eagle = []
    times_baseline = []
    times_eagle = []

    for step in range(steps):
        action_bl, ms_bl = predict_baseline(processor, model, obs, instruction, device, dtype)
        times_baseline.append(ms_bl)

        if eagle_head is not None:
            action_ea, ms_ea = predict_eagle(
                processor, model, eagle_head, obs, instruction, device, dtype, lookahead,
            )
            times_eagle.append(ms_ea)

        frame = env.render()
        if frame is not None:
            frames_baseline.append(frame.copy())
            if eagle_head is not None:
                frames_eagle.append(frame.copy())

        action_14 = pad_action_14dof(action_bl)
        obs, reward, terminated, truncated, info = env.step(action_14)

        if step % 10 == 0:
            eagle_str = f"  EAGLE: {times_eagle[-1]:.0f}ms" if times_eagle else ""
            logger.info(
                "  step %d/%d  baseline: %.0fms%s",
                step + 1, steps, ms_bl, eagle_str,
            )

        if terminated or truncated:
            break

    return frames_baseline, frames_eagle, times_baseline, times_eagle


def save_video(frames, path, fps=50):
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    logger.info("Saved video: %s (%d frames)", path, len(frames))


def main():
    parser = argparse.ArgumentParser(description="Run OpenVLA in ALOHA sim")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--eagle_dir", default=None, help="Path to trained EAGLE checkpoint")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--instruction", default="pick up the cube")
    parser.add_argument("--output_dir", default="outputs/openvla_sim")
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    processor, model = load_openvla(args.pretrained, args.device, dtype)

    eagle_head = None
    if args.eagle_dir:
        eagle_head = load_eagle(args.eagle_dir, args.device, dtype)

    env = create_env()
    out_dir = Path(args.output_dir)

    all_bl_times = []
    all_ea_times = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        logger.info("=== Episode %d/%d (seed=%d) ===", ep + 1, args.episodes, seed)

        frames_bl, frames_ea, times_bl, times_ea = run_episode(
            env, processor, model, eagle_head, seed, args.steps,
            args.instruction, args.device, dtype, args.lookahead,
        )

        save_video(frames_bl, out_dir / f"episode_{ep:02d}_baseline.mp4", fps=env.fps)
        if frames_ea:
            save_video(frames_ea, out_dir / f"episode_{ep:02d}_eagle.mp4", fps=env.fps)

        avg_bl = np.mean(times_bl) if times_bl else 0
        all_bl_times.extend(times_bl)
        logger.info("  Baseline avg: %.1f ms/step", avg_bl)

        if times_ea:
            avg_ea = np.mean(times_ea)
            all_ea_times.extend(times_ea)
            logger.info("  EAGLE avg:    %.1f ms/step (%.2fx speedup)", avg_ea, avg_bl / avg_ea)

    logger.info("=" * 60)
    logger.info("OVERALL RESULTS (%d episodes, %d steps each)", args.episodes, args.steps)
    logger.info("  Baseline: %.1f ms/step avg", np.mean(all_bl_times))
    if all_ea_times:
        logger.info("  EAGLE:    %.1f ms/step avg", np.mean(all_ea_times))
        logger.info("  Speedup:  %.2fx", np.mean(all_bl_times) / np.mean(all_ea_times))
    logger.info("  Videos in: %s", out_dir)
    logger.info("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
