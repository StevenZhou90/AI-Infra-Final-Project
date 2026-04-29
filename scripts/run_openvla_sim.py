#!/usr/bin/env python3
"""Run OpenVLA in SimplerEnv with optional EAGLE speculative decoding.

Produces side-by-side video output comparing baseline vs EAGLE inference,
with per-step timing overlay.

Usage:
    python scripts/run_openvla_sim.py --episodes 3 --steps 50
    python scripts/run_openvla_sim.py --episodes 3 --steps 50 --eagle_dir checkpoints/eagle_openvla_v2
    python scripts/run_openvla_sim.py --task widowx_spoon_on_towel --unnorm_key bridge_orig
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
from transforms3d.euler import euler2axangle

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
        attn_implementation="eager",
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


def load_trajectory_decoder(
    model,
    device: str,
    band_radius: int,
    max_residual_bins: float,
    tree_width: int,
    max_tree_depth: int,
    allow_approx_tree: bool,
    fast_draft_only: bool,
    fast_min_confident_tokens: int,
    head_checkpoint: str | None,
    head_threshold: float,
    head_top_k: int,
):
    from serving.trajectory_speculative_decoder import TrajectorySpeculativeDecoder
    from serving.trajectory_draft_head import TinyTrajectoryHead

    draft_head = TinyTrajectoryHead.load(head_checkpoint, device=device) if head_checkpoint else None

    return TrajectorySpeculativeDecoder(
        model=model,
        device=device,
        band_radius=band_radius,
        max_residual_bins=max_residual_bins,
        tree_width=tree_width,
        max_tree_depth=max_tree_depth,
        allow_approx_tree=allow_approx_tree,
        fast_draft_only=fast_draft_only,
        fast_min_confident_tokens=fast_min_confident_tokens,
        draft_head=draft_head,
        head_threshold=head_threshold,
        head_top_k=head_top_k,
    )


def coke_can_variant_kwargs(task: str) -> dict:
    if task == "google_robot_pick_horizontal_coke_can":
        return {"lr_switch": True}
    if task == "google_robot_pick_vertical_coke_can":
        return {"laid_vertically": True}
    if task == "google_robot_pick_standing_coke_can":
        return {"upright": True}
    return {}


def create_env(
    task: str,
    max_episode_steps: int | None,
    use_published_eval_setup: bool,
    obj_init_x: float | None,
    obj_init_y: float | None,
):
    from envs.simpler_env import SimplerEnv, SimplerEnvConfig

    return SimplerEnv(
        SimplerEnvConfig(
            task=task,
            max_episode_steps=max_episode_steps,
            use_published_eval_setup=use_published_eval_setup,
            obj_init_x=obj_init_x,
            obj_init_y=obj_init_y,
            additional_env_build_kwargs=coke_can_variant_kwargs(task) if use_published_eval_setup else None,
        ),
        device="cpu",
    )


def default_unnorm_key(task: str) -> str:
    if task.startswith("google_robot"):
        return "fractal20220817_data"
    if task.startswith("widowx"):
        return "bridge_orig"
    raise ValueError(f"Cannot infer OpenVLA unnorm_key for SimplerEnv task: {task}")


def sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def obs_to_pil(obs: dict, image_size: int = 224) -> "Image.Image":
    """Extract a camera image from the observation dict and convert to PIL."""
    from PIL import Image
    for key in obs:
        if "image" in key:
            t = obs[key]
            if t.dim() == 3 and t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = (t.cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr).resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    raise ValueError(f"No image key in obs: {list(obs.keys())}")


def ensure_openvla_action_prompt_token(inputs, device: str):
    """Add OpenVLA's empty action-prompt token and keep attention_mask aligned."""
    if torch.all(inputs["input_ids"][:, -1] == 29871):
        return inputs

    token = torch.tensor([[29871]], dtype=inputs["input_ids"].dtype, device=device)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], token], dim=1)
    if "attention_mask" in inputs:
        mask_token = torch.ones(
            (inputs["attention_mask"].shape[0], 1),
            dtype=inputs["attention_mask"].dtype,
            device=device,
        )
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], mask_token], dim=1)
    return inputs


def build_prompt(instruction: str, prompt_style: str) -> str:
    if prompt_style == "plain":
        return instruction
    if prompt_style == "official":
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    raise ValueError(f"Unknown prompt_style: {prompt_style}")


class SimplerActionProcessor:
    """Convert raw OpenVLA actions into SimplerEnv controller actions."""

    def __init__(self, task: str, action_scale: float = 1.0) -> None:
        self.policy_setup = "google_robot" if task.startswith("google_robot") else "widowx_bridge"
        self.action_scale = action_scale
        self.sticky_gripper_num_repeat = 15 if self.policy_setup == "google_robot" else 1
        self.reset()

    def reset(self) -> None:
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float64)
        world_vector = raw_action[:3] * self.action_scale

        roll, pitch, yaw = raw_action[3:6]
        rot_axis, rot_angle = euler2axangle(roll, pitch, yaw)
        rot_axangle = np.asarray(rot_axis, dtype=np.float64) * rot_angle * self.action_scale

        open_gripper = raw_action[6:7]
        if self.policy_setup == "google_robot":
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0.0])
            else:
                # Google Robot controller expects relative gripper commands:
                # positive closes, negative opens.
                relative_gripper_action = self.previous_gripper_action - open_gripper
            self.previous_gripper_action = open_gripper

            if np.abs(relative_gripper_action).max() > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            gripper = np.asarray(relative_gripper_action, dtype=np.float64)
        else:
            # Bridge/WidowX controller uses absolute binarized gripper:
            # 1 opens, -1 closes.
            gripper = 2.0 * (open_gripper > 0.5).astype(np.float64) - 1.0

        return np.concatenate([world_vector, rot_axangle, gripper]).astype(np.float32)


def predict_baseline(processor, model, obs, instruction, unnorm_key, prompt_style, image_size, device, dtype):
    """Standard autoregressive inference — returns (action_7dof, elapsed_ms)."""
    pil_image = obs_to_pil(obs, image_size=image_size)
    prompt = build_prompt(instruction, prompt_style)
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)
    inputs = ensure_openvla_action_prompt_token(inputs, device)

    sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            use_cache=True,
        )
    sync_if_cuda(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return action, elapsed_ms


def predict_trajectory_spec(spec_decoder, processor, obs, instruction, unnorm_key, prompt_style, image_size, device, dtype):
    pil_image = obs_to_pil(obs, image_size=image_size)
    prompt = build_prompt(instruction, prompt_style)
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)
    inputs = ensure_openvla_action_prompt_token(inputs, device)

    sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        action = spec_decoder.predict_action(inputs, unnorm_key=unnorm_key)
    sync_if_cuda(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return action, elapsed_ms


def predict_eagle(processor, model, eagle_head, obs, instruction, unnorm_key, prompt_style, image_size, device, dtype, lookahead=1):
    """EAGLE speculative decoding — returns (action_7dof, elapsed_ms)."""
    from serving.kv_cache_manager import trim_kv, kv_seq_len

    pil_image = obs_to_pil(obs, image_size=image_size)
    prompt = build_prompt(instruction, prompt_style)
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)
    inputs = ensure_openvla_action_prompt_token(inputs, device)

    llm = model.language_model
    norm = llm.model.norm
    lm_head = llm.lm_head

    captured = {}
    handle = llm.model.layers[-2].register_forward_hook(
        lambda m, inp, out: captured.update({"h": (out[0] if isinstance(out, tuple) else out).detach()})
    )

    sync_if_cuda(device)
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

    sync_if_cuda(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    handle.remove()

    gen_ids = torch.cat(generated[:max_new], dim=1)
    action_ids = gen_ids[0]
    action = decode_action_tokens(processor, model, action_ids, unnorm_key)
    return action, elapsed_ms


def decode_action_tokens(processor, model, action_ids, unnorm_key):
    predicted_action_token_ids = action_ids.detach().cpu().numpy()
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]

    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"], dtype=np.float32)
    action_low = np.array(action_norm_stats["q01"], dtype=np.float32)
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


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


def info_success(info: dict) -> bool:
    """Normalize success flags returned by different robot env wrappers."""
    for key in ("success", "is_success"):
        if key in info:
            value = info[key]
            if isinstance(value, np.ndarray):
                return bool(value.any())
            return bool(value)
    return False


def run_episode(
    env,
    processor,
    model,
    eagle_head,
    trajectory_decoder,
    decoder,
    seed,
    steps,
    instruction_override,
    unnorm_key,
    prompt_style,
    image_size,
    device,
    dtype,
    lookahead,
):
    obs = env.reset(seed=seed)
    if trajectory_decoder is not None:
        trajectory_decoder.reset()
    baseline_action_processor = SimplerActionProcessor(getattr(env, "_config").task)
    eagle_action_processor = SimplerActionProcessor(getattr(env, "_config").task)
    frames_baseline = []
    frames_eagle = []
    times_baseline = []
    times_eagle = []
    total_reward = 0.0
    success = False
    executed_steps = 0

    for step in range(steps):
        instruction = instruction_override or env.get_language_instruction()
        if decoder == "trajectory-spec":
            raw_action_bl, ms_bl = predict_trajectory_spec(
                trajectory_decoder,
                processor,
                obs,
                instruction,
                unnorm_key,
                prompt_style,
                image_size,
                device,
                dtype,
            )
        else:
            raw_action_bl, ms_bl = predict_baseline(
                processor, model, obs, instruction, unnorm_key, prompt_style, image_size, device, dtype,
            )
        action_bl = baseline_action_processor.process(raw_action_bl)
        times_baseline.append(ms_bl)

        if eagle_head is not None:
            raw_action_ea, ms_ea = predict_eagle(
                processor, model, eagle_head, obs, instruction, unnorm_key, prompt_style, image_size, device, dtype, lookahead,
            )
            action_ea = eagle_action_processor.process(raw_action_ea)
            times_eagle.append(ms_ea)

        frame = env.render()
        if frame is not None:
            frames_baseline.append(frame.copy())
            if eagle_head is not None:
                frames_eagle.append(frame.copy())

        obs, reward, terminated, truncated, info = env.step(action_bl)
        total_reward += float(reward)
        success = success or info_success(info)
        executed_steps = step + 1

        if step % 10 == 0:
            eagle_str = f"  EAGLE: {times_eagle[-1]:.0f}ms" if times_eagle else ""
            logger.info(
                "  step %d/%d  baseline: %.0fms%s  raw=%s  env=%s",
                step + 1,
                steps,
                ms_bl,
                eagle_str,
                np.array2string(raw_action_bl, precision=3, suppress_small=True),
                np.array2string(action_bl, precision=3, suppress_small=True),
            )

        if terminated or truncated:
            break

    metrics = {
        "success": success,
        "reward": total_reward,
        "steps": executed_steps,
        "avg_ms": float(np.mean(times_baseline)) if times_baseline else 0.0,
        "decoder": decoder,
    }
    if trajectory_decoder is not None:
        metrics["spec_stats"] = trajectory_decoder.stats.summary()
    return frames_baseline, frames_eagle, times_baseline, times_eagle, metrics


def save_video(frames, path, fps=50):
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    logger.info("Saved video: %s (%d frames)", path, len(frames))


def main():
    parser = argparse.ArgumentParser(description="Run OpenVLA in SimplerEnv")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--eagle_dir", default=None, help="Path to trained EAGLE checkpoint")
    parser.add_argument("--decoder", choices=["baseline", "trajectory-spec"], default="baseline")
    parser.add_argument("--trajectory-band-radius", type=int, default=2)
    parser.add_argument("--trajectory-max-residual-bins", type=float, default=8.0)
    parser.add_argument("--trajectory-tree-width", type=int, default=8)
    parser.add_argument("--trajectory-max-tree-depth", type=int, default=1)
    parser.add_argument(
        "--trajectory-allow-approx-tree",
        action="store_true",
        help="Enable non-exact multi-token cached tree verification for speed/success experiments.",
    )
    parser.add_argument(
        "--trajectory-fast-draft-only",
        action="store_true",
        help="Use the trajectory head action tokens directly after warmup; fastest but approximate.",
    )
    parser.add_argument("--trajectory-fast-min-confident-tokens", type=int, default=7)
    parser.add_argument("--trajectory-head-checkpoint", default=None)
    parser.add_argument("--trajectory-head-threshold", type=float, default=0.5)
    parser.add_argument("--trajectory-head-top-k", type=int, default=3)
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--instruction", default=None, help="Override SimplerEnv's task instruction")
    parser.add_argument("--unnorm_key", default=None, help="OpenVLA action unnormalization key")
    parser.add_argument("--prompt-style", choices=["official", "plain"], default="plain")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--published-eval-setup", action="store_true")
    parser.add_argument("--obj-init-x", type=float, default=None)
    parser.add_argument("--obj-init-y", type=float, default=None)
    parser.add_argument("--output_dir", default="outputs/openvla_sim")
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    unnorm_key = args.unnorm_key or default_unnorm_key(args.task)
    logger.info("Task: %s  unnorm_key: %s", args.task, unnorm_key)

    processor, model = load_openvla(args.pretrained, args.device, dtype)

    eagle_head = None
    if args.eagle_dir:
        eagle_head = load_eagle(args.eagle_dir, args.device, dtype)
    trajectory_decoder = (
        load_trajectory_decoder(
            model,
            args.device,
            args.trajectory_band_radius,
            args.trajectory_max_residual_bins,
            args.trajectory_tree_width,
            args.trajectory_max_tree_depth,
            args.trajectory_allow_approx_tree,
            args.trajectory_fast_draft_only,
            args.trajectory_fast_min_confident_tokens,
            args.trajectory_head_checkpoint,
            args.trajectory_head_threshold,
            args.trajectory_head_top_k,
        )
        if args.decoder == "trajectory-spec"
        else None
    )

    env = create_env(
        args.task,
        args.steps,
        args.published_eval_setup,
        args.obj_init_x,
        args.obj_init_y,
    )
    out_dir = Path(args.output_dir)

    all_bl_times = []
    all_ea_times = []
    episode_metrics = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        logger.info("=== Episode %d/%d (seed=%d) ===", ep + 1, args.episodes, seed)

        frames_bl, frames_ea, times_bl, times_ea, metrics = run_episode(
            env, processor, model, eagle_head, trajectory_decoder, args.decoder, seed, args.steps,
            args.instruction, unnorm_key, args.prompt_style, args.image_size, args.device, dtype, args.lookahead,
        )
        episode_metrics.append(metrics)

        save_video(frames_bl, out_dir / f"episode_{ep:02d}_baseline.mp4", fps=env.fps)
        if frames_ea:
            save_video(frames_ea, out_dir / f"episode_{ep:02d}_eagle.mp4", fps=env.fps)

        avg_bl = np.mean(times_bl) if times_bl else 0
        all_bl_times.extend(times_bl)
        logger.info(
            "  %s avg: %.1f ms/step  success=%s  reward=%.3f  steps=%d",
            args.decoder,
            avg_bl,
            metrics["success"],
            metrics["reward"],
            metrics["steps"],
        )
        if metrics.get("spec_stats"):
            logger.info("  Spec stats: %s", metrics["spec_stats"])

        if times_ea:
            avg_ea = np.mean(times_ea)
            all_ea_times.extend(times_ea)
            logger.info("  EAGLE avg:    %.1f ms/step (%.2fx speedup)", avg_ea, avg_bl / avg_ea)

    logger.info("=" * 60)
    logger.info("OVERALL RESULTS (%d episodes, %d steps each)", args.episodes, args.steps)
    logger.info("  Baseline: %.1f ms/step avg", np.mean(all_bl_times))
    successes = sum(1 for m in episode_metrics if m["success"])
    logger.info(
        "  Success:  %d/%d (%.1f%%)",
        successes,
        len(episode_metrics),
        100 * successes / max(len(episode_metrics), 1),
    )
    logger.info("  Avg reward: %.3f", np.mean([m["reward"] for m in episode_metrics]))
    if all_ea_times:
        logger.info("  EAGLE:    %.1f ms/step avg", np.mean(all_ea_times))
        logger.info("  Speedup:  %.2fx", np.mean(all_bl_times) / np.mean(all_ea_times))
    logger.info("  Videos in: %s", out_dir)
    logger.info("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
