"""Run ACT policy rollouts in the ALOHA sim and save videos.

Usage:
    uv run python -m eval.run_rollout
    uv run python -m eval.run_rollout --episodes 10
    uv run python -m eval.run_rollout --episodes 50 --no-video
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

# Auto-set headless rendering for MuJoCo on servers without a display
if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import imageio
import numpy as np
import torch
import yaml

from envs.aloha_env import AlohaEnv, AlohaEnvConfig
from policies.act_policy import ACTPolicyWrapper, ACTPolicyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
)
logger = logging.getLogger("eval")


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run_episode(
    env: AlohaEnv,
    policy: ACTPolicyWrapper,
    seed: int | None = None,
    record_video: bool = True,
) -> dict:
    """Run a single episode. Returns metrics dict and optional list of frames."""
    policy.reset()
    obs = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    steps = 0
    success = False

    t0 = time.perf_counter()
    done = False

    while not done:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if "is_success" in info:
            success = success or bool(info["is_success"])

        if record_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    elapsed = time.perf_counter() - t0
    return {
        "reward": total_reward,
        "steps": steps,
        "success": success,
        "elapsed_s": round(elapsed, 2),
        "fps_sim": round(steps / elapsed, 1) if elapsed > 0 else 0,
        "frames": frames if record_video else [],
    }


def save_video(frames: list[np.ndarray], path: Path, fps: int = 50) -> None:
    if not frames:
        logger.warning("No frames to save")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    logger.info("Saved video: %s  (%d frames, %d fps)", path, len(frames), fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACT policy rollouts in ALOHA sim")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--device", type=str, default=None, help="Torch device (auto-detects CUDA)")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model path")
    parser.add_argument("--task", type=str, default=None, help="Gym task name")
    parser.add_argument("--output-dir", type=str, default="outputs/eval", help="Video output dir")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    env_cfg = AlohaEnvConfig(
        task=args.task or cfg.get("env", {}).get("task", "AlohaTransferCube-v0"),
        obs_type=cfg.get("env", {}).get("obs_type", "pixels_agent_pos"),
        max_episode_steps=cfg.get("env", {}).get("max_episode_steps", 400),
    )
    policy_cfg = ACTPolicyConfig(
        pretrained_path=args.model or cfg.get("policy", {}).get("pretrained_path", "lerobot/act_aloha_sim_transfer_cube_human"),
        device=args.device,
    )

    logger.info("=== ALOHA ACT Evaluation ===")
    logger.info("Task:     %s", env_cfg.task)
    logger.info("Model:    %s", policy_cfg.pretrained_path)
    logger.info("Episodes: %d", args.episodes)
    logger.info("Device:   %s", args.device)

    env = AlohaEnv(env_cfg, device=args.device)
    policy = ACTPolicyWrapper(policy_cfg)

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        logger.info("Episode %d/%d  seed=%d", ep + 1, args.episodes, seed)

        result = run_episode(env, policy, seed=seed, record_video=not args.no_video)
        results.append(result)

        logger.info(
            "  reward=%.2f  steps=%d  success=%s  elapsed=%.1fs",
            result["reward"], result["steps"], result["success"], result["elapsed_s"],
        )

        if not args.no_video and result["frames"]:
            video_path = Path(args.output_dir) / f"episode_{ep:03d}.mp4"
            save_video(result["frames"], video_path, fps=env.fps)

    successes = sum(1 for r in results if r["success"])
    avg_reward = np.mean([r["reward"] for r in results])
    logger.info("=== Summary ===")
    logger.info("Success rate: %d/%d (%.1f%%)", successes, len(results), 100 * successes / len(results))
    logger.info("Avg reward:   %.3f", avg_reward)
    logger.info("Avg steps:    %.1f", np.mean([r["steps"] for r in results]))

    env.close()


if __name__ == "__main__":
    main()
