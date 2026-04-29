#!/usr/bin/env python3
"""Generate DAgger-style data for the trajectory draft head.

Rolls out a fast trajectory-head policy, but labels each visited state with
teacher OpenVLA action tokens. This trains on the state distribution induced by
the fast policy rather than only baseline trajectories.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_openvla_sim import (  # noqa: E402
    SimplerActionProcessor,
    build_prompt,
    create_env,
    default_unnorm_key,
    ensure_openvla_action_prompt_token,
    info_success,
    load_openvla,
    obs_to_pil,
)
from serving.trajectory_draft_head import TinyTrajectoryHead, token_ids_to_bins  # noqa: E402
from serving.trajectory_speculative_decoder import (  # noqa: E402
    TrajectorySpeculativeDecoder,
    capture_prefill_hidden_openvla,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_trajectory_head_dagger_data")

ORIENTATION_TO_TASK = {
    "horizontal": "google_robot_pick_horizontal_coke_can",
    "vertical": "google_robot_pick_vertical_coke_can",
    "standing": "google_robot_pick_standing_coke_can",
}
XS = [-0.35, -0.2925, -0.235, -0.1775, -0.12]
YS = [-0.02, 0.09, 0.20, 0.31, 0.42]


def points(sweep: str) -> list[tuple[str, float, float]]:
    ys = [-0.02] if sweep == "mini" else YS
    return [(orientation, x, y) for orientation in ORIENTATION_TO_TASK for y in ys for x in XS]


def make_inputs(processor, obs, instruction, prompt_style, image_size, device, dtype):
    pil_image = obs_to_pil(obs, image_size=image_size)
    prompt = build_prompt(instruction, prompt_style)
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)
    return ensure_openvla_action_prompt_token(inputs, device)


@torch.no_grad()
def teacher_action_ids(model, inputs, action_dim: int) -> torch.Tensor:
    out = model.generate(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=action_dim,
        do_sample=False,
        use_cache=True,
    )
    return out[0, -action_dim:].detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DAgger data for trajectory head")
    parser.add_argument("--policy-head-checkpoint", required=True)
    parser.add_argument("--sweep", choices=["mini", "full"], default="mini")
    parser.add_argument("--out-dir", default="data/trajectory_head_dagger_mini")
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--prompt-style", choices=["plain", "official"], default="plain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-threshold", type=float, default=0.2)
    parser.add_argument("--head-top-k", type=int, default=3)
    parser.add_argument("--fast-min-confident-tokens", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    processor, model = load_openvla(args.pretrained, args.device, dtype)
    unnorm_key = "fractal20220817_data"
    action_dim = model.get_action_dim(unnorm_key)
    draft_head = TinyTrajectoryHead.load(args.policy_head_checkpoint, device=args.device)

    examples = []
    rollouts = []
    t_start = time.perf_counter()
    for rollout_idx, (orientation, x, y) in enumerate(points(args.sweep)):
        task = ORIENTATION_TO_TASK[orientation]
        env = create_env(task, args.steps, True, x, y)
        action_processor = SimplerActionProcessor(task)
        spec = TrajectorySpeculativeDecoder(
            model=model,
            device=args.device,
            draft_head=draft_head,
            head_threshold=args.head_threshold,
            head_top_k=args.head_top_k,
            fast_draft_only=True,
            fast_min_confident_tokens=args.fast_min_confident_tokens,
        )
        obs = env.reset(seed=args.seed)
        executed_bins_hist: list[torch.Tensor] = []
        success = False
        steps = 0
        matches = 0
        logger.info("Rollout %d: %s x=%.4f y=%.4f", rollout_idx, task, x, y)

        for step in range(args.steps):
            instruction = env.get_language_instruction()
            inputs = make_inputs(processor, obs, instruction, args.prompt_style, args.image_size, args.device, dtype)
            teacher_ids = teacher_action_ids(model, inputs, action_dim)
            prefill_kw = {
                key: value
                for key, value in {
                    "pixel_values": inputs.get("pixel_values"),
                    "attention_mask": inputs.get("attention_mask"),
                }.items()
                if value is not None
            }
            _prefill, prefill_h = capture_prefill_hidden_openvla(
                model,
                input_ids=inputs["input_ids"],
                device=torch.device(args.device),
                **prefill_kw,
            )
            executed_ids = spec.generate_action_ids(inputs, unnorm_key=unnorm_key, update_history=True)
            if torch.equal(executed_ids, teacher_ids):
                matches += 1

            if len(executed_bins_hist) >= args.history_size:
                rec = {
                    "history_bins": torch.stack(executed_bins_hist[-args.history_size:]).short(),
                    "target_bins": token_ids_to_bins(teacher_ids, model.vocab_size).short(),
                    "orientation": orientation,
                    "task": task,
                    "obj_init_x": x,
                    "obj_init_y": y,
                    "timestep": step,
                    "rollout_idx": rollout_idx,
                    "source": "dagger",
                }
                rec["prefill_hidden"] = prefill_h[0].detach().float().cpu().to(torch.bfloat16)
                examples.append(rec)

            executed_bins_hist.append(token_ids_to_bins(executed_ids, model.vocab_size).cpu())
            action = spec.decode_action_ids(executed_ids, unnorm_key)
            obs, reward, terminated, truncated, info = env.step(action_processor.process(action))
            success = success or info_success(info)
            steps = step + 1
            if terminated or truncated:
                break

        rollouts.append(
            {
                "orientation": orientation,
                "task": task,
                "obj_init_x": x,
                "obj_init_y": y,
                "success": success,
                "steps": steps,
                "policy_teacher_match_rate": matches / max(steps, 1),
            }
        )
        env.close()
        logger.info("  success=%s steps=%d examples=%d match_rate=%.3f", success, steps, len(examples), matches / max(steps, 1))

    torch.save(
        {
            "examples": examples,
            "history_size": args.history_size,
            "rollouts": rollouts,
            "unnorm_key": unnorm_key,
            "use_prefill_hidden": True,
            "llm_hidden_size": int(model.language_model.config.hidden_size),
            "data_source": "dagger_fast_policy_teacher_labels",
            "policy_head_checkpoint": args.policy_head_checkpoint,
        },
        out_dir / "dataset.pt",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "examples": len(examples),
                "rollouts": len(rollouts),
                "successes": sum(1 for r in rollouts if r["success"]),
                "avg_policy_teacher_match_rate": sum(r["policy_teacher_match_rate"] for r in rollouts) / max(len(rollouts), 1),
                "elapsed_s": time.perf_counter() - t_start,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Saved %d examples to %s", len(examples), out_dir / "dataset.pt")


if __name__ == "__main__":
    main()
