#!/usr/bin/env python3
"""Generate supervised data for TinyTrajectoryHead from OpenVLA rollouts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
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
from serving.trajectory_draft_head import token_ids_to_bins  # noqa: E402
from serving.trajectory_speculative_decoder import TrajectorySpeculativeDecoder, capture_prefill_hidden_openvla  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_trajectory_head_data")

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


@torch.no_grad()
def generate_action_tokens(
    processor,
    model,
    obs,
    instruction,
    prompt_style,
    image_size,
    device,
    dtype,
    unnorm_key,
    capture_prefill: bool,
):
    """Greedy action tokens; optional prefill hidden (layer -2, last position)."""
    pil_image = obs_to_pil(obs, image_size=image_size)
    prompt = build_prompt(instruction, prompt_style)
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)
    inputs = ensure_openvla_action_prompt_token(inputs, device)
    prefill_kw = {
        key: value
        for key, value in {
            "pixel_values": inputs.get("pixel_values"),
            "attention_mask": inputs.get("attention_mask"),
        }.items()
        if value is not None
    }

    if capture_prefill:
        prefill, prefill_hidden = capture_prefill_hidden_openvla(
            model,
            input_ids=inputs["input_ids"],
            device=torch.device(device),
            **prefill_kw,
        )
        llm = model.language_model
        past = prefill.past_key_values
        last_logits = prefill.logits[:, -1:, :]
        gen: list[torch.Tensor] = []
        dim = model.get_action_dim(unnorm_key)
        for _ in range(dim):
            next_id = last_logits.argmax(dim=-1)
            gen.append(next_id)
            out = llm(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            last_logits = out.logits[:, -1:, :]
        stacked = torch.cat(gen, dim=1)
        action_ids = stacked[0].detach().cpu()
        prefill_h = prefill_hidden[0].detach().float().cpu()
    else:
        out = model.generate(
            inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=model.get_action_dim(unnorm_key),
            do_sample=False,
            use_cache=True,
        )
        action_ids = out[0, -model.get_action_dim(unnorm_key) :]
        prefill_h = None

    decoder = TrajectorySpeculativeDecoder(model=model, device=device)
    action = decoder.decode_action_ids(action_ids, unnorm_key)
    return action_ids.detach().cpu(), action, prefill_h


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TinyTrajectoryHead rollout data")
    parser.add_argument("--sweep", choices=["mini", "full"], default="mini")
    parser.add_argument("--out-dir", default="data/trajectory_head_mini")
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--no-prefill-hidden", action="store_true", help="Do not store VLM prefill hidden (smaller data, history-only head)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--prompt-style", choices=["plain", "official"], default="plain")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    processor, model = load_openvla(args.pretrained, args.device, dtype)
    unnorm_key = "fractal20220817_data"

    examples = []
    rollouts = []
    t_start = time.perf_counter()
    for rollout_idx, (orientation, x, y) in enumerate(points(args.sweep)):
        task = ORIENTATION_TO_TASK[orientation]
        env = create_env(task, args.steps, True, x, y)
        action_processor = SimplerActionProcessor(task)
        obs = env.reset(seed=42)
        history_bins: list[torch.Tensor] = []
        success = False
        steps = 0
        logger.info("Rollout %d: %s x=%.4f y=%.4f", rollout_idx, task, x, y)

        for step in range(args.steps):
            instruction = env.get_language_instruction()
            action_ids, raw_action, prefill_h = generate_action_tokens(
                processor,
                model,
                obs,
                instruction,
                args.prompt_style,
                args.image_size,
                args.device,
                dtype,
                unnorm_key,
                capture_prefill=not args.no_prefill_hidden,
            )
            bins = token_ids_to_bins(action_ids, model.vocab_size).cpu()
            if len(history_bins) >= args.history_size:
                rec = {
                    "history_bins": torch.stack(history_bins[-args.history_size :]).short(),
                    "target_bins": bins.short(),
                    "orientation": orientation,
                    "task": task,
                    "obj_init_x": x,
                    "obj_init_y": y,
                    "timestep": step,
                    "rollout_idx": rollout_idx,
                }
                if prefill_h is not None:
                    rec["prefill_hidden"] = prefill_h.to(torch.bfloat16)
                examples.append(rec)
            history_bins.append(bins)
            obs, reward, terminated, truncated, info = env.step(action_processor.process(raw_action))
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
            }
        )
        env.close()
        logger.info("  success=%s steps=%d examples=%d", success, steps, len(examples))

    torch.save(
        {
            "examples": examples,
            "history_size": args.history_size,
            "rollouts": rollouts,
            "unnorm_key": unnorm_key,
            "use_prefill_hidden": not args.no_prefill_hidden,
            "llm_hidden_size": int(model.language_model.config.hidden_size),
        },
        out_dir / "dataset.pt",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "examples": len(examples),
                "rollouts": len(rollouts),
                "successes": sum(1 for r in rollouts if r["success"]),
                "elapsed_s": time.perf_counter() - t_start,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Saved %d examples to %s", len(examples), out_dir / "dataset.pt")


if __name__ == "__main__":
    main()
