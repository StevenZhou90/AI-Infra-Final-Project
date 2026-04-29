#!/usr/bin/env python3
"""Check whether speculative decoding exactly matches OpenVLA greedy tokens."""

from __future__ import annotations

import argparse
import json
import sys
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
from serving.trajectory_draft_head import TinyTrajectoryHead  # noqa: E402
from serving.trajectory_speculative_decoder import TrajectorySpeculativeDecoder  # noqa: E402


def make_inputs(processor, obs, instruction, prompt_style, image_size, device, dtype):
    pil_image = obs_to_pil(obs, image_size=image_size)
    prompt = build_prompt(instruction, prompt_style)
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)
    return ensure_openvla_action_prompt_token(inputs, device)


@torch.no_grad()
def baseline_action_ids(model, inputs, unnorm_key):
    out = model.generate(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=model.get_action_dim(unnorm_key),
        do_sample=False,
        use_cache=True,
    )
    return out[0, -model.get_action_dim(unnorm_key):].detach().cpu()


def run_case(args, processor, model, task, x, y) -> dict:
    dtype = getattr(torch, args.dtype)
    unnorm_key = args.unnorm_key or default_unnorm_key(task)
    draft_head = TinyTrajectoryHead.load(args.trajectory_head_checkpoint, device=args.device) if args.trajectory_head_checkpoint else None
    spec = TrajectorySpeculativeDecoder(
        model=model,
        device=args.device,
        band_radius=args.trajectory_band_radius,
        max_residual_bins=args.trajectory_max_residual_bins,
        tree_width=args.trajectory_tree_width,
        max_tree_depth=args.trajectory_max_tree_depth,
        allow_approx_tree=args.trajectory_allow_approx_tree,
        fast_draft_only=args.trajectory_fast_draft_only,
        fast_min_confident_tokens=args.trajectory_fast_min_confident_tokens,
        draft_head=draft_head,
        head_threshold=args.trajectory_head_threshold,
        head_top_k=args.trajectory_head_top_k,
    )
    env = create_env(task, args.steps, True, x, y)
    action_processor = SimplerActionProcessor(task)
    obs = env.reset(seed=args.seed)
    rows = []
    success = False

    for step in range(args.steps):
        instruction = env.get_language_instruction()
        inputs = make_inputs(processor, obs, instruction, args.prompt_style, args.image_size, args.device, dtype)
        base_ids = baseline_action_ids(model, inputs, unnorm_key)
        spec_ids = spec.generate_action_ids(inputs, unnorm_key=unnorm_key, update_history=True)
        match = torch.equal(base_ids, spec_ids)
        mismatch_positions = (base_ids != spec_ids).nonzero(as_tuple=False).flatten().tolist()
        rows.append(
            {
                "step": step,
                "match": match,
                "baseline_ids": [int(x) for x in base_ids.tolist()],
                "spec_ids": [int(x) for x in spec_ids.tolist()],
                "mismatch_positions": mismatch_positions,
            }
        )

        action = spec.decode_action_ids(base_ids, unnorm_key)
        obs, reward, terminated, truncated, info = env.step(action_processor.process(action))
        success = success or info_success(info)
        if terminated or truncated:
            break

    env.close()
    mismatches = [r for r in rows if not r["match"]]
    return {
        "task": task,
        "obj_init_x": x,
        "obj_init_y": y,
        "steps": len(rows),
        "success": success,
        "matches": len(rows) - len(mismatches),
        "mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "spec_stats": spec.stats.summary(),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check speculative decoding exactness")
    parser.add_argument("--task", default="google_robot_pick_horizontal_coke_can")
    parser.add_argument("--obj-init-x", type=float, default=-0.35)
    parser.add_argument("--obj-init-y", type=float, default=-0.02)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--unnorm_key", default=None)
    parser.add_argument("--prompt-style", choices=["plain", "official"], default="plain")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trajectory-band-radius", type=int, default=2)
    parser.add_argument("--trajectory-max-residual-bins", type=float, default=8.0)
    parser.add_argument("--trajectory-tree-width", type=int, default=8)
    parser.add_argument("--trajectory-max-tree-depth", type=int, default=1)
    parser.add_argument("--trajectory-allow-approx-tree", action="store_true")
    parser.add_argument("--trajectory-fast-draft-only", action="store_true")
    parser.add_argument("--trajectory-fast-min-confident-tokens", type=int, default=7)
    parser.add_argument("--trajectory-head-checkpoint", default=None)
    parser.add_argument("--trajectory-head-threshold", type=float, default=0.7)
    parser.add_argument("--trajectory-head-top-k", type=int, default=3)
    parser.add_argument("--output", default="outputs/spec_exactness.json")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    processor, model = load_openvla(args.pretrained, args.device, dtype)
    result = run_case(args, processor, model, args.task, args.obj_init_x, args.obj_init_y)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task": result["task"],
                "steps": result["steps"],
                "matches": result["matches"],
                "mismatches": result["mismatches"],
                "first_mismatch": result["first_mismatch"],
                "spec_stats": result["spec_stats"],
                "output": str(out),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["mismatches"] == 0 else 1)


if __name__ == "__main__":
    main()
