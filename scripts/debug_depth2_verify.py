#!/usr/bin/env python3
"""Debug depth-2 trajectory tree verification against sequential decode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_spec_exactness import baseline_action_ids, make_inputs  # noqa: E402
from scripts.run_openvla_sim import (  # noqa: E402
    SimplerActionProcessor,
    create_env,
    default_unnorm_key,
    load_openvla,
)
from serving.trajectory_speculative_decoder import TrajectorySpeculativeDecoder  # noqa: E402


def attn_mask(kv, new_tokens: int, batch_size: int, device: torch.device) -> torch.Tensor:
    total = kv[0][0].shape[2] + new_tokens
    return torch.ones(batch_size, total, dtype=torch.long, device=device)


def pos_ids(kv, new_tokens: int, batch_size: int, device: torch.device) -> torch.Tensor:
    start = kv[0][0].shape[2]
    pos = torch.arange(start, start + new_tokens, dtype=torch.long, device=device)
    return pos.unsqueeze(0).expand(batch_size, -1)


def cache_pos(kv, new_tokens: int, device: torch.device) -> torch.Tensor:
    start = kv[0][0].shape[2]
    return torch.arange(start, start + new_tokens, dtype=torch.long, device=device)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="google_robot_pick_horizontal_coke_can")
    parser.add_argument("--obj-init-x", type=float, default=-0.35)
    parser.add_argument("--obj-init-y", type=float, default=-0.02)
    parser.add_argument("--step", type=int, default=9)
    parser.add_argument("--candidate-token", type=int, default=31813)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt-style", default="plain")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--output", default="outputs/debug_depth2_verify.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor, model = load_openvla(args.pretrained, args.device, dtype)
    llm = model.language_model
    unnorm_key = default_unnorm_key(args.task)
    env = create_env(args.task, 80, True, args.obj_init_x, args.obj_init_y)
    action_processor = SimplerActionProcessor(args.task)
    obs = env.reset(seed=args.seed)
    decoder = TrajectorySpeculativeDecoder(model=model, device=args.device)

    base_ids = None
    inputs = None
    for step in range(args.step + 1):
        instruction = env.get_language_instruction()
        inputs = make_inputs(processor, obs, instruction, args.prompt_style, args.image_size, args.device, dtype)
        base_ids = baseline_action_ids(model, inputs, unnorm_key)
        action = decoder.decode_action_ids(base_ids, unnorm_key)
        if step < args.step:
            obs, _reward, terminated, truncated, _info = env.step(action_processor.process(action))
            if terminated or truncated:
                raise RuntimeError(f"Episode ended before debug step {args.step}: ended at {step}")

    assert inputs is not None and base_ids is not None
    prefill = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        attention_mask=inputs.get("attention_mask"),
        use_cache=True,
    )
    kv = prefill.past_key_values
    first = prefill.logits[:, -1:, :].argmax(dim=-1)
    target_second = base_ids[1].view(1, 1).to(device)
    candidate_second = torch.tensor([[args.candidate_token]], dtype=torch.long, device=device)

    single = llm(
        first,
        past_key_values=kv,
        attention_mask=attn_mask(kv, 1, 1, device),
        position_ids=pos_ids(kv, 1, 1, device),
        cache_position=cache_pos(kv, 1, device),
        use_cache=True,
    )
    single_pred = int(single.logits[:, -1, :].argmax(dim=-1).item())

    seq_preds = []
    seq_kv = kv
    seq_next = first
    for _ in range(6):
        out = llm(
            seq_next,
            past_key_values=seq_kv,
            attention_mask=attn_mask(seq_kv, 1, 1, device),
            position_ids=pos_ids(seq_kv, 1, 1, device),
            cache_position=cache_pos(seq_kv, 1, device),
            use_cache=True,
        )
        pred = out.logits[:, -1:, :].argmax(dim=-1)
        seq_preds.append(int(pred.item()))
        seq_kv = out.past_key_values
        seq_next = pred

    rows = []
    for name, second in [("baseline_second", target_second), ("candidate_second", candidate_second)]:
        seq = torch.cat([first, second], dim=1)
        variants = {
            "explicit_all": {
                "attention_mask": attn_mask(kv, 2, 1, device),
                "position_ids": pos_ids(kv, 2, 1, device),
                "cache_position": cache_pos(kv, 2, device),
            },
            "mask_only": {
                "attention_mask": attn_mask(kv, 2, 1, device),
            },
            "pos_only": {
                "position_ids": pos_ids(kv, 2, 1, device),
                "cache_position": cache_pos(kv, 2, device),
            },
            "none": {},
        }
        for variant, kwargs in variants.items():
            multi = llm(
                seq,
                past_key_values=kv,
                use_cache=True,
                **kwargs,
            )
            rows.append(
                {
                    "name": name,
                    "variant": variant,
                    "input": [int(x) for x in seq[0].tolist()],
                    "pred_after_first_from_multi": int(multi.logits[:, 0, :].argmax(dim=-1).item()),
                    "pred_after_second_from_multi": int(multi.logits[:, 1, :].argmax(dim=-1).item()),
                }
            )

    result = {
        "task": args.task,
        "step": args.step,
        "baseline_ids": [int(x) for x in base_ids.tolist()],
        "first_from_prefill": int(first.item()),
        "single_pred_after_first": single_pred,
        "sequential_after_first_preds": seq_preds,
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
