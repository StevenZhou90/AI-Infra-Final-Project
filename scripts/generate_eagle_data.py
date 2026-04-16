#!/usr/bin/env python3
"""Generate training data for an EAGLE draft head from frozen OpenVLA.

Uses BATCHED forward passes to capture hidden states — matching how the verify
pass works at inference.  This eliminates the train/inference distribution
mismatch that caused low acceptance with verify-pass hidden states.

Per sample saves:
  hidden_states: [7, 4096]  batched layer[-2] outputs (shifted by 1 position)
  token_ids:     [7]        action tokens
  target_ids:    [7]        next-token targets

Usage:
    python scripts/generate_eagle_data.py --num_samples 10000 --out_dir data/eagle_train_v2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ACTION_DIM = 7

INSTRUCTIONS = [
    "pick up the cube",
    "move the red block to the right",
    "grasp the object on the table",
    "push the blue cylinder forward",
    "lift the cup and place it on the plate",
    "stack the green block on top of the yellow block",
    "slide the box to the left",
    "pick up the bottle",
    "move the object closer",
    "place the item in the bin",
]


def make_random_image(size: int = 224) -> Image.Image:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


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


def generate_samples(
    processor,
    model,
    num_samples: int,
    out_dir: Path,
    shard_size: int,
    device: str,
    dtype: torch.dtype,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = get_llm(model)
    layers = llm.model.layers
    hook_layer = layers[-2]

    captured_hidden = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured_hidden["h"] = output[0].detach()
        else:
            captured_hidden["h"] = output.detach()

    handle = hook_layer.register_forward_hook(hook_fn)

    shard_idx = 0
    shard_samples = []
    t_start = time.perf_counter()

    try:
        for i in range(num_samples):
            instruction = INSTRUCTIONS[i % len(INSTRUCTIONS)]
            pil_image = make_random_image()
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            inputs = processor(prompt, pil_image).to(device, dtype=dtype)

            input_ids = inputs["input_ids"]
            pixel_values = inputs.get("pixel_values")
            attention_mask = inputs.get("attention_mask")

            # --- Step 1: Prefill to get KV cache + first token ---
            with torch.no_grad():
                prefill_out = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                )

            prefill_kv = prefill_out.past_key_values
            prefill_hidden = captured_hidden["h"][:, -1:, :].clone()  # [1,1,H]

            # --- Step 2: Generate 7 action tokens autoregressively ---
            kv_cache = prefill_kv
            action_tokens = []
            current_logits = prefill_out.logits[:, -1:, :]

            for step in range(ACTION_DIM):
                tok = current_logits.argmax(dim=-1)  # [1,1]
                action_tokens.append(tok)
                with torch.no_grad():
                    out = llm(tok, past_key_values=kv_cache, use_cache=True)
                kv_cache = out.past_key_values
                current_logits = out.logits[:, -1:, :]

            # action_tokens = [a0, a1, ..., a6], each [1,1]
            action_seq = torch.cat(action_tokens, dim=1)  # [1, 7]

            # --- Step 3: Batched forward of all 7 tokens (like verify pass) ---
            with torch.no_grad():
                batch_out = llm(action_seq, past_key_values=prefill_kv, use_cache=False)

            batch_hidden = captured_hidden["h"][0].clone().cpu()  # [7, H]
            batch_logits = batch_out.logits[0]                    # [7, vocab]

            # --- Step 4: Build training data with shifted hidden states ---
            # EAGLE's formulation: (hidden_prev, token_cur) → predict token_next
            # hidden_prev[t] = hidden BEFORE token_cur[t] was processed
            # For t=0: hidden_prev = prefill_hidden (before any action tokens)
            # For t>0: hidden_prev = batch_hidden[t-1] (after processing action[t-1])

            shifted_hidden = torch.cat([
                prefill_hidden[0].cpu(),       # [1, H]
                batch_hidden[:6],              # [6, H] — positions 0..5
            ], dim=0)                          # [7, H]

            token_ids = action_seq[0].cpu()    # [7]

            target_ids = torch.cat([
                action_seq[0, 1:].cpu(),                           # [6] — a1..a6
                batch_logits[-1:].argmax(dim=-1).cpu(),            # [1] — pred after a6
            ], dim=0)                          # [7]

            sample = {
                "hidden_states": shifted_hidden.to(torch.bfloat16),
                "token_ids": token_ids,
                "target_ids": target_ids,
            }
            shard_samples.append(sample)

            if len(shard_samples) >= shard_size:
                shard_path = out_dir / f"shard_{shard_idx:04d}.pt"
                torch.save(shard_samples, shard_path)
                logger.info(
                    "Saved shard %d (%d samples) -> %s",
                    shard_idx, len(shard_samples), shard_path,
                )
                shard_idx += 1
                shard_samples = []

            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed
                eta = (num_samples - i - 1) / rate
                logger.info(
                    "Progress: %d/%d (%.1f samples/s, ETA %.0fs)",
                    i + 1, num_samples, rate, eta,
                )

        if shard_samples:
            shard_path = out_dir / f"shard_{shard_idx:04d}.pt"
            torch.save(shard_samples, shard_path)
            logger.info("Saved final shard %d (%d samples)", shard_idx, len(shard_samples))

    finally:
        handle.remove()

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Data generation complete: %d samples in %.1fs (%.1f samples/s)",
        num_samples, elapsed, num_samples / elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate EAGLE training data from OpenVLA")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--shard_size", type=int, default=500)
    parser.add_argument("--out_dir", default="data/eagle_train_v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    processor, model = load_model(args.pretrained, args.device, dtype)
    generate_samples(
        processor, model, args.num_samples,
        Path(args.out_dir), args.shard_size,
        args.device, dtype,
    )


if __name__ == "__main__":
    main()
