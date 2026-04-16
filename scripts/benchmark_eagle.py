#!/usr/bin/env python3
"""Benchmark a trained EAGLE draft head against baseline OpenVLA inference.

Uses verify-pass hidden states directly (no extra forward pass), matching the
v2 training data that uses batched hidden states.

Usage:
    python scripts/benchmark_eagle.py \
        --eagle_dir checkpoints/eagle_openvla \
        --pretrained openvla/openvla-7b \
        --num_samples 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.eagle_draft import EagleDraftHead
from serving.kv_cache_manager import trim_kv, kv_seq_len

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ACTION_DIM = 7


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


def make_inputs(processor, device, dtype, idx: int = 0):
    instructions = [
        "pick up the cube",
        "move the red block to the right",
        "grasp the object on the table",
        "push the blue cylinder forward",
        "lift the cup and place it on the plate",
    ]
    instruction = instructions[idx % len(instructions)]
    from PIL import Image
    pil_image = Image.fromarray(
        np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    )
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)
    return inputs


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def benchmark_baseline(processor, model, device, dtype, num_samples: int):
    logger.info("--- Baseline (standard autoregressive) ---")
    times = []
    for i in range(num_samples):
        inputs = make_inputs(processor, device, dtype, i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=ACTION_DIM,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = times[2:]
    avg_ms = np.mean(times) * 1000
    logger.info(
        "Baseline: %.1f ms/inference (%.1f ms/token) over %d samples",
        avg_ms, avg_ms / ACTION_DIM, len(times),
    )
    return avg_ms


# ---------------------------------------------------------------------------
# EAGLE speculative decoding (no extra forward pass)
# ---------------------------------------------------------------------------

class EagleSpeculativeGenerator:
    """Correct speculative decoding using verify-pass hidden states directly.

    The correction/bonus token is NOT processed separately — its KV is added
    by the NEXT round's verify pass (where it appears as first_token).
    This eliminates the ~25ms extra forward pass per round.
    """

    def __init__(self, model, eagle_head: EagleDraftHead, device, dtype):
        self.model = model
        self.eagle = eagle_head
        self.device = device
        self.dtype = dtype

        self.llm = model.language_model
        self.norm = self.llm.model.norm
        self.lm_head = self.llm.lm_head
        self.layers = self.llm.model.layers

        self._captured_hidden = {}
        self._hook_handle = self.layers[-2].register_forward_hook(self._hook_fn)

        self.total_drafted = 0
        self.total_accepted = 0

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self._captured_hidden["h"] = output[0].detach()
        else:
            self._captured_hidden["h"] = output.detach()

    def generate(self, inputs, max_new_tokens: int = ACTION_DIM, lookahead: int = 4):
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")

        # --- Prefill ---
        with torch.no_grad():
            prefill_out = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                use_cache=True,
            )

        target_kv = prefill_out.past_key_values
        prev_logits = prefill_out.logits[:, -1:, :]
        prev_hidden = self._captured_hidden["h"][:, -1:, :]

        generated = []

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            K = min(lookahead + 1, remaining)

            first_token = prev_logits.argmax(dim=-1)  # [1,1]

            if K == 1:
                generated.append(first_token)
                with torch.no_grad():
                    out = self.llm(first_token, past_key_values=target_kv, use_cache=True)
                target_kv = out.past_key_values
                prev_logits = out.logits[:, -1:, :]
                prev_hidden = self._captured_hidden["h"][:, -1:, :]
                continue

            # --- Draft K-1 tokens with EAGLE ---
            self.eagle.reset_kv()
            draft_tokens = [first_token]
            draft_hidden = prev_hidden

            for _ in range(K - 1):
                with torch.no_grad():
                    eagle_out = self.eagle(
                        draft_hidden, draft_tokens[-1], use_cache=True,
                    )
                    normed = self.norm(eagle_out)
                    logits = self.lm_head(normed)
                next_tok = logits[:, -1:, :].argmax(dim=-1)
                draft_tokens.append(next_tok)
                draft_hidden = eagle_out[:, -1:, :]

            # --- Verify all K tokens ---
            verify_input = torch.cat(draft_tokens, dim=1)  # [1, K]
            with torch.no_grad():
                verify_out = self.llm(
                    verify_input, past_key_values=target_kv, use_cache=True,
                )

            verify_logits = verify_out.logits    # [1, K, vocab]
            verify_hidden = self._captured_hidden["h"]  # [1, K, H]

            # --- Accept / reject ---
            # first_token (index 0) is always correct.
            # verify_logits[i] predicts what comes after draft_tokens[i].
            # Check: verify_logits[i].argmax() == draft_tokens[i+1] ?
            n_accepted = 1  # first_token
            for i in range(K - 1):
                target_pred = verify_logits[:, i, :].argmax(dim=-1)
                self.total_drafted += 1
                if target_pred.item() == draft_tokens[i + 1].view(-1).item():
                    n_accepted += 1
                    self.total_accepted += 1
                else:
                    break

            all_matched = (n_accepted == K)

            # Emit accepted draft tokens
            for j in range(n_accepted):
                generated.append(draft_tokens[j])

            if all_matched:
                bonus = verify_logits[:, K - 1, :].argmax(dim=-1).view(1, 1)
                generated.append(bonus)
                # KV has entries for all K draft tokens — keep all
                target_kv = trim_kv(verify_out.past_key_values,
                                    kv_seq_len(target_kv) + K)
                # Next round: bonus is first_token, hidden at last draft position
                prev_logits = verify_logits[:, K - 1:K, :]
                prev_hidden = verify_hidden[:, K - 1:K, :]
            else:
                correction = verify_logits[:, n_accepted - 1, :].argmax(dim=-1).view(1, 1)
                generated.append(correction)
                # KV: keep only accepted draft tokens (0..n_accepted-1)
                target_kv = trim_kv(verify_out.past_key_values,
                                    kv_seq_len(target_kv) + n_accepted)
                # Next round: correction is first_token, hidden at last accepted position
                prev_logits = verify_logits[:, n_accepted - 1:n_accepted, :]
                prev_hidden = verify_hidden[:, n_accepted - 1:n_accepted, :]

        gen_ids = torch.cat(generated[:max_new_tokens], dim=1)
        all_ids = torch.cat([input_ids, gen_ids], dim=1)
        return all_ids

    def cleanup(self):
        self._hook_handle.remove()

    @property
    def acceptance_rate(self):
        return self.total_accepted / max(self.total_drafted, 1)


def benchmark_eagle(
    processor, model, eagle_head, device, dtype, num_samples: int, lookahead: int,
):
    logger.info("--- EAGLE speculative decoding (trained, no extra fwd) ---")
    gen = EagleSpeculativeGenerator(model, eagle_head, device, dtype)

    times = []
    for i in range(num_samples):
        inputs = make_inputs(processor, device, dtype, i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen.generate(inputs, max_new_tokens=ACTION_DIM, lookahead=lookahead)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    gen.cleanup()

    times = times[2:]
    avg_ms = np.mean(times) * 1000

    logger.info(
        "EAGLE: %.1f ms/inference (%.1f ms/token) | acceptance=%.1f%% | drafted=%d accepted=%d",
        avg_ms, avg_ms / ACTION_DIM,
        gen.acceptance_rate * 100,
        gen.total_drafted, gen.total_accepted,
    )
    return avg_ms, gen.acceptance_rate


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained EAGLE head")
    parser.add_argument("--eagle_dir", default="checkpoints/eagle_openvla")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    processor, model = load_openvla(args.pretrained, args.device, dtype)

    eagle_head = EagleDraftHead.from_pretrained(
        args.eagle_dir, device=args.device, dtype=dtype,
    )
    logger.info(
        "EAGLE head loaded: %.1fM params (%d layers)",
        sum(p.numel() for p in eagle_head.parameters()) / 1e6,
        len(eagle_head.layers),
    )

    baseline_ms = benchmark_baseline(processor, model, args.device, dtype, args.num_samples)
    eagle_ms, acc_rate = benchmark_eagle(
        processor, model, eagle_head, args.device, dtype, args.num_samples, args.lookahead,
    )

    speedup = baseline_ms / eagle_ms
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("  Baseline:     %.1f ms/inference", baseline_ms)
    logger.info("  EAGLE:        %.1f ms/inference", eagle_ms)
    logger.info("  Speedup:      %.2fx", speedup)
    logger.info("  Acceptance:   %.1f%%", acc_rate * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
