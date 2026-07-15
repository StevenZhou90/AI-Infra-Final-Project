#!/usr/bin/env python
"""Benchmark PI0-FAST single-request decode without creating a LIBERO env.

This uses the real LeRobot PI0-FAST checkpoint and synthetic, shape-correct
image/text inputs.  It is meant for runtime-backend experiments when MuJoCo/EGL
is unavailable; accuracy must still be validated on real LIBERO rollouts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic PI0-FAST single-request latency benchmark.")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--image-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-tokens", type=int, default=48)
    parser.add_argument(
        "--prompt",
        default="put the alphabet soup in the basket",
        help="Prompt used to build synthetic language token IDs.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-decoding-steps", type=int, default=-1)
    parser.add_argument(
        "--attn-implementations",
        default="default,eager,sdpa",
        help="Comma-separated language-model attention implementations to test.",
    )
    parser.add_argument(
        "--decode-attn-implementation",
        choices=["default", "eager", "sdpa", "flash_attention_2"],
        default="default",
        help="Optionally switch only one-token decode forwards to this backend after prefill.",
    )
    parser.add_argument("--action-vocab-size", type=int, default=None)
    parser.add_argument("--text-vocab-size", type=int, default=None)
    parser.add_argument("--structural-token-radius", type=int, default=512)
    parser.add_argument("--full-head-margin", type=float, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--compile-language-model-forward",
        action="store_true",
        help="Experimental latency probe; compare token output against an uncompiled run before using.",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("outputs/pi0fast_system_components/synthetic_single_request.json"))
    return parser.parse_args()


def load_dotenv_token() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"} and value:
            os.environ.setdefault("HF_TOKEN", value)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", value)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", value)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_ms(device: torch.device, fn):
    sync_if_cuda(device)
    start = time.perf_counter()
    out = fn()
    sync_if_cuda(device)
    return out, (time.perf_counter() - start) * 1000.0


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def language_model(policy: PI0FastPolicy):
    return policy.model.paligemma_with_expert.paligemma.model.language_model


def set_attn_implementation(policy: PI0FastPolicy, implementation: str, original: str | None) -> str | None:
    model = language_model(policy)
    if implementation == "default":
        model.config._attn_implementation = original
    else:
        model.config._attn_implementation = implementation
    return getattr(model.config, "_attn_implementation", None)


def build_synthetic_inputs(
    *,
    policy: PI0FastPolicy,
    device: torch.device,
    image_dtype: torch.dtype,
    batch_size: int,
    text_tokens: int,
    prompt: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    height, width = tuple(policy.config.image_resolution)
    image_count = max(1, len(getattr(policy.config, "image_features", [])))
    generator = torch.Generator(device="cpu").manual_seed(0)
    images = [
        torch.rand((batch_size, 3, height, width), generator=generator, dtype=torch.float32)
        .to(device=device, dtype=image_dtype)
        for _ in range(image_count)
    ]
    img_masks = [torch.ones((batch_size,), dtype=torch.bool, device=device) for _ in range(image_count)]

    tokenizer = policy.model._paligemma_tokenizer
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    encoded = [int(token_id) for token_id in encoded[:text_tokens]]
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    token_row = encoded + [pad_id] * max(0, text_tokens - len(encoded))
    mask_row = [True] * len(encoded) + [False] * max(0, text_tokens - len(encoded))
    tokens = torch.tensor([token_row] * batch_size, dtype=torch.long, device=device)
    masks = torch.tensor([mask_row] * batch_size, dtype=torch.bool, device=device)
    return images, img_masks, tokens, masks


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    load_dotenv_token()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    image_dtype = getattr(torch, args.image_dtype)

    policy_config = PreTrainedConfig.from_pretrained(args.policy)
    if hasattr(policy_config, "device"):
        policy_config.device = str(device)
    if hasattr(policy_config, "dtype"):
        policy_config.dtype = args.dtype
    if args.disable_gradient_checkpointing and hasattr(policy_config, "gradient_checkpointing"):
        policy_config.gradient_checkpointing = False
    policy = PI0FastPolicy.from_pretrained(args.policy, config=policy_config).to(device=device, dtype=dtype).eval()
    if args.disable_gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
        policy.eval()
    if args.compile_language_model_forward:
        language_model(policy).forward = torch.compile(language_model(policy).forward, mode=args.compile_mode)

    adapter = PI0FastTokenLogitAdapter(policy)
    adapter._profile_action_end_decode = bool(args.profile)

    images, img_masks, tokens, masks = build_synthetic_inputs(
        policy=policy,
        device=device,
        image_dtype=image_dtype,
        batch_size=args.batch_size,
        text_tokens=args.text_tokens,
        prompt=args.prompt,
    )
    max_decoding_steps = (
        int(getattr(policy.config, "max_decoding_steps", 0) or 0)
        if args.max_decoding_steps < 0
        else int(args.max_decoding_steps)
    )
    original_attn = getattr(language_model(policy).config, "_attn_implementation", None)

    rows: list[dict[str, Any]] = []
    reference_tokens: torch.Tensor | None = None
    reference_actions: torch.Tensor | None = None
    for implementation in [item.strip() for item in args.attn_implementations.split(",") if item.strip()]:
        active_attn = set_attn_implementation(policy, implementation, original_attn)
        row: dict[str, Any] = {"attn_implementation": implementation, "active_attn_implementation": active_attn}
        times: list[float] = []
        token_counts: list[int] = []
        profiles: list[dict[str, Any]] = []
        try:
            for _ in range(args.warmup):
                adapter.sample_actions_fast_kv_cache_action_end_constrained(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    max_decoding_steps=max_decoding_steps,
                    action_vocab_size=args.action_vocab_size,
                    text_vocab_size=args.text_vocab_size,
                    structural_token_radius=args.structural_token_radius,
                    full_head_margin=args.full_head_margin,
                    decode_attn_implementation=args.decode_attn_implementation,
                )
            last_tokens = None
            last_actions = None
            for _ in range(args.steps):
                token_ids, elapsed = timed_ms(
                    device,
                    lambda: adapter.sample_actions_fast_kv_cache_action_end_constrained(
                        images,
                        img_masks,
                        tokens,
                        masks,
                        max_decoding_steps=max_decoding_steps,
                        action_vocab_size=args.action_vocab_size,
                        text_vocab_size=args.text_vocab_size,
                        structural_token_radius=args.structural_token_radius,
                        full_head_margin=args.full_head_margin,
                        decode_attn_implementation=args.decode_attn_implementation,
                    ),
                )
                actions = adapter._detokenize_generated_actions(token_ids)
                sync_if_cuda(device)
                last_tokens = token_ids.detach().clone()
                last_actions = actions.detach().clone()
                times.append(float(elapsed))
                token_counts.append(int(token_ids.shape[-1]))
                profile = getattr(adapter, "_last_action_end_profile", None)
                if isinstance(profile, dict):
                    profiles.append(jsonable(profile))

            if reference_tokens is None:
                reference_tokens = last_tokens
                reference_actions = last_actions
                tokens_equal = True
                action_max_abs_diff = 0.0
            else:
                tokens_equal = bool(torch.equal(reference_tokens, last_tokens))
                if reference_actions is not None and last_actions is not None:
                    action_max_abs_diff = float(torch.max(torch.abs(reference_actions - last_actions)).item())
                else:
                    action_max_abs_diff = float("nan")

            row.update(
                {
                    "latency": summarize(times),
                    "times_ms": times,
                    "token_counts": token_counts,
                    "token_count_mean": float(np.mean(token_counts)) if token_counts else 0.0,
                    "tokens_equal_to_reference": tokens_equal,
                    "action_max_abs_diff_to_reference": action_max_abs_diff,
                    "last_token_ids": jsonable(last_tokens[0]) if last_tokens is not None else None,
                    "last_profile": profiles[-1] if profiles else None,
                    "error": None,
                }
            )
        except Exception as exc:  # keep sweeping if one backend is unavailable
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    summary = {
        "config": {
            "policy": args.policy,
            "device": str(device),
            "dtype": args.dtype,
            "image_dtype": args.image_dtype,
            "batch_size": args.batch_size,
            "text_tokens": args.text_tokens,
            "prompt": args.prompt,
            "warmup": args.warmup,
            "steps": args.steps,
            "max_decoding_steps": max_decoding_steps,
            "action_vocab_size": args.action_vocab_size,
            "text_vocab_size": args.text_vocab_size,
            "structural_token_radius": args.structural_token_radius,
            "full_head_margin": args.full_head_margin,
            "decode_attn_implementation": args.decode_attn_implementation,
            "profile": bool(args.profile),
            "disable_gradient_checkpointing": bool(args.disable_gradient_checkpointing),
            "compile_language_model_forward": bool(args.compile_language_model_forward),
            "compile_mode": args.compile_mode,
        },
        "language_model_attn_original": original_attn,
        "image_count": len(images),
        "image_shape": list(images[0].shape),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(json.dumps(jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
