#!/usr/bin/env python3
"""Benchmark spatial K/V reuse plus video patch compression.

This benchmark is intentionally self-contained so it can run without downloading
OpenVLA or starting SimplerEnv.  It creates a synthetic robot-camera stream with
a moving object, then compares:

- baseline: recompute patch embeddings and K/V for every patch every frame
- optimized: detect changed patches, refresh only those patch K/V tensors, and
  reuse cached K/V for unchanged spatial patches

The toy model is not meant to be a policy.  It isolates the inference-side
systems question: how much work and bandwidth can be skipped when adjacent
frames have stable spatial content?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from math import sin
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.spatial_video_cache import PatchGrid, SpatialKVCache, VideoPatchCompressor, patchify


class ToySpatialAttention(nn.Module):
    """Small patch encoder + attention block used to exercise K/V reuse."""

    def __init__(self, patch_dim: int, hidden_dim: int, output_dim: int = 7) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Parameter(torch.randn(hidden_dim) / hidden_dim**0.5)
        self.head = nn.Linear(hidden_dim, output_dim)

    def encode_patches(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(patches)
        return self.key(hidden), self.value(hidden)

    def attend(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = (key * self.query).sum(dim=-1) / key.shape[-1] ** 0.5
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(weights[:, None] * value, dim=0)
        return self.head(pooled)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        key, value = self.encode_patches(patches)
        return self.attend(key, value)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_synthetic_stream(
    frames: int,
    image_size: int,
    object_size: int,
    noise: float,
    device: torch.device,
) -> torch.Tensor:
    """Create C,H,W frames with a smooth background and moving bright square."""
    coords = torch.linspace(0, 1, image_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    base = torch.stack(
        [
            0.20 + 0.25 * xx,
            0.25 + 0.20 * yy,
            0.30 + 0.15 * (xx + yy),
        ],
        dim=0,
    )

    stream = []
    max_pos = image_size - object_size - 1
    for idx in range(frames):
        frame = base.clone()
        x = int((idx * 3) % max_pos)
        y = int(image_size * 0.35 + sin(idx / 6.0) * image_size * 0.15)
        y = max(0, min(max_pos, y))
        frame[:, y : y + object_size, x : x + object_size] = torch.tensor(
            [0.95, 0.15, 0.10],
            device=device,
        )[:, None, None]
        if noise > 0:
            frame = frame + torch.randn_like(frame) * noise
        stream.append(frame.clamp(0, 1))
    return torch.stack(stream, dim=0)


def frame_to_flat_patches(frame: torch.Tensor, grid: PatchGrid) -> torch.Tensor:
    patches = patchify(frame.unsqueeze(0), grid)
    return patches.reshape(grid.num_patches, -1)


def run_baseline(
    model: ToySpatialAttention,
    stream: torch.Tensor,
    grid: PatchGrid,
    device: torch.device,
) -> tuple[list[torch.Tensor], float]:
    outputs = []
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for frame in stream:
            patches = frame_to_flat_patches(frame, grid)
            outputs.append(model(patches))
    sync(device)
    return outputs, (time.perf_counter() - start) * 1000


def run_cached(
    model: ToySpatialAttention,
    stream: torch.Tensor,
    grid: PatchGrid,
    threshold: float,
    device: torch.device,
) -> tuple[list[torch.Tensor], float, SpatialKVCache, VideoPatchCompressor]:
    cache = SpatialKVCache(grid.num_patches)
    compressor = VideoPatchCompressor(grid, threshold=threshold)
    outputs = []

    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for frame in stream:
            changed_mask, _ = compressor.update(frame.unsqueeze(0))
            patches = frame_to_flat_patches(frame, grid)

            if cache.stats.frames == 0:
                key, value = model.encode_patches(patches)
            else:
                changed_idx = changed_mask.nonzero(as_tuple=False).flatten()
                key = torch.empty(
                    grid.num_patches,
                    model.query.shape[0],
                    dtype=patches.dtype,
                    device=device,
                )
                value = torch.empty_like(key)
                if changed_idx.numel() > 0:
                    changed_key, changed_value = model.encode_patches(patches[changed_idx])
                    key[changed_idx] = changed_key
                    value[changed_idx] = changed_value

            merged_key, merged_value = cache.update(key, value, changed_mask)
            outputs.append(model.attend(merged_key, merged_value))

    sync(device)
    return outputs, (time.perf_counter() - start) * 1000, cache, compressor


def max_output_error(baseline: list[torch.Tensor], cached: list[torch.Tensor]) -> float:
    errors = [
        (base.detach().float().cpu() - opt.detach().float().cpu()).abs().max().item()
        for base, opt in zip(baseline, cached, strict=True)
    ]
    return float(max(errors)) if errors else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--object-size", type=int, default=36)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    grid = PatchGrid(args.image_size, args.patch_size)
    patch_dim = 3 * args.patch_size * args.patch_size

    model = ToySpatialAttention(patch_dim=patch_dim, hidden_dim=args.hidden_dim).to(device).eval()
    stream = make_synthetic_stream(
        frames=args.frames,
        image_size=args.image_size,
        object_size=args.object_size,
        noise=args.noise,
        device=device,
    )

    # Warm up kernels and allocator.
    warmup = stream[: min(8, args.frames)]
    run_baseline(model, warmup, grid, device)
    run_cached(model, warmup, grid, args.threshold, device)

    baseline_outputs, baseline_ms = run_baseline(model, stream, grid, device)
    cached_outputs, cached_ms, cache, compressor = run_cached(
        model,
        stream,
        grid,
        args.threshold,
        device,
    )

    metrics = {
        "frames": args.frames,
        "device": str(device),
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "num_patches": grid.num_patches,
        "threshold": args.threshold,
        "baseline_ms_total": baseline_ms,
        "cached_ms_total": cached_ms,
        "baseline_ms_per_frame": baseline_ms / args.frames,
        "cached_ms_per_frame": cached_ms / args.frames,
        "speedup": baseline_ms / cached_ms if cached_ms > 0 else 0.0,
        "max_output_error": max_output_error(baseline_outputs, cached_outputs),
        "spatial_cache": asdict(cache.stats),
        "spatial_cache_reuse_ratio": cache.stats.reuse_ratio,
        "video_compression": asdict(compressor.stats),
        "video_compression_ratio": compressor.stats.compression_ratio,
        "video_changed_patch_ratio": compressor.stats.changed_ratio,
    }

    print(json.dumps(metrics, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
