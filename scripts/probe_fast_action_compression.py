#!/usr/bin/env python3
"""Probe FAST-style action chunk compression on saved trajectory-head data.

This is an offline feasibility check. It does not modify OpenVLA or run LIBERO;
it asks whether contiguous future action-bin chunks can be reconstructed from a
small discrete codebook with tolerable error.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("probe_fast_action_compression")


def group_examples(examples: list[dict[str, Any]]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        groups[(str(ex["suite"]), int(ex["task_id"]), int(ex["trial"]))].append(ex)
    for rows in groups.values():
        rows.sort(key=lambda ex: int(ex["timestep"]))
    return groups


def build_chunks(
    examples: list[dict[str, Any]],
    horizon: int,
    phase_filter: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    chunks: list[torch.Tensor] = []
    meta: list[dict[str, Any]] = []
    for (suite, task_id, trial), rows in group_examples(examples).items():
        by_t = {int(ex["timestep"]): ex for ex in rows}
        for ex in rows:
            phase = str(ex.get("phase_label", "unknown"))
            if phase_filter != "all" and phase != phase_filter:
                continue
            t0 = int(ex["timestep"])
            seq = []
            ok = True
            for offset in range(horizon):
                nxt = by_t.get(t0 + offset)
                if nxt is None:
                    ok = False
                    break
                seq.append(nxt["target_bins"].long())
            if not ok:
                continue
            chunks.append(torch.stack(seq, dim=0))
            meta.append({"suite": suite, "task_id": task_id, "trial": trial, "timestep": t0, "phase": phase})
    if not chunks:
        return torch.empty(0, horizon, 7, dtype=torch.long), []
    return torch.stack(chunks, dim=0), meta


def train_val_split(n: int, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = max(1, int(round(n * val_frac)))
    return perm[n_val:], perm[:n_val]


def init_centroids(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:k]
    if idx.numel() < k:
        repeats = torch.randint(0, x.shape[0], (k - idx.numel(),), generator=gen, device=x.device)
        idx = torch.cat([idx, repeats], dim=0)
    return x[idx].clone()


def fit_kmeans(
    train_x: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    k = min(k, train_x.shape[0])
    centroids = init_centroids(train_x, k, seed)
    for _ in range(iters):
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(k, dtype=torch.float32, device=train_x.device)
        for start in range(0, train_x.shape[0], batch_size):
            xb = train_x[start : start + batch_size]
            labels = torch.cdist(xb, centroids).argmin(dim=1)
            sums.index_add_(0, labels, xb)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    return centroids


def assign(x: torch.Tensor, centroids: torch.Tensor, batch_size: int) -> torch.Tensor:
    labels = []
    for start in range(0, x.shape[0], batch_size):
        labels.append(torch.cdist(x[start : start + batch_size], centroids).argmin(dim=1).cpu())
    return torch.cat(labels, dim=0)


def metrics(chunks: torch.Tensor, recon_flat: torch.Tensor) -> dict[str, Any]:
    recon = recon_flat.round().clamp(0, 255).view_as(chunks).long()
    err = (recon - chunks).abs().float()
    per_dim = err.mean(dim=(0, 1))
    exact_action = (recon == chunks).all(dim=2).float().mean()
    exact_chunk = (recon == chunks).all(dim=(1, 2)).float().mean()
    gripper_mismatch = (recon[:, :, 6] != chunks[:, :, 6]).float().mean()
    return {
        "mae": float(err.mean().item()),
        "p95_abs": float(torch.quantile(err.flatten(), 0.95).item()),
        "max_abs": float(err.max().item()),
        "pos_mae": float(err[:, :, :3].mean().item()),
        "rot_mae": float(err[:, :, 3:6].mean().item()),
        "gripper_mismatch": float(gripper_mismatch.item()),
        "exact_action_rate": float(exact_action.item()),
        "exact_chunk_rate": float(exact_chunk.item()),
        "per_dim_mae": [float(x) for x in per_dim.tolist()],
    }


def task_metrics(chunks: torch.Tensor, recon_flat: torch.Tensor, meta: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(meta):
        by_task[int(row["task_id"])].append(idx)
    out = {}
    for task_id, idxs in sorted(by_task.items()):
        idx = torch.tensor(idxs, dtype=torch.long)
        out[str(task_id)] = metrics(chunks[idx], recon_flat[idx])
        out[str(task_id)]["n"] = int(len(idxs))
    return out


def run_probe(
    chunks: torch.Tensor,
    meta: list[dict[str, Any]],
    codebook_sizes: list[int],
    seed: int,
    iters: int,
    batch_size: int,
    val_frac: float,
) -> list[dict[str, Any]]:
    train_idx, val_idx = train_val_split(chunks.shape[0], val_frac, seed)
    train = chunks[train_idx].float().flatten(start_dim=1)
    val_chunks = chunks[val_idx]
    val = val_chunks.float().flatten(start_dim=1)
    val_meta = [meta[int(i)] for i in val_idx.tolist()]
    results = []
    for k in codebook_sizes:
        centroids = fit_kmeans(train, k=k, iters=iters, seed=seed + k, batch_size=batch_size)
        labels = assign(val, centroids, batch_size=batch_size)
        recon = centroids[labels].cpu()
        m = metrics(val_chunks.cpu(), recon)
        m["codebook_size"] = int(min(k, train.shape[0]))
        m["bits_per_chunk"] = float(torch.log2(torch.tensor(float(m["codebook_size"]))).item())
        m["baseline_action_tokens"] = int(val_chunks.shape[1] * val_chunks.shape[2])
        m["compression_vs_7k_tokens"] = float(m["baseline_action_tokens"] / max(m["bits_per_chunk"] / 8.0, 1e-6))
        m["n_train"] = int(train.shape[0])
        m["n_val"] = int(val.shape[0])
        m["by_task"] = task_metrics(val_chunks.cpu(), recon, val_meta)
        results.append(m)
    return results


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/libero_goal/useful_6task_5trial_160/dataset.pt"))
    parser.add_argument("--horizons", default="2,4,6")
    parser.add_argument("--codebook-sizes", default="64,128,256,512")
    parser.add_argument("--phase-filter", choices=["all", "smooth", "complex"], default="all")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = torch.load(args.data, map_location="cpu")
    examples = data["examples"]
    horizons = parse_ints(args.horizons)
    codebook_sizes = parse_ints(args.codebook_sizes)

    payload: dict[str, Any] = {
        "data": str(args.data),
        "phase_filter": args.phase_filter,
        "horizons": {},
    }
    for horizon in horizons:
        chunks, meta = build_chunks(examples, horizon=horizon, phase_filter=args.phase_filter)
        logger.info("horizon=%d chunks=%d phase_filter=%s", horizon, chunks.shape[0], args.phase_filter)
        if chunks.shape[0] < 10:
            payload["horizons"][str(horizon)] = {"n_chunks": int(chunks.shape[0]), "results": []}
            continue
        results = run_probe(
            chunks,
            meta,
            codebook_sizes=codebook_sizes,
            seed=args.seed,
            iters=args.iters,
            batch_size=args.batch_size,
            val_frac=args.val_frac,
        )
        payload["horizons"][str(horizon)] = {"n_chunks": int(chunks.shape[0]), "results": results}

    print(json.dumps(payload, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
