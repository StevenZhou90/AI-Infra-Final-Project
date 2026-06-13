#!/usr/bin/env python3
"""Train a conservative PI0-FAST prefix cutoff safety gate."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_prefix_gate import (  # noqa: E402
    PI0FastPrefixGate,
    PI0FastPrefixGateConfig,
    save_prefix_gate,
    vectorize_prefix_gate_row,
)

logger = logging.getLogger("train_pi0fast_prefix_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST prefix cutoff gate")
    parser.add_argument(
        "--data",
        required=True,
        help="JSONL from generate_pi0fast_prefix_gate_data.py; comma-separated paths are concatenated in memory.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-task-ids", default="", help="Comma-separated held-out task ids; random split if empty")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_ids(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _load_all_rows(value: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        loaded = _load_rows(Path(part))
        logger.info("loaded %d rows from %s", len(loaded), part)
        rows.extend(loaded)
    if not rows:
        raise ValueError(f"No rows from {value}")
    return rows


def _split_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[int], list[int]]:
    val_task_ids = _parse_ids(args.val_task_ids)
    if val_task_ids:
        train = [idx for idx, row in enumerate(rows) if int(row["task_id"]) not in val_task_ids]
        val = [idx for idx, row in enumerate(rows) if int(row["task_id"]) in val_task_ids]
        return train, val
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - args.val_frac)))
    return indices[:split], indices[split:]


def _tensorize(rows: list[dict[str, Any]], indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([vectorize_prefix_gate_row(rows[idx]) for idx in indices], dtype=torch.float32)
    y = torch.tensor([float(rows[idx]["label"]) for idx in indices], dtype=torch.float32)
    return x, y


@torch.no_grad()
def _eval(model: PI0FastPrefixGate, x: torch.Tensor, y: torch.Tensor, threshold: float, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits = model(x.to(device))
    probs = torch.sigmoid(logits).cpu()
    pred = probs >= threshold
    y_bool = y.bool()
    tp = int((pred & y_bool).sum().item())
    fp = int((pred & ~y_bool).sum().item())
    fn = int((~pred & y_bool).sum().item())
    tn = int((~pred & ~y_bool).sum().item())
    return {
        "rows": int(y.numel()),
        "positive_rate": float(y.mean().item()) if y.numel() else 0.0,
        "loss": float(F.binary_cross_entropy_with_logits(logits.cpu(), y).item()) if y.numel() else 0.0,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accept_rate": int(pred.sum().item()) / max(int(y.numel()), 1),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    rows = _load_all_rows(args.data)
    train_idx, val_idx = _split_rows(rows, args)
    x_train, y_train = _tensorize(rows, train_idx)
    x_val, y_val = _tensorize(rows, val_idx)
    pos_weight = ((y_train.numel() - y_train.sum()) / y_train.sum().clamp_min(1.0)).to(device)

    model = PI0FastPrefixGate(
        PI0FastPrefixGateConfig(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_score = (-1.0, -1.0)
    history = []
    gen = torch.Generator().manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], generator=gen)
        total_loss = 0.0
        for start in range(0, order.numel(), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * int(idx.numel())
        metrics = _eval(model, x_val, y_val, args.threshold, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(order.numel(), 1)
        history.append(metrics)
        score = (metrics["precision"], metrics["recall"])
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            logger.info("epoch=%d train_loss=%.4f val=%s", epoch, metrics["train_loss"], metrics)

    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "data": args.data,
        "rows": len(rows),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "val_task_ids": sorted(_parse_ids(args.val_task_ids)),
        "threshold": args.threshold,
        "best_val": _eval(model, x_val, y_val, args.threshold, device),
        "history": history,
    }
    save_prefix_gate(args.output, model, threshold=args.threshold, summary=summary)
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["best_val"], indent=2))


if __name__ == "__main__":
    main()
