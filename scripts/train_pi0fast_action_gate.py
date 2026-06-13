#!/usr/bin/env python3
"""Train a conservative action-space gate for DFlash PI0-FAST candidates."""

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

from serving.pi0fast_action_gate import (  # noqa: E402
    PI0FastActionGate,
    PI0FastActionGateConfig,
    save_action_gate,
    vectorize_action_gate_row,
)

logger = logging.getLogger("train_pi0fast_action_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST DFlash action gate")
    parser.add_argument("--data", required=True, help="JSONL from generate_pi0fast_action_gate_data.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-task-ids", default="", help="Comma-separated held-out task ids; random split if empty")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--target-precision", type=float, default=0.98)
    parser.add_argument(
        "--require-guard-accepted-label",
        action="store_true",
        help="Treat rows as positive only when the generator label is positive and the handcrafted guard accepted.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_ids(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _load_rows(paths: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in paths.split(","):
        part = part.strip()
        if not part:
            continue
        path = Path(part)
        loaded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        logger.info("loaded %d rows from %s", len(loaded), path)
        rows.extend(loaded)
    if not rows:
        raise ValueError(f"No rows in {paths}")
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


def _tensorize(
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    require_guard_accepted_label: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([vectorize_action_gate_row(rows[idx]) for idx in indices], dtype=torch.float32)
    if require_guard_accepted_label:
        y = torch.tensor(
            [float(rows[idx]["label"] and rows[idx].get("guard_accepted", False)) for idx in indices],
            dtype=torch.float32,
        )
    else:
        y = torch.tensor([float(rows[idx]["label"]) for idx in indices], dtype=torch.float32)
    return x, y


@torch.no_grad()
def _probs(model: PI0FastActionGate, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, x.shape[0], 4096):
        chunks.append(model.probability(x[start : start + 4096].to(device)).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)


def _metrics_from_probs(probs: torch.Tensor, y: torch.Tensor, threshold: float) -> dict[str, Any]:
    pred = probs >= threshold
    y_bool = y.bool()
    tp = int((pred & y_bool).sum().item())
    fp = int((pred & ~y_bool).sum().item())
    fn = int((~pred & y_bool).sum().item())
    tn = int((~pred & ~y_bool).sum().item())
    return {
        "rows": int(y.numel()),
        "positive_rate": float(y.mean().item()) if y.numel() else 0.0,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accept_rate": int(pred.sum().item()) / max(int(y.numel()), 1),
        "mean_probability": float(probs.mean().item()) if probs.numel() else 0.0,
    }


def _select_threshold(probs: torch.Tensor, y: torch.Tensor, target_precision: float, fallback: float) -> tuple[float, dict[str, Any]]:
    candidates = [round(v, 3) for v in np.linspace(0.05, 0.995, 190)]
    best = None
    for threshold in candidates:
        metrics = _metrics_from_probs(probs, y, threshold)
        if metrics["precision"] >= target_precision and metrics["tp"] > 0:
            score = (metrics["recall"], metrics["precision"], metrics["accept_rate"])
        else:
            score = (-1.0, metrics["precision"], metrics["accept_rate"])
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    if best is None or best[0][0] < 0:
        metrics = _metrics_from_probs(probs, y, fallback)
        return fallback, metrics
    return float(best[1]), best[2]


@torch.no_grad()
def _eval(model: PI0FastActionGate, x: torch.Tensor, y: torch.Tensor, threshold: float, device: torch.device) -> dict[str, Any]:
    probs = _probs(model, x, device)
    metrics = _metrics_from_probs(probs, y, threshold)
    logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
    metrics["loss"] = float(F.binary_cross_entropy_with_logits(logits, y).item()) if y.numel() else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    rows = _load_rows(args.data)
    train_idx, val_idx = _split_rows(rows, args)
    x_train, y_train = _tensorize(
        rows,
        train_idx,
        require_guard_accepted_label=args.require_guard_accepted_label,
    )
    x_val, y_val = _tensorize(
        rows,
        val_idx,
        require_guard_accepted_label=args.require_guard_accepted_label,
    )
    if y_train.sum() == 0:
        raise ValueError("Training split has no positive rows; collect more safe candidates or relax labels")

    model = PI0FastActionGate(
        PI0FastActionGateConfig(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = ((y_train.numel() - y_train.sum()) / y_train.sum().clamp_min(1.0)).to(device)
    best_state = None
    best_score = (-1.0, -1.0, -1.0)
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)

    logger.info(
        "train_rows=%d val_rows=%d train_pos=%.3f val_pos=%.3f",
        int(y_train.numel()),
        int(y_val.numel()),
        float(y_train.mean().item()),
        float(y_val.mean().item()) if y_val.numel() else 0.0,
    )
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator)
        total_loss = 0.0
        for start in range(0, order.numel(), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(idx.numel())
        probs = _probs(model, x_val, device)
        selected_threshold, metrics = _select_threshold(probs, y_val, args.target_precision, args.threshold)
        metrics["selected_threshold"] = selected_threshold
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(order.numel(), 1)
        history.append(metrics)
        score = (metrics["precision"], metrics["recall"], metrics["accept_rate"])
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = selected_threshold
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            logger.info("epoch=%d train_loss=%.4f val=%s", epoch, metrics["train_loss"], metrics)

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_threshold = args.threshold
    best_val = _eval(model, x_val, y_val, best_threshold, device)
    summary = {
        "data": args.data,
        "rows": len(rows),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "val_task_ids": sorted(_parse_ids(args.val_task_ids)),
        "target_precision": args.target_precision,
        "require_guard_accepted_label": args.require_guard_accepted_label,
        "threshold": best_threshold,
        "best_val": best_val,
        "history": history,
    }
    save_action_gate(args.output, model, threshold=best_threshold, summary=summary)
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(best_val, indent=2))


if __name__ == "__main__":
    main()
