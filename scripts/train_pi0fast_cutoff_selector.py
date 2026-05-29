#!/usr/bin/env python3
"""Train a prefix-hidden selector for PI0-FAST block SD cutoff choice."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_cutoff_selector import (  # noqa: E402
    PI0FastCutoffSelector,
    PI0FastCutoffSelectorConfig,
    save_cutoff_selector,
)

logger = logging.getLogger("train_pi0fast_cutoff_selector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST cutoff selector")
    parser.add_argument("--data", required=True, help="data.pt from generate_pi0fast_action_corrector_data.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutoffs", default="52,56,60,64")
    parser.add_argument("--val-task-ids", default="2,5")
    parser.add_argument("--accept-pos-l2", type=float, default=0.05)
    parser.add_argument("--accept-rot-l2", type=float, default=0.25)
    parser.add_argument("--min-accept-len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _accept_len(init: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> int:
    horizon = min(init.shape[0], target.shape[0])
    err = init[:horizon] - target[:horizon]
    pos = torch.linalg.norm(err[:, :3], dim=-1)
    rot = torch.linalg.norm(err[:, 3:6], dim=-1) if init.shape[-1] >= 6 else torch.zeros_like(pos)
    grip_ok = torch.ones_like(pos, dtype=torch.bool)
    if init.shape[-1] >= 7:
        grip_ok = torch.sign(init[:horizon, 6]) == torch.sign(target[:horizon, 6])
    within = (pos <= args.accept_pos_l2) & (rot <= args.accept_rot_l2) & grip_ok
    return int(within.int().cumprod(dim=0).sum().item())


def _build_examples(rows: list[dict[str, Any]], cutoffs: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        cutoff = int(row["cutoff"])
        if cutoff in cutoffs:
            grouped[(int(row["task_id"]), int(row["seed"]), int(row["chunk"]))][cutoff] = row

    examples: list[dict[str, Any]] = []
    for key, by_cutoff in grouped.items():
        if not all(cutoff in by_cutoff for cutoff in cutoffs):
            continue
        label_idx = len(cutoffs) - 1
        accept_by_cutoff = {}
        for idx, cutoff in enumerate(cutoffs):
            row = by_cutoff[cutoff]
            accept = _accept_len(row["init_actions"].float(), row["target_actions"].float(), args)
            accept_by_cutoff[str(cutoff)] = accept
            if accept >= args.min_accept_len:
                label_idx = idx
                break
        first = by_cutoff[cutoffs[0]]
        examples.append(
            {
                "task_id": int(key[0]),
                "seed": int(key[1]),
                "chunk": int(key[2]),
                "hidden": first["hidden"].float(),
                "label_idx": label_idx,
                "label_cutoff": cutoffs[label_idx],
                "accept_by_cutoff": accept_by_cutoff,
            }
        )
    return examples


def _split(examples: list[dict[str, Any]], val_task_ids: set[int], seed: int) -> tuple[list[int], list[int]]:
    train = [idx for idx, ex in enumerate(examples) if int(ex["task_id"]) not in val_task_ids]
    val = [idx for idx, ex in enumerate(examples) if int(ex["task_id"]) in val_task_ids]
    if train and val:
        return train, val
    ids = list(range(len(examples)))
    random.Random(seed).shuffle(ids)
    split = max(1, int(0.8 * len(ids)))
    return ids[:split], ids[split:]


def _tensors(examples: list[dict[str, Any]], indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([examples[idx]["hidden"].float() for idx in indices])
    y = torch.tensor([int(examples[idx]["label_idx"]) for idx in indices], dtype=torch.long)
    return x, y


@torch.no_grad()
def _eval(
    model: PI0FastCutoffSelector,
    x: torch.Tensor,
    y: torch.Tensor,
    cutoffs: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits = model(x.to(device))
    pred = torch.argmax(logits, dim=-1).cpu()
    probs = torch.softmax(logits.cpu(), dim=-1)
    pred_cutoffs = torch.tensor([cutoffs[int(idx)] for idx in pred], dtype=torch.float32)
    gold_cutoffs = torch.tensor([cutoffs[int(idx)] for idx in y], dtype=torch.float32)
    return {
        "rows": int(y.numel()),
        "loss": float(F.cross_entropy(logits.cpu(), y).item()) if y.numel() else 0.0,
        "accuracy": float((pred == y).float().mean().item()) if y.numel() else 0.0,
        "safe_or_later": float((pred >= y).float().mean().item()) if y.numel() else 0.0,
        "too_early_rate": float((pred < y).float().mean().item()) if y.numel() else 0.0,
        "mean_pred_cutoff": float(pred_cutoffs.mean().item()) if y.numel() else 0.0,
        "mean_gold_cutoff": float(gold_cutoffs.mean().item()) if y.numel() else 0.0,
        "p_pred": {str(c): float((pred == idx).float().mean().item()) for idx, c in enumerate(cutoffs)},
        "p_gold": {str(c): float((y == idx).float().mean().item()) for idx, c in enumerate(cutoffs)},
        "mean_confidence": float(probs.max(dim=-1).values.mean().item()) if y.numel() else 0.0,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    cutoffs = _parse_ints(args.cutoffs)
    val_task_ids = set(_parse_ints(args.val_task_ids))
    examples = _build_examples(list(payload["rows"]), cutoffs, args)
    if not examples:
        raise ValueError("No grouped cutoff examples found")
    train_idx, val_idx = _split(examples, val_task_ids, args.seed)
    x_train, y_train = _tensors(examples, train_idx)
    x_val, y_val = _tensors(examples, val_idx)

    model = PI0FastCutoffSelector(
        PI0FastCutoffSelectorConfig(
            hidden_dim=int(x_train.shape[-1]),
            cutoffs=tuple(cutoffs),
            model_dim=args.model_dim,
            dropout=args.dropout,
        )
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_counts = torch.bincount(y_train, minlength=len(cutoffs)).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0)).to(device)
    class_weights = class_weights / class_weights.mean().clamp_min(1e-6)
    history: list[dict[str, Any]] = []
    best_score = -1e9
    best_state = None
    gen = torch.Generator().manual_seed(args.seed)

    logger.info(
        "examples=%d train=%d val=%d train_gold=%s val_gold=%s",
        len(examples),
        len(train_idx),
        len(val_idx),
        torch.bincount(y_train, minlength=len(cutoffs)).tolist(),
        torch.bincount(y_val, minlength=len(cutoffs)).tolist(),
    )
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], generator=gen)
        total = 0.0
        for start in range(0, order.numel(), args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = model(x_train[idx].to(device))
            loss = F.cross_entropy(logits, y_train[idx].to(device), weight=class_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * int(idx.numel())
        val = _eval(model, x_val, y_val, cutoffs, device)
        val["epoch"] = epoch
        val["train_loss"] = total / max(int(order.numel()), 1)
        history.append(val)
        score = val["safe_or_later"] - 0.25 * val["too_early_rate"] - 0.002 * val["mean_pred_cutoff"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            logger.info("epoch=%d val=%s", epoch, val)

    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = _eval(model, x_train, y_train, cutoffs, device)
    val_metrics = _eval(model, x_val, y_val, cutoffs, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "cutoffs": cutoffs,
        "examples": len(examples),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "val_task_ids": sorted(val_task_ids),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
        "checkpoint": str(out_dir / "pi0fast_cutoff_selector.pt"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_cutoff_selector(out_dir / "pi0fast_cutoff_selector.pt", model.cpu(), extra=summary)
    print(json.dumps({"val_metrics": val_metrics, "checkpoint": summary["checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
