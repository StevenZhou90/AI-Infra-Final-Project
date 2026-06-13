#!/usr/bin/env python3
"""Train a small PI0-FAST continuous trajectory-tail draft head."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_trajectory_head import (  # noqa: E402
    PI0FastTrajectoryTailConfig,
    PI0FastTrajectoryTailHead,
    save_trajectory_tail_checkpoint,
)

logger = logging.getLogger("train_pi0fast_trajectory_head")


def _parse_data_dirs(value: str) -> list[Path]:
    dirs = [Path(part.strip()) for part in value.split(",") if part.strip()]
    if not dirs:
        raise ValueError("--data-dirs must include at least one directory")
    return dirs


def load_rows(data_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_idx, data_dir in enumerate(data_dirs):
        for shard in sorted(data_dir.glob("shard_*.pt")):
            shard_rows = torch.load(shard, map_location="cpu", weights_only=False)
            for row in shard_rows:
                copied = dict(row)
                copied["task_id"] = suite_idx * 1000 + int(row.get("task_id", -1))
                copied["trace_id"] = f"{data_dir.name}/{row.get('trace_id', shard.stem)}"
                rows.append(copied)
    if not rows:
        raise FileNotFoundError(f"No rows found in {', '.join(str(d) for d in data_dirs)}")
    return rows


def split_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[int], list[int]]:
    if args.split == "task":
        heldout = {int(part.strip()) for part in args.heldout_task_ids.split(",") if part.strip()}
        train = [idx for idx, row in enumerate(rows) if int(row["task_id"]) not in heldout]
        val = [idx for idx, row in enumerate(rows) if int(row["task_id"]) in heldout]
    else:
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(rows), generator=generator).tolist()
        n_val = max(1, int(round(len(rows) * args.val_fraction)))
        n_val = min(n_val, len(rows) - 1)
        val_set = set(perm[:n_val])
        train = [idx for idx in range(len(rows)) if idx not in val_set]
        val = [idx for idx in range(len(rows)) if idx in val_set]
    if not train or not val:
        raise ValueError(f"Empty split: train={len(train)} val={len(val)}")
    return train, val


def tensors_for(rows: list[dict[str, Any]], indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.stack([rows[idx]["input_chunk"].to(dtype=torch.float32) for idx in indices])
    y = torch.stack([rows[idx]["target_tail"].to(dtype=torch.float32) for idx in indices])
    task_ids = torch.tensor([int(rows[idx]["task_id"]) for idx in indices], dtype=torch.long)
    return X, y, task_ids


def damped_tail(chunk: torch.Tensor, tail_horizon: int, damping: float) -> torch.Tensor:
    last = chunk[:, -1].clone()
    delta = chunk[:, -1] - chunk[:, -2]
    if delta.shape[-1] >= 7:
        delta[:, 6] = 0.0
    outs = []
    for _ in range(tail_horizon):
        delta = delta * damping
        last = (last + delta).clamp(-1.0, 1.0)
        if last.shape[-1] >= 7:
            last[:, 6] = chunk[:, -1, 6]
        outs.append(last.clone())
    return torch.stack(outs, dim=1)


@torch.inference_mode()
def evaluate(model: PI0FastTrajectoryTailHead, X: torch.Tensor, y: torch.Tensor, task_ids: torch.Tensor, *, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds = []
    for start in range(0, X.shape[0], batch_size):
        preds.append(model(X[start : start + batch_size].to(device)).cpu())
    pred = torch.cat(preds, dim=0)
    damped = damped_tail(X, y.shape[1], damping=0.65)
    metrics = _metrics(pred, y, task_ids)
    metrics["damped_baseline"] = _metrics(damped, y, task_ids, include_by_task=False)
    return metrics


def _metrics(pred: torch.Tensor, y: torch.Tensor, task_ids: torch.Tensor, *, include_by_task: bool = True) -> dict[str, Any]:
    err = pred - y
    pos = torch.linalg.norm(err[:, :, :3], dim=-1)
    rot = torch.linalg.norm(err[:, :, 3:6], dim=-1) if pred.shape[-1] >= 6 else torch.zeros_like(pos)
    grip_match = torch.ones_like(pos, dtype=torch.bool)
    if pred.shape[-1] >= 7:
        grip_match = torch.isclose(pred[:, :, 6], y[:, :, 6], atol=1e-4, rtol=0.0)
    within = (pos <= 0.035) & (rot <= 0.25) & grip_match
    row_accept = within.cumprod(dim=1).sum(dim=1).to(dtype=torch.float32)
    step_delta = torch.diff(pred, dim=1)
    if step_delta.shape[1] > 0:
        pos_step = torch.linalg.norm(step_delta[:, :, :3], dim=-1).max(dim=1).values
        rot_step = torch.linalg.norm(step_delta[:, :, 3:6], dim=-1).max(dim=1).values if pred.shape[-1] >= 6 else torch.zeros_like(pos_step)
    else:
        pos_step = torch.zeros(pred.shape[0])
        rot_step = torch.zeros(pred.shape[0])
    out: dict[str, Any] = {
        "rows": int(pred.shape[0]),
        "mae": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(err.square())).item()),
        "max_pos_err_mean": float(pos.max(dim=1).values.mean().item()),
        "max_rot_err_mean": float(rot.max(dim=1).values.mean().item()),
        "mean_relaxed_tail_accept": float(row_accept.mean().item()),
        "p_accept_ge_2": float((row_accept >= 2).to(dtype=torch.float32).mean().item()),
        "p_accept_ge_4": float((row_accept >= 4).to(dtype=torch.float32).mean().item()),
        "p_accept_full": float((row_accept >= pred.shape[1]).to(dtype=torch.float32).mean().item()),
        "max_tail_pos_step_mean": float(pos_step.mean().item()),
        "max_tail_rot_step_mean": float(rot_step.mean().item()),
    }
    if include_by_task:
        by_task: dict[str, Any] = {}
        for task_id in sorted(set(int(t) for t in task_ids.tolist())):
            mask = task_ids == task_id
            by_task[str(task_id)] = _metrics(pred[mask], y[mask], task_ids[mask], include_by_task=False)
        out["by_task"] = by_task
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(_parse_data_dirs(args.data_dirs))
    train_idx, val_idx = split_rows(rows, args)
    X_train, y_train, _train_tasks = tensors_for(rows, train_idx)
    X_val, y_val, val_tasks = tensors_for(rows, val_idx)

    cfg = PI0FastTrajectoryTailConfig(
        input_horizon=args.input_horizon,
        tail_horizon=args.tail_horizon,
        action_dim=int(X_train.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        residual=not args.no_residual,
        residual_from_damped=args.residual_from_damped,
        residual_scale=args.residual_scale,
        damping=args.damping,
        freeze_gripper=not args.no_freeze_gripper,
    )
    device = torch.device(args.train_device if torch.cuda.is_available() or not args.train_device.startswith("cuda") else "cpu")
    model = PI0FastTrajectoryTailHead(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(X_train.shape[0])
        total_loss = 0.0
        total_rows = 0
        for start in range(0, perm.numel(), args.batch_size):
            idx = perm[start : start + args.batch_size]
            x = X_train[idx].to(device)
            target = y_train[idx].to(device)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, target, beta=args.huber_beta)
            velocity = torch.diff(torch.cat([x[:, -1:, :], pred], dim=1), dim=1)
            smooth_loss = velocity.square().mean()
            jerk = torch.diff(torch.cat([x[:, -2:, :], pred], dim=1), n=2, dim=1)
            jerk_loss = jerk.square().mean()
            first_step_loss = (pred[:, 0, :6] - x[:, -1, :6]).square().mean()
            loss = loss + args.smoothness_weight * smooth_loss
            loss = loss + args.jerk_weight * jerk_loss
            loss = loss + args.first_step_weight * first_step_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(idx.numel())
            total_rows += int(idx.numel())

        val_metrics = evaluate(model, X_val, y_val, val_tasks, batch_size=args.batch_size, device=device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            "val": {k: v for k, v in val_metrics.items() if k != "by_task"},
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.6f val_mae=%.4f val_accept=%.2f damped_accept=%.2f",
            epoch,
            row["train_loss"],
            val_metrics["mae"],
            val_metrics["mean_relaxed_tail_accept"],
            val_metrics["damped_baseline"]["mean_relaxed_tail_accept"],
        )
        score = float(val_metrics["mae"]) - args.accept_score_weight * float(val_metrics["mean_relaxed_tail_accept"])
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = evaluate(model, X_val, y_val, val_tasks, batch_size=args.batch_size, device=device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_tail_checkpoint(
        out_dir / "pi0fast_trajectory_tail.pt",
        model.cpu(),
        extra={"args": vars(args), "train_rows": len(train_idx), "val_rows": len(val_idx), "val_metrics": val_metrics},
    )
    summary = {
        "config": vars(args),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "history": history,
        "val_metrics": val_metrics,
        "checkpoint": str(out_dir / "pi0fast_trajectory_tail.pt"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST trajectory-tail draft head")
    parser.add_argument("--data-dirs", required=True)
    parser.add_argument("--output-dir", default="outputs/pi0fast_trajectory_head")
    parser.add_argument("--split", choices=["trace", "task"], default="trace")
    parser.add_argument("--heldout-task-ids", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--input-horizon", type=int, default=10)
    parser.add_argument("--tail-horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--residual-from-damped", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.04)
    parser.add_argument("--damping", type=float, default=0.65)
    parser.add_argument("--no-freeze-gripper", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--jerk-weight", type=float, default=0.0)
    parser.add_argument("--first-step-weight", type=float, default=0.0)
    parser.add_argument("--accept-score-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
