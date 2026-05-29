#!/usr/bin/env python3
"""Train a DFlash-style action corrector for PI0-FAST cutoff candidates."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_action_dflash import (  # noqa: E402
    PI0FastActionDFlashConfig,
    PI0FastActionDFlashHead,
    save_action_dflash_checkpoint,
)

logger = logging.getLogger("train_pi0fast_action_corrector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST cutoff action corrector")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-task-ids", default="2,5")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--action-residual-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--huber-beta", type=float, default=0.02)
    parser.add_argument("--init-noise-std", type=float, default=0.08)
    parser.add_argument("--mask-prob", type=float, default=0.03)
    parser.add_argument("--prefix-loss-decay", type=float, default=0.88)
    parser.add_argument("--prefix-loss-floor", type=float, default=0.35)
    parser.add_argument("--velocity-weight", type=float, default=0.10)
    parser.add_argument("--jerk-weight", type=float, default=0.03)
    parser.add_argument("--gripper-weight", type=float, default=0.25)
    parser.add_argument("--gripper-sign-weight", type=float, default=0.10)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument("--eval-refine-steps", type=int, default=2)
    parser.add_argument(
        "--preserve-prefix-actions",
        type=int,
        default=0,
        help="Force the first N corrected actions to remain equal to the cutoff candidate.",
    )
    parser.add_argument(
        "--eval-blend",
        type=float,
        default=1.0,
        help="Blend evaluated prediction as init + blend * (pred - init).",
    )
    parser.add_argument("--accept-pos-l2", type=float, default=0.05)
    parser.add_argument("--accept-rot-l2", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_ids(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = list(payload["rows"])
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows, payload


def _split(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[int], list[int]]:
    val_task_ids = _parse_ids(args.val_task_ids)
    if val_task_ids:
        train = [idx for idx, row in enumerate(rows) if int(row["task_id"]) not in val_task_ids]
        val = [idx for idx, row in enumerate(rows) if int(row["task_id"]) in val_task_ids]
        if train and val:
            return train, val
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - args.val_frac)))
    return indices[:split], indices[split:]


def _tensorize(rows: list[dict[str, Any]], indices: list[int], horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.stack([rows[idx]["hidden"].float() for idx in indices])
    init = torch.stack([rows[idx]["init_actions"].float()[:horizon] for idx in indices])
    target = torch.stack([rows[idx]["target_actions"].float()[:horizon] for idx in indices])
    task_ids = torch.tensor([int(rows[idx]["task_id"]) for idx in indices], dtype=torch.long)
    cutoffs = torch.tensor([int(rows[idx]["cutoff"]) for idx in indices], dtype=torch.long)
    return hidden, init, target, task_ids, cutoffs


def _accept_lengths(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    err = pred - target
    pos = torch.linalg.norm(err[:, :, :3], dim=-1)
    rot = torch.linalg.norm(err[:, :, 3:6], dim=-1) if pred.shape[-1] >= 6 else torch.zeros_like(pos)
    grip_ok = torch.ones_like(pos, dtype=torch.bool)
    if pred.shape[-1] >= 7:
        grip_ok = torch.sign(pred[:, :, 6]) == torch.sign(target[:, :, 6])
    within = (pos <= args.accept_pos_l2) & (rot <= args.accept_rot_l2) & grip_ok
    return within.cumprod(dim=1).sum(dim=1).float()


def _metrics(pred: torch.Tensor, target: torch.Tensor, init: torch.Tensor, cutoffs: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    err = pred - target
    accept = _accept_lengths(pred, target, args)
    init_accept = _accept_lengths(init, target, args)
    out: dict[str, Any] = {
        "rows": int(pred.shape[0]),
        "mae": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt(err.square().mean()).item()),
        "init_mae": float((init - target).abs().mean().item()),
        "mean_accept": float(accept.mean().item()),
        "init_mean_accept": float(init_accept.mean().item()),
        "p_accept_ge_3": float((accept >= 3).float().mean().item()),
        "p_accept_ge_5": float((accept >= 5).float().mean().item()),
        "p_accept_full": float((accept >= pred.shape[1]).float().mean().item()),
        "init_p_accept_full": float((init_accept >= pred.shape[1]).float().mean().item()),
    }
    by_cutoff: dict[str, Any] = {}
    for cutoff in sorted(set(int(v) for v in cutoffs.tolist())):
        mask = cutoffs == cutoff
        if not int(mask.sum().item()):
            continue
        pred_c = pred[mask]
        target_c = target[mask]
        init_c = init[mask]
        accept_c = _accept_lengths(pred_c, target_c, args)
        init_accept_c = _accept_lengths(init_c, target_c, args)
        by_cutoff[str(cutoff)] = {
            "rows": int(mask.sum().item()),
            "mean_accept": float(accept_c.mean().item()),
            "init_mean_accept": float(init_accept_c.mean().item()),
            "p_accept_full": float((accept_c >= pred_c.shape[1]).float().mean().item()),
            "init_p_accept_full": float((init_accept_c >= pred_c.shape[1]).float().mean().item()),
            "mae": float((pred_c - target_c).abs().mean().item()),
        }
    out["by_cutoff"] = by_cutoff
    return out


@torch.no_grad()
def evaluate(
    model: PI0FastActionDFlashHead,
    hidden: torch.Tensor,
    init: torch.Tensor,
    target: torch.Tensor,
    cutoffs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    preds = []
    confs = []
    for start in range(0, hidden.shape[0], args.batch_size):
        h = hidden[start : start + args.batch_size].to(device)
        init_b = init[start : start + args.batch_size].to(device)
        pred, conf, _stats = model.draft(h, refine_steps=args.eval_refine_steps, init=init_b)
        blend = float(np.clip(args.eval_blend, 0.0, 1.0))
        if blend < 1.0:
            pred = init_b + blend * (pred - init_b)
        preserve = min(max(int(args.preserve_prefix_actions), 0), pred.shape[1])
        if preserve:
            pred = torch.cat([init_b[:, :preserve], pred[:, preserve:]], dim=1)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
    pred = torch.cat(preds, dim=0)
    conf = torch.cat(confs, dim=0)
    metrics = _metrics(pred, target, init, cutoffs, args)
    metrics["mean_confidence"] = float(conf.mean().item())
    metrics["min_confidence_mean"] = float(conf.min(dim=1).values.mean().item())
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    rows, payload = _load_rows(Path(args.data))
    train_idx, val_idx = _split(rows, args)
    x_train, init_train, y_train, _train_tasks, cutoff_train = _tensorize(rows, train_idx, args.action_horizon)
    x_val, init_val, y_val, _val_tasks, cutoff_val = _tensorize(rows, val_idx, args.action_horizon)
    logger.info(
        "rows train=%d val=%d init_accept=%.2f val_init_accept=%.2f",
        x_train.shape[0],
        x_val.shape[0],
        float(_accept_lengths(init_train, y_train, args).mean().item()),
        float(_accept_lengths(init_val, y_val, args).mean().item()),
    )

    cfg = PI0FastActionDFlashConfig(
        hidden_dim=int(x_train.shape[-1]),
        action_horizon=int(y_train.shape[1]),
        action_dim=int(y_train.shape[2]),
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        action_residual_scale=args.action_residual_scale,
    )
    model = PI0FastActionDFlashHead(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    prefix_weights = torch.pow(
        torch.full((cfg.action_horizon,), float(args.prefix_loss_decay), device=device),
        torch.arange(cfg.action_horizon, device=device),
    ).clamp_min(args.prefix_loss_floor)
    prefix_weights = prefix_weights / prefix_weights.mean().clamp_min(1e-6)
    best_score = -1e9
    initial_val = evaluate(model, x_val, init_val, y_val, cutoff_val, args, device)
    best_score = initial_val["mean_accept"] + 5.0 * initial_val["p_accept_full"] - 0.05 * initial_val["mae"]
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(x_train.shape[0])
        total_loss = 0.0
        for start in range(0, order.numel(), args.batch_size):
            idx = order[start : start + args.batch_size]
            h = x_train[idx].to(device)
            init = init_train[idx].to(device)
            target = y_train[idx].to(device)
            noise_level = torch.rand((idx.numel(), 1), device=device, dtype=target.dtype)
            noisy = init + torch.randn_like(init) * args.init_noise_std * noise_level.view(-1, 1, 1)
            if args.mask_prob > 0:
                mask = torch.rand((idx.numel(), cfg.action_horizon, 1), device=device) < args.mask_prob
                noisy = torch.where(mask, init, noisy)
            pred, conf = model(h, noisy, noise_level)
            preserve = min(max(int(args.preserve_prefix_actions), 0), pred.shape[1])
            if preserve:
                pred = torch.cat([init[:, :preserve], pred[:, preserve:]], dim=1)
            action_loss = F.smooth_l1_loss(pred, target, beta=args.huber_beta, reduction="none")
            action_loss = (action_loss.mean(dim=-1) * prefix_weights.view(1, -1)).mean()
            velocity_loss = F.smooth_l1_loss(torch.diff(pred, dim=1), torch.diff(target, dim=1), beta=args.huber_beta)
            jerk = torch.diff(pred, n=2, dim=1)
            jerk_loss = jerk.square().mean() if jerk.numel() else pred.sum() * 0.0
            grip_loss = pred.sum() * 0.0
            grip_sign_loss = pred.sum() * 0.0
            if pred.shape[-1] >= 7:
                grip_loss = F.smooth_l1_loss(pred[:, :, 6], target[:, :, 6], beta=args.huber_beta)
                target_sign = torch.where(target[:, :, 6] >= 0, 1.0, -1.0).to(dtype=pred.dtype)
                grip_sign_loss = F.softplus(-target_sign * pred[:, :, 6] / 0.2).mean()
            with torch.no_grad():
                good = (_accept_lengths(pred.detach(), target, args).unsqueeze(1) > torch.arange(cfg.action_horizon, device=device)).float()
            conf_loss = F.binary_cross_entropy(conf.clamp(1e-4, 1 - 1e-4), good)
            loss = (
                action_loss
                + args.velocity_weight * velocity_loss
                + args.jerk_weight * jerk_loss
                + args.gripper_weight * grip_loss
                + args.gripper_sign_weight * grip_sign_loss
                + args.confidence_weight * conf_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(idx.numel())
        val = evaluate(model, x_val, init_val, y_val, cutoff_val, args, device)
        score = val["mean_accept"] + 5.0 * val["p_accept_full"] - 0.05 * val["mae"]
        row = {"epoch": epoch, "train_loss": total_loss / max(int(order.numel()), 1), "val": val}
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info(
                "epoch=%d loss=%.4f init_accept=%.2f pred_accept=%.2f full=%.3f mae=%.4f",
                epoch,
                row["train_loss"],
                val["init_mean_accept"],
                val["mean_accept"],
                val["p_accept_full"],
                val["mae"],
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = evaluate(model, x_train, init_train, y_train, cutoff_train, args, device)
    final_val = evaluate(model, x_val, init_val, y_val, cutoff_val, args, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "source_data_config": payload.get("config", {}),
        "rows": len(rows),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "val_task_ids": sorted(_parse_ids(args.val_task_ids)),
        "initial_val_metrics": initial_val,
        "history": history,
        "best_epoch": max(history, key=lambda item: item["val"]["mean_accept"] + 5.0 * item["val"]["p_accept_full"] - 0.05 * item["val"]["mae"]),
        "final_train_metrics": final_train,
        "final_val_metrics": final_val,
        "elapsed_s": time.perf_counter() - start_time,
        "checkpoint": str(out_dir / "pi0fast_action_corrector.pt"),
        "target_space": "postprocessed",
        "model_kind": "action_dflash_corrector",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_action_dflash_checkpoint(out_dir / "pi0fast_action_corrector.pt", model.cpu(), extra=summary)
    print(json.dumps({"final_val_metrics": final_val, "checkpoint": summary["checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
