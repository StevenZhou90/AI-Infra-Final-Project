#!/usr/bin/env python3
"""Train an action-space DFlash-style chunk head from PI0-FAST traces."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_pi0fast_eagle_data import _ensure_libero_config  # noqa: E402
from scripts.run_pi0fast_chunk_eval import _import_lerobot, _to_numpy_action  # noqa: E402
from scripts.train_pi0fast_medusa_from_traces import _parse_data_dirs, load_all_records, split_indices  # noqa: E402
from serving.pi0fast_action_dflash import (  # noqa: E402
    PI0FastActionDFlashConfig,
    PI0FastActionDFlashHead,
    save_action_dflash_checkpoint,
)
from serving.pi0fast_eagle import PI0FastTraceRecord, _trim_record_tokens_and_hidden  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

logger = logging.getLogger("train_pi0fast_action_dflash")


@torch.inference_mode()
def build_action_rows(
    records: list[PI0FastTraceRecord],
    *,
    indices: list[int],
    adapter: PI0FastTokenLogitAdapter,
    postprocessor,
    action_horizon: int,
    device: torch.device,
    target_space: str,
    init_shift_actions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    init_rows: list[torch.Tensor] = []
    task_rows: list[int] = []
    cached_actions: dict[int, torch.Tensor] = {}

    def actions_for(record_idx: int) -> torch.Tensor | None:
        if record_idx in cached_actions:
            return cached_actions[record_idx]
        record = records[record_idx]
        token_ids_cpu, _hidden_states = _trim_record_tokens_and_hidden(record, (adapter.action_end_token_id,))
        token_ids = token_ids_cpu.unsqueeze(0).to(device)
        try:
            raw_actions = adapter._detokenize_generated_actions(token_ids)
            if target_space == "postprocessed":
                try:
                    raw_actions = postprocessor(raw_actions)
                except Exception:
                    pass
            actions = torch.as_tensor(_to_numpy_action(raw_actions), dtype=torch.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: detokenize failed: %r", record.trace_id, exc)
            return None
        if actions.ndim != 2 or actions.shape[0] < action_horizon:
            logger.warning("Skipping %s: action shape %s", record.trace_id, tuple(actions.shape))
            return None
        cached_actions[record_idx] = actions[:action_horizon].to(dtype=torch.float32)
        return cached_actions[record_idx]

    def shifted_prior(actions: torch.Tensor) -> torch.Tensor:
        shift = max(0, int(init_shift_actions))
        if shift <= 0:
            return actions.clone()
        prior = torch.empty_like(actions)
        if shift < actions.shape[0]:
            keep = actions.shape[0] - shift
            prior[:keep] = actions[shift:]
            prior[keep:] = actions[-1:].expand(shift, -1)
        else:
            prior[:] = actions[-1:]
        return prior

    for idx in indices:
        record = records[idx]
        _token_ids_cpu, hidden_states = _trim_record_tokens_and_hidden(record, (adapter.action_end_token_id,))
        actions = actions_for(idx)
        if actions is None:
            continue
        if hidden_states.shape[0] == 0:
            continue
        init = torch.zeros_like(actions)
        prev_idx = idx - 1
        if prev_idx >= 0 and records[prev_idx].task_id == record.task_id:
            prev_actions = actions_for(prev_idx)
            if prev_actions is not None:
                init = shifted_prior(prev_actions)
        hidden_rows.append(hidden_states[0].to(dtype=torch.float32))
        action_rows.append(actions)
        init_rows.append(init)
        task_rows.append(int(record.task_id))
    if not hidden_rows:
        raise ValueError("No action-DFlash rows built")
    return torch.stack(hidden_rows), torch.stack(action_rows), torch.stack(init_rows), torch.tensor(task_rows, dtype=torch.long)


def _corrupt_actions(target: torch.Tensor, init: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = target.shape[0]
    device = target.device
    noise_level = torch.rand((bsz, 1, 1), dtype=target.dtype, device=device)
    base = init if args.init_source == "previous" else target
    noisy = base + torch.randn_like(target) * float(args.noise_std) * noise_level
    if args.mask_prob > 0:
        mask = torch.rand((bsz, target.shape[1], 1), device=device) < float(args.mask_prob)
        noisy = torch.where(mask, torch.zeros_like(noisy), noisy)
    if args.random_prob > 0:
        rand_mask = torch.rand((bsz, target.shape[1], 1), device=device) < float(args.random_prob)
        scale = target.flatten(0, 1).std(dim=0).view(1, 1, -1).clamp_min(0.05)
        random_actions = torch.randn_like(noisy) * scale
        noisy = torch.where(rand_mask, random_actions, noisy)
    return noisy, noise_level.view(bsz, 1)


def _accept_lengths(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    err = pred - target
    pos = torch.linalg.norm(err[:, :, :3], dim=-1)
    rot = torch.linalg.norm(err[:, :, 3:6], dim=-1) if pred.shape[-1] >= 6 else torch.zeros_like(pos)
    grip = torch.ones_like(pos, dtype=torch.bool)
    if pred.shape[-1] >= 7:
        grip = torch.sign(pred[:, :, 6]) == torch.sign(target[:, :, 6])
    within = (pos <= args.accept_pos_l2) & (rot <= args.accept_rot_l2) & grip
    return within.cumprod(dim=1).sum(dim=1).to(dtype=torch.float32)


def _metrics(pred: torch.Tensor, target: torch.Tensor, task_ids: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    err = pred - target
    pos = torch.linalg.norm(err[:, :, :3], dim=-1)
    rot = torch.linalg.norm(err[:, :, 3:6], dim=-1) if pred.shape[-1] >= 6 else torch.zeros_like(pos)
    grip_mismatch = torch.zeros_like(pos)
    if pred.shape[-1] >= 7:
        grip_mismatch = (torch.sign(pred[:, :, 6]) != torch.sign(target[:, :, 6])).to(dtype=torch.float32)
    accept = _accept_lengths(pred, target, args)
    velocity = torch.diff(pred, dim=1)
    jerk = torch.diff(pred, n=2, dim=1)
    out: dict[str, Any] = {
        "rows": int(pred.shape[0]),
        "mae": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt(err.square().mean()).item()),
        "max_abs_mean": float(err.abs().amax(dim=(1, 2)).mean().item()),
        "max_pos_l2_mean": float(pos.max(dim=1).values.mean().item()),
        "max_rot_l2_mean": float(rot.max(dim=1).values.mean().item()),
        "gripper_clean_rate": float((grip_mismatch.sum(dim=1) == 0).to(dtype=torch.float32).mean().item()),
        "mean_accept": float(accept.mean().item()),
        "p_accept_ge_3": float((accept >= 3).to(dtype=torch.float32).mean().item()),
        "p_accept_ge_5": float((accept >= 5).to(dtype=torch.float32).mean().item()),
        "p_accept_full": float((accept >= pred.shape[1]).to(dtype=torch.float32).mean().item()),
        "max_velocity_mean": float(velocity.abs().amax(dim=(1, 2)).mean().item()) if velocity.numel() else 0.0,
        "max_jerk_mean": float(jerk.abs().amax(dim=(1, 2)).mean().item()) if jerk.numel() else 0.0,
    }
    by_task: dict[str, Any] = {}
    for task_id in sorted(set(int(v) for v in task_ids.tolist())):
        mask = task_ids == task_id
        if int(mask.sum().item()) == pred.shape[0]:
            continue
        task_pred = pred[mask]
        task_target = target[mask]
        task_accept = _accept_lengths(task_pred, task_target, args)
        by_task[str(task_id)] = {
            "rows": int(mask.sum().item()),
            "mean_accept": float(task_accept.mean().item()),
            "p_accept_full": float((task_accept >= task_pred.shape[1]).to(dtype=torch.float32).mean().item()),
            "mae": float((task_pred - task_target).abs().mean().item()),
        }
    out["by_task"] = by_task
    return out


@torch.inference_mode()
def evaluate(
    model: PI0FastActionDFlashHead,
    hidden: torch.Tensor,
    target: torch.Tensor,
    init: torch.Tensor,
    task_ids: torch.Tensor,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    preds = []
    confs = []
    for start in range(0, hidden.shape[0], args.batch_size):
        h = hidden[start : start + args.batch_size].to(device)
        init_batch = init[start : start + args.batch_size].to(device) if args.init_source == "previous" else None
        pred, conf, _stats = model.draft(h, refine_steps=args.eval_refine_steps, init=init_batch)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
    pred = torch.cat(preds, dim=0)
    conf = torch.cat(confs, dim=0)
    metrics = _metrics(pred, target, task_ids, args)
    metrics["mean_confidence"] = float(conf.mean().item())
    metrics["min_confidence_mean"] = float(conf.min(dim=1).values.mean().item())
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_libero_config(args.libero_config_path)
    _make_env, _make_env_pre_post_processors, _preprocess_observation, _LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()
    records = load_all_records(_parse_data_dirs(args.data_dirs))
    train_idx, val_idx = split_indices(records, args)

    device = torch.device(args.train_device if torch.cuda.is_available() or not args.train_device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    logger.info("Loading policy %s for FAST detokenization", args.policy)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    _preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    X_train, y_train, init_train, train_tasks = build_action_rows(
        records,
        indices=train_idx,
        adapter=adapter,
        postprocessor=postprocessor,
        action_horizon=args.action_horizon,
        device=device,
        target_space=args.target_space,
        init_shift_actions=args.init_shift_actions,
    )
    X_val, y_val, init_val, val_tasks = build_action_rows(
        records,
        indices=val_idx,
        adapter=adapter,
        postprocessor=postprocessor,
        action_horizon=args.action_horizon,
        device=device,
        target_space=args.target_space,
        init_shift_actions=args.init_shift_actions,
    )
    logger.info("rows train=%d val=%d hidden=%d action=%s", X_train.shape[0], X_val.shape[0], X_train.shape[-1], tuple(y_train.shape[1:]))

    cfg = PI0FastActionDFlashConfig(
        hidden_dim=int(X_train.shape[-1]),
        action_horizon=args.action_horizon,
        action_dim=int(y_train.shape[-1]),
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        action_residual_scale=args.action_residual_scale,
    )
    model = PI0FastActionDFlashHead(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    prefix_weights = torch.pow(
        torch.full((args.action_horizon,), float(args.prefix_loss_decay), device=device),
        torch.arange(args.action_horizon, device=device),
    ).clamp_min(args.prefix_loss_floor)
    prefix_weights = prefix_weights / prefix_weights.mean().clamp_min(1e-6)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(X_train.shape[0])
        total_loss = 0.0
        total_rows = 0
        for start in range(0, perm.numel(), args.batch_size):
            idx = perm[start : start + args.batch_size]
            h = X_train[idx].to(device)
            target = y_train[idx].to(device)
            init = init_train[idx].to(device)
            noisy, noise_level = _corrupt_actions(target, init, args)
            pred, conf = model(h, noisy, noise_level)
            action_loss = F.smooth_l1_loss(pred, target, beta=args.huber_beta, reduction="none")
            action_loss = (action_loss.mean(dim=-1) * prefix_weights.view(1, -1)).mean()
            velocity = torch.diff(pred, dim=1)
            target_velocity = torch.diff(target, dim=1)
            velocity_loss = F.smooth_l1_loss(velocity, target_velocity, beta=args.huber_beta) if velocity.numel() else pred.sum() * 0.0
            jerk = torch.diff(pred, n=2, dim=1)
            jerk_loss = jerk.square().mean() if jerk.numel() else pred.sum() * 0.0
            grip_loss = pred.sum() * 0.0
            grip_sign_loss = pred.sum() * 0.0
            if pred.shape[-1] >= 7:
                grip_loss = F.smooth_l1_loss(pred[:, :, 6], target[:, :, 6], beta=args.huber_beta)
                target_sign = torch.where(target[:, :, 6] >= 0, 1.0, -1.0).to(dtype=pred.dtype, device=pred.device)
                grip_sign_loss = F.softplus(-target_sign * pred[:, :, 6] / max(args.gripper_margin, 1e-6)).mean()
            with torch.no_grad():
                err = pred.detach() - target
                pos = torch.linalg.norm(err[:, :, :3], dim=-1)
                rot = torch.linalg.norm(err[:, :, 3:6], dim=-1) if pred.shape[-1] >= 6 else torch.zeros_like(pos)
                grip_ok = torch.ones_like(pos, dtype=torch.bool)
                if pred.shape[-1] >= 7:
                    grip_ok = torch.sign(pred[:, :, 6]) == torch.sign(target[:, :, 6])
                conf_target = ((pos <= args.accept_pos_l2) & (rot <= args.accept_rot_l2) & grip_ok).to(dtype=conf.dtype)
            conf_loss = F.binary_cross_entropy(conf.clamp(1e-4, 1.0 - 1e-4), conf_target)
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
            total_rows += int(idx.numel())

        val = evaluate(model, X_val, y_val, init_val, val_tasks, args, device=device)
        score = (
            float(val["mean_accept"])
            + args.full_accept_score_weight * float(val["p_accept_full"])
            - args.mae_score_weight * float(val["mae"])
        )
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_rows, 1), "val": {k: v for k, v in val.items() if k != "by_task"}})
        logger.info(
            "epoch=%d loss=%.5f val_accept=%.2f val_full=%.3f val_mae=%.4f conf=%.3f",
            epoch,
            total_loss / max(total_rows, 1),
            val["mean_accept"],
            val["p_accept_full"],
            val["mae"],
            val["mean_confidence"],
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = evaluate(model, X_train, y_train, init_train, train_tasks, args, device=device)
    final_val = evaluate(model, X_val, y_val, init_val, val_tasks, args, device=device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "model_kind": "action_dflash_head",
        "records": len(records),
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "history": history,
        "final_train_metrics": final_train,
        "final_val_metrics": final_val,
        "best_epoch": max(
            history,
            key=lambda row: row["val"]["mean_accept"]
            + args.full_accept_score_weight * row["val"]["p_accept_full"]
            - args.mae_score_weight * row["val"]["mae"],
        )
        if history
        else None,
        "elapsed_s": time.perf_counter() - START_TIME,
        "checkpoint": str(out_dir / "pi0fast_action_dflash.pt"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_action_dflash_checkpoint(
        out_dir / "pi0fast_action_dflash.pt",
        model.cpu(),
        extra=summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST action-space DFlash chunk head")
    parser.add_argument("--data-dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--split", choices=["trace", "task"], default="task")
    parser.add_argument("--heldout-task-ids", default="2,5,1002,1005,2002,2005")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--target-space", choices=["raw", "postprocessed"], default="raw")
    parser.add_argument("--init-source", choices=["zero", "previous"], default="zero")
    parser.add_argument(
        "--init-shift-actions",
        type=int,
        default=0,
        help="When using the previous chunk as DFlash prior, shift off this many already-executed actions.",
    )
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--action-residual-scale", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.35)
    parser.add_argument("--mask-prob", type=float, default=0.20)
    parser.add_argument("--random-prob", type=float, default=0.04)
    parser.add_argument("--eval-refine-steps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--huber-beta", type=float, default=0.02)
    parser.add_argument("--prefix-loss-decay", type=float, default=0.92)
    parser.add_argument("--prefix-loss-floor", type=float, default=0.45)
    parser.add_argument("--velocity-weight", type=float, default=0.08)
    parser.add_argument("--jerk-weight", type=float, default=0.02)
    parser.add_argument("--gripper-weight", type=float, default=0.20)
    parser.add_argument("--gripper-sign-weight", type=float, default=0.0)
    parser.add_argument("--gripper-margin", type=float, default=0.20)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument("--full-accept-score-weight", type=float, default=5.0)
    parser.add_argument("--mae-score-weight", type=float, default=0.02)
    parser.add_argument("--accept-pos-l2", type=float, default=0.05)
    parser.add_argument("--accept-rot-l2", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


START_TIME = time.perf_counter()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
