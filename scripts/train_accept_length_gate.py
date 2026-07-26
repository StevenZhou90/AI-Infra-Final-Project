#!/usr/bin/env python3
"""Train an accept-length classifier for direct trajectory chunk heads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.accept_length_gate import (  # noqa: E402
    AcceptLengthGate,
    AcceptLengthGateConfig,
    build_accept_length_features,
)
from serving.trajectory_draft_head import TinyTrajectoryHead  # noqa: E402
from serving.trajectory_phase import PhaseThresholds, label_phase  # noqa: E402


def parse_int_set(raw: str) -> set[int]:
    if not raw:
        return set()
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def longest_safe_prefix(
    pred: np.ndarray,
    target: np.ndarray,
    probs: np.ndarray,
    *,
    pos_tol: float,
    rot_tol: float,
    min_confident_tokens: int,
    confidence_threshold: float,
) -> int:
    length = 0
    horizon = min(pred.shape[0], target.shape[0])
    for idx in range(horizon):
        pos_ok = np.mean(np.abs(pred[idx, :3] - target[idx, :3])) <= pos_tol
        rot_ok = np.mean(np.abs(pred[idx, 3:6] - target[idx, 3:6])) <= rot_tol
        grip_ok = int(pred[idx, 6]) == int(target[idx, 6])
        conf_ok = int(np.sum(probs[idx] >= confidence_threshold)) >= min_confident_tokens
        if not (pos_ok and rot_ok and grip_ok and conf_ok):
            break
        length += 1
    return length


def build_examples(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(Path(args.data_dir) / "dataset.pt", map_location="cpu")
    examples = payload["examples"]
    history_size = int(payload.get("history_size", 4))
    vocab_size = int(args.vocab_size)
    smooth_head = TinyTrajectoryHead.load(args.smooth_head, device=args.device)
    complex_head = TinyTrajectoryHead.load(args.complex_head, device=args.device)
    thresholds = PhaseThresholds(
        smooth_curvature=args.smooth_phase_curvature,
        smooth_acceleration=args.smooth_phase_acceleration,
        min_displacement=args.smooth_phase_min_displacement,
    )

    groups: dict[tuple[str, int, int], dict[int, dict]] = {}
    for ex in examples:
        key = (str(ex["suite"]), int(ex["task_id"]), int(ex["trial"]))
        groups.setdefault(key, {})[int(ex["timestep"])] = ex

    val_task_ids = parse_int_set(args.val_task_ids)
    features = []
    labels = []
    tasks = []
    counts = {"smooth": 0, "complex": 0, "label_counts": {}}
    for ex in examples:
        hist = ex["history_bins"].long()
        if hist.shape[0] != history_size:
            continue
        phase = label_phase(hist, thresholds=thresholds)
        head = smooth_head if phase == "smooth" else complex_head
        horizon = int(head.config.action_horizon)
        key = (str(ex["suite"]), int(ex["task_id"]), int(ex["trial"]))
        t0 = int(ex["timestep"])
        targets = []
        for offset in range(horizon):
            nxt = groups[key].get(t0 + offset)
            if nxt is None:
                targets = []
                break
            targets.append(nxt["target_bins"].long().numpy())
        if not targets:
            continue

        bins = hist.to(args.device)
        top_bins, _top_probs, max_probs = head.predict(bins.unsqueeze(0), top_k=1)
        pred = top_bins[0, :, :, 0].detach().cpu().numpy()
        probs = max_probs[0].detach().cpu().numpy()
        target = np.stack(targets, axis=0)
        label = longest_safe_prefix(
            pred,
            target,
            probs,
            pos_tol=args.pos_tol,
            rot_tol=args.rot_tol,
            min_confident_tokens=args.min_confident_tokens,
            confidence_threshold=args.confidence_threshold,
        )
        feature = build_accept_length_features(
            history_bins=hist.numpy(),
            predicted_bins=pred,
            max_probs=probs,
            phase=phase,
            timestep=float(ex.get("timestep", 0)),
        )
        features.append(feature)
        labels.append(label)
        tasks.append(int(ex["task_id"]))
        counts[phase] += 1
        counts["label_counts"][str(label)] = counts["label_counts"].get(str(label), 0) + 1

    counts["val_task_ids"] = sorted(val_task_ids)
    return (
        torch.tensor(np.stack(features), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(tasks, dtype=torch.long),
        counts,
    )


def evaluate(model: AcceptLengthGate, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    under = 0
    over = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=-1)
            total += y.numel()
            correct += int((pred == y).sum().item())
            under += int((pred < y).sum().item())
            over += int((pred > y).sum().item())
            loss_sum += float(loss.item()) * y.numel()
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "under_rate": under / max(total, 1),
        "over_rate": over / max(total, 1),
    }


def class_weights(labels: torch.Tensor, max_accept: int, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels.cpu(), minlength=max_accept + 1).float().clamp_min(1.0)
    weights = counts.sum() / counts
    if mode == "sqrt":
        weights = torch.sqrt(weights)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train accept-length gate")
    parser.add_argument("--data-dir", default="artifacts/data/libero_goal/useful_6task_5trial_160")
    parser.add_argument("--smooth-head", required=True)
    parser.add_argument("--complex-head", required=True)
    parser.add_argument("--out-dir", default="artifacts/checkpoints/libero_goal/accept_length_gate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=32064)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--min-confident-tokens", type=int, default=6)
    parser.add_argument("--pos-tol", type=float, default=4.0)
    parser.add_argument("--rot-tol", type=float, default=6.0)
    parser.add_argument(
        "--val-task-ids",
        default="",
        help="Comma-separated task ids held out for validation. Defaults to random 80/20 split.",
    )
    parser.add_argument("--smooth-phase-curvature", type=float, default=18.0)
    parser.add_argument("--smooth-phase-acceleration", type=float, default=24.0)
    parser.add_argument("--smooth-phase-min-displacement", type=float, default=0.5)
    parser.add_argument(
        "--class-balance",
        choices=["none", "sqrt", "inverse"],
        default="none",
        help="Reweight labels so rare nonzero accept lengths are learned instead of predicting all zero.",
    )
    parser.add_argument(
        "--over-penalty",
        type=float,
        default=2.0,
        help="Model-selection penalty for validation overaccept rate.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    x, y, tasks, counts = build_examples(args)
    dataset = TensorDataset(x, y)
    val_task_ids = parse_int_set(args.val_task_ids)
    if val_task_ids:
        train_idx = torch.tensor([i for i, task in enumerate(tasks.tolist()) if task not in val_task_ids], dtype=torch.long)
        val_idx = torch.tensor([i for i, task in enumerate(tasks.tolist()) if task in val_task_ids], dtype=torch.long)
        if train_idx.numel() == 0 or val_idx.numel() == 0:
            raise ValueError(f"Invalid --val-task-ids={sorted(val_task_ids)} train={train_idx.numel()} val={val_idx.numel()}")
        train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
        split = f"task_heldout:{sorted(val_task_ids)}"
    else:
        n_val = max(1, int(0.2 * len(dataset)))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        split = "random:0.2"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    max_accept = max(int(y.max().item()), 1)
    print(
        f"examples={len(dataset)} split={split} train={len(train_ds)} val={len(val_ds)} "
        f"feature_dim={x.shape[1]} max_accept={max_accept} counts={counts}",
        flush=True,
    )
    cfg = AcceptLengthGateConfig(
        feature_dim=x.shape[1],
        max_accept=max_accept,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    model = AcceptLengthGate(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce_weight = class_weights(y, max_accept, args.class_balance, device)
    best = {"score": -1e9}
    history = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by, weight=ce_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        score = val_metrics["acc"] - args.over_penalty * val_metrics["over_rate"]
        print(
            f"epoch={epoch} train_acc={train_metrics['acc']:.3f} "
            f"val_acc={val_metrics['acc']:.3f} val_over={val_metrics['over_rate']:.3f} score={score:.3f}",
            flush=True,
        )
        if score > best["score"]:
            best = {"epoch": epoch, "score": score, **val_metrics}
            model.save(out_dir / "best.pt")

    model.save(out_dir / "last.pt")
    (out_dir / "metrics.json").write_text(
        json.dumps({"best": best, "history": history, "counts": counts, "split": split, "args": vars(args)}, indent=2)
        + "\n"
    )
    print(f"saved {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
