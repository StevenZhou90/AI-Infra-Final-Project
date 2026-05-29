#!/usr/bin/env python3
"""Train an accept-length gate from closed-loop chunk decision logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.accept_length_gate import AcceptLengthGate, AcceptLengthGateConfig, build_accept_length_features  # noqa: E402


def token_ids_to_bins_np(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    return (vocab_size - 1 - token_ids).clip(0, 255).astype(np.int64)


def build_dataset(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    trials: list[int] = []
    counts = {"success": 0, "failure": 0, "label_counts": {}}
    for log_path in args.logs:
        with Path(log_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                history_tokens = np.asarray(row.get("history_tokens", []), dtype=np.int64)
                predicted_bins = np.asarray(row.get("predicted_bins", []), dtype=np.int64)
                max_probs = np.asarray(row.get("max_probs", []), dtype=np.float32)
                if history_tokens.ndim != 2 or predicted_bins.ndim != 2 or max_probs.ndim != 2:
                    continue
                if len(predicted_bins) == 0:
                    continue
                history_bins = token_ids_to_bins_np(history_tokens, args.vocab_size)
                feature = build_accept_length_features(
                    history_bins=history_bins,
                    predicted_bins=predicted_bins,
                    max_probs=max_probs,
                    phase=str(row.get("phase", "complex")),
                    timestep=float(row.get("action_step", 0)),
                )
                accepted = int(row.get("accepted_count", 0))
                if bool(row.get("episode_success", False)):
                    label = accepted
                    counts["success"] += 1
                else:
                    label = max(0, accepted - int(args.failure_margin))
                    counts["failure"] += 1
                label = max(0, min(label, int(args.max_accept)))
                features.append(feature)
                labels.append(label)
                trials.append(int(row.get("trial", 0)))
                counts["label_counts"][str(label)] = counts["label_counts"].get(str(label), 0) + 1

    if not features:
        raise ValueError("No examples built from chunk logs")
    return (
        torch.tensor(np.stack(features), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(trials, dtype=torch.long),
        counts,
    )


def evaluate(model: AcceptLengthGate, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    under = 0
    over = 0
    loss_sum = 0.0
    pred_counts: dict[int, int] = {}
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
            for item in pred.cpu().tolist():
                pred_counts[int(item)] = pred_counts.get(int(item), 0) + 1
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "under_rate": under / max(total, 1),
        "over_rate": over / max(total, 1),
        "pred_counts": dict(sorted(pred_counts.items())),
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
    parser = argparse.ArgumentParser(description="Train accept-length gate from closed-loop chunk logs.")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-accept", type=int, default=2)
    parser.add_argument("--failure-margin", type=int, default=1)
    parser.add_argument("--val-trials", default="8,9")
    parser.add_argument("--class-balance", choices=["none", "sqrt", "inverse"], default="sqrt")
    parser.add_argument("--over-penalty", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    x, y, trials, counts = build_dataset(args)
    val_trials = {int(item) for item in args.val_trials.split(",") if item.strip()}
    train_idx = [i for i, trial in enumerate(trials.tolist()) if trial not in val_trials]
    val_idx = [i for i, trial in enumerate(trials.tolist()) if trial in val_trials]
    if not train_idx or not val_idx:
        raise ValueError(f"Bad --val-trials={sorted(val_trials)} train={len(train_idx)} val={len(val_idx)}")
    dataset = TensorDataset(x, y)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    cfg = AcceptLengthGateConfig(
        feature_dim=x.shape[1],
        max_accept=args.max_accept,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    model = AcceptLengthGate(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = class_weights(y, args.max_accept, args.class_balance, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1e9
    history = []

    print(
        f"examples={len(dataset)} train={len(train_ds)} val={len(val_ds)} "
        f"feature_dim={x.shape[1]} counts={counts}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx)
            loss = F.cross_entropy(logits, by, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        score = val_metrics["acc"] - args.over_penalty * val_metrics["over_rate"]
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": score})
        if score > best_score:
            best_score = score
            model.save(out_dir / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} train_acc={train_metrics['acc']:.3f} "
                f"val_acc={val_metrics['acc']:.3f} val_over={val_metrics['over_rate']:.3f} "
                f"val_pred={val_metrics['pred_counts']}",
                flush=True,
            )
    model.save(out_dir / "last.pt")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"counts": counts, "history": history, "best_score": best_score}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
