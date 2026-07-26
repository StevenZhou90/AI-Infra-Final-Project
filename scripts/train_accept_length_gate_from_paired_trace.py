#!/usr/bin/env python3
"""Train accept-length gate from paired AR/spec action traces."""

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


def load_action_trace(path: Path) -> dict[tuple[int, int, int], np.ndarray]:
    grouped: dict[tuple[int, int], dict[int, list[float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row["task_id"]), int(row["trial"]))
            grouped.setdefault(key, {})[int(row["action_step"])] = row["action"]
    out: dict[tuple[int, int, int], np.ndarray] = {}
    for (task_id, trial), steps in grouped.items():
        for step, action in steps.items():
            out[(task_id, trial, step)] = np.asarray(action, dtype=np.float32)
    return out


def safe_prefix_against_ar(
    *,
    task_id: int,
    trial: int,
    action_step: int,
    accepted_count: int,
    ar_actions: dict[tuple[int, int, int], np.ndarray],
    spec_actions: dict[tuple[int, int, int], np.ndarray],
    pos_tol: float,
    rot_tol: float,
    grip_tol: float,
) -> int:
    safe = 0
    for offset in range(1, accepted_count + 1):
        # Offset 1 is the first buffered action after the anchor model call.
        step = action_step + offset
        ar = ar_actions.get((task_id, trial, step))
        spec = spec_actions.get((task_id, trial, step))
        if ar is None or spec is None:
            break
        pos_ok = float(np.mean(np.abs(ar[:3] - spec[:3]))) <= pos_tol
        rot_ok = float(np.mean(np.abs(ar[3:6] - spec[3:6]))) <= rot_tol
        grip_ok = abs(float(ar[6]) - float(spec[6])) <= grip_tol
        if not (pos_ok and rot_ok and grip_ok):
            break
        safe += 1
    return safe


def build_dataset(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    ar_actions = load_action_trace(Path(args.ar_actions))
    spec_actions = load_action_trace(Path(args.spec_actions))
    features: list[np.ndarray] = []
    labels: list[int] = []
    trials: list[int] = []
    counts = {"label_counts": {}, "skipped_no_accept": 0, "examples": 0}
    with Path(args.chunk_decisions).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            accepted_count = int(row.get("accepted_count", 0))
            if accepted_count <= 0 and not args.include_zero_accept:
                counts["skipped_no_accept"] += 1
                continue
            history_tokens = np.asarray(row.get("history_tokens", []), dtype=np.int64)
            predicted_bins = np.asarray(row.get("predicted_bins", []), dtype=np.int64)
            max_probs = np.asarray(row.get("max_probs", []), dtype=np.float32)
            if history_tokens.ndim != 2 or predicted_bins.ndim != 2 or max_probs.ndim != 2:
                continue
            if len(predicted_bins) == 0:
                continue
            task_id = int(row["task_id"])
            trial = int(row["trial"])
            action_step = int(row["action_step"])
            label = safe_prefix_against_ar(
                task_id=task_id,
                trial=trial,
                action_step=action_step,
                accepted_count=min(accepted_count, args.max_accept),
                ar_actions=ar_actions,
                spec_actions=spec_actions,
                pos_tol=args.pos_tol,
                rot_tol=args.rot_tol,
                grip_tol=args.grip_tol,
            )
            if accepted_count == 0:
                label = 0
            history_bins = token_ids_to_bins_np(history_tokens, args.vocab_size)
            feature = build_accept_length_features(
                history_bins=history_bins,
                predicted_bins=predicted_bins,
                max_probs=max_probs,
                phase=str(row.get("phase", "complex")),
                timestep=float(action_step),
            )
            features.append(feature)
            labels.append(max(0, min(int(label), args.max_accept)))
            trials.append(trial)
            counts["examples"] += 1
            counts["label_counts"][str(labels[-1])] = counts["label_counts"].get(str(labels[-1]), 0) + 1
    if not features:
        raise ValueError("No paired-trace examples built")
    return (
        torch.tensor(np.stack(features), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(trials, dtype=torch.long),
        counts,
    )


def evaluate(model: AcceptLengthGate, loader: DataLoader, device: torch.device) -> dict:
    total = correct = under = over = 0
    loss_sum = 0.0
    pred_counts: dict[int, int] = {}
    model.eval()
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
    parser = argparse.ArgumentParser(description="Train accept gate from paired AR/spec traces.")
    parser.add_argument("--chunk-decisions", required=True)
    parser.add_argument("--ar-actions", required=True)
    parser.add_argument("--spec-actions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-accept", type=int, default=2)
    parser.add_argument("--pos-tol", type=float, default=0.04)
    parser.add_argument("--rot-tol", type=float, default=0.08)
    parser.add_argument("--grip-tol", type=float, default=0.1)
    parser.add_argument("--val-trials", default="8,9")
    parser.add_argument("--class-balance", choices=["none", "sqrt", "inverse"], default="sqrt")
    parser.add_argument("--over-penalty", type=float, default=2.0)
    parser.add_argument("--include-zero-accept", action="store_true")
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
