#!/usr/bin/env python3
"""Train TinyTrajectoryHead on OpenVLA action-bin trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.trajectory_draft_head import TinyTrajectoryHead, TrajectoryHeadConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_trajectory_head")


class TrajectoryDataset(Dataset):
    def __init__(self, path: Path):
        data = torch.load(path, map_location="cpu")
        self.examples = data["examples"]
        self.history_size = int(data.get("history_size", 4))
        self.llm_hidden_size = int(data.get("llm_hidden_size", 4096))
        self.use_prefill_hidden = bool(data.get("use_prefill_hidden"))
        if self.examples:
            self.use_prefill_hidden = self.use_prefill_hidden and ("prefill_hidden" in self.examples[0])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        h = ex["history_bins"].long()
        t = ex["target_bins"].long()
        timestep = torch.tensor(float(ex.get("timestep", 0)), dtype=torch.float32)
        if self.use_prefill_hidden:
            ph = ex["prefill_hidden"].float()
            return h, t, ph, timestep
        return h, t, timestep


def metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    pred = logits.argmax(dim=-1)
    top3 = logits.topk(k=3, dim=-1).indices
    top5 = logits.topk(k=5, dim=-1).indices
    acc_dim = (pred == target).float().mean(dim=0)
    return {
        "loss": float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)).item()),
        "top1": float((pred == target).float().mean().item()),
        "top3": float((top3 == target.unsqueeze(-1)).any(dim=-1).float().mean().item()),
        "top5": float((top5 == target.unsqueeze(-1)).any(dim=-1).float().mean().item()),
        "per_dim_top1": [float(x) for x in acc_dim.tolist()],
    }


def parse_dim_weights(raw: str, action_dim: int, device: torch.device) -> torch.Tensor:
    vals = [float(x) for x in raw.split(",")]
    if len(vals) != action_dim:
        raise ValueError(f"--dim-weights must have {action_dim} comma-separated values")
    return torch.tensor(vals, dtype=torch.float32, device=device)


def rollout_weighted_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    timestep: torch.Tensor,
    dim_weights: torch.Tensor,
    change_weight: float,
    gripper_change_weight: float,
    late_timestep: float,
    late_weight: float,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).view_as(target)
    weights = dim_weights.view(1, -1).expand_as(ce).clone()

    last = history[:, -1, :]
    changed = target.ne(last)
    weights = weights + changed.float() * change_weight

    gripper_changed = target[:, 6].ne(last[:, 6])
    weights[:, 6] = weights[:, 6] + gripper_changed.float() * gripper_change_weight

    late = timestep.to(weights.device).ge(late_timestep).float().view(-1, 1)
    weights = weights * (1.0 + late * late_weight)
    return (ce * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: TinyTrajectoryHead,
    loader: DataLoader,
    device: torch.device,
    use_hidden: bool,
) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0.0
    top3 = 0.0
    top5 = 0.0
    per_dim = torch.zeros(model.config.action_dim, device=device)
    for batch in loader:
        if use_hidden:
            history, target, ph, _timestep = batch
            history = history.to(device)
            target = target.to(device)
            ph = ph.to(device)
            logits = model(history, prefill_hidden=ph)
        else:
            history, target, _timestep = batch
            history = history.to(device)
            target = target.to(device)
            logits = model(history)
        bs = history.shape[0]
        m = metrics(logits, target)
        total_loss += m["loss"] * bs
        top1 += m["top1"] * bs
        top3 += m["top3"] * bs
        top5 += m["top5"] * bs
        per_dim += torch.tensor(m["per_dim_top1"], device=device) * bs
        total += bs
    return {
        "loss": total_loss / max(total, 1),
        "top1": top1 / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "per_dim_top1": [float(x) for x in (per_dim / max(total, 1)).tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyTrajectoryHead")
    parser.add_argument("--data-dir", default="data/trajectory_head_mini")
    parser.add_argument("--out-dir", default="checkpoints/trajectory_head")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-fusion-dim", type=int, default=256)
    parser.add_argument("--dim-weights", default="1,1,1.25,1.5,1.5,1.5,3.0")
    parser.add_argument("--change-weight", type=float, default=1.0)
    parser.add_argument("--gripper-change-weight", type=float, default=5.0)
    parser.add_argument("--late-timestep", type=float, default=20.0)
    parser.add_argument("--late-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-prefill-hidden",
        action="store_true",
        help="Force history-only model even when dataset contains prefill_hidden",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir) / "dataset.pt"
    dataset = TrajectoryDataset(data_path)
    if len(dataset) < 4:
        raise ValueError(f"Dataset too small: {len(dataset)} examples")

    llm_hs = dataset.llm_hidden_size
    use_prefill_hidden = dataset.use_prefill_hidden and not args.no_prefill_hidden

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device(args.device)
    config = TrajectoryHeadConfig(
        history_size=dataset.history_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_prefill_hidden=use_prefill_hidden,
        llm_hidden_size=llm_hs,
        hidden_fusion_dim=args.hidden_fusion_dim,
    )
    model = TinyTrajectoryHead(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dim_weights = parse_dim_weights(args.dim_weights, config.action_dim, device)

    logger.info(
        "use_prefill_hidden=%s llm_hidden_size=%d hidden_fusion_dim=%d dim_weights=%s",
        use_prefill_hidden,
        llm_hs,
        args.hidden_fusion_dim,
        args.dim_weights,
    )

    best = {"top3": -1.0}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            if use_prefill_hidden:
                hist, target, ph, timestep = batch
                hist = hist.to(device)
                target = target.to(device)
                ph = ph.to(device)
                timestep = timestep.to(device)
                logits = model(hist, prefill_hidden=ph)
            else:
                hist, target, timestep = batch
                hist = hist.to(device)
                target = target.to(device)
                timestep = timestep.to(device)
                logits = model(hist)
            loss = rollout_weighted_loss(
                logits,
                target,
                hist,
                timestep,
                dim_weights,
                args.change_weight,
                args.gripper_change_weight,
                args.late_timestep,
                args.late_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        train_metrics = evaluate(model, train_loader, device, use_prefill_hidden)
        val_metrics = evaluate(model, val_loader, device, use_prefill_hidden)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        logger.info(
            "epoch=%d train_top1=%.3f train_top3=%.3f val_top1=%.3f val_top3=%.3f",
            epoch,
            train_metrics["top1"],
            train_metrics["top3"],
            val_metrics["top1"],
            val_metrics["top3"],
        )
        if val_metrics["top3"] > best["top3"]:
            best = {"epoch": epoch, **val_metrics}
            model.save(out_dir / "best.pt")

    model.save(out_dir / "last.pt")
    (out_dir / "metrics.json").write_text(json.dumps({"best": best, "history": history}, indent=2) + "\n")
    logger.info("Saved best checkpoint to %s", out_dir / "best.pt")


if __name__ == "__main__":
    main()
