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
from torch.utils.data import DataLoader, Dataset, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.trajectory_draft_head import TinyTrajectoryHead, TrajectoryHeadConfig  # noqa: E402
from serving.trajectory_phase import PhaseThresholds, label_phase  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_trajectory_head")


def parse_weight_map(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not raw:
        return weights
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", maxsplit=1)
        weights[key.strip()] = float(value)
    return weights


def parse_int_set(raw: str) -> set[int]:
    if not raw:
        return set()
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def parse_float_list(raw: str) -> list[float]:
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        path: Path,
        phase_filter: str = "all",
        action_horizon: int = 1,
        task_weights: dict[str, float] | None = None,
        phase_weights: dict[str, float] | None = None,
        source_weights: dict[str, float] | None = None,
        phase_thresholds: PhaseThresholds | None = None,
        recompute_phase: bool = False,
    ):
        data = torch.load(path, map_location="cpu")
        examples = data["examples"]
        self.phase_thresholds = phase_thresholds
        self.recompute_phase = recompute_phase
        if phase_filter != "all":
            examples = [ex for ex in examples if self._phase(ex) == phase_filter]
        if action_horizon > 1:
            examples = self._with_chunk_targets(examples, action_horizon)
        self.task_weights = task_weights or {}
        self.phase_weights = phase_weights or {}
        self.source_weights = source_weights or {}
        self.examples = examples
        self.history_size = int(data.get("history_size", 4))
        self.action_horizon = int(action_horizon)
        self.llm_hidden_size = int(data.get("llm_hidden_size", 4096))
        self.use_prefill_hidden = bool(data.get("use_prefill_hidden"))
        if self.examples:
            self.use_prefill_hidden = self.use_prefill_hidden and ("prefill_hidden" in self.examples[0])

    def _phase(self, ex: dict) -> str:
        if "phase_label" in ex and not self.recompute_phase:
            return str(ex["phase_label"])
        return label_phase(ex["history_bins"], thresholds=self.phase_thresholds)

    @staticmethod
    def _with_chunk_targets(examples: list[dict], action_horizon: int) -> list[dict]:
        groups: dict[tuple[str, int, int], dict[int, dict]] = {}
        for ex in examples:
            key = (str(ex["suite"]), int(ex["task_id"]), int(ex["trial"]))
            groups.setdefault(key, {})[int(ex["timestep"])] = ex

        chunked = []
        for ex in examples:
            key = (str(ex["suite"]), int(ex["task_id"]), int(ex["trial"]))
            t0 = int(ex["timestep"])
            seq = []
            for offset in range(action_horizon):
                nxt = groups[key].get(t0 + offset)
                if nxt is None:
                    seq = []
                    break
                seq.append(nxt["target_bins"].long())
            if not seq:
                continue
            rec = dict(ex)
            rec["target_bins"] = torch.stack(seq, dim=0).short()
            rec["action_horizon"] = action_horizon
            chunked.append(rec)
        return chunked

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        h = ex["history_bins"].long()
        t = ex["target_bins"].long()
        timestep = torch.tensor(float(ex.get("timestep", 0)), dtype=torch.float32)
        weight = float(self.task_weights.get(str(int(ex.get("task_id", -1))), 1.0))
        weight *= float(self.phase_weights.get(self._phase(ex), 1.0))
        weight *= float(self.source_weights.get(str(ex.get("source", "teacher")), 1.0))
        sample_weight = torch.tensor(weight, dtype=torch.float32)
        if self.use_prefill_hidden:
            ph = ex["prefill_hidden"].float()
            return h, t, ph, timestep, sample_weight
        return h, t, timestep, sample_weight


def metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    if logits.dim() == 3 and target.dim() == 3:
        logits = logits.unsqueeze(1)
    if logits.dim() == 4 and target.dim() == 2:
        target = target.unsqueeze(1)
    pred = logits.argmax(dim=-1)
    top3 = logits.topk(k=3, dim=-1).indices
    top5 = logits.topk(k=5, dim=-1).indices
    acc_dim = (pred == target).float().mean(dim=tuple(range(pred.dim() - 1)))
    result = {
        "loss": float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)).item()),
        "top1": float((pred == target).float().mean().item()),
        "top3": float((top3 == target.unsqueeze(-1)).any(dim=-1).float().mean().item()),
        "top5": float((top5 == target.unsqueeze(-1)).any(dim=-1).float().mean().item()),
        "per_dim_top1": [float(x) for x in acc_dim.tolist()],
    }
    if target.dim() == 3:
        horizon_top1 = (pred == target).float().mean(dim=(0, 2))
        horizon_top3 = (top3 == target.unsqueeze(-1)).any(dim=-1).float().mean(dim=(0, 2))
        result["per_horizon_top1"] = [float(x) for x in horizon_top1.tolist()]
        result["per_horizon_top3"] = [float(x) for x in horizon_top3.tolist()]
    return result


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
    horizon_weights: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).view_as(target)
    if target.dim() == 3:
        weights = dim_weights.view(1, 1, -1).expand_as(ce).clone()
        last = history[:, -1, :].unsqueeze(1)
    else:
        weights = dim_weights.view(1, -1).expand_as(ce).clone()
        last = history[:, -1, :]

    changed = target.ne(last)
    weights = weights + changed.float() * change_weight

    gripper_changed = target[..., 6].ne(last[..., 6])
    weights[..., 6] = weights[..., 6] + gripper_changed.float() * gripper_change_weight

    late_shape = (-1,) + (1,) * (weights.dim() - 1)
    late = timestep.to(weights.device).ge(late_timestep).float().view(*late_shape)
    weights = weights * (1.0 + late * late_weight)
    if horizon_weights is not None and target.dim() == 3:
        weights = weights * horizon_weights.to(weights.device).view(1, -1, 1)
    if sample_weight is not None:
        weights = weights * sample_weight.to(weights.device).view(*late_shape)
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
    per_horizon_top1 = None
    per_horizon_top3 = None
    for batch in loader:
        if use_hidden:
            history, target, ph, _timestep, _sample_weight = batch
            history = history.to(device)
            target = target.to(device)
            ph = ph.to(device)
            logits = model(history, prefill_hidden=ph)
        else:
            history, target, _timestep, _sample_weight = batch
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
        if "per_horizon_top1" in m:
            if per_horizon_top1 is None:
                per_horizon_top1 = torch.zeros(len(m["per_horizon_top1"]), device=device)
                per_horizon_top3 = torch.zeros(len(m["per_horizon_top3"]), device=device)
            per_horizon_top1 += torch.tensor(m["per_horizon_top1"], device=device) * bs
            per_horizon_top3 += torch.tensor(m["per_horizon_top3"], device=device) * bs
        total += bs
    result = {
        "loss": total_loss / max(total, 1),
        "top1": top1 / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "per_dim_top1": [float(x) for x in (per_dim / max(total, 1)).tolist()],
    }
    if per_horizon_top1 is not None and per_horizon_top3 is not None:
        result["per_horizon_top1"] = [float(x) for x in (per_horizon_top1 / max(total, 1)).tolist()]
        result["per_horizon_top3"] = [float(x) for x in (per_horizon_top3 / max(total, 1)).tolist()]
    return result


def split_dataset(dataset: TrajectoryDataset, val_task_ids: set[int]) -> tuple[Subset, Subset, str]:
    if val_task_ids:
        train_idx = []
        val_idx = []
        for idx, ex in enumerate(dataset.examples):
            task_id = int(ex.get("task_id", -1))
            if task_id in val_task_ids:
                val_idx.append(idx)
            else:
                train_idx.append(idx)
        if not train_idx or not val_idx:
            raise ValueError(
                f"Invalid --val-task-ids={sorted(val_task_ids)} for dataset; "
                f"train_examples={len(train_idx)} val_examples={len(val_idx)}"
            )
        return Subset(dataset, train_idx), Subset(dataset, val_idx), f"task_heldout:{sorted(val_task_ids)}"

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    return train_ds, val_ds, "random:0.2"


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
    parser.add_argument("--action-horizon", type=int, default=1)
    parser.add_argument(
        "--dependent-horizon",
        action="store_true",
        help="Predict chunk steps sequentially, Hydra-style, conditioned on previous draft actions.",
    )
    parser.add_argument("--dim-weights", default="1,1,1.25,1.5,1.5,1.5,3.0")
    parser.add_argument("--change-weight", type=float, default=1.0)
    parser.add_argument("--gripper-change-weight", type=float, default=5.0)
    parser.add_argument("--late-timestep", type=float, default=20.0)
    parser.add_argument("--late-weight", type=float, default=1.0)
    parser.add_argument(
        "--horizon-weights",
        default="",
        help="Comma-separated weights for chunk positions, e.g. '2,1.5,1,0.75' for K4 prefix correctness.",
    )
    parser.add_argument(
        "--val-task-ids",
        default="",
        help="Comma-separated task ids held out for validation. Defaults to random 80/20 split.",
    )
    parser.add_argument(
        "--task-weights",
        default="",
        help="Comma-separated task_id:weight pairs, e.g. '3:3,7:1.5'.",
    )
    parser.add_argument(
        "--phase-weights",
        default="",
        help="Comma-separated phase:weight pairs, e.g. 'complex:1.5,smooth:1'.",
    )
    parser.add_argument(
        "--source-weights",
        default="",
        help="Comma-separated source:weight pairs, e.g. 'libero_sparse_mid_dagger:0.25'.",
    )
    parser.add_argument(
        "--phase-filter",
        choices=["all", "smooth", "complex"],
        default="all",
        help="Train on all examples or only one phase label for two-head decoding.",
    )
    parser.add_argument("--recompute-phase-labels", action="store_true")
    parser.add_argument("--smooth-phase-curvature", type=float, default=6.0)
    parser.add_argument("--smooth-phase-acceleration", type=float, default=8.0)
    parser.add_argument("--smooth-phase-min-displacement", type=float, default=1.5)
    parser.add_argument(
        "--no-prefill-hidden",
        action="store_true",
        help="Force history-only model even when dataset contains prefill_hidden",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir) / "dataset.pt"
    task_weights = parse_weight_map(args.task_weights)
    phase_weights = parse_weight_map(args.phase_weights)
    source_weights = parse_weight_map(args.source_weights)
    phase_thresholds = PhaseThresholds(
        smooth_curvature=args.smooth_phase_curvature,
        smooth_acceleration=args.smooth_phase_acceleration,
        min_displacement=args.smooth_phase_min_displacement,
    )
    dataset = TrajectoryDataset(
        data_path,
        phase_filter=args.phase_filter,
        action_horizon=args.action_horizon,
        task_weights=task_weights,
        phase_weights=phase_weights,
        source_weights=source_weights,
        phase_thresholds=phase_thresholds,
        recompute_phase=args.recompute_phase_labels,
    )
    if len(dataset) < 4:
        raise ValueError(f"Dataset too small after phase_filter={args.phase_filter}: {len(dataset)} examples")

    llm_hs = dataset.llm_hidden_size
    use_prefill_hidden = dataset.use_prefill_hidden and not args.no_prefill_hidden
    dataset.use_prefill_hidden = use_prefill_hidden

    val_task_ids = parse_int_set(args.val_task_ids)
    train_ds, val_ds, split_name = split_dataset(dataset, val_task_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device(args.device)
    config = TrajectoryHeadConfig(
        history_size=dataset.history_size,
        action_horizon=dataset.action_horizon,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_prefill_hidden=use_prefill_hidden,
        llm_hidden_size=llm_hs,
        hidden_fusion_dim=args.hidden_fusion_dim,
        dependent_horizon=bool(args.dependent_horizon),
    )
    model = TinyTrajectoryHead(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dim_weights = parse_dim_weights(args.dim_weights, config.action_dim, device)
    horizon_weights_raw = parse_float_list(args.horizon_weights)
    if horizon_weights_raw and len(horizon_weights_raw) != dataset.action_horizon:
        raise ValueError(f"--horizon-weights must have {dataset.action_horizon} values")
    horizon_weights = (
        torch.tensor(horizon_weights_raw, dtype=torch.float32, device=device) if horizon_weights_raw else None
    )

    logger.info(
        "use_prefill_hidden=%s llm_hidden_size=%d hidden_fusion_dim=%d action_horizon=%d dependent_horizon=%s dim_weights=%s",
        use_prefill_hidden,
        llm_hs,
        args.hidden_fusion_dim,
        args.action_horizon,
        args.dependent_horizon,
        args.dim_weights,
    )
    logger.info(
        "phase_filter=%s examples=%d task_weights=%s phase_weights=%s source_weights=%s",
        args.phase_filter,
        len(dataset),
        task_weights,
        phase_weights,
        source_weights,
    )
    logger.info("recompute_phase=%s phase_thresholds=%s", args.recompute_phase_labels, phase_thresholds)
    logger.info(
        "split=%s train_examples=%d val_examples=%d horizon_weights=%s",
        split_name,
        len(train_ds),
        len(val_ds),
        horizon_weights_raw,
    )

    best = {"top3": -1.0}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            if use_prefill_hidden:
                hist, target, ph, timestep, sample_weight = batch
                hist = hist.to(device)
                target = target.to(device)
                ph = ph.to(device)
                timestep = timestep.to(device)
                sample_weight = sample_weight.to(device)
                if model.config.dependent_horizon:
                    logits = model(hist, prefill_hidden=ph, target_bins=target)
                else:
                    logits = model(hist, prefill_hidden=ph)
            else:
                hist, target, timestep, sample_weight = batch
                hist = hist.to(device)
                target = target.to(device)
                timestep = timestep.to(device)
                sample_weight = sample_weight.to(device)
                if model.config.dependent_horizon:
                    logits = model(hist, target_bins=target)
                else:
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
                horizon_weights,
                sample_weight,
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
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best": best,
                "history": history,
                "phase_filter": args.phase_filter,
                "action_horizon": args.action_horizon,
                "dependent_horizon": args.dependent_horizon,
                "horizon_weights": horizon_weights_raw,
                "split": split_name,
                "val_task_ids": sorted(val_task_ids),
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Saved best checkpoint to %s", out_dir / "best.pt")


if __name__ == "__main__":
    main()
