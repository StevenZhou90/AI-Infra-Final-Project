#!/usr/bin/env python3
"""Train Medusa-style future FAST-token heads from PI0-FAST trace shards."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_eagle import CompactTokenMap, PI0FastTraceRecord, load_trace_records  # noqa: E402

logger = logging.getLogger("train_pi0fast_medusa_from_traces")


@dataclass
class MedusaConfig:
    hidden_dim: int
    vocab_size: int
    lookahead: int
    hidden_proj_dim: int = 0
    dropout: float = 0.0


class MedusaFutureTokenHeads(nn.Module):
    """Parallel future-token classifiers over target PI0-FAST hidden states."""

    def __init__(self, config: MedusaConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.hidden_dim
        if config.hidden_proj_dim > 0:
            self.backbone = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_proj_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            in_dim = config.hidden_proj_dim
        else:
            self.backbone = nn.LayerNorm(config.hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(in_dim, config.vocab_size) for _ in range(config.lookahead)])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        x = self.backbone(hidden)
        return [head(x) for head in self.heads]


def _parse_data_dirs(value: str) -> list[Path]:
    dirs = [Path(part.strip()) for part in value.split(",") if part.strip()]
    if not dirs:
        raise ValueError("--data-dirs must include at least one directory")
    return dirs


def load_all_records(data_dirs: list[Path]) -> list[PI0FastTraceRecord]:
    records: list[PI0FastTraceRecord] = []
    for suite_idx, data_dir in enumerate(data_dirs):
        suite_records = load_trace_records(data_dir)
        suite_name = data_dir.name
        for record in suite_records:
            records.append(
                PI0FastTraceRecord(
                    hidden_states=record.hidden_states,
                    token_ids=record.token_ids,
                    task_id=suite_idx * 1000 + record.task_id,
                    seed=record.seed,
                    trace_id=f"{suite_name}/{record.trace_id}",
                    decode_ms=record.decode_ms,
                    success=record.success,
                )
            )
    return records


def split_indices(records: list[PI0FastTraceRecord], args: argparse.Namespace) -> tuple[list[int], list[int]]:
    if args.split == "task":
        heldout = {int(part.strip()) for part in args.heldout_task_ids.split(",") if part.strip()}
        train = [idx for idx, record in enumerate(records) if record.task_id not in heldout]
        val = [idx for idx, record in enumerate(records) if record.task_id in heldout]
    else:
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(records), generator=generator).tolist()
        n_val = max(1, int(round(len(records) * args.val_fraction)))
        n_val = min(n_val, len(records) - 1)
        val_set = set(perm[:n_val])
        train = [idx for idx in range(len(records)) if idx not in val_set]
        val = [idx for idx in range(len(records)) if idx in val_set]
    if not train or not val:
        raise ValueError(f"Empty split: train={len(train)} val={len(val)}")
    return train, val


def build_token_map(records: list[PI0FastTraceRecord], args: argparse.Namespace, train_idx: list[int]) -> CompactTokenMap:
    indices = list(range(len(records))) if args.vocab_source == "all" else train_idx
    tokens: list[int] = []
    for idx in indices:
        tokens.extend(int(token) for token in records[idx].token_ids.tolist())
    return CompactTokenMap(tokens)


def build_rows(
    records: list[PI0FastTraceRecord],
    indices: list[int],
    token_map: CompactTokenMap,
    lookahead: int,
    *,
    drop_oov: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    hidden_rows: list[torch.Tensor] = []
    labels_by_head: list[list[int]] = [[] for _ in range(lookahead)]
    trace_rows: list[int] = []
    for record_idx in indices:
        record = records[record_idx]
        tokens = record.token_ids
        encoded = token_map.encode_tensor(tokens)
        for pos in range(max(0, tokens.numel() - lookahead)):
            future = encoded[pos + 1 : pos + 1 + lookahead]
            if drop_oov and bool((future < 0).any().item()):
                continue
            hidden_rows.append(record.hidden_states[pos].to(dtype=torch.float32))
            trace_rows.append(record_idx)
            for head_idx in range(lookahead):
                labels_by_head[head_idx].append(int(future[head_idx]))
    if not hidden_rows:
        raise ValueError("No Medusa training rows; reduce lookahead or check vocab coverage")
    return (
        torch.stack(hidden_rows),
        [torch.tensor(labels, dtype=torch.long) for labels in labels_by_head],
        torch.tensor(trace_rows, dtype=torch.long),
    )


def subset_rows_by_record(trace_rows: torch.Tensor, record_indices: set[int]) -> torch.Tensor:
    mask = torch.tensor([int(idx) in record_indices for idx in trace_rows.tolist()], dtype=torch.bool)
    return torch.nonzero(mask, as_tuple=False).flatten()


@torch.inference_mode()
def evaluate_by_task(
    model: MedusaFutureTokenHeads,
    X: torch.Tensor,
    ys: list[torch.Tensor],
    trace_rows: torch.Tensor,
    records: list[PI0FastTraceRecord],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    by_task: dict[int, set[int]] = {}
    for record_idx in set(int(idx) for idx in trace_rows.tolist()):
        by_task.setdefault(records[record_idx].task_id, set()).add(record_idx)
    metrics: dict[str, Any] = {}
    for task_id, record_indices in sorted(by_task.items()):
        row_idx = subset_rows_by_record(trace_rows, record_indices)
        if row_idx.numel() == 0:
            continue
        metrics[str(task_id)] = evaluate_heads(
            model,
            X[row_idx],
            [y[row_idx] for y in ys],
            batch_size=batch_size,
            device=device,
        )
    return metrics


@torch.inference_mode()
def evaluate_heads(
    model: MedusaFutureTokenHeads,
    X: torch.Tensor,
    ys: list[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    accepted_counts: list[int] = []
    correct = [0 for _ in ys]
    total = 0
    for start in range(0, X.shape[0], batch_size):
        x = X[start : start + batch_size].to(device)
        y_batch = [y[start : start + batch_size].to(device) for y in ys]
        logits = model(x)
        preds = [logit.argmax(dim=-1) for logit in logits]
        bsz = x.shape[0]
        total += bsz
        for head_idx, pred in enumerate(preds):
            correct[head_idx] += int((pred == y_batch[head_idx]).sum().item())
        for row in range(bsz):
            accepted = 0
            for head_idx, pred in enumerate(preds):
                if int(pred[row]) != int(y_batch[head_idx][row]):
                    break
                accepted += 1
            accepted_counts.append(accepted)
    accepted_arr = np.asarray(accepted_counts, dtype=np.float32)
    return {
        "rows": int(total),
        "head_accuracy": [c / max(total, 1) for c in correct],
        "mean_spec_accept": float(accepted_arr.mean()) if accepted_counts else 0.0,
        "p_accept_ge_1": float(np.mean(accepted_arr >= 1)) if accepted_counts else 0.0,
        "p_accept_ge_2": float(np.mean(accepted_arr >= 2)) if accepted_counts else 0.0,
        "p_accept_ge_3": float(np.mean(accepted_arr >= 3)) if accepted_counts else 0.0,
        "p_accept_full": float(np.mean(accepted_arr >= len(ys))) if accepted_counts else 0.0,
    }


def train_model(
    X_train: torch.Tensor,
    y_train: list[torch.Tensor],
    X_val: torch.Tensor,
    y_val: list[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[MedusaFutureTokenHeads, list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.train_device if torch.cuda.is_available() or not args.train_device.startswith("cuda") else "cpu")
    config = MedusaConfig(
        hidden_dim=int(X_train.shape[-1]),
        vocab_size=int(max(y.max().item() for y in [*y_train, *y_val]) + 1),
        lookahead=args.lookahead,
        hidden_proj_dim=args.hidden_proj_dim,
        dropout=args.dropout,
    )
    model = MedusaFutureTokenHeads(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(X_train.shape[0])
        total_loss = 0.0
        total_rows = 0
        for start in range(0, perm.numel(), args.batch_size):
            idx = perm[start : start + args.batch_size]
            x = X_train[idx].to(device)
            labels = [y[idx].to(device) for y in y_train]
            logits = model(x)
            losses = []
            for head_idx, (logit, label) in enumerate(zip(logits, labels)):
                weight = args.future_loss_decay**head_idx
                losses.append(weight * F.cross_entropy(logit, label))
            loss = sum(losses) / max(sum(args.future_loss_decay**i for i in range(len(losses))), 1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(idx.numel())
            total_rows += int(idx.numel())
        val = evaluate_heads(model, X_val, y_val, batch_size=args.batch_size, device=device)
        row = {"epoch": epoch, "train_loss": total_loss / max(total_rows, 1), "val": val}
        history.append(row)
        metric = float(val["mean_spec_accept"])
        logger.info(
            "epoch=%d loss=%.4f val_accept=%.3f val_p>=2=%.1f%% head1=%.1f%%",
            epoch,
            row["train_loss"],
            metric,
            val["p_accept_ge_2"] * 100,
            val["head_accuracy"][0] * 100 if val["head_accuracy"] else 0.0,
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, asdict(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST Medusa future-token heads from trace shards")
    parser.add_argument("--data-dirs", default="outputs/pi0fast_eagle_tasks0_4_20trace_data")
    parser.add_argument("--output-dir", default="outputs/pi0fast_medusa")
    parser.add_argument("--split", choices=["trace", "task"], default="trace")
    parser.add_argument("--heldout-task-ids", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--vocab-source", choices=["train", "all"], default="all")
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--future-loss-decay", type=float, default=0.75)
    parser.add_argument("--hidden-proj-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_records(_parse_data_dirs(args.data_dirs))
    train_idx, val_idx = split_indices(records, args)
    token_map = build_token_map(records, args, train_idx)
    X_train, y_train, train_trace_rows = build_rows(records, train_idx, token_map, args.lookahead, drop_oov=True)
    X_val, y_val, val_trace_rows = build_rows(records, val_idx, token_map, args.lookahead, drop_oov=True)
    logger.info(
        "records=%d train_records=%d val_records=%d train_rows=%d val_rows=%d vocab=%d hidden=%d",
        len(records),
        len(train_idx),
        len(val_idx),
        int(X_train.shape[0]),
        int(X_val.shape[0]),
        len(token_map),
        int(X_train.shape[-1]),
    )

    model, history, model_config = train_model(X_train, y_train, X_val, y_val, args)
    device = next(model.parameters()).device
    train_metrics = evaluate_heads(model, X_train, y_train, batch_size=args.batch_size, device=device)
    val_metrics = evaluate_heads(model, X_val, y_val, batch_size=args.batch_size, device=device)
    val_metrics_by_task = evaluate_by_task(
        model,
        X_val,
        y_val,
        val_trace_rows,
        records,
        batch_size=args.batch_size,
        device=device,
    )
    summary = {
        "config": vars(args),
        "model_config": model_config,
        "records": len(records),
        "train_record_indices": train_idx,
        "val_record_indices": val_idx,
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "train_trace_rows": int(train_trace_rows.numel()),
        "val_trace_rows": int(val_trace_rows.numel()),
        "compact_vocab_size": len(token_map),
        "history": history,
        "best_epoch": max(history, key=lambda row: row["val"]["mean_spec_accept"]) if history else None,
        "final_train_metrics": train_metrics,
        "final_val_metrics": val_metrics,
        "final_val_metrics_by_task": val_metrics_by_task,
        "elapsed_s": time.perf_counter() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": model_config,
            "token_map": token_map.to_dict(),
            "summary": summary,
        },
        out_dir / "pi0fast_medusa.pt",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
