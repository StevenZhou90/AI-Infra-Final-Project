#!/usr/bin/env python3
"""Train a gate for PI0-FAST block speculative verification."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_pi0fast_block_drafter import _parse_data_dirs, _parse_stop_token_ids  # noqa: E402
from scripts.train_pi0fast_medusa_from_traces import load_all_records, split_indices  # noqa: E402
from serving.pi0fast_block_drafter import load_block_drafter_checkpoint  # noqa: E402
from serving.pi0fast_block_gate import (  # noqa: E402
    PI0FastBlockGate,
    PI0FastBlockGateConfig,
    block_gate_features,
    save_block_gate,
)
from serving.pi0fast_eagle import _trim_record_tokens_and_hidden  # noqa: E402

logger = logging.getLogger("train_pi0fast_block_gate")


def build_gate_rows(
    *,
    records,
    indices: list[int],
    block_drafter,
    token_map,
    conditioning_mode: str,
    lookahead: int,
    context_len: int,
    min_position: int,
    max_position: int,
    positive_accept_tokens: int,
    stop_token_ids: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []
    labels: list[float] = []
    pad_id = len(token_map)
    dtype = next(block_drafter.parameters()).dtype
    pending: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = []

    def flush() -> None:
        if not pending:
            return
        h = torch.stack([row[0] for row in pending]).to(device=device, dtype=dtype)
        ctx = torch.stack([row[1] for row in pending]).to(device=device)
        futures = torch.stack([row[2] for row in pending])
        positions = [row[3] for row in pending]
        max_steps = [row[4] for row in pending]
        with torch.inference_mode():
            drafts, conf = block_drafter.draft(h, ctx, steps=lookahead, refine_steps=1)
        drafts_cpu = drafts.cpu()
        conf_cpu = conf.cpu()
        for row_idx in range(len(pending)):
            matches = drafts_cpu[row_idx].eq(futures[row_idx])
            mismatches = torch.nonzero(~matches, as_tuple=False)
            accepted_prefix = lookahead if mismatches.numel() == 0 else int(mismatches[0].item())
            hidden_rows.append(pending[row_idx][0].to(dtype=torch.float32))
            feature_rows.append(
                block_gate_features(
                    position=positions[row_idx],
                    max_decoding_steps=max_steps[row_idx],
                    confidences=conf_cpu[row_idx],
                    lookahead=lookahead,
                )
            )
            if positive_accept_tokens > 0:
                labels.append(float(accepted_prefix >= positive_accept_tokens))
            else:
                labels.append(float(accepted_prefix == lookahead))
        pending.clear()

    for record_idx in indices:
        tokens, hidden_states = _trim_record_tokens_and_hidden(records[record_idx], stop_token_ids)
        if tokens.numel() <= lookahead:
            continue
        encoded = token_map.encode_tensor(tokens)
        last_pos = tokens.numel() - lookahead - 1
        first_pos = 1 if conditioning_mode == "pre_known" else 0
        for pos in range(first_pos, last_pos + 1):
            if pos < min_position:
                continue
            if max_position > 0 and pos > max_position:
                continue
            future = encoded[pos + 1 : pos + 1 + lookahead]
            ctx = encoded[max(0, pos + 1 - context_len) : pos + 1]
            if bool((future < 0).any().item()) or bool((ctx < 0).any().item()):
                continue
            padded = torch.full((context_len,), pad_id, dtype=torch.long)
            padded[-ctx.numel() :] = ctx
            hidden_idx = pos - 1 if conditioning_mode == "pre_known" else pos
            pending.append(
                (
                    hidden_states[hidden_idx].to(dtype=torch.float32),
                    padded,
                    future.to(dtype=torch.long),
                    pos,
                    int(tokens.numel()),
                )
            )
            if len(pending) >= batch_size:
                flush()
    flush()
    if not hidden_rows:
        raise ValueError("No gate rows built")
    return torch.stack(hidden_rows), torch.stack(feature_rows), torch.tensor(labels, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: PI0FastBlockGate, hidden: torch.Tensor, features: torch.Tensor, labels: torch.Tensor, threshold: float, device: torch.device) -> dict[str, Any]:
    model.eval()
    probs: list[torch.Tensor] = []
    for start in range(0, labels.numel(), 4096):
        probs.append(model.probability(hidden[start : start + 4096].to(device), features[start : start + 4096].to(device)).cpu())
    p = torch.cat(probs)
    pred = p >= threshold
    y = labels.bool()
    tp = int((pred & y).sum().item())
    fp = int((pred & ~y).sum().item())
    fn = int((~pred & y).sum().item())
    tn = int((~pred & ~y).sum().item())
    return {
        "rows": int(labels.numel()),
        "positive_rate": float(labels.mean().item()),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accept_rate": int(pred.sum().item()) / max(int(labels.numel()), 1),
        "mean_probability": float(p.mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST block verification gate")
    parser.add_argument("--data-dirs", required=True)
    parser.add_argument("--block-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["trace", "task"], default="trace")
    parser.add_argument("--heldout-task-ids", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--min-position", type=int, default=42)
    parser.add_argument("--max-position", type=int, default=0)
    parser.add_argument(
        "--positive-accept-tokens",
        type=int,
        default=0,
        help="Label drafts positive when their verified prefix is at least this long. 0 keeps full-block labels.",
    )
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-token-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    records = load_all_records(_parse_data_dirs(args.data_dirs))
    train_idx, val_idx = split_indices(records, args)
    stop_token_ids = _parse_stop_token_ids(args.stop_token_ids)
    block_drafter, token_map, block_summary = load_block_drafter_checkpoint(args.block_checkpoint, device=device)
    context_len = int(block_drafter.config.context_len)
    conditioning_mode = str(block_summary.get("config", {}).get("conditioning_mode", "post_known"))

    logger.info("Building train rows")
    xh_train, xf_train, y_train = build_gate_rows(
        records=records,
        indices=train_idx,
        block_drafter=block_drafter,
        token_map=token_map,
        conditioning_mode=conditioning_mode,
        lookahead=args.lookahead,
        context_len=context_len,
        min_position=args.min_position,
        max_position=args.max_position,
        positive_accept_tokens=args.positive_accept_tokens,
        stop_token_ids=stop_token_ids,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info("Building val rows")
    xh_val, xf_val, y_val = build_gate_rows(
        records=records,
        indices=val_idx,
        block_drafter=block_drafter,
        token_map=token_map,
        conditioning_mode=conditioning_mode,
        lookahead=args.lookahead,
        context_len=context_len,
        min_position=args.min_position,
        max_position=args.max_position,
        positive_accept_tokens=args.positive_accept_tokens,
        stop_token_ids=stop_token_ids,
        device=device,
        batch_size=args.batch_size,
    )
    logger.info(
        "train_rows=%d val_rows=%d train_pos=%.3f val_pos=%.3f",
        int(y_train.numel()),
        int(y_val.numel()),
        float(y_train.mean().item()),
        float(y_val.mean().item()),
    )

    config = PI0FastBlockGateConfig(
        hidden_dim=int(xh_train.shape[-1]),
        lookahead=args.lookahead,
        feature_dim=int(xf_train.shape[-1]),
        model_dim=args.model_dim,
        dropout=args.dropout,
    )
    model = PI0FastBlockGate(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = ((y_train.numel() - y_train.sum()) / y_train.sum().clamp_min(1.0)).to(device)
    best_state = None
    best_score = (-1.0, -1.0, -1.0)
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(y_train.numel(), generator=generator)
        total_loss = 0.0
        for start in range(0, order.numel(), args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = model(xh_train[idx].to(device), xf_train[idx].to(device))
            target = y_train[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(idx.numel())
        metrics = evaluate(model, xh_val, xf_val, y_val, args.threshold, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(int(order.numel()), 1)
        history.append(metrics)
        precision_ok = 1.0 if metrics["precision"] >= args.target_precision else 0.0
        score = (precision_ok, metrics["recall"], metrics["precision"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info("epoch=%d loss=%.4f val=%s", epoch, metrics["train_loss"], metrics)

    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "config": vars(args),
        "block_best_epoch": block_summary.get("best_epoch"),
        "conditioning_mode": conditioning_mode,
        "train_rows": int(y_train.numel()),
        "val_rows": int(y_val.numel()),
        "train_positive_rate": float(y_train.mean().item()),
        "val_positive_rate": float(y_val.mean().item()),
        "best_val": evaluate(model, xh_val, xf_val, y_val, args.threshold, device),
        "history": history,
    }
    save_block_gate(args.output, model, threshold=args.threshold, summary=summary)
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["best_val"], indent=2))


if __name__ == "__main__":
    main()
