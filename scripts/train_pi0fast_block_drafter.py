#!/usr/bin/env python3
"""Train a masked-block FAST-token drafter from PI0-FAST traces."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_pi0fast_medusa_from_traces import load_all_records, split_indices  # noqa: E402
from serving.pi0fast_block_drafter import (  # noqa: E402
    PI0FastBlockDrafter,
    PI0FastBlockDrafterConfig,
    load_block_drafter_checkpoint,
)
from serving.pi0fast_eagle import CompactTokenMap, PI0FastTraceRecord, _trim_record_tokens_and_hidden  # noqa: E402

logger = logging.getLogger("train_pi0fast_block_drafter")


def _parse_data_dirs(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def _parse_stop_token_ids(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_token_map(
    records: list[PI0FastTraceRecord],
    indices: list[int],
    *,
    stop_token_ids: tuple[int, ...],
    vocab_source: str,
) -> CompactTokenMap:
    source = range(len(records)) if vocab_source == "all" else indices
    tokens: list[int] = []
    for idx in source:
        trimmed_tokens, _hidden = _trim_record_tokens_and_hidden(records[idx], stop_token_ids)
        tokens.extend(int(token) for token in trimmed_tokens.tolist())
    return CompactTokenMap(tokens)


def build_rows(
    records: list[PI0FastTraceRecord],
    indices: list[int],
    token_map: CompactTokenMap,
    *,
    context_len: int,
    block_len: int,
    stop_token_ids: tuple[int, ...],
    drop_oov: bool,
    min_position: int,
    max_position: int,
    conditioning_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build training rows for block drafting.

    ``post_known`` matches the after-known online decoder:
    hidden[pos] + tokens[:pos+1] predict tokens[pos+1:].

    ``pre_known`` matches LLM-style SD:
    hidden[pos-1] + tokens[:pos+1] predict tokens[pos+1:], where token[pos]
    is the already-known target greedy token that will be verified with the
    drafted future in one target pass.
    """

    hidden_rows: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    trace_rows: list[int] = []
    pad_id = len(token_map)
    for record_idx in indices:
        tokens, hidden_states = _trim_record_tokens_and_hidden(records[record_idx], stop_token_ids)
        if tokens.numel() <= block_len:
            continue
        encoded = token_map.encode_tensor(tokens)
        first_pos = 1 if conditioning_mode == "pre_known" else 0
        for pos in range(first_pos, tokens.numel() - block_len):
            if pos < min_position:
                continue
            if max_position > 0 and pos > max_position:
                continue
            label = encoded[pos + 1 : pos + 1 + block_len]
            if drop_oov and bool((label < 0).any().item()):
                continue
            start = max(0, pos + 1 - context_len)
            ctx = encoded[start : pos + 1]
            if drop_oov and bool((ctx < 0).any().item()):
                continue
            padded = torch.full((context_len,), pad_id, dtype=torch.long)
            padded[-ctx.numel() :] = ctx
            hidden_idx = pos - 1 if conditioning_mode == "pre_known" else pos
            hidden_rows.append(hidden_states[hidden_idx].to(dtype=torch.float32))
            contexts.append(padded)
            labels.append(label.to(dtype=torch.long))
            trace_rows.append(record_idx)
    if not hidden_rows:
        raise ValueError("No block-drafter rows; check traces, positions, and vocab")
    return (
        torch.stack(hidden_rows),
        torch.stack(contexts),
        torch.stack(labels),
        torch.tensor(trace_rows, dtype=torch.long),
    )


def corrupt_block_inputs(labels: torch.Tensor, model: PI0FastBlockDrafter, args: argparse.Namespace) -> torch.Tensor:
    block = torch.full_like(labels, model.mask_class_id)
    if args.block_visible_prob > 0:
        visible = torch.rand(labels.shape, device=labels.device) < args.block_visible_prob
        block[visible] = labels[visible]
    if args.block_random_prob > 0:
        random_mask = torch.rand(labels.shape, device=labels.device) < args.block_random_prob
        random_classes = torch.randint(0, model.config.vocab_size, labels.shape, dtype=torch.long, device=labels.device)
        block[random_mask] = random_classes[random_mask]
    return block


def slot_loss_weights(block_len: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """Return per-slot loss weights, biased toward prefix accuracy."""

    slots = torch.arange(block_len, dtype=torch.float32, device=device)
    if args.slot_loss_decay <= 0:
        weights = torch.ones(block_len, dtype=torch.float32, device=device)
    else:
        weights = torch.pow(torch.full_like(slots, float(args.slot_loss_decay)), slots)
    if args.slot_loss_floor > 0:
        weights = torch.clamp(weights, min=float(args.slot_loss_floor))
    if args.slot_loss_boost_first > 1:
        boost = int(min(args.slot_loss_boost_first, block_len))
        weights[:boost] *= float(args.slot_loss_boost)
    return weights / weights.mean().clamp_min(1e-6)


def weighted_slot_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    """Cross entropy over slots with explicit prefix-weighting."""

    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(labels)
    return (flat_loss * weights.view(1, -1)).mean()


@torch.inference_mode()
def mine_hard_rows(
    model: PI0FastBlockDrafter,
    hidden: torch.Tensor,
    contexts: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    refine_steps: int,
    accept_threshold: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return rows where the current drafter would accept too short a prefix."""

    model.eval()
    hard_indices: list[int] = []
    accepted_future: list[int] = []
    correct_by_slot = torch.zeros(labels.shape[1], dtype=torch.long)
    for start in range(0, labels.shape[0], batch_size):
        h = hidden[start : start + batch_size].to(device)
        ctx = contexts[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        draft, _conf = model.draft(h, ctx, steps=labels.shape[1], refine_steps=refine_steps)
        pred = draft[:, : y.shape[1]]
        correct_by_slot += (pred == y).sum(dim=0).detach().cpu()
        for row in range(y.shape[0]):
            accepted = 0
            for slot in range(y.shape[1]):
                if int(pred[row, slot].item()) != int(y[row, slot].item()):
                    break
                accepted += 1
            accepted_future.append(accepted)
            if accepted <= accept_threshold:
                hard_indices.append(start + row)
    accepted_arr = np.asarray(accepted_future, dtype=np.float32)
    total = int(labels.shape[0])
    summary = {
        "rows": total,
        "hard_rows": len(hard_indices),
        "hard_fraction": float(len(hard_indices) / max(total, 1)),
        "accept_threshold": int(accept_threshold),
        "mean_future_accept": float(accepted_arr.mean()) if accepted_arr.size else 0.0,
        "p_future_ge_1": float(np.mean(accepted_arr >= 1)) if accepted_arr.size else 0.0,
        "p_future_ge_2": float(np.mean(accepted_arr >= 2)) if accepted_arr.size else 0.0,
        "p_future_ge_4": float(np.mean(accepted_arr >= 4)) if accepted_arr.size else 0.0,
        "slot_top1": [float(v) / max(total, 1) for v in correct_by_slot.tolist()],
    }
    return torch.tensor(hard_indices, dtype=torch.long), summary


@torch.inference_mode()
def evaluate_next_block(
    model: PI0FastBlockDrafter,
    hidden: torch.Tensor,
    contexts: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    refine_steps: int,
) -> dict[str, Any]:
    model.eval()
    total = 0
    loss_sum = 0.0
    correct_by_slot = torch.zeros(labels.shape[1], dtype=torch.long)
    accepted_future: list[int] = []
    confidence_rows: list[float] = []
    for start in range(0, labels.shape[0], batch_size):
        h = hidden[start : start + batch_size].to(device)
        ctx = contexts[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        draft, conf = model.draft(h, ctx, steps=labels.shape[1], refine_steps=refine_steps)
        logits = model(h, ctx)
        loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item())
        pred = draft[:, : y.shape[1]]
        correct_by_slot += (pred == y).sum(dim=0).detach().cpu()
        total += int(y.shape[0])
        confidence_rows.extend(float(v) for v in conf.mean(dim=1).detach().cpu().tolist())
        for row in range(y.shape[0]):
            accepted = 0
            for slot in range(y.shape[1]):
                if int(pred[row, slot].item()) != int(y[row, slot].item()):
                    break
                accepted += 1
            accepted_future.append(accepted)
    arr = np.asarray(accepted_future, dtype=np.float32)
    return {
        "rows": int(total),
        "loss": loss_sum / max(total * labels.shape[1], 1),
        "slot_top1": [float(v) / max(total, 1) for v in correct_by_slot.tolist()],
        "mean_future_accept": float(arr.mean()) if arr.size else 0.0,
        "mean_total_accept_with_known_first": float((arr + 1).mean()) if arr.size else 0.0,
        "p_future_ge_1": float(np.mean(arr >= 1)) if arr.size else 0.0,
        "p_future_ge_2": float(np.mean(arr >= 2)) if arr.size else 0.0,
        "p_future_ge_4": float(np.mean(arr >= 4)) if arr.size else 0.0,
        "p_future_full": float(np.mean(arr >= labels.shape[1])) if arr.size else 0.0,
        "mean_draft_confidence": float(np.mean(confidence_rows)) if confidence_rows else 0.0,
    }


def train_model(
    X_train: torch.Tensor,
    ctx_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    ctx_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    vocab_size: int,
) -> tuple[PI0FastBlockDrafter, list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.train_device if torch.cuda.is_available() or not args.train_device.startswith("cuda") else "cpu")
    config = PI0FastBlockDrafterConfig(
        hidden_dim=int(X_train.shape[-1]),
        vocab_size=vocab_size,
        context_len=args.context_len,
        block_len=args.block_len,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = PI0FastBlockDrafter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(y_train.shape[0])
        total_loss = 0.0
        total = 0
        for start in range(0, perm.numel(), args.batch_size):
            idx = perm[start : start + args.batch_size]
            h = X_train[idx].to(device)
            ctx = ctx_train[idx].to(device)
            y = y_train[idx].to(device)
            block_inputs = corrupt_block_inputs(y, model, args)
            logits = model(h, ctx, block_inputs)
            loss = weighted_slot_cross_entropy(
                logits,
                y,
                slot_loss_weights(y.shape[1], args, device),
                label_smoothing=args.label_smoothing,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
        val = evaluate_next_block(
            model,
            X_val,
            ctx_val,
            y_val,
            batch_size=args.batch_size,
            device=device,
            refine_steps=args.eval_refine_steps,
        )
        row = {"epoch": epoch, "train_loss": total_loss / max(total, 1), "val": val}
        history.append(row)
        metric = float(val["mean_total_accept_with_known_first"])
        logger.info(
            "epoch=%d loss=%.4f val_total_accept=%.3f val_future=%.3f slot1=%.1f%%",
            epoch,
            row["train_loss"],
            metric,
            val["mean_future_accept"],
            100 * (val["slot_top1"][0] if val["slot_top1"] else 0.0),
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, asdict(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST masked-block FAST-token drafter")
    parser.add_argument("--data-dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["trace", "task"], default="trace")
    parser.add_argument("--heldout-task-ids", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--vocab-source", choices=["train", "all"], default="all")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--block-len", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--block-visible-prob", type=float, default=0.10)
    parser.add_argument("--block-random-prob", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument(
        "--slot-loss-decay",
        type=float,
        default=1.0,
        help="Geometric per-slot loss decay. Values below 1 prioritize early accepted-prefix slots.",
    )
    parser.add_argument("--slot-loss-floor", type=float, default=0.0)
    parser.add_argument("--slot-loss-boost-first", type=int, default=0)
    parser.add_argument("--slot-loss-boost", type=float, default=1.0)
    parser.add_argument("--eval-refine-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--stop-token-ids", default="")
    parser.add_argument("--min-position", type=int, default=0)
    parser.add_argument("--max-position", type=int, default=0)
    parser.add_argument(
        "--conditioning-mode",
        choices=["post_known", "pre_known"],
        default="post_known",
        help="post_known trains for draft-after-known; pre_known trains for one-pass LLM-style verification.",
    )
    parser.add_argument(
        "--hard-mine-checkpoint",
        default="",
        help="Optional drafter checkpoint used to find low-acceptance rows to oversample.",
    )
    parser.add_argument(
        "--hard-repeat",
        type=int,
        default=0,
        help="Number of extra copies to add for each mined hard row.",
    )
    parser.add_argument(
        "--hard-accept-threshold",
        type=int,
        default=0,
        help="Rows with accepted future tokens <= this value are hard-mined.",
    )
    parser.add_argument("--hard-mine-refine-steps", type=int, default=1)
    parser.add_argument(
        "--hard-max-extra-rows",
        type=int,
        default=0,
        help="Cap extra hard-mined rows. Zero means no cap.",
    )
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
    stop_token_ids = _parse_stop_token_ids(args.stop_token_ids)
    token_map = build_token_map(records, train_idx, stop_token_ids=stop_token_ids, vocab_source=args.vocab_source)
    X_train, ctx_train, y_train, train_trace_rows = build_rows(
        records,
        train_idx,
        token_map,
        context_len=args.context_len,
        block_len=args.block_len,
        stop_token_ids=stop_token_ids,
        drop_oov=True,
        min_position=args.min_position,
        max_position=args.max_position,
        conditioning_mode=args.conditioning_mode,
    )
    X_val, ctx_val, y_val, val_trace_rows = build_rows(
        records,
        val_idx,
        token_map,
        context_len=args.context_len,
        block_len=args.block_len,
        stop_token_ids=stop_token_ids,
        drop_oov=True,
        min_position=args.min_position,
        max_position=args.max_position,
        conditioning_mode=args.conditioning_mode,
    )
    original_train_rows = int(y_train.shape[0])
    hard_mining_summary: dict[str, Any] | None = None
    if args.hard_mine_checkpoint and args.hard_repeat > 0:
        mine_device = torch.device(
            args.train_device if torch.cuda.is_available() or not args.train_device.startswith("cuda") else "cpu"
        )
        miner, miner_token_map, _miner_summary = load_block_drafter_checkpoint(args.hard_mine_checkpoint, device=mine_device)
        if miner_token_map.token_values != token_map.token_values:
            raise ValueError("Hard-mining checkpoint token map does not match this training token map")
        if miner.config.block_len < args.block_len:
            raise ValueError(
                f"Hard-mining checkpoint block_len={miner.config.block_len} is shorter than requested block_len={args.block_len}"
            )
        hard_idx, hard_mining_summary = mine_hard_rows(
            miner,
            X_train,
            ctx_train,
            y_train,
            batch_size=args.batch_size,
            device=mine_device,
            refine_steps=args.hard_mine_refine_steps,
            accept_threshold=args.hard_accept_threshold,
        )
        max_extra_rows = int(args.hard_max_extra_rows)
        if max_extra_rows > 0 and hard_idx.numel() * int(args.hard_repeat) > max_extra_rows:
            generator = torch.Generator().manual_seed(args.seed)
            needed = max(1, int(np.ceil(max_extra_rows / max(int(args.hard_repeat), 1))))
            order = torch.randperm(hard_idx.numel(), generator=generator)[:needed]
            hard_idx = hard_idx[order]
        if hard_idx.numel() > 0:
            repeat_idx = hard_idx.repeat(int(args.hard_repeat))
            X_train = torch.cat([X_train, X_train[repeat_idx]], dim=0)
            ctx_train = torch.cat([ctx_train, ctx_train[repeat_idx]], dim=0)
            y_train = torch.cat([y_train, y_train[repeat_idx]], dim=0)
            train_trace_rows = torch.cat([train_trace_rows, train_trace_rows[repeat_idx]], dim=0)
        if hard_mining_summary is not None:
            hard_mining_summary = {
                **hard_mining_summary,
                "checkpoint": args.hard_mine_checkpoint,
                "hard_repeat": int(args.hard_repeat),
                "original_train_rows": original_train_rows,
                "extra_rows": int(y_train.shape[0]) - original_train_rows,
                "augmented_train_rows": int(y_train.shape[0]),
            }
        logger.info("hard_mining=%s", json.dumps(hard_mining_summary, sort_keys=True))
    logger.info(
        "records=%d train_records=%d val_records=%d train_rows=%d val_rows=%d vocab=%d hidden=%d",
        len(records),
        len(train_idx),
        len(val_idx),
        int(y_train.shape[0]),
        int(y_val.shape[0]),
        len(token_map),
        int(X_train.shape[-1]),
    )

    model, history, model_config = train_model(X_train, ctx_train, y_train, X_val, ctx_val, y_val, args, len(token_map))
    device = next(model.parameters()).device
    train_metrics = evaluate_next_block(
        model,
        X_train,
        ctx_train,
        y_train,
        batch_size=args.batch_size,
        device=device,
        refine_steps=args.eval_refine_steps,
    )
    val_metrics = evaluate_next_block(
        model,
        X_val,
        ctx_val,
        y_val,
        batch_size=args.batch_size,
        device=device,
        refine_steps=args.eval_refine_steps,
    )
    summary = {
        "config": vars(args),
        "model_kind": "masked_block_drafter",
        "model_config": model_config,
        "records": len(records),
        "train_record_indices": train_idx,
        "val_record_indices": val_idx,
        "train_rows": int(y_train.shape[0]),
        "original_train_rows": original_train_rows,
        "val_rows": int(y_val.shape[0]),
        "train_trace_rows": int(train_trace_rows.numel()),
        "val_trace_rows": int(val_trace_rows.numel()),
        "hard_mining": hard_mining_summary,
        "compact_vocab_size": len(token_map),
        "history": history,
        "best_epoch": max(history, key=lambda row: row["val"]["mean_total_accept_with_known_first"]) if history else None,
        "final_train_metrics": train_metrics,
        "final_val_metrics": val_metrics,
        "elapsed_s": time.perf_counter() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": model_config,
            "model_kind": "masked_block_drafter",
            "token_map": token_map.to_dict(),
            "summary": summary,
        },
        out_dir / "pi0fast_block_drafter.pt",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
