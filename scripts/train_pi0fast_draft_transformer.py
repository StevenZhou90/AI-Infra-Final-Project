#!/usr/bin/env python3
"""Train a small causal transformer drafter from PI0-FAST trace shards."""

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
from serving.pi0fast_draft_transformer import (  # noqa: E402
    PI0FastDraftTransformer,
    PI0FastDraftTransformerConfig,
    PI0FastParallelDraftTransformer,
    PI0FastParallelDraftTransformerConfig,
)
from serving.pi0fast_eagle import CompactTokenMap, PI0FastTraceRecord, _trim_record_tokens_and_hidden  # noqa: E402

logger = logging.getLogger("train_pi0fast_draft_transformer")


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
    stop_token_ids: tuple[int, ...],
    drop_oov: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_rows: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    labels: list[int] = []
    trace_rows: list[int] = []
    pad_id = len(token_map)
    for record_idx in indices:
        tokens, hidden_states = _trim_record_tokens_and_hidden(records[record_idx], stop_token_ids)
        if tokens.numel() < 2:
            continue
        encoded = token_map.encode_tensor(tokens)
        for pos in range(tokens.numel() - 1):
            label = int(encoded[pos + 1])
            if drop_oov and label < 0:
                continue
            start = max(0, pos + 1 - context_len)
            ctx = encoded[start : pos + 1]
            if drop_oov and bool((ctx < 0).any().item()):
                continue
            padded = torch.full((context_len,), pad_id, dtype=torch.long)
            padded[-ctx.numel() :] = ctx
            hidden_rows.append(hidden_states[pos].to(dtype=torch.float32))
            contexts.append(padded)
            labels.append(label)
            trace_rows.append(record_idx)
    if not hidden_rows:
        raise ValueError("No draft-transformer rows; check traces/vocab")
    return (
        torch.stack(hidden_rows),
        torch.stack(contexts),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(trace_rows, dtype=torch.long),
    )


def build_parallel_rows(
    records: list[PI0FastTraceRecord],
    indices: list[int],
    token_map: CompactTokenMap,
    *,
    context_len: int,
    lookahead: int,
    stop_token_ids: tuple[int, ...],
    drop_oov: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_rows: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    trace_rows: list[int] = []
    pad_id = len(token_map)
    for record_idx in indices:
        tokens, hidden_states = _trim_record_tokens_and_hidden(records[record_idx], stop_token_ids)
        if tokens.numel() < lookahead + 1:
            continue
        encoded = token_map.encode_tensor(tokens)
        for pos in range(tokens.numel() - lookahead):
            label = encoded[pos + 1 : pos + 1 + lookahead]
            if drop_oov and bool((label < 0).any().item()):
                continue
            start = max(0, pos + 1 - context_len)
            ctx = encoded[start : pos + 1]
            if drop_oov and bool((ctx < 0).any().item()):
                continue
            padded = torch.full((context_len,), pad_id, dtype=torch.long)
            padded[-ctx.numel() :] = ctx
            hidden_rows.append(hidden_states[pos].to(dtype=torch.float32))
            contexts.append(padded)
            labels.append(label.to(dtype=torch.long))
            trace_rows.append(record_idx)
    if not hidden_rows:
        raise ValueError("No parallel draft-transformer rows; check traces/vocab")
    return (
        torch.stack(hidden_rows),
        torch.stack(contexts),
        torch.stack(labels),
        torch.tensor(trace_rows, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate_next_token(
    model: PI0FastDraftTransformer | PI0FastParallelDraftTransformer,
    hidden: torch.Tensor,
    contexts: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for start in range(0, labels.numel(), batch_size):
        h = hidden[start : start + batch_size].to(device)
        ctx = contexts[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        logits = model(h, ctx)
        if logits.ndim == 3:
            logits_for_loss = logits.reshape(-1, logits.shape[-1])
            y_for_loss = y.reshape(-1)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        else:
            logits_for_loss = logits
            y_for_loss = y
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        loss_sum += float(F.cross_entropy(logits_for_loss, y_for_loss, reduction="sum").item())
    return {"rows": total, "loss": loss_sum / max(total, 1), "top1": correct / max(total, 1)}


@torch.inference_mode()
def evaluate_spec_accept(
    model: PI0FastDraftTransformer | PI0FastParallelDraftTransformer,
    records: list[PI0FastTraceRecord],
    indices: list[int],
    token_map: CompactTokenMap,
    *,
    context_len: int,
    lookahead: int,
    stop_token_ids: tuple[int, ...],
    device: torch.device,
    max_positions: int | None = None,
) -> dict[str, Any]:
    accepted_counts: list[int] = []
    pad_id = len(token_map)
    for record_idx in indices:
        tokens, hidden_states = _trim_record_tokens_and_hidden(records[record_idx], stop_token_ids)
        if tokens.numel() < lookahead + 1:
            continue
        encoded = token_map.encode_tensor(tokens)
        for pos in range(tokens.numel() - lookahead):
            future = encoded[pos + 1 : pos + 1 + lookahead]
            if bool((future < 0).any().item()):
                continue
            start = max(0, pos + 1 - context_len)
            ctx = encoded[start : pos + 1]
            if bool((ctx < 0).any().item()):
                continue
            padded = torch.full((1, context_len), pad_id, dtype=torch.long, device=device)
            padded[0, -ctx.numel() :] = ctx.to(device)
            hidden = hidden_states[pos : pos + 1].to(device=device, dtype=torch.float32)
            draft, _conf = model.draft(hidden, padded, steps=lookahead)
            accepted = 0
            for idx in range(lookahead):
                if int(draft[0, idx].item()) != int(future[idx].item()):
                    break
                accepted += 1
            accepted_counts.append(accepted)
            if max_positions is not None and len(accepted_counts) >= max_positions:
                arr = np.asarray(accepted_counts, dtype=np.float32)
                return {
                    "rows": int(arr.size),
                    "mean_spec_accept": float(arr.mean()) if arr.size else 0.0,
                    "p_accept_ge_1": float(np.mean(arr >= 1)) if arr.size else 0.0,
                    "p_accept_ge_2": float(np.mean(arr >= 2)) if arr.size else 0.0,
                    "p_accept_ge_3": float(np.mean(arr >= 3)) if arr.size else 0.0,
                    "p_accept_full": float(np.mean(arr >= lookahead)) if arr.size else 0.0,
                    "capped": True,
                }
    arr = np.asarray(accepted_counts, dtype=np.float32)
    return {
        "rows": int(arr.size),
        "mean_spec_accept": float(arr.mean()) if arr.size else 0.0,
        "p_accept_ge_1": float(np.mean(arr >= 1)) if arr.size else 0.0,
        "p_accept_ge_2": float(np.mean(arr >= 2)) if arr.size else 0.0,
        "p_accept_ge_3": float(np.mean(arr >= 3)) if arr.size else 0.0,
        "p_accept_full": float(np.mean(arr >= lookahead)) if arr.size else 0.0,
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
) -> tuple[PI0FastDraftTransformer | PI0FastParallelDraftTransformer, list[dict[str, Any]], dict[str, Any], str]:
    device = torch.device(args.train_device if torch.cuda.is_available() or not args.train_device.startswith("cuda") else "cpu")
    if args.parallel_lookahead > 1:
        config = PI0FastParallelDraftTransformerConfig(
            hidden_dim=int(X_train.shape[-1]),
            vocab_size=vocab_size,
            context_len=args.context_len,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            parallel_lookahead=args.parallel_lookahead,
        )
        model: PI0FastDraftTransformer | PI0FastParallelDraftTransformer = PI0FastParallelDraftTransformer(config).to(device)
        model_kind = "parallel_draft_transformer"
    else:
        config = PI0FastDraftTransformerConfig(
            hidden_dim=int(X_train.shape[-1]),
            vocab_size=vocab_size,
            context_len=args.context_len,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
        model = PI0FastDraftTransformer(config).to(device)
        model_kind = "draft_transformer"
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
            logits = model(h, ctx)
            if logits.ndim == 3:
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            else:
                loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
        val = evaluate_next_token(model, X_val, ctx_val, y_val, batch_size=args.batch_size, device=device)
        row = {"epoch": epoch, "train_loss": total_loss / max(total, 1), "val": val}
        history.append(row)
        logger.info("epoch=%d loss=%.4f val_top1=%.1f%% val_loss=%.4f", epoch, row["train_loss"], val["top1"] * 100, val["loss"])
        if val["top1"] > best_metric:
            best_metric = float(val["top1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, asdict(config), model_kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST small draft transformer")
    parser.add_argument("--data-dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["trace", "task"], default="trace")
    parser.add_argument("--heldout-task-ids", default="")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--vocab-source", choices=["train", "all"], default="all")
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--parallel-lookahead", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--stop-token-ids", default="")
    parser.add_argument("--max-offline-spec-positions", type=int, default=0)
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
    if args.parallel_lookahead > 1:
        row_lookahead = args.parallel_lookahead
        X_train, ctx_train, y_train, train_trace_rows = build_parallel_rows(
            records, train_idx, token_map, context_len=args.context_len, lookahead=row_lookahead,
            stop_token_ids=stop_token_ids, drop_oov=True
        )
        X_val, ctx_val, y_val, val_trace_rows = build_parallel_rows(
            records, val_idx, token_map, context_len=args.context_len, lookahead=row_lookahead,
            stop_token_ids=stop_token_ids, drop_oov=True
        )
    else:
        X_train, ctx_train, y_train, train_trace_rows = build_rows(
            records, train_idx, token_map, context_len=args.context_len, stop_token_ids=stop_token_ids, drop_oov=True
        )
        X_val, ctx_val, y_val, val_trace_rows = build_rows(
            records, val_idx, token_map, context_len=args.context_len, stop_token_ids=stop_token_ids, drop_oov=True
        )
    logger.info(
        "records=%d train_records=%d val_records=%d train_rows=%d val_rows=%d vocab=%d hidden=%d",
        len(records), len(train_idx), len(val_idx), int(y_train.numel()), int(y_val.numel()), len(token_map), int(X_train.shape[-1])
    )
    model, history, model_config, model_kind = train_model(X_train, ctx_train, y_train, X_val, ctx_val, y_val, args, len(token_map))
    device = next(model.parameters()).device
    train_next = evaluate_next_token(model, X_train, ctx_train, y_train, batch_size=args.batch_size, device=device)
    val_next = evaluate_next_token(model, X_val, ctx_val, y_val, batch_size=args.batch_size, device=device)
    train_spec = evaluate_spec_accept(
        model, records, train_idx, token_map, context_len=args.context_len, lookahead=args.lookahead,
        stop_token_ids=stop_token_ids, device=device,
        max_positions=args.max_offline_spec_positions or None,
    )
    val_spec = evaluate_spec_accept(
        model, records, val_idx, token_map, context_len=args.context_len, lookahead=args.lookahead,
        stop_token_ids=stop_token_ids, device=device,
        max_positions=args.max_offline_spec_positions or None,
    )
    summary = {
        "config": vars(args),
        "model_kind": model_kind,
        "model_config": model_config,
        "records": len(records),
        "train_record_indices": train_idx,
        "val_record_indices": val_idx,
        "train_rows": int(y_train.numel()),
        "val_rows": int(y_val.numel()),
        "train_trace_rows": int(train_trace_rows.numel()),
        "val_trace_rows": int(val_trace_rows.numel()),
        "compact_vocab_size": len(token_map),
        "history": history,
        "best_epoch": max(history, key=lambda row: row["val"]["top1"]) if history else None,
        "final_train_next": train_next,
        "final_val_next": val_next,
        "final_train_spec": train_spec,
        "final_val_spec": val_spec,
        "elapsed_s": time.perf_counter() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": model_config,
            "model_kind": model_kind,
            "token_map": token_map.to_dict(),
            "summary": summary,
        },
        out_dir / "pi0fast_draft_transformer.pt",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
