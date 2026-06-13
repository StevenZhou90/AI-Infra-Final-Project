#!/usr/bin/env python3
"""Train a PI0-FAST EAGLE-style recurrent FAST-token drafter."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pi0fast_chunk_eval import _import_lerobot  # noqa: E402
from serving.pi0fast_eagle import (  # noqa: E402
    PI0FastEagleConfig,
    PI0FastEagleHead,
    build_compact_token_map,
    evaluate_offline_acceptance,
    load_trace_records,
    make_teacher_forcing_rows,
    save_checkpoint,
    split_trace_records,
)

logger = logging.getLogger("train_pi0fast_eagle")


def _ensure_libero_config(config_path: str | None) -> None:
    if config_path:
        os.environ["LIBERO_CONFIG_PATH"] = config_path
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")).expanduser()
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        return
    libero_root = None
    for entry in sys.path:
        candidate = Path(entry) / "libero" / "libero"
        if (candidate / "bddl_files").exists() and (candidate / "init_files").exists():
            libero_root = candidate.resolve()
            break
    if libero_root is None:
        raise RuntimeError("Could not locate installed LIBERO package root before loading PI0 policy")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(
        "\n".join(
            [
                f"assets: {libero_root / 'assets'}",
                f"bddl_files: {libero_root / 'bddl_files'}",
                f"benchmark_root: {libero_root}",
                f"datasets: {libero_root.parent / 'datasets'}",
                f"init_states: {libero_root / 'init_files'}",
                "",
            ]
        )
    )
    logger.info("Wrote noninteractive LIBERO config to %s", cfg_file)


def _target_embedding_weight(policy) -> torch.Tensor:
    paligemma = policy.model.paligemma_with_expert.paligemma
    candidates = [
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
    ]
    for path in candidates:
        obj = paligemma
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.Tensor):
                return obj.detach()
        except AttributeError:
            continue
    raise RuntimeError("Could not locate PI0-FAST PaliGemma token embedding weight")


def maybe_initialize_embeddings(model: PI0FastEagleHead, token_map, args, device: torch.device, dtype: torch.dtype) -> None:
    if args.skip_policy_embedding_init:
        logger.info("Skipping target embedding initialization")
        return
    _ensure_libero_config(args.libero_config_path)
    _make_env, _make_env_pre_post_processors, _preprocess_observation, _LiberoEnv, _make_pre_post_processors, PI0FastPolicy = _import_lerobot()
    logger.info("Loading policy embeddings from %s", args.policy)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    model.initialize_embeddings_from_target(_target_embedding_weight(policy), token_map)
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Initialized compact EAGLE embeddings from target token embeddings")


def train_epoch(
    model: PI0FastEagleHead,
    hidden: torch.Tensor,
    input_classes: torch.Tensor,
    target_classes: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    perm = torch.randperm(hidden.shape[0])
    total_loss = 0.0
    total_correct = 0
    total = 0
    autocast_enabled = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)

    for start in range(0, len(perm), batch_size):
        idx = perm[start : start + batch_size]
        h = hidden[idx].to(device=device, dtype=dtype).unsqueeze(1)
        inp = input_classes[idx].to(device=device).unsqueeze(1)
        tgt = target_classes[idx].to(device=device)

        model.reset_kv()
        with torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            logits = model(h, inp, use_cache=False)[:, -1, :]
            loss = F.cross_entropy(logits.float(), tgt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        pred = logits.argmax(dim=-1)
        total_correct += int((pred == tgt).sum().item())
        total += int(tgt.numel())
        total_loss += float(loss.item()) * int(tgt.numel())

    return {
        "train_loss": total_loss / max(total, 1),
        "train_next_token_accuracy": total_correct / max(total, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0-FAST EAGLE draft head")
    parser.add_argument("--data-dir", default="data/pi0fast_eagle")
    parser.add_argument("--output-dir", default="outputs/pi0fast_eagle")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--split", choices=["trace", "task"], default="trace")
    parser.add_argument(
        "--vocab-source",
        choices=["train", "all"],
        default="all",
        help="Build compact FAST-token class map from train traces only or all collected traces.",
    )
    parser.add_argument("--heldout-task-id", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument(
        "--stop-token-ids",
        default="",
        help="Comma-separated token ids to trim traces through before training/eval.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--skip-policy-embedding-init", action="store_true")
    parser.add_argument(
        "--libero-config-path",
        default=os.environ.get("LIBERO_CONFIG_PATH"),
        help="Directory for LIBERO config.yaml. Created noninteractively if missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)

    records = load_trace_records(args.data_dir)
    stop_token_ids = tuple(int(part.strip()) for part in args.stop_token_ids.split(",") if part.strip())
    train_idx, val_idx = split_trace_records(
        records,
        split=args.split,
        val_fraction=args.val_fraction,
        seed=args.seed,
        heldout_task_id=args.heldout_task_id,
    )
    vocab_indices = train_idx if args.vocab_source == "train" else list(range(len(records)))
    token_map = build_compact_token_map(records, vocab_indices, stop_token_ids=stop_token_ids)
    X_train, input_train, target_train, trace_train = make_teacher_forcing_rows(
        records,
        train_idx,
        token_map,
        drop_oov=True,
        stop_token_ids=stop_token_ids,
    )

    config = PI0FastEagleConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        train_embeddings=args.train_embeddings,
    )
    model = PI0FastEagleHead(config, len(token_map))
    maybe_initialize_embeddings(model, token_map, args, device, dtype)
    model.to(device=device, dtype=dtype)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(X_train) / args.batch_size) * args.epochs)
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 5))

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_summary: dict[str, Any] | None = None
    t0 = time.perf_counter()
    logger.info(
        "Training rows=%d train_traces=%d val_traces=%d compact_vocab=%d device=%s",
        int(X_train.shape[0]),
        len(train_idx),
        len(val_idx),
        len(token_map),
        device,
    )

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model,
            X_train,
            input_train,
            target_train,
            optimizer,
            scheduler,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate_offline_acceptance(
            model,
            records,
            val_idx,
            token_map,
            lookahead=args.lookahead,
            device=device,
            dtype=dtype,
            stop_token_ids=stop_token_ids,
        )
        row = {"epoch": epoch, **train_metrics, "val": val_metrics}
        history.append(row)
        metric = float(val_metrics["mean_spec_accept"])
        logger.info(
            "epoch=%d loss=%.4f train_acc=%.1f%% val_accept=%.3f val_p>=2=%.1f%%",
            epoch,
            train_metrics["train_loss"],
            train_metrics["train_next_token_accuracy"] * 100,
            metric,
            val_metrics["p_accept_ge_2"] * 100,
        )
        if metric > best_metric:
            best_metric = metric
            best_summary = row
            summary = {
                "config": vars(args),
                "model_config": asdict(config),
                "records": len(records),
                "train_trace_indices": train_idx,
                "val_trace_indices": val_idx,
                "train_rows": int(X_train.shape[0]),
                "compact_vocab_size": len(token_map),
                "history": history,
                "best_epoch": row,
                "elapsed_s": time.perf_counter() - t0,
            }
            save_checkpoint(model, out_dir, config=config, token_map=token_map, summary=summary)

    summary = {
        "config": vars(args),
        "model_config": asdict(config),
        "records": len(records),
        "train_trace_indices": train_idx,
        "val_trace_indices": val_idx,
        "train_rows": int(X_train.shape[0]),
        "train_trace_rows": int(trace_train.numel()),
        "compact_vocab_size": len(token_map),
        "history": history,
        "best_epoch": best_summary,
        "final_epoch": history[-1] if history else None,
        "elapsed_s": time.perf_counter() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_checkpoint(model, out_dir / "final", config=config, token_map=token_map, summary=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
