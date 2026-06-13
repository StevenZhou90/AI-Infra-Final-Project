#!/usr/bin/env python3
"""Evaluate a trained PI0-FAST EAGLE drafter on held-out trace shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_eagle import (  # noqa: E402
    evaluate_offline_acceptance,
    load_checkpoint,
    load_trace_records,
    split_trace_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline PI0-FAST EAGLE acceptance evaluation")
    parser.add_argument("--checkpoint", default="outputs/pi0fast_eagle/pi0fast_eagle.pt")
    parser.add_argument("--data-dir", default="data/pi0fast_eagle")
    parser.add_argument("--split", choices=["trace", "task", "all"], default="trace")
    parser.add_argument("--heldout-task-id", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument(
        "--stop-token-ids",
        default="",
        help="Comma-separated token ids to trim traces through before eval.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    stop_token_ids = tuple(int(part.strip()) for part in args.stop_token_ids.split(",") if part.strip())
    model, token_map, ckpt_summary = load_checkpoint(args.checkpoint, device=device)
    records = load_trace_records(args.data_dir)
    if args.split == "all":
        indices = list(range(len(records)))
    else:
        _train_idx, indices = split_trace_records(
            records,
            split=args.split,
            val_fraction=args.val_fraction,
            seed=args.seed,
            heldout_task_id=args.heldout_task_id,
        )
    metrics = evaluate_offline_acceptance(
        model,
        records,
        indices,
        token_map,
        lookahead=args.lookahead,
        device=device,
        dtype=dtype,
        stop_token_ids=stop_token_ids,
    )
    summary = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "heldout_task_id": args.heldout_task_id,
        "eval_trace_indices": indices,
        "checkpoint_best_epoch": ckpt_summary.get("best_epoch"),
        "metrics": metrics,
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
