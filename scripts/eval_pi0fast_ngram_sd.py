#!/usr/bin/env python3
"""Evaluate n-gram FAST-token speculative drafting on PI0-FAST traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_eagle import load_trace_records, split_trace_records  # noqa: E402
from serving.pi0fast_ngram import NgramDraftConfig, NgramFastTokenDrafter, evaluate_ngram_drafter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate n-gram PI0-FAST speculative drafter")
    parser.add_argument("--data-dir", default="outputs/pi0fast_eagle_tasks0_4_20trace_data")
    parser.add_argument("--split", choices=["trace", "task"], default="trace")
    parser.add_argument("--heldout-task-id", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument(
        "--stop-token-ids",
        default="",
        help="Comma-separated token ids to trim traces through, e.g. FAST action end and EOS.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_trace_records(args.data_dir)
    train_idx, val_idx = split_trace_records(
        records,
        split=args.split,
        val_fraction=args.val_fraction,
        seed=args.seed,
        heldout_task_id=args.heldout_task_id,
    )
    stop_token_ids = tuple(int(part.strip()) for part in args.stop_token_ids.split(",") if part.strip())
    drafter = NgramFastTokenDrafter(
        NgramDraftConfig(
            max_context=args.max_context,
            min_count=args.min_count,
            lookahead=args.lookahead,
            stop_token_ids=stop_token_ids,
        )
    )
    drafter.fit(records[idx] for idx in train_idx)
    metrics = evaluate_ngram_drafter(
        drafter,
        (records[idx] for idx in val_idx),
        lookahead=args.lookahead,
    )
    summary = {
        "config": vars(args),
        "records": len(records),
        "train_trace_indices": train_idx,
        "val_trace_indices": val_idx,
        "metrics": metrics,
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
