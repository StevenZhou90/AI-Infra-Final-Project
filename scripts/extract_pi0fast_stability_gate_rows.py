#!/usr/bin/env python3
"""Extract labeled PI0-FAST adaptive stability-stop rows from token traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from serving.pi0fast_prefix_gate import PREFIX_GATE_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, skip rows without fallback_action_max_diff exact-validation labels.",
    )
    return parser.parse_args()


def iter_trace_rows(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in sorted(trace_dir.glob("*.pt")):
        loaded = torch.load(shard, map_location="cpu", weights_only=False)
        if not isinstance(loaded, list):
            raise ValueError(f"{shard} did not contain a list of trace rows")
        rows.extend(loaded)
    return rows


def stability_row(row: dict[str, Any], *, require_label: bool) -> dict[str, Any] | None:
    stats = row.get("stats") or {}
    if not stats.get("stopped_on_stability") and not stats.get("continued_after_stability_stop"):
        return None
    has_label = "fallback_action_max_diff" in stats
    if require_label and not has_label:
        return None
    max_diff = float(stats.get("fallback_action_max_diff", 0.0))
    out = {
        "task": row.get("task"),
        "task_id": int(row.get("task_id", -1)),
        "episode": int(row.get("episode", -1)),
        "seed": int(row.get("seed", -1)),
        "step": int(row.get("step", -1)),
        "mode": row.get("mode"),
        "token_count": int(row.get("token_count", 0)),
        "label": int(has_label and max_diff == 0.0),
        "fallback_action_max_diff": max_diff,
    }
    for key, value in sorted(stats.items()):
        if key.startswith("stability_stop_"):
            out[key] = float(value)
            unprefixed = key.removeprefix("stability_stop_")
            if unprefixed in PREFIX_GATE_FEATURES:
                out[unprefixed] = float(value)
    return out


def main() -> None:
    args = parse_args()
    rows = [
        extracted
        for row in iter_trace_rows(args.trace_dir)
        if (extracted := stability_row(row, require_label=args.require_label)) is not None
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
