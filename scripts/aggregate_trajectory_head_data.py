#!/usr/bin/env python3
"""Aggregate multiple trajectory-head datasets into one for proper DAgger training.

Standard DAgger trains on the union of all data collected so far. This concatenates
the `examples` lists across the input dataset directories while preserving the shared
metadata (history_size, llm_hidden_size, use_prefill_hidden).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def load(p: Path) -> dict:
    return torch.load(p / "dataset.pt", map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Dataset directories to aggregate")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [Path(p) for p in args.inputs]
    bundles = [load(p) for p in inputs]

    history_sizes = {int(b.get("history_size", 4)) for b in bundles}
    if len(history_sizes) != 1:
        print(f"ERROR: mismatched history_size across inputs: {history_sizes}", file=sys.stderr)
        sys.exit(2)

    llm_hidden = {int(b.get("llm_hidden_size", 0)) for b in bundles if b.get("llm_hidden_size")}
    if len(llm_hidden) > 1:
        print(f"ERROR: mismatched llm_hidden_size across inputs: {llm_hidden}", file=sys.stderr)
        sys.exit(2)

    use_hidden = all(bool(b.get("use_prefill_hidden")) for b in bundles)

    examples: list = []
    rollouts: list = []
    sources: list = []
    for path, bundle in zip(inputs, bundles):
        ex = bundle.get("examples", [])
        examples.extend(ex)
        rollouts.extend(bundle.get("rollouts", []))
        sources.append({"path": str(path), "examples": len(ex)})

    out = {
        "examples": examples,
        "history_size": history_sizes.pop(),
        "rollouts": rollouts,
        "use_prefill_hidden": use_hidden,
        "data_source": "aggregated",
        "sources": sources,
    }
    if llm_hidden:
        out["llm_hidden_size"] = llm_hidden.pop()

    torch.save(out, out_dir / "dataset.pt")
    summary = {"examples": len(examples), "rollouts": len(rollouts), "sources": sources}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
