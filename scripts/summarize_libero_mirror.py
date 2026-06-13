#!/usr/bin/env python3
"""Summarize AR vs Spec outputs from mirror/distributed runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    steps = sum(r["steps"] for r in rows)
    successes = sum(1 for r in rows if r["success"])
    infer_ms = sum(r["inference_ms_total"] for r in rows)
    episodes = len(rows)
    return {
        "episodes": float(episodes),
        "successes": float(successes),
        "success_rate": successes / max(episodes, 1),
        "steps_total": float(steps),
        "avg_ms_per_step": infer_ms / max(steps, 1),
    }


def merge_distributed(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ar_rows = load_jsonl(run_dir / "ar_episodes.global.jsonl")
    spec_rows = load_jsonl(run_dir / "spec_episodes.global.jsonl")
    if ar_rows and spec_rows:
        return ar_rows, spec_rows

    rank_dirs = sorted([p for p in run_dir.glob("rank_*") if p.is_dir()])
    merged_ar: list[dict[str, Any]] = []
    merged_spec: list[dict[str, Any]] = []
    for rank_dir in rank_dirs:
        merged_ar.extend(load_jsonl(rank_dir / "ar_episodes.jsonl"))
        merged_spec.extend(load_jsonl(rank_dir / "spec_episodes.jsonl"))
    if merged_ar:
        with (run_dir / "ar_episodes.global.jsonl").open("w", encoding="utf-8") as handle:
            for row in merged_ar:
                handle.write(json.dumps(row) + "\n")
    if merged_spec:
        with (run_dir / "spec_episodes.global.jsonl").open("w", encoding="utf-8") as handle:
            for row in merged_spec:
                handle.write(json.dumps(row) + "\n")
    return merged_ar, merged_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to smoke/ or full/ run directory.")
    parser.add_argument("--distributed", action="store_true", help="Summarize distributed rank artifacts.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_candidates = [
        run_dir / "summary.global.json",
        run_dir / "summary.json",
    ]
    summary_path = next((p for p in summary_candidates if p.exists()), None)
    if summary_path is None:
        raise SystemExit("Missing summary file (expected summary.global.json or summary.json)")
    summary = load_json(summary_path)

    if args.distributed:
        ar_rows, spec_rows = merge_distributed(run_dir)
    else:
        ar_path = run_dir / "ar_episodes.jsonl"
        spec_path = run_dir / "spec_episodes.jsonl"
        if not ar_path.exists() or not spec_path.exists():
            raise SystemExit("Missing expected run artifacts: ar_episodes.jsonl and spec_episodes.jsonl")
        ar_rows = load_jsonl(ar_path)
        spec_rows = load_jsonl(spec_path)

    ar = aggregate(ar_rows)
    spec = aggregate(spec_rows)
    speedup = ar["avg_ms_per_step"] / max(spec["avg_ms_per_step"], 1e-9)
    success_delta = spec["success_rate"] - ar["success_rate"]

    result = {
        "mode": summary.get("mode"),
        "world_size": summary.get("world_size", 1),
        "gate": summary.get("gate"),
        "computed": {
            "speedup_ar_over_spec": speedup,
            "success_rate_ar": ar["success_rate"],
            "success_rate_spec": spec["success_rate"],
            "success_delta_spec_minus_ar": success_delta,
            "episodes_ar": ar["episodes"],
            "episodes_spec": spec["episodes"],
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
