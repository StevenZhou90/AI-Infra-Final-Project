from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("*/*/metrics.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(rows: list[dict], mode: str, suite: str | None = None) -> dict:
    filtered = [row for row in rows if row["mode"] == mode and (suite is None or row["task"] == suite)]
    if not filtered:
        return {"episodes": 0, "successes": 0, "success_rate": 0.0, "avg_ms": 0.0}
    successes = sum(int(row["success"]) for row in filtered)
    return {
        "episodes": len(filtered),
        "successes": successes,
        "success_rate": successes / len(filtered),
        "avg_ms": _mean([row["avg_ms_per_control_step"] for row in filtered]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pi0fast target-eos speed sweeps.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("outputs/pi0fast_target_eos_speed_90_separate"),
    )
    args = parser.parse_args()

    rows = _load_rows(args.root)
    suites = ["libero_object", "libero_spatial", "libero_goal"]
    print("| suite | baseline | target_eos | baseline ms | target_eos ms | speedup | drop |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for suite in [None, *suites]:
        baseline = _summarize(rows, "baseline", suite)
        target = _summarize(rows, "target_eos", suite)
        if baseline["episodes"] == 0 or target["episodes"] == 0:
            continue
        speedup = baseline["avg_ms"] / target["avg_ms"] if target["avg_ms"] else 0.0
        drop = baseline["success_rate"] - target["success_rate"]
        label = "overall" if suite is None else suite
        print(
            f"| {label} | {baseline['successes']}/{baseline['episodes']} | "
            f"{target['successes']}/{target['episodes']} | {baseline['avg_ms']:.1f} | "
            f"{target['avg_ms']:.1f} | {speedup:.2f}x | {drop:.1%} |"
        )


if __name__ == "__main__":
    main()
