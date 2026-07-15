#!/usr/bin/env python3
"""Run the CI-friendly robotics action-token speculative decode benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.robotics_spec_benchmark import RobotSpecBenchmarkConfig, run_robotics_spec_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark exact robotics action-token speculative decoding.")
    parser.add_argument("--num-tasks", type=int, default=120)
    parser.add_argument("--action-steps", type=int, default=36)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--lookahead", type=int, default=14)
    parser.add_argument("--phase-len", type=int, default=12)
    parser.add_argument("--target-forward-ms", type=float, default=1.0)
    parser.add_argument("--draft-token-ms", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def format_markdown(summary: dict) -> str:
    status = "PASS" if summary["gate_passed"] else "FAIL"
    return "\n".join(
        [
            f"Synthetic robotics spec decode gate: {status}",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| tasks | {summary['tasks']} |",
            f"| exact token match | {summary['all_exact']} |",
            f"| baseline success | {summary['baseline_successes']}/{summary['tasks']} |",
            f"| spec success | {summary['spec_successes']}/{summary['tasks']} |",
            f"| accuracy drop | {summary['accuracy_drop_abs']:.2%} |",
            f"| speedup | {summary['speedup']:.2f}x |",
            f"| target forward reduction | {summary['target_forward_reduction']:.2f}x |",
            f"| acceptance rate | {summary['acceptance_rate']:.2%} |",
            f"| rejected blocks | {summary['rejected_blocks']} |",
        ]
    )


def main() -> int:
    args = parse_args()
    config = RobotSpecBenchmarkConfig(
        num_tasks=args.num_tasks,
        action_steps=args.action_steps,
        action_dim=args.action_dim,
        vocab_size=args.vocab_size,
        lookahead=args.lookahead,
        phase_len=args.phase_len,
        target_forward_ms=args.target_forward_ms,
        draft_token_ms=args.draft_token_ms,
        seed=args.seed,
    )
    summary = run_robotics_spec_benchmark(config)
    checks = {
        "all_exact": bool(summary["all_exact"]),
        "speedup": float(summary["speedup"]) >= args.min_speedup,
        "accuracy_drop": float(summary["accuracy_drop_abs"]) <= args.max_accuracy_drop,
    }
    summary["thresholds"] = {
        "min_speedup": args.min_speedup,
        "max_accuracy_drop": args.max_accuracy_drop,
    }
    summary["checks"] = checks
    summary["gate_passed"] = all(checks.values())

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(format_markdown(summary) if args.markdown else json.dumps(summary, indent=2))
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
