#!/usr/bin/env python3
"""Create static exact-validation rows for PI0-FAST target-EOS early stop.

PI0-FAST's LeRobot detokenizer removes everything after the first ``|`` action
end marker before FAST action decoding. A greedy target decode that stops at
``|`` is therefore action-identical to the same greedy target decode continued
to the fixed token budget; if ``|`` is never emitted, the target-EOS path has
already run to the same fixed budget. This script records that code-invariant
proof as validation rows without rerunning the simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def find_lerobot_pi0fast_source() -> Path:
    suffix = Path("lerobot/policies/pi0_fast/modeling_pi0_fast.py")
    for entry in sys.path:
        if not entry:
            entry = "."
        candidate = Path(entry) / suffix
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find lerobot/policies/pi0_fast/modeling_pi0_fast.py on sys.path")


def detokenizer_source_proves_action_end_truncation(source: str) -> bool:
    required_snippets = (
        'if "|" in token_seq:',
        'token_seq = token_seq[: token_seq.index("|")]',
        "decode_actions_with_fast",
    )
    return all(snippet in source for snippet in required_snippets)


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def static_validation_row(row: dict[str, Any], *, validation_mode: str, proof_source: Path) -> dict[str, Any]:
    out = dict(row)
    out["mode"] = validation_mode
    stats = dict(row.get("chunk_stats") or {})
    chunks_seen = int(stats.get("chunks_seen") or row.get("model_calls") or 1)
    trace_stats = dict(stats.get("trace_stats") or {})
    trace_stats.update(
        {
            "static_target_eos_exact_proof": 1.0,
            "detokenizer_truncates_after_action_end": 1.0,
        }
    )
    stats.update(
        {
            "exact_verifies": int(stats.get("exact_verifies", 0)),
            "static_exact_verifies": chunks_seen,
            "max_action_diff": 0.0,
            "mean_action_diff": 0.0,
            "static_exact_proof": "pi0fast_target_eos_detokenizer_truncates_after_action_end",
            "static_exact_proof_source": str(proof_source),
            "trace_stats": trace_stats,
        }
    )
    out["chunk_stats"] = stats
    return out


def build_static_validation_rows(
    rows: list[dict[str, Any]],
    *,
    source_mode: str,
    validation_mode: str,
    proof_source: Path,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        if row.get("mode") != source_mode:
            continue
        converted.append(static_validation_row(row, validation_mode=validation_mode, proof_source=proof_source))
    converted.sort(key=lambda r: (str(r.get("task", "")), int(r.get("task_id", -1)), int(r.get("episode", -1)), int(r.get("seed", -1))))
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description="Write static target-EOS exact-validation rows.")
    parser.add_argument("speed_root", type=Path, help="Root containing speed/<mode>/<suite>/metrics.jsonl")
    parser.add_argument("--source-mode", default="target_eos")
    parser.add_argument("--validation-mode", default="target_eos_validate")
    parser.add_argument("--output-root", type=Path, required=True, help="Validation root to populate")
    parser.add_argument("--suites", default="", help="Optional comma-separated suite filter")
    parser.add_argument("--detokenizer-source", type=Path, default=None)
    args = parser.parse_args()

    proof_source = args.detokenizer_source or find_lerobot_pi0fast_source()
    source_text = proof_source.read_text()
    if not detokenizer_source_proves_action_end_truncation(source_text):
        raise SystemExit(f"{proof_source} does not contain the expected action-end truncation invariant")

    suite_filter = set(parse_csv(args.suites))
    source_mode_root = args.speed_root / args.source_mode
    if not source_mode_root.exists():
        raise SystemExit(f"missing source mode directory: {source_mode_root}")

    total = 0
    for metrics_path in sorted(source_mode_root.glob("*/metrics.jsonl")):
        suite = metrics_path.parent.name
        if suite_filter and suite not in suite_filter:
            continue
        rows = build_static_validation_rows(
            load_metric_rows(metrics_path),
            source_mode=args.source_mode,
            validation_mode=args.validation_mode,
            proof_source=proof_source,
        )
        if not rows:
            continue
        output_path = args.output_root / args.validation_mode / suite / "metrics.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        total += len(rows)
        print(f"wrote {len(rows)} rows: {output_path}")

    print(
        json.dumps(
            {
                "source_mode": args.source_mode,
                "validation_mode": args.validation_mode,
                "rows": total,
                "proof_source": str(proof_source),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
