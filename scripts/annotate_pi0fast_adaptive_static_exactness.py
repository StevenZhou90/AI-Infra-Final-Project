#!/usr/bin/env python3
"""Annotate PI0-FAST adaptive validation rows with static target-EOS proof.

``target_eos_adaptive_validate --adaptive-validate-stability-stops-only``
runtime-checks only chunks that stop on adaptive stability. The remaining
chunks may be ordinary target-EOS/action-end chunks or explicit target-EOS
fallbacks. For those chunks, the same detokenizer invariant used by
``prove_pi0fast_target_eos_exactness.py`` applies: LeRobot truncates FAST action
tokens at the first ``|`` before action decoding, so a greedy decode that stops
at ``|`` is action-identical to continuing the same greedy target decode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.prove_pi0fast_target_eos_exactness import (
    detokenizer_source_proves_action_end_truncation,
    find_lerobot_pi0fast_source,
    parse_csv,
)


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trace_count(trace_stats: dict[str, Any], name: str, chunks_seen: int) -> int:
    return round(float(trace_stats.get(name, 0.0)) * chunks_seen)


def _chunks_seen(row: dict[str, Any], stats: dict[str, Any]) -> int:
    value = stats.get("chunks_seen") or row.get("model_calls") or 0
    chunks_seen = int(value)
    if chunks_seen <= 0:
        raise ValueError("row has no positive chunk count")
    return chunks_seen


def adaptive_static_exact_counts(row: dict[str, Any]) -> dict[str, int]:
    stats = row.get("chunk_stats") or {}
    chunks_seen = _chunks_seen(row, stats)
    runtime_exact = int(stats.get("exact_verifies", 0))
    if runtime_exact < 0 or runtime_exact > chunks_seen:
        raise ValueError(f"runtime exact count {runtime_exact} is outside [0, {chunks_seen}]")
    trace_stats = stats.get("trace_stats") or {}
    if not isinstance(trace_stats, dict):
        trace_stats = {}

    stability_runtime = trace_count(trace_stats, "stopped_on_stability", chunks_seen) + trace_count(
        trace_stats,
        "continued_after_stability_stop",
        chunks_seen,
    )
    target_eos_chunks = trace_count(trace_stats, "stopped_on_action_end", chunks_seen)
    target_eos_chunks = max(
        target_eos_chunks,
        trace_count(trace_stats, "stability_risk_target_eos_fallback", chunks_seen),
        trace_count(trace_stats, "forced_full_refresh", chunks_seen),
    )
    static_exact = min(max(chunks_seen - runtime_exact, 0), target_eos_chunks)
    accounted = min(chunks_seen, runtime_exact + static_exact)
    return {
        "chunks_seen": chunks_seen,
        "runtime_exact_verifies": runtime_exact,
        "stability_runtime_chunks": stability_runtime,
        "target_eos_static_chunks": static_exact,
        "accounted_exact_chunks": accounted,
        "unaccounted_chunks": chunks_seen - accounted,
    }


def annotate_row(
    row: dict[str, Any],
    *,
    validation_mode: str,
    proof_source: Path,
    max_action_diff: float,
    require_runtime_stability_accounting: bool,
) -> dict[str, Any]:
    if row.get("mode") != validation_mode:
        return row

    out = dict(row)
    stats = dict(row.get("chunk_stats") or {})
    max_diff = float(stats.get("max_action_diff", 0.0))
    if max_diff > max_action_diff:
        raise ValueError(
            f"{row.get('task')} task_id={row.get('task_id')} episode={row.get('episode')} "
            f"has max_action_diff={max_diff}"
        )

    counts = adaptive_static_exact_counts(row)
    if require_runtime_stability_accounting and counts["runtime_exact_verifies"] not in (
        counts["stability_runtime_chunks"],
        counts["chunks_seen"],
    ):
        raise ValueError(
            f"{row.get('task')} task_id={row.get('task_id')} episode={row.get('episode')} "
            f"has runtime exact {counts['runtime_exact_verifies']} but stability trace count "
            f"{counts['stability_runtime_chunks']}"
        )

    existing_static = int(stats.get("static_exact_verifies", 0))
    trace_stats = dict(stats.get("trace_stats") or {})
    trace_stats.update(
        {
            "static_target_eos_adaptive_exact_proof": 1.0,
            "static_target_eos_adaptive_accounted_chunks": float(counts["accounted_exact_chunks"]),
            "static_target_eos_adaptive_unaccounted_chunks": float(counts["unaccounted_chunks"]),
        }
    )
    stats.update(
        {
            "static_exact_verifies": max(existing_static, counts["target_eos_static_chunks"]),
            "static_exact_proof": "pi0fast_adaptive_stability_runtime_plus_target_eos_static",
            "static_exact_proof_source": str(proof_source),
            "static_exact_accounted_chunks": counts["accounted_exact_chunks"],
            "static_exact_unaccounted_chunks": counts["unaccounted_chunks"],
            "trace_stats": trace_stats,
        }
    )
    out["chunk_stats"] = stats
    return out


def annotate_rows(
    rows: list[dict[str, Any]],
    *,
    validation_mode: str,
    proof_source: Path,
    max_action_diff: float = 0.0,
    require_runtime_stability_accounting: bool = True,
) -> list[dict[str, Any]]:
    return [
        annotate_row(
            row,
            validation_mode=validation_mode,
            proof_source=proof_source,
            max_action_diff=max_action_diff,
            require_runtime_stability_accounting=require_runtime_stability_accounting,
        )
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate adaptive validation rows with static exact proof.")
    parser.add_argument("validation_root", type=Path, help="Root containing <mode>/<suite>/metrics.jsonl")
    parser.add_argument("--validation-mode", default="target_eos_adaptive_validate")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suites", default="", help="Optional comma-separated suite filter")
    parser.add_argument("--detokenizer-source", type=Path, default=None)
    parser.add_argument("--max-action-diff", type=float, default=0.0)
    parser.add_argument(
        "--allow-runtime-stability-mismatch",
        action="store_true",
        help="Do not require runtime exact counts to match the aggregate stability-stop trace count.",
    )
    args = parser.parse_args()

    proof_source = args.detokenizer_source or find_lerobot_pi0fast_source()
    if not detokenizer_source_proves_action_end_truncation(proof_source.read_text()):
        raise SystemExit(f"{proof_source} does not contain the expected action-end truncation invariant")

    suite_filter = set(parse_csv(args.suites))
    mode_root = args.validation_root / args.validation_mode
    if not mode_root.exists():
        raise SystemExit(f"missing validation mode directory: {mode_root}")

    total_rows = 0
    total_static = 0
    total_unaccounted = 0
    for metrics_path in sorted(mode_root.glob("*/metrics.jsonl")):
        suite = metrics_path.parent.name
        if suite_filter and suite not in suite_filter:
            continue
        rows = annotate_rows(
            load_metric_rows(metrics_path),
            validation_mode=args.validation_mode,
            proof_source=proof_source,
            max_action_diff=args.max_action_diff,
            require_runtime_stability_accounting=not args.allow_runtime_stability_mismatch,
        )
        for row in rows:
            stats = row.get("chunk_stats") or {}
            total_static += int(stats.get("static_exact_verifies", 0))
            total_unaccounted += int(stats.get("static_exact_unaccounted_chunks", 0))
        output_path = args.output_root / args.validation_mode / suite / "metrics.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        total_rows += len(rows)
        print(f"wrote {len(rows)} rows: {output_path}")

    print(
        json.dumps(
            {
                "validation_mode": args.validation_mode,
                "rows": total_rows,
                "static_exact_verifies": total_static,
                "unaccounted_chunks": total_unaccounted,
                "proof_source": str(proof_source),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
