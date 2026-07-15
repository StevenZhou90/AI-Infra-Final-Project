#!/usr/bin/env python3
"""Strict matched gate for OpenVLA AR vs SpecVLA-style episode artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("suite"),
        row.get("task_id", row.get("task")),
        row.get("trial", row.get("episode")),
        row.get("seed"),
    )


def _row_ms_per_step(row: dict[str, Any]) -> float:
    return float(row.get("inference_ms_total", 0.0)) / max(int(row.get("steps", 0)), 1)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = sum(int(row.get("steps", 0)) for row in rows)
    ms = sum(float(row.get("inference_ms_total", 0.0)) for row in rows)
    successes = sum(1 for row in rows if bool(row.get("success")))
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / max(len(rows), 1),
        "steps": steps,
        "inference_ms_total": ms,
        "avg_ms_per_step": ms / max(steps, 1),
    }


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _spec_stats(row: dict[str, Any]) -> dict[str, Any]:
    stats = row.get("spec_stats")
    return stats if isinstance(stats, dict) else {}


def _aggregate_spec_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_spec_stats = sum(1 for row in rows if not isinstance(row.get("spec_stats"), dict))
    fast_draft_calls = 0
    fast_draft_tokens = 0
    chunk_buffer_hits = 0
    chunk_buffered_actions = 0
    relaxed_group_accepts = 0
    tree_verifies = 0
    max_tree_depth_used = 0
    unverified_action_shortcuts = 0
    unverified_draft_tokens = 0

    for row in rows:
        stats = _spec_stats(row)
        fast = _as_int(stats.get("fast_draft_calls"))
        chunks = _as_int(stats.get("chunk_buffer_hits"))
        fast_tokens = _as_int(stats.get("fast_draft_tokens"))
        fast_draft_calls += fast
        fast_draft_tokens += fast_tokens
        chunk_buffer_hits += chunks
        chunk_buffered_actions += _as_int(stats.get("chunk_buffered_actions"))
        relaxed_group_accepts += _as_int(stats.get("relaxed_group_accepts"))
        tree_verifies += _as_int(stats.get("tree_verifies"))
        max_tree_depth_used = max(max_tree_depth_used, _as_int(stats.get("max_tree_depth_used")))
        unverified_action_shortcuts += _as_int(stats.get("unverified_action_shortcuts", fast + chunks))
        unverified_draft_tokens += _as_int(stats.get("unverified_draft_tokens", fast_tokens))

    return {
        "candidate_rows_missing_spec_stats": missing_spec_stats,
        "fast_draft_calls": fast_draft_calls,
        "fast_draft_tokens": fast_draft_tokens,
        "chunk_buffer_hits": chunk_buffer_hits,
        "chunk_buffered_actions": chunk_buffered_actions,
        "relaxed_group_accepts": relaxed_group_accepts,
        "tree_verifies": tree_verifies,
        "max_tree_depth_used": max_tree_depth_used,
        "unverified_action_shortcuts": unverified_action_shortcuts,
        "unverified_draft_tokens": unverified_draft_tokens,
    }


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = episode_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate episode key in rows: {key!r}")
        indexed[key] = row
    return indexed


def build_gate_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_mode: str = "ar",
    candidate_mode: str = "spec",
    min_pairs: int = 120,
    min_unique_tasks: int = 0,
    min_speedup: float = 2.0,
    min_baseline_successes: int = 1,
    min_suite_matched_pairs: int = 0,
    min_suite_baseline_successes: int = 0,
    min_suite_speedup: float | None = None,
    max_success_drop: float = 0.0,
    max_baseline_success_regressions: int = 0,
    require_matched_steps: bool = True,
    require_spec_stats: bool = False,
    max_unverified_action_shortcuts: int | None = None,
    max_fast_draft_calls: int | None = None,
    max_chunk_buffer_hits: int | None = None,
    max_relaxed_group_accepts: int | None = None,
    max_tree_depth_used: int | None = None,
) -> dict[str, Any]:
    baseline_by_key = _index_rows(baseline_rows)
    candidate_by_key = _index_rows(candidate_rows)
    matched_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    matched_baseline = [baseline_by_key[key] for key in matched_keys]
    matched_candidate = [candidate_by_key[key] for key in matched_keys]
    missing_candidate = len(set(baseline_by_key) - set(candidate_by_key))
    extra_candidate = len(set(candidate_by_key) - set(baseline_by_key))

    baseline = _aggregate(matched_baseline)
    candidate = _aggregate(matched_candidate)
    quality = _aggregate_spec_quality(matched_candidate)
    unique_tasks = {(key[0], key[1]) for key in matched_keys}
    speedup = baseline["avg_ms_per_step"] / max(candidate["avg_ms_per_step"], 1e-9)
    success_drop = baseline["success_rate"] - candidate["success_rate"]
    regressions = 0
    improvements = 0
    step_mismatch_count = 0
    per_episode: list[dict[str, Any]] = []
    for key, base, cand in zip(matched_keys, matched_baseline, matched_candidate):
        base_success = bool(base.get("success"))
        cand_success = bool(cand.get("success"))
        baseline_steps = _as_int(base.get("steps"))
        candidate_steps = _as_int(cand.get("steps"))
        if base_success and not cand_success:
            regressions += 1
        if cand_success and not base_success:
            improvements += 1
        if baseline_steps != candidate_steps:
            step_mismatch_count += 1
        per_episode.append(
            {
                "key": list(key),
                "baseline_success": base_success,
                "candidate_success": cand_success,
                "baseline_steps": baseline_steps,
                "candidate_steps": candidate_steps,
                "baseline_ms_per_step": _row_ms_per_step(base),
                "candidate_ms_per_step": _row_ms_per_step(cand),
            }
        )

    suites: dict[str, dict[str, Any]] = {}
    for suite in sorted({str(key[0]) for key in matched_keys}):
        suite_keys = [key for key in matched_keys if str(key[0]) == suite]
        suite_unique_tasks = {key[1] for key in suite_keys}
        suite_baseline = [baseline_by_key[key] for key in suite_keys]
        suite_candidate = [candidate_by_key[key] for key in suite_keys]
        suite_base = _aggregate(suite_baseline)
        suite_cand = _aggregate(suite_candidate)
        suite_regressions = sum(
            1
            for key in suite_keys
            if bool(baseline_by_key[key].get("success")) and not bool(candidate_by_key[key].get("success"))
        )
        suites[suite] = {
            "matched_pairs": len(suite_keys),
            "unique_tasks": len(suite_unique_tasks),
            "baseline_successes": suite_base["successes"],
            "candidate_successes": suite_cand["successes"],
            "success_drop_abs": suite_base["success_rate"] - suite_cand["success_rate"],
            "baseline_success_regressions": suite_regressions,
            "baseline_avg_ms_per_step": suite_base["avg_ms_per_step"],
            "candidate_avg_ms_per_step": suite_cand["avg_ms_per_step"],
            "speedup": suite_base["avg_ms_per_step"] / max(suite_cand["avg_ms_per_step"], 1e-9),
        }

    checks = {
        "matched_pairs": len(matched_keys) >= int(min_pairs),
        "key_matching": missing_candidate == 0 and extra_candidate == 0,
        "unique_tasks": len(unique_tasks) >= int(min_unique_tasks),
        "baseline_successes": baseline["successes"] >= int(min_baseline_successes),
        "speedup": speedup >= float(min_speedup),
        "success_drop": success_drop <= float(max_success_drop),
        "baseline_success_regressions": regressions <= int(max_baseline_success_regressions),
        "matched_steps": (not require_matched_steps) or step_mismatch_count == 0,
    }
    if min_suite_matched_pairs > 0:
        checks["suite_matched_pairs"] = bool(suites) and all(
            int(suite.get("matched_pairs", 0)) >= int(min_suite_matched_pairs) for suite in suites.values()
        )
    if min_suite_baseline_successes > 0:
        checks["suite_baseline_successes"] = bool(suites) and all(
            int(suite.get("baseline_successes", 0)) >= int(min_suite_baseline_successes) for suite in suites.values()
        )
    if min_suite_speedup is not None:
        checks["suite_speedup"] = bool(suites) and all(
            float(suite.get("speedup", 0.0)) >= float(min_suite_speedup) for suite in suites.values()
        )
    if require_spec_stats:
        checks["spec_stats_present"] = quality["candidate_rows_missing_spec_stats"] == 0
    if max_unverified_action_shortcuts is not None:
        checks["unverified_action_shortcuts"] = (
            quality["unverified_action_shortcuts"] <= int(max_unverified_action_shortcuts)
        )
    if max_fast_draft_calls is not None:
        checks["fast_draft_calls"] = quality["fast_draft_calls"] <= int(max_fast_draft_calls)
    if max_chunk_buffer_hits is not None:
        checks["chunk_buffer_hits"] = quality["chunk_buffer_hits"] <= int(max_chunk_buffer_hits)
    if max_relaxed_group_accepts is not None:
        checks["relaxed_group_accepts"] = quality["relaxed_group_accepts"] <= int(max_relaxed_group_accepts)
    if max_tree_depth_used is not None:
        checks["max_tree_depth_used"] = quality["max_tree_depth_used"] <= int(max_tree_depth_used)

    return {
        "gate_passed": all(checks.values()),
        "platform": "openvla_specvla",
        "checks": checks,
        "speed": {
            "baseline_mode": baseline_mode,
            "candidate_mode": candidate_mode,
            "matched_pairs": len(matched_keys),
            "unique_tasks": len(unique_tasks),
            "baseline_episodes": len(baseline_rows),
            "candidate_episodes": len(candidate_rows),
            "missing_candidate_rows": missing_candidate,
            "extra_candidate_rows": extra_candidate,
            "baseline_successes": baseline["successes"],
            "candidate_successes": candidate["successes"],
            "baseline_success_rate": baseline["success_rate"],
            "candidate_success_rate": candidate["success_rate"],
            "success_drop_abs": success_drop,
            "baseline_success_regressions": regressions,
            "candidate_success_improvements": improvements,
            "step_mismatch_count": step_mismatch_count,
            "baseline_avg_ms_per_step": baseline["avg_ms_per_step"],
            "candidate_avg_ms_per_step": candidate["avg_ms_per_step"],
            "speedup": speedup,
            "suites": suites,
        },
        "quality": quality,
        "thresholds": {
            "min_pairs": int(min_pairs),
            "min_unique_tasks": int(min_unique_tasks),
            "min_speedup": float(min_speedup),
            "min_baseline_successes": int(min_baseline_successes),
            "min_suite_matched_pairs": int(min_suite_matched_pairs),
            "min_suite_baseline_successes": int(min_suite_baseline_successes),
            "min_suite_speedup": min_suite_speedup,
            "max_success_drop": float(max_success_drop),
            "max_baseline_success_regressions": int(max_baseline_success_regressions),
            "require_matched_steps": bool(require_matched_steps),
            "require_spec_stats": bool(require_spec_stats),
            "max_unverified_action_shortcuts": max_unverified_action_shortcuts,
            "max_fast_draft_calls": max_fast_draft_calls,
            "max_chunk_buffer_hits": max_chunk_buffer_hits,
            "max_relaxed_group_accepts": max_relaxed_group_accepts,
            "max_tree_depth_used": max_tree_depth_used,
        },
        "per_episode": per_episode,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    speed = summary["speed"]
    quality = summary.get("quality", {})
    status = "PASS" if summary["gate_passed"] else "FAIL"
    return "\n".join(
        [
            f"OpenVLA/SpecVLA matched gate: {status}",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| matched pairs | {speed['matched_pairs']} |",
            f"| baseline success | {speed['baseline_successes']}/{speed['matched_pairs']} |",
            f"| candidate success | {speed['candidate_successes']}/{speed['matched_pairs']} |",
            f"| success drop | {speed['success_drop_abs']:.2%} |",
            f"| baseline-success regressions | {speed['baseline_success_regressions']} |",
            f"| step mismatches | {speed.get('step_mismatch_count', 0)} |",
            f"| baseline ms/step | {speed['baseline_avg_ms_per_step']:.1f} |",
            f"| candidate ms/step | {speed['candidate_avg_ms_per_step']:.1f} |",
            f"| speedup | {speed['speedup']:.2f}x |",
            f"| rows missing spec_stats | {quality.get('candidate_rows_missing_spec_stats', 0)} |",
            f"| unverified action shortcuts | {quality.get('unverified_action_shortcuts', 0)} |",
            f"| fast draft calls | {quality.get('fast_draft_calls', 0)} |",
            f"| chunk buffer hits | {quality.get('chunk_buffer_hits', 0)} |",
            f"| max tree depth used | {quality.get('max_tree_depth_used', 0)} |",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate OpenVLA AR vs SpecVLA matched episode artifacts.")
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-mode", default="ar")
    parser.add_argument("--candidate-mode", default="spec")
    parser.add_argument("--min-pairs", type=int, default=120)
    parser.add_argument("--min-unique-tasks", type=int, default=0)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--min-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-matched-pairs", type=int, default=0)
    parser.add_argument("--min-suite-baseline-successes", type=int, default=0)
    parser.add_argument("--min-suite-speedup", type=float, default=None)
    parser.add_argument("--max-success-drop", type=float, default=0.0)
    parser.add_argument("--max-baseline-success-regressions", type=int, default=0)
    parser.add_argument("--require-matched-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-spec-stats", action="store_true")
    parser.add_argument("--max-unverified-action-shortcuts", type=int, default=None)
    parser.add_argument("--max-fast-draft-calls", type=int, default=None)
    parser.add_argument("--max-chunk-buffer-hits", type=int, default=None)
    parser.add_argument("--max-relaxed-group-accepts", type=int, default=None)
    parser.add_argument("--max-tree-depth-used", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--include-per-episode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_gate_summary(
        baseline_rows=load_jsonl(args.baseline_jsonl),
        candidate_rows=load_jsonl(args.candidate_jsonl),
        baseline_mode=args.baseline_mode,
        candidate_mode=args.candidate_mode,
        min_pairs=args.min_pairs,
        min_unique_tasks=args.min_unique_tasks,
        min_speedup=args.min_speedup,
        min_baseline_successes=args.min_baseline_successes,
        min_suite_matched_pairs=args.min_suite_matched_pairs,
        min_suite_baseline_successes=args.min_suite_baseline_successes,
        min_suite_speedup=args.min_suite_speedup,
        max_success_drop=args.max_success_drop,
        max_baseline_success_regressions=args.max_baseline_success_regressions,
        require_matched_steps=args.require_matched_steps,
        require_spec_stats=args.require_spec_stats,
        max_unverified_action_shortcuts=args.max_unverified_action_shortcuts,
        max_fast_draft_calls=args.max_fast_draft_calls,
        max_chunk_buffer_hits=args.max_chunk_buffer_hits,
        max_relaxed_group_accepts=args.max_relaxed_group_accepts,
        max_tree_depth_used=args.max_tree_depth_used,
    )
    if not args.include_per_episode:
        summary = {key: value for key, value in summary.items() if key != "per_episode"}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(format_markdown(summary) if args.markdown else json.dumps(summary, indent=2))
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
