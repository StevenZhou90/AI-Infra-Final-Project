#!/usr/bin/env python3
"""Gate PI0-FAST speedup results against matched baselines.

The speed sweep is intentionally run in separate processes for ``baseline`` and
accelerated modes such as ``target_eos``. This script rebuilds the matched eval
set from metrics.jsonl rows, then checks the claims we care about:

* enough matched robot evaluations,
* no success drop, including no baseline-success episode regressions,
* latency speedup,
* optional comparison against an early-stop reference mode,
* optional exact-action validation against fixed-budget decode.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, order=True)
class EvalKey:
    task: str
    task_id: int | None
    episode: int
    seed: int


def load_metric_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def eval_key(row: dict[str, Any]) -> EvalKey:
    task_id = row.get("task_id")
    return EvalKey(
        task=str(row.get("task", "")),
        task_id=None if task_id is None else int(task_id),
        episode=int(row.get("episode", 0)),
        seed=int(row.get("seed", 0)),
    )


def rows_by_key(rows: list[dict[str, Any]], mode: str) -> dict[EvalKey, dict[str, Any]]:
    keyed: dict[EvalKey, dict[str, Any]] = {}
    duplicates: list[EvalKey] = []
    for row in rows:
        if row.get("mode") != mode:
            continue
        key = eval_key(row)
        if key in keyed:
            duplicates.append(key)
        keyed[key] = row
    if duplicates:
        preview = ", ".join(str(key) for key in sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate rows for mode={mode}: {preview}")
    return keyed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _success(row: dict[str, Any]) -> bool:
    return bool(row.get("success", False))


def _latency(row: dict[str, Any]) -> float:
    return float(row.get("avg_ms_per_control_step", 0.0))


def _steps(row: dict[str, Any]) -> int | None:
    if "steps" in row and row["steps"] is not None:
        return int(row["steps"])
    if "control_steps" in row and row["control_steps"] is not None:
        return int(row["control_steps"])
    return None


def _avg_chunk_stat(rows: list[dict[str, Any]], name: str) -> float:
    values = []
    for row in rows:
        stats = row.get("chunk_stats") or {}
        if name in stats:
            values.append(float(stats[name]))
    return _mean(values)


def _trace_stats(row: dict[str, Any]) -> dict[str, Any]:
    chunk_stats = row.get("chunk_stats") or {}
    trace_stats = chunk_stats.get("trace_stats") or {}
    return trace_stats if isinstance(trace_stats, dict) else {}


def _safe_check_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def parse_stat_thresholds(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Trace-stat threshold must use NAME=VALUE syntax, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Trace-stat threshold has an empty name: {value!r}")
        parsed[key] = float(raw)
    return parsed


def parse_metadata(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Metadata must use NAME=VALUE syntax, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Metadata has an empty name: {value!r}")
        parsed[key] = raw.strip()
    return parsed


def summarize_candidate_trace_stats(
    rows: list[dict[str, Any]],
    *,
    names: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in sorted(set(names)):
        values = []
        missing = 0
        for row in rows:
            trace_stats = _trace_stats(row)
            if name in trace_stats:
                values.append(float(trace_stats[name]))
            else:
                missing += 1
        effective_for_max = [*values, *([0.0] * missing)]
        summary[name] = {
            "rows": len(rows),
            "rows_with_stat": len(values),
            "missing_rows": missing,
            "total": sum(values),
            "avg": _mean(values),
            "min": min(values) if values else 0.0,
            "max": max(effective_for_max) if effective_for_max else 0.0,
        }
    return summary


def summarize_matched_speed(
    rows: list[dict[str, Any]],
    *,
    baseline_mode: str,
    candidate_mode: str,
) -> dict[str, Any]:
    baseline = rows_by_key(rows, baseline_mode)
    candidate = rows_by_key(rows, candidate_mode)
    matched_keys = sorted(set(baseline) & set(candidate))
    missing_candidate = sorted(set(baseline) - set(candidate))
    missing_baseline = sorted(set(candidate) - set(baseline))
    baseline_rows = [baseline[key] for key in matched_keys]
    candidate_rows = [candidate[key] for key in matched_keys]

    baseline_ms = _mean([_latency(row) for row in baseline_rows])
    candidate_ms = _mean([_latency(row) for row in candidate_rows])
    baseline_successes = sum(_success(row) for row in baseline_rows)
    candidate_successes = sum(_success(row) for row in candidate_rows)
    unique_tasks = {(key.task, key.task_id) for key in matched_keys}
    regressions = [
        key
        for key in matched_keys
        if _success(baseline[key]) and not _success(candidate[key])
    ]
    gains = [
        key
        for key in matched_keys
        if not _success(baseline[key]) and _success(candidate[key])
    ]
    step_mismatches = [
        key
        for key in matched_keys
        if _steps(baseline[key]) is not None
        and _steps(candidate[key]) is not None
        and _steps(baseline[key]) != _steps(candidate[key])
    ]
    missing_step_keys = [
        key
        for key in matched_keys
        if _steps(baseline[key]) is None or _steps(candidate[key]) is None
    ]

    suites: dict[str, dict[str, Any]] = {}
    for suite in sorted({key.task for key in matched_keys}):
        suite_keys = [key for key in matched_keys if key.task == suite]
        suite_base = [baseline[key] for key in suite_keys]
        suite_candidate = [candidate[key] for key in suite_keys]
        suite_base_ms = _mean([_latency(row) for row in suite_base])
        suite_candidate_ms = _mean([_latency(row) for row in suite_candidate])
        suite_unique_tasks = {key.task_id for key in suite_keys}
        suite_baseline_successes = sum(_success(row) for row in suite_base)
        suite_candidate_successes = sum(_success(row) for row in suite_candidate)
        suite_regressions = sum(
            1 for key in suite_keys if _success(baseline[key]) and not _success(candidate[key])
        )
        suite_step_mismatches = sum(
            1
            for key in suite_keys
            if _steps(baseline[key]) is not None
            and _steps(candidate[key]) is not None
            and _steps(baseline[key]) != _steps(candidate[key])
        )
        suite_missing_steps = sum(
            1 for key in suite_keys if _steps(baseline[key]) is None or _steps(candidate[key]) is None
        )
        suites[suite] = {
            "matched_pairs": len(suite_keys),
            "unique_tasks": len(suite_unique_tasks),
            "baseline_successes": int(suite_baseline_successes),
            "candidate_successes": int(suite_candidate_successes),
            "success_drop_abs": (suite_baseline_successes - suite_candidate_successes) / max(len(suite_keys), 1),
            "baseline_success_regressions": suite_regressions,
            "step_mismatch_count": suite_step_mismatches,
            "missing_step_count": suite_missing_steps,
            "baseline_avg_ms": suite_base_ms,
            "candidate_avg_ms": suite_candidate_ms,
            "speedup": suite_base_ms / suite_candidate_ms if suite_candidate_ms else 0.0,
        }

    return {
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "matched_pairs": len(matched_keys),
        "unique_tasks": len(unique_tasks),
        "missing_candidate_rows": len(missing_candidate),
        "missing_baseline_rows": len(missing_baseline),
        "baseline_successes": int(baseline_successes),
        "candidate_successes": int(candidate_successes),
        "baseline_success_rate": baseline_successes / max(len(matched_keys), 1),
        "candidate_success_rate": candidate_successes / max(len(matched_keys), 1),
        "success_drop_abs": (baseline_successes - candidate_successes) / max(len(matched_keys), 1),
        "baseline_success_regressions": len(regressions),
        "candidate_success_gains": len(gains),
        "step_mismatch_count": len(step_mismatches),
        "missing_step_count": len(missing_step_keys),
        "baseline_avg_ms": baseline_ms,
        "candidate_avg_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms if candidate_ms else 0.0,
        "baseline_avg_model_call_ms": _avg_chunk_stat(baseline_rows, "avg_model_call_ms"),
        "candidate_avg_model_call_ms": _avg_chunk_stat(candidate_rows, "avg_model_call_ms"),
        "baseline_avg_fast_token_count": _avg_chunk_stat(baseline_rows, "avg_fast_token_count"),
        "candidate_avg_fast_token_count": _avg_chunk_stat(candidate_rows, "avg_fast_token_count"),
        "suites": suites,
        "regression_examples": [key.__dict__ for key in regressions[:10]],
        "step_mismatch_examples": [
            {
                **key.__dict__,
                "baseline_steps": _steps(baseline[key]),
                "candidate_steps": _steps(candidate[key]),
            }
            for key in step_mismatches[:10]
        ],
        "missing_step_examples": [
            {
                **key.__dict__,
                "baseline_steps": _steps(baseline[key]),
                "candidate_steps": _steps(candidate[key]),
            }
            for key in missing_step_keys[:10]
        ],
        "missing_candidate_examples": [key.__dict__ for key in missing_candidate[:10]],
        "missing_baseline_examples": [key.__dict__ for key in missing_baseline[:10]],
    }


def matched_speed_keys(rows: list[dict[str, Any]], *, baseline_mode: str, candidate_mode: str) -> set[EvalKey]:
    baseline = rows_by_key(rows, baseline_mode)
    candidate = rows_by_key(rows, candidate_mode)
    return set(baseline) & set(candidate)


def summarize_exact_validation(
    rows: list[dict[str, Any]],
    *,
    validation_mode: str,
    expected_keys: set[EvalKey] | None = None,
) -> dict[str, Any]:
    validation_by_key = rows_by_key(rows, validation_mode)
    if expected_keys is None:
        selected_keys = set(validation_by_key)
        missing_validation = set()
        extra_validation = set()
    else:
        selected_keys = set(validation_by_key) & expected_keys
        missing_validation = expected_keys - set(validation_by_key)
        extra_validation = set(validation_by_key) - expected_keys

    exact_verifies = 0
    static_exact_verifies = 0
    max_action_diff = 0.0
    mean_action_diffs: list[float] = []
    successes = 0
    rows_with_exact_verifies = 0
    rows_missing_exact_verifies: list[EvalKey] = []
    for key in sorted(selected_keys):
        row = validation_by_key[key]
        stats = row.get("chunk_stats") or {}
        row_runtime_exact_verifies = int(stats.get("exact_verifies", 0))
        row_static_exact_verifies = int(stats.get("static_exact_verifies", 0))
        row_exact_verifies = row_runtime_exact_verifies + row_static_exact_verifies
        exact_verifies += row_exact_verifies
        static_exact_verifies += row_static_exact_verifies
        if row_exact_verifies > 0:
            rows_with_exact_verifies += 1
        else:
            rows_missing_exact_verifies.append(key)
        max_action_diff = max(max_action_diff, float(stats.get("max_action_diff", 0.0)))
        mean_action_diffs.append(float(stats.get("mean_action_diff", 0.0)))
        successes += int(_success(row))
    return {
        "validation_mode": validation_mode,
        "episodes": len(selected_keys),
        "available_rows": len(validation_by_key),
        "expected_rows": None if expected_keys is None else len(expected_keys),
        "missing_validation_rows": len(missing_validation),
        "extra_validation_rows": len(extra_validation),
        "rows_with_exact_verifies": rows_with_exact_verifies,
        "rows_missing_exact_verifies": len(rows_missing_exact_verifies),
        "successes": successes,
        "success_rate": successes / max(len(selected_keys), 1),
        "exact_verifies": exact_verifies,
        "runtime_exact_verifies": exact_verifies - static_exact_verifies,
        "static_exact_verifies": static_exact_verifies,
        "max_action_diff": max_action_diff,
        "mean_action_diff": _mean(mean_action_diffs),
        "missing_validation_examples": [key.__dict__ for key in sorted(missing_validation)[:10]],
        "extra_validation_examples": [key.__dict__ for key in sorted(extra_validation)[:10]],
        "rows_missing_exact_verify_examples": [key.__dict__ for key in rows_missing_exact_verifies[:10]],
    }


def build_gate_summary(
    *,
    speed_rows: list[dict[str, Any]],
    baseline_mode: str,
    candidate_mode: str,
    min_pairs: int,
    min_speedup: float,
    max_success_drop: float,
    max_baseline_success_regressions: int,
    min_unique_tasks: int = 0,
    min_baseline_successes: int = 1,
    min_suite_matched_pairs: int = 0,
    min_suite_baseline_successes: int = 0,
    min_suite_speedup: float | None = None,
    require_matched_steps: bool = True,
    validation_rows: list[dict[str, Any]] | None = None,
    validation_mode: str = "target_eos_validate",
    extra_validation_modes: list[str] | None = None,
    require_exact_validation: bool = True,
    min_validation_episodes: int = 120,
    min_exact_verifies: int = 1,
    max_action_diff: float = 0.0,
    reference_mode: str | None = None,
    min_reference_speedup: float | None = None,
    max_reference_success_drop: float | None = None,
    max_reference_success_regressions: int | None = None,
    min_candidate_trace_stats: dict[str, float] | None = None,
    max_candidate_trace_stats: dict[str, float] | None = None,
    min_candidate_trace_stat_totals: dict[str, float] | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    speed = summarize_matched_speed(speed_rows, baseline_mode=baseline_mode, candidate_mode=candidate_mode)
    speed_keys = matched_speed_keys(speed_rows, baseline_mode=baseline_mode, candidate_mode=candidate_mode)
    candidate_by_key = rows_by_key(speed_rows, candidate_mode)
    matched_candidate_rows = [candidate_by_key[key] for key in sorted(speed_keys)]
    min_candidate_trace_stats = min_candidate_trace_stats or {}
    max_candidate_trace_stats = max_candidate_trace_stats or {}
    min_candidate_trace_stat_totals = min_candidate_trace_stat_totals or {}
    candidate_trace_stats = summarize_candidate_trace_stats(
        matched_candidate_rows,
        names=[*min_candidate_trace_stats, *max_candidate_trace_stats, *min_candidate_trace_stat_totals],
    )
    checks = {
        "matched_pair_count": speed["matched_pairs"] >= min_pairs,
        "complete_matching": speed["missing_candidate_rows"] == 0 and speed["missing_baseline_rows"] == 0,
        "unique_tasks": speed["unique_tasks"] >= min_unique_tasks,
        "baseline_successes": speed["baseline_successes"] >= min_baseline_successes,
        "speedup": speed["speedup"] >= min_speedup,
        "success_drop": speed["success_drop_abs"] <= max_success_drop,
        "baseline_success_regressions": speed["baseline_success_regressions"] <= max_baseline_success_regressions,
    }
    if require_matched_steps:
        checks["matched_steps"] = speed["step_mismatch_count"] == 0 and speed["missing_step_count"] == 0
    suite_stats = speed.get("suites") or {}
    if min_suite_matched_pairs > 0:
        checks["suite_matched_pairs"] = bool(suite_stats) and all(
            int(suite.get("matched_pairs", 0)) >= min_suite_matched_pairs for suite in suite_stats.values()
        )
    if min_suite_baseline_successes > 0:
        checks["suite_baseline_successes"] = bool(suite_stats) and all(
            int(suite.get("baseline_successes", 0)) >= min_suite_baseline_successes
            for suite in suite_stats.values()
        )
    if min_suite_speedup is not None:
        checks["suite_speedup"] = bool(suite_stats) and all(
            float(suite.get("speedup", 0.0)) >= min_suite_speedup for suite in suite_stats.values()
        )
    for name, threshold in min_candidate_trace_stats.items():
        stat = candidate_trace_stats.get(name, {})
        checks[f"candidate_trace_min_{_safe_check_name(name)}"] = (
            int(stat.get("missing_rows", 0)) == 0 and float(stat.get("min", 0.0)) >= float(threshold)
        )
    for name, threshold in max_candidate_trace_stats.items():
        stat = candidate_trace_stats.get(name, {})
        checks[f"candidate_trace_max_{_safe_check_name(name)}"] = float(stat.get("max", 0.0)) <= float(threshold)
    for name, threshold in min_candidate_trace_stat_totals.items():
        stat = candidate_trace_stats.get(name, {})
        checks[f"candidate_trace_total_{_safe_check_name(name)}"] = float(stat.get("total", 0.0)) >= float(
            threshold
        )

    exact = None
    extra_exact: dict[str, dict[str, Any]] = {}
    if validation_rows is not None:
        exact = summarize_exact_validation(validation_rows, validation_mode=validation_mode, expected_keys=speed_keys)
        checks.update(
            {
                "validation_episode_count": exact["episodes"] >= min_validation_episodes,
                "validation_key_matching": exact["missing_validation_rows"] == 0,
                "exact_verify_count": exact["exact_verifies"] >= min_exact_verifies,
                "per_row_exact_verify": exact["rows_missing_exact_verifies"] == 0,
                "exact_action_diff": exact["max_action_diff"] <= max_action_diff,
            }
        )
        for extra_mode in extra_validation_modes or []:
            if extra_mode == validation_mode:
                continue
            extra = summarize_exact_validation(validation_rows, validation_mode=extra_mode, expected_keys=speed_keys)
            extra_exact[extra_mode] = extra
            check_prefix = f"extra_validation_{extra_mode}"
            checks.update(
                {
                    f"{check_prefix}_episode_count": extra["episodes"] >= min_validation_episodes,
                    f"{check_prefix}_key_matching": extra["missing_validation_rows"] == 0,
                    f"{check_prefix}_exact_verify_count": extra["exact_verifies"] >= min_exact_verifies,
                    f"{check_prefix}_per_row_exact_verify": extra["rows_missing_exact_verifies"] == 0,
                    f"{check_prefix}_exact_action_diff": extra["max_action_diff"] <= max_action_diff,
                }
            )
    elif require_exact_validation:
        checks.update(
            {
                "validation_episode_count": False,
                "validation_key_matching": False,
                "exact_verify_count": False,
                "per_row_exact_verify": False,
                "exact_action_diff": False,
            }
        )

    reference = None
    if reference_mode is not None:
        reference = summarize_matched_speed(
            speed_rows,
            baseline_mode=reference_mode,
            candidate_mode=candidate_mode,
        )
        checks["reference_complete_matching"] = (
            reference["missing_candidate_rows"] == 0 and reference["missing_baseline_rows"] == 0
        )
        if require_matched_steps:
            checks["reference_matched_steps"] = (
                reference["step_mismatch_count"] == 0 and reference["missing_step_count"] == 0
            )
        if min_reference_speedup is not None:
            checks["reference_speedup"] = reference["speedup"] >= min_reference_speedup
        if max_reference_success_drop is not None:
            checks["reference_success_drop"] = reference["success_drop_abs"] <= max_reference_success_drop
        if max_reference_success_regressions is not None:
            checks["reference_success_regressions"] = (
                reference["baseline_success_regressions"] <= max_reference_success_regressions
            )

    return {
        "gate_passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_pairs": min_pairs,
            "min_unique_tasks": min_unique_tasks,
            "min_speedup": min_speedup,
            "min_baseline_successes": min_baseline_successes,
            "max_success_drop": max_success_drop,
            "max_baseline_success_regressions": max_baseline_success_regressions,
            "require_matched_steps": require_matched_steps,
            "require_exact_validation": require_exact_validation,
            "min_suite_matched_pairs": min_suite_matched_pairs,
            "min_suite_baseline_successes": min_suite_baseline_successes,
            "min_suite_speedup": min_suite_speedup,
            "min_validation_episodes": min_validation_episodes,
            "min_exact_verifies": min_exact_verifies,
            "max_action_diff": max_action_diff,
            "extra_validation_modes": extra_validation_modes or [],
            "reference_mode": reference_mode,
            "min_reference_speedup": min_reference_speedup,
            "max_reference_success_drop": max_reference_success_drop,
            "max_reference_success_regressions": max_reference_success_regressions,
            "min_candidate_trace_stats": min_candidate_trace_stats,
            "max_candidate_trace_stats": max_candidate_trace_stats,
            "min_candidate_trace_stat_totals": min_candidate_trace_stat_totals,
        },
        "speed": speed,
        "reference": reference,
        "exact_validation": exact,
        "extra_exact_validations": extra_exact,
        "candidate_trace_stats": candidate_trace_stats,
        "metadata": metadata or {},
    }


def format_markdown(summary: dict[str, Any]) -> str:
    speed = summary["speed"]
    reference = summary.get("reference")
    exact = summary.get("exact_validation")
    extra_exact = summary.get("extra_exact_validations") or {}
    candidate_trace_stats = summary.get("candidate_trace_stats") or {}
    metadata = summary.get("metadata") or {}
    status = "PASS" if summary["gate_passed"] else "FAIL"
    candidate_label = speed["candidate_mode"]
    lines = [
        f"PI0-FAST speedup gate: {status}",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| matched eval pairs | {speed['matched_pairs']} |",
        f"| baseline success | {speed['baseline_successes']}/{speed['matched_pairs']} |",
        f"| {candidate_label} success | {speed['candidate_successes']}/{speed['matched_pairs']} |",
        f"| success drop | {speed['success_drop_abs']:.2%} |",
        f"| baseline-success regressions | {speed['baseline_success_regressions']} |",
        f"| step mismatches | {speed['step_mismatch_count']} |",
        f"| missing step rows | {speed['missing_step_count']} |",
        f"| baseline avg ms/control | {speed['baseline_avg_ms']:.1f} |",
        f"| {candidate_label} avg ms/control | {speed['candidate_avg_ms']:.1f} |",
        f"| speedup | {speed['speedup']:.2f}x |",
    ]
    if reference is not None:
        lines.extend(
            [
                f"| reference mode | {reference['baseline_mode']} |",
                f"| {candidate_label} speedup vs reference | {reference['speedup']:.2f}x |",
                f"| {candidate_label} drop vs reference | {reference['success_drop_abs']:.2%} |",
                f"| {candidate_label} reference-success regressions | {reference['baseline_success_regressions']} |",
                f"| {candidate_label} reference step mismatches | {reference['step_mismatch_count']} |",
                f"| {candidate_label} reference missing step rows | {reference['missing_step_count']} |",
            ]
        )
    for key, value in sorted(metadata.items()):
        lines.append(f"| metadata {key} | {value} |")
    if exact is not None:
        lines.extend(
            [
                f"| validation episodes | {exact['episodes']} |",
                f"| missing validation rows | {exact['missing_validation_rows']} |",
                f"| rows missing exact verifies | {exact['rows_missing_exact_verifies']} |",
                f"| exact verifies | {exact['exact_verifies']} |",
                f"| max action diff | {exact['max_action_diff']:.6g} |",
            ]
        )
    for name, stat in sorted(candidate_trace_stats.items()):
        lines.extend(
            [
                f"| candidate trace {name} min | {stat['min']:.6g} |",
                f"| candidate trace {name} max | {stat['max']:.6g} |",
                f"| candidate trace {name} total | {stat['total']:.6g} |",
                f"| candidate trace {name} missing rows | {stat['missing_rows']} |",
            ]
        )
    for mode, extra in sorted(extra_exact.items()):
        lines.extend(
            [
                f"| {mode} validation episodes | {extra['episodes']} |",
                f"| {mode} max action diff | {extra['max_action_diff']:.6g} |",
            ]
        )
    lines.extend(["", "Checks:"])
    for name, passed in summary["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate PI0-FAST speedup claims.")
    parser.add_argument("speed_root", type=Path, help="Root containing per-mode metrics.jsonl files.")
    parser.add_argument("--baseline-mode", default="baseline")
    parser.add_argument("--candidate-mode", default="target_eos")
    parser.add_argument(
        "--reference-mode",
        default=None,
        help="Optional early-stop/reference mode to compare the candidate against, usually target_eos.",
    )
    parser.add_argument(
        "--min-reference-speedup",
        type=float,
        default=None,
        help="Optional minimum candidate speedup relative to --reference-mode.",
    )
    parser.add_argument(
        "--max-reference-success-drop",
        type=float,
        default=None,
        help="Optional maximum candidate success drop relative to --reference-mode.",
    )
    parser.add_argument(
        "--max-reference-success-regressions",
        type=int,
        default=None,
        help="Optional maximum candidate regressions on reference-success episodes.",
    )
    parser.add_argument("--validation-root", type=Path, default=None)
    parser.add_argument("--validation-mode", default="target_eos_validate")
    parser.add_argument(
        "--extra-validation-mode",
        action="append",
        default=[],
        help="Additional exact-validation modes that must cover the matched speed keys.",
    )
    parser.add_argument("--min-pairs", type=int, default=120)
    parser.add_argument("--min-unique-tasks", type=int, default=0)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--min-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-matched-pairs", type=int, default=0)
    parser.add_argument("--min-suite-baseline-successes", type=int, default=0)
    parser.add_argument("--min-suite-speedup", type=float, default=None)
    parser.add_argument("--max-success-drop", type=float, default=0.0)
    parser.add_argument("--max-baseline-success-regressions", type=int, default=0)
    parser.add_argument(
        "--require-matched-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require matched baseline/candidate rows to report identical control-step counts.",
    )
    parser.add_argument("--require-exact-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-validation-episodes", type=int, default=120)
    parser.add_argument("--min-exact-verifies", type=int, default=1)
    parser.add_argument("--max-action-diff", type=float, default=0.0)
    parser.add_argument(
        "--min-candidate-trace-stat",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require every matched candidate row to report trace_stats[NAME] >= VALUE.",
    )
    parser.add_argument(
        "--max-candidate-trace-stat",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require matched candidate trace_stats[NAME] <= VALUE. Missing rows count as 0.",
    )
    parser.add_argument(
        "--min-candidate-trace-stat-total",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require total matched candidate trace_stats[NAME] across rows to be >= VALUE.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Attach run metadata such as policy_kind=pi05 to the gate JSON artifact.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    speed_rows = load_metric_rows(args.speed_root)
    validation_rows = load_metric_rows(args.validation_root) if args.validation_root is not None else None
    summary = build_gate_summary(
        speed_rows=speed_rows,
        baseline_mode=args.baseline_mode,
        candidate_mode=args.candidate_mode,
        min_pairs=args.min_pairs,
        min_unique_tasks=args.min_unique_tasks,
        min_speedup=args.min_speedup,
        min_baseline_successes=args.min_baseline_successes,
        min_suite_matched_pairs=args.min_suite_matched_pairs,
        min_suite_baseline_successes=args.min_suite_baseline_successes,
        min_suite_speedup=args.min_suite_speedup,
        require_matched_steps=args.require_matched_steps,
        max_success_drop=args.max_success_drop,
        max_baseline_success_regressions=args.max_baseline_success_regressions,
        validation_rows=validation_rows,
        validation_mode=args.validation_mode,
        extra_validation_modes=args.extra_validation_mode,
        require_exact_validation=args.require_exact_validation,
        min_validation_episodes=args.min_validation_episodes,
        min_exact_verifies=args.min_exact_verifies,
        max_action_diff=args.max_action_diff,
        reference_mode=args.reference_mode,
        min_reference_speedup=args.min_reference_speedup,
        max_reference_success_drop=args.max_reference_success_drop,
        max_reference_success_regressions=args.max_reference_success_regressions,
        min_candidate_trace_stats=parse_stat_thresholds(args.min_candidate_trace_stat),
        max_candidate_trace_stats=parse_stat_thresholds(args.max_candidate_trace_stat),
        min_candidate_trace_stat_totals=parse_stat_thresholds(args.min_candidate_trace_stat_total),
        metadata=parse_metadata(args.metadata),
    )
    text = format_markdown(summary) if args.markdown else json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(text)
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
