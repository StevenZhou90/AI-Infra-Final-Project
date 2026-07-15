#!/usr/bin/env python3
"""Audit artifacts for the robotics speculative decoding objective.

This script does not run the expensive simulator. It checks whether the output
artifacts from the real PI0-FAST gate and the CI synthetic draft-verify check
are strong enough to support the final claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _as_float(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_int(row: dict[str, Any] | None, key: str, default: int = 0) -> int:
    if row is None:
        return default
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _optional_float(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None or key not in row or row.get(key) is None:
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return float("inf")


CHECK_DESCRIPTIONS = {
    "pi0fast_gate_present": "missing PI0-FAST gate JSON artifact",
    "pi0fast_gate_passed": "PI0-FAST gate did not pass",
    "pi0fast_policy_kind": "PI0-FAST/PI0.5 gate policy kind does not match expected policy",
    "pi0fast_manifest_present": "missing PI0-FAST run manifest artifact",
    "pi0fast_manifest_candidate_mode": "PI0-FAST run manifest candidate mode does not match gate candidate",
    "pi0fast_manifest_reference_mode": "PI0-FAST run manifest does not record target_eos reference mode",
    "pi0fast_manifest_policy_kind": "PI0-FAST run manifest policy kind does not match expected policy",
    "pi0fast_pattern_source_coverage": "pattern sweep selection lacks required evaluated source coverage",
    "pi0fast_pattern_heldout_evidence": "pattern sweep selection lacks required heldout selection evidence",
    "pi0fast_matched_pairs": "real PI0-FAST matched eval count is below threshold",
    "pi0fast_unique_tasks": "real PI0-FAST unique task coverage is below threshold",
    "pi0fast_baseline_successes": "real PI0-FAST baseline success count is below threshold",
    "pi0fast_suite_matched_pairs": "real PI0-FAST per-suite matched eval count is below threshold",
    "pi0fast_suite_baseline_successes": "real PI0-FAST per-suite baseline success count is below threshold",
    "pi0fast_suite_speedup": "real PI0-FAST per-suite speedup is below threshold",
    "pi0fast_speedup": "real PI0-FAST speedup is below threshold",
    "pi0fast_success_drop": "real PI0-FAST success drop exceeds threshold",
    "pi0fast_baseline_success_regressions": "candidate regressed at least one baseline-success episode",
    "pi0fast_matched_steps": "PI0-FAST gate did not enforce matched control-step counts",
    "pi0fast_exact_validation_present": "missing candidate exact-validation artifact",
    "pi0fast_candidate_validation_mode": "candidate exact-validation mode does not match the candidate",
    "pi0fast_validation_rows_match": "candidate exact-validation rows do not cover the matched speed eval keys",
    "pi0fast_validation_episodes": "candidate exact-validation episode count is below threshold",
    "pi0fast_per_row_exact_verify": "at least one candidate validation row has no exact verify",
    "pi0fast_action_diff": "candidate validation max action diff exceeds threshold",
    "pi0fast_early_stop_exact_validation": "missing passing target_eos_validate exactness rows for the early-stop reference",
    "pi0fast_early_stop_reference": "missing target_eos early-stop reference comparison",
    "pi0fast_reference_candidate_mode": "target_eos reference comparison does not use the candidate mode",
    "pi0fast_reference_matched_pairs": "target_eos reference comparison matched eval count is below threshold",
    "pi0fast_reference_speedup": "candidate speedup versus target_eos is below threshold",
    "pi0fast_reference_success_drop": "candidate success drop versus target_eos exceeds threshold",
    "pi0fast_reference_success_regressions": "candidate regressed at least one target_eos-success episode",
    "pi0fast_reference_matched_steps": "candidate and target_eos reference rows do not have matched control-step counts",
    "openvla_gate_present": "missing OpenVLA/SpecVLA gate JSON artifact",
    "openvla_gate_passed": "OpenVLA/SpecVLA gate did not pass",
    "openvla_matched_pairs": "real OpenVLA/SpecVLA matched eval count is below threshold",
    "openvla_unique_tasks": "real OpenVLA/SpecVLA unique task coverage is below threshold",
    "openvla_baseline_successes": "real OpenVLA/SpecVLA baseline success count is below threshold",
    "openvla_suite_matched_pairs": "real OpenVLA/SpecVLA per-suite matched eval count is below threshold",
    "openvla_suite_baseline_successes": "real OpenVLA/SpecVLA per-suite baseline success count is below threshold",
    "openvla_suite_speedup": "real OpenVLA/SpecVLA per-suite speedup is below threshold",
    "openvla_speedup": "real OpenVLA/SpecVLA speedup is below threshold",
    "openvla_success_drop": "real OpenVLA/SpecVLA success drop exceeds threshold",
    "openvla_baseline_success_regressions": "SpecVLA candidate regressed at least one AR-success episode",
    "openvla_matched_steps": "OpenVLA/SpecVLA gate did not enforce matched control-step counts",
    "openvla_spec_stats_present": "OpenVLA/SpecVLA candidate rows are missing spec_stats",
    "openvla_strict_spec_thresholds": "OpenVLA/SpecVLA gate did not enforce strict zero-shortcut thresholds",
    "openvla_unverified_action_shortcuts": "OpenVLA/SpecVLA artifact used unverified action shortcuts",
    "openvla_unverified_draft_tokens": "OpenVLA/SpecVLA artifact used unverified draft tokens",
    "openvla_fast_draft_calls": "OpenVLA/SpecVLA artifact used fast draft calls",
    "openvla_chunk_buffer_hits": "OpenVLA/SpecVLA artifact used chunk-buffer shortcuts",
    "openvla_chunk_buffered_actions": "OpenVLA/SpecVLA artifact used chunk-buffered actions",
    "openvla_relaxed_group_accepts": "OpenVLA/SpecVLA artifact used relaxed group accepts",
    "openvla_tree_depth": "OpenVLA/SpecVLA artifact used tree depth above 1",
    "synthetic_gate_present": "missing synthetic draft-verify gate artifact",
    "synthetic_gate_passed": "synthetic draft-verify gate did not pass",
    "synthetic_tasks": "synthetic task count is below threshold",
    "synthetic_exact": "synthetic draft-verify outputs are not exact",
    "synthetic_speedup": "synthetic modeled speedup is below threshold",
    "synthetic_accuracy_drop": "synthetic accuracy drop exceeds threshold",
}


def missing_evidence_from_checks(checks: dict[str, bool]) -> list[str]:
    missing: list[str] = []
    for name, passed in checks.items():
        if passed:
            continue
        missing.append(CHECK_DESCRIPTIONS.get(name, name))
    return missing


def expected_candidate_validation_mode(candidate_mode: str, early_stop_mode: str) -> str:
    if candidate_mode in {"", early_stop_mode}:
        return f"{early_stop_mode}_validate"
    if candidate_mode.startswith("pattern_sd"):
        return "pattern_sd_validate"
    if candidate_mode.startswith("ngram_sd"):
        return "ngram_sd_validate"
    if candidate_mode.startswith("block_sd"):
        return "block_sd_validate"
    if candidate_mode.startswith("target_eos_adaptive"):
        return "target_eos_adaptive_validate"
    return f"{candidate_mode}_validate"


def _exact_validation_passes(
    exact: dict[str, Any] | None,
    *,
    expected_mode: str,
    speed: dict[str, Any] | None,
    min_validation_episodes: int,
    max_action_diff: float,
) -> bool:
    return bool(
        exact
        and exact.get("validation_mode") == expected_mode
        and _as_int(exact, "missing_validation_rows", 10**9) == 0
        and _as_int(exact, "expected_rows", -1) == _as_int(speed, "matched_pairs", -2)
        and _as_int(exact, "episodes") >= min_validation_episodes
        and _as_int(exact, "rows_missing_exact_verifies", 10**9) == 0
        and _as_float(exact, "max_action_diff", 1.0) <= max_action_diff
    )


def _as_str_dict(row: Any) -> dict[str, Any]:
    return row if isinstance(row, dict) else {}


def _as_str_list(row: Any) -> list[str]:
    if isinstance(row, list):
        return [str(value) for value in row]
    return []


def _positive_counts_for_sources(counts: dict[str, Any], sources: list[str]) -> bool:
    if not sources:
        return False
    for source in sources:
        try:
            if int(counts.get(source, 0)) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _parse_metric_threshold(value: str) -> tuple[str, float] | None:
    if "=" not in value:
        return None
    name, raw = value.split("=", 1)
    name = name.strip()
    if not name:
        return None
    try:
        return name, float(raw)
    except (TypeError, ValueError):
        return None


def _pattern_heldout_evidence_passes(
    selection: dict[str, Any],
    *,
    required_suites: list[str],
) -> bool:
    selected = _as_str_dict(selection.get("selected"))
    if not selected or _as_int(selected, "heldout_task_count") <= 0:
        return False
    if str(selection.get("heldout_split", "")).strip().lower() in {"", "none", "null"}:
        return False
    if _as_int(selection, "heldout_trace_count") <= 0:
        return False
    if selection.get("heldout_task_disjoint") is not True:
        return False
    if _as_int(selection, "heldout_task_overlap_count", 1) != 0:
        return False
    if not isinstance(selection.get("heldout_task_overlap"), list):
        return False
    required_suite_set = {suite for suite in required_suites if suite}
    if required_suite_set:
        heldout_suite_keys = set(_as_str_list(selection.get("heldout_suite_keys")))
        if not heldout_suite_keys:
            return False
        if not required_suite_set.issubset(heldout_suite_keys):
            return False
    scalar_metrics = {
        "min_heldout_modeled_speedup": "heldout_modeled_speedup",
        "min_heldout_forward_reduction": "heldout_target_forward_reduction",
        "min_heldout_task_forward_reduction": "heldout_min_task_target_forward_reduction",
        "min_heldout_task_acceptance_rate": "heldout_min_task_acceptance_rate",
        "min_heldout_task_count": "heldout_task_count",
    }
    saw_requirement = False
    for threshold_name, metric_name in scalar_metrics.items():
        threshold = selection.get(threshold_name)
        if threshold is None:
            continue
        saw_requirement = True
        try:
            expected = float(threshold)
        except (TypeError, ValueError):
            return False
        if _as_float(selected, metric_name, float("-inf")) < expected:
            return False
    for raw_threshold in _as_str_list(selection.get("effective_min_heldout_metric")):
        parsed = _parse_metric_threshold(raw_threshold)
        if parsed is None:
            return False
        name, threshold = parsed
        metric_name = name if name.startswith("heldout_") else f"heldout_{name}"
        saw_requirement = True
        if _as_float(selected, metric_name, float("-inf")) < threshold:
            return False
    return saw_requirement


def _suite_metric_passes(
    speed: dict[str, Any] | None,
    *,
    metric: str,
    threshold: float,
) -> bool:
    suites = _as_str_dict(speed.get("suites") if speed else None)
    if not suites:
        return False
    return all(_as_float(_as_str_dict(suite), metric, float("-inf")) >= threshold for suite in suites.values())


def audit_pi0fast_gate(
    gate: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    *,
    min_pairs: int,
    min_unique_tasks: int,
    min_speedup: float,
    min_baseline_successes: int,
    min_suite_matched_pairs: int,
    min_suite_baseline_successes: int,
    min_suite_speedup: float | None,
    max_success_drop: float,
    max_baseline_success_regressions: int,
    min_validation_episodes: int,
    max_action_diff: float,
    early_stop_mode: str,
    require_early_stop_reference: bool,
    expected_policy_kind: str | None,
    require_manifest: bool,
    require_pattern_source_coverage: bool,
    require_pattern_heldout_evidence: bool,
) -> dict[str, Any]:
    speed = gate.get("speed") if gate else None
    exact = gate.get("exact_validation") if gate else None
    extra_exact = gate.get("extra_exact_validations") if gate else None
    extra_exact = extra_exact if isinstance(extra_exact, dict) else {}
    reference = gate.get("reference") if gate else None
    thresholds = _as_str_dict(gate.get("thresholds") if gate else None)
    metadata = gate.get("metadata") if gate else None
    metadata = metadata if isinstance(metadata, dict) else {}
    manifest_metadata = _as_str_dict(manifest.get("eval_metadata") if manifest else None)
    pattern_sweep_selection = _as_str_dict(manifest.get("pattern_sweep_selection") if manifest else None)
    required_suites = _as_str_list(manifest.get("suites") if manifest else None)
    required_sources = _as_str_list(pattern_sweep_selection.get("required_source_coverage"))
    required_source_counts = _as_str_dict(pattern_sweep_selection.get("required_source_counts"))
    candidate_mode = str(speed.get("candidate_mode", "")) if isinstance(speed, dict) else ""
    needs_reference = require_early_stop_reference and candidate_mode not in {"", early_stop_mode}
    expected_validation_mode = expected_candidate_validation_mode(candidate_mode, early_stop_mode)
    early_stop_validation_mode = f"{early_stop_mode}_validate"
    early_stop_exact = extra_exact.get(early_stop_validation_mode)
    min_reference_speedup = _optional_float(thresholds, "min_reference_speedup")

    checks = {
        "pi0fast_gate_present": gate is not None,
        "pi0fast_gate_passed": bool(gate and gate.get("gate_passed")),
        "pi0fast_policy_kind": (
            expected_policy_kind is None
            or str(metadata.get("policy_kind", "")) == expected_policy_kind
        ),
        "pi0fast_manifest_present": (manifest is not None) or not require_manifest,
        "pi0fast_manifest_candidate_mode": (
            manifest is None
            or str(manifest.get("candidate_mode", "")) == candidate_mode
        ),
        "pi0fast_manifest_reference_mode": (
            manifest is None
            or not needs_reference
            or str(manifest.get("reference_mode", "")) == early_stop_mode
        ),
        "pi0fast_manifest_policy_kind": (
            manifest is None
            or expected_policy_kind is None
            or str(manifest_metadata.get("policy_kind", "")) == expected_policy_kind
        ),
        "pi0fast_pattern_source_coverage": (
            not require_pattern_source_coverage
            or bool(pattern_sweep_selection)
            and _positive_counts_for_sources(required_source_counts, required_sources)
        ),
        "pi0fast_pattern_heldout_evidence": (
            not require_pattern_heldout_evidence
            or bool(pattern_sweep_selection)
            and _pattern_heldout_evidence_passes(
                pattern_sweep_selection,
                required_suites=required_suites,
            )
        ),
        "pi0fast_matched_pairs": _as_int(speed, "matched_pairs") >= min_pairs,
        "pi0fast_unique_tasks": _as_int(speed, "unique_tasks") >= min_unique_tasks,
        "pi0fast_baseline_successes": _as_int(speed, "baseline_successes") >= min_baseline_successes,
        "pi0fast_suite_matched_pairs": (
            min_suite_matched_pairs <= 0
            or _suite_metric_passes(speed, metric="matched_pairs", threshold=float(min_suite_matched_pairs))
        ),
        "pi0fast_suite_baseline_successes": (
            min_suite_baseline_successes <= 0
            or _suite_metric_passes(
                speed,
                metric="baseline_successes",
                threshold=float(min_suite_baseline_successes),
            )
        ),
        "pi0fast_suite_speedup": (
            min_suite_speedup is None
            or _suite_metric_passes(speed, metric="speedup", threshold=float(min_suite_speedup))
        ),
        "pi0fast_speedup": _as_float(speed, "speedup") >= min_speedup,
        "pi0fast_success_drop": _as_float(speed, "success_drop_abs", 1.0) <= max_success_drop,
        "pi0fast_baseline_success_regressions": (
            _as_int(speed, "baseline_success_regressions", 10**9) <= max_baseline_success_regressions
        ),
        "pi0fast_matched_steps": bool(
            gate
            and thresholds.get("require_matched_steps") is True
            and _as_int(speed, "step_mismatch_count", 10**9) == 0
            and _as_int(speed, "missing_step_count", 10**9) == 0
        ),
        "pi0fast_exact_validation_present": exact is not None,
        "pi0fast_candidate_validation_mode": bool(
            exact is not None and exact.get("validation_mode") == expected_validation_mode
        ),
        "pi0fast_validation_rows_match": bool(
            exact
            and _as_int(exact, "missing_validation_rows", 10**9) == 0
            and _as_int(exact, "expected_rows", -1) == _as_int(speed, "matched_pairs", -2)
        ),
        "pi0fast_validation_episodes": _as_int(exact, "episodes") >= min_validation_episodes,
        "pi0fast_per_row_exact_verify": _as_int(exact, "rows_missing_exact_verifies", 10**9) == 0,
        "pi0fast_action_diff": _as_float(exact, "max_action_diff", 1.0) <= max_action_diff,
        "pi0fast_early_stop_exact_validation": (
            not needs_reference
            or _exact_validation_passes(
                early_stop_exact,
                expected_mode=early_stop_validation_mode,
                speed=speed,
                min_validation_episodes=min_validation_episodes,
                max_action_diff=max_action_diff,
            )
        ),
        "pi0fast_early_stop_reference": (
            not needs_reference
            or bool(reference and reference.get("baseline_mode") == early_stop_mode)
        ),
        "pi0fast_reference_candidate_mode": (
            not needs_reference
            or bool(reference and reference.get("candidate_mode") == candidate_mode)
        ),
        "pi0fast_reference_matched_pairs": (
            not needs_reference
            or bool(reference and _as_int(reference, "matched_pairs") >= min_pairs)
        ),
        "pi0fast_reference_speedup": (
            not needs_reference
            or min_reference_speedup is None
            or bool(reference and _as_float(reference, "speedup") >= min_reference_speedup)
        ),
        "pi0fast_reference_success_drop": (
            not needs_reference
            or bool(reference and _as_float(reference, "success_drop_abs", 1.0) <= max_success_drop)
        ),
        "pi0fast_reference_success_regressions": (
            not needs_reference
            or bool(
                reference
                and _as_int(reference, "baseline_success_regressions", 10**9) <= max_baseline_success_regressions
            )
        ),
        "pi0fast_reference_matched_steps": (
            not needs_reference
            or bool(
                reference
                and thresholds.get("require_matched_steps") is True
                and _as_int(reference, "step_mismatch_count", 10**9) == 0
                and _as_int(reference, "missing_step_count", 10**9) == 0
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_evidence": missing_evidence_from_checks(checks),
        "candidate_mode": candidate_mode,
        "expected_validation_mode": expected_validation_mode,
        "speed": speed,
        "exact_validation": exact,
        "extra_exact_validations": extra_exact,
        "reference": reference,
        "thresholds": thresholds,
        "metadata": metadata,
        "manifest": manifest,
        "candidate_trace_stats": gate.get("candidate_trace_stats") if gate else {},
    }


def audit_synthetic_gate(
    gate: dict[str, Any] | None,
    *,
    required: bool,
    min_tasks: int,
    min_speedup: float,
    max_accuracy_drop: float,
) -> dict[str, Any]:
    checks = {
        "synthetic_gate_present": (gate is not None) or not required,
        "synthetic_gate_passed": bool(gate and gate.get("gate_passed")) or not required,
        "synthetic_tasks": (gate is not None and _as_int(gate, "tasks") >= min_tasks) or not required,
        "synthetic_exact": bool(gate and gate.get("all_exact")) or not required,
        "synthetic_speedup": (gate is not None and _as_float(gate, "speedup") >= min_speedup) or not required,
        "synthetic_accuracy_drop": (
            gate is not None and _as_float(gate, "accuracy_drop_abs", 1.0) <= max_accuracy_drop
        )
        or not required,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_evidence": missing_evidence_from_checks(checks),
        "summary": gate,
    }


def audit_openvla_gate(
    gate: dict[str, Any] | None,
    *,
    min_pairs: int,
    min_unique_tasks: int,
    min_speedup: float,
    min_baseline_successes: int,
    min_suite_matched_pairs: int,
    min_suite_baseline_successes: int,
    min_suite_speedup: float | None,
    max_success_drop: float,
    max_baseline_success_regressions: int,
    require_strict_spec: bool,
) -> dict[str, Any]:
    speed = gate.get("speed") if gate else None
    quality = _as_str_dict(gate.get("quality") if gate else None)
    thresholds = _as_str_dict(gate.get("thresholds") if gate else None)
    strict_thresholds = bool(
        thresholds.get("require_spec_stats") is True
        and _as_int(thresholds, "max_unverified_action_shortcuts", 10**9) == 0
        and _as_int(thresholds, "max_fast_draft_calls", 10**9) == 0
        and _as_int(thresholds, "max_chunk_buffer_hits", 10**9) == 0
        and _as_int(thresholds, "max_relaxed_group_accepts", 10**9) == 0
        and _as_int(thresholds, "max_tree_depth_used", 10**9) <= 1
    )
    checks = {
        "openvla_gate_present": gate is not None,
        "openvla_gate_passed": bool(gate and gate.get("gate_passed")),
        "openvla_matched_pairs": _as_int(speed, "matched_pairs") >= min_pairs,
        "openvla_unique_tasks": _as_int(speed, "unique_tasks") >= min_unique_tasks,
        "openvla_baseline_successes": _as_int(speed, "baseline_successes") >= min_baseline_successes,
        "openvla_suite_matched_pairs": (
            min_suite_matched_pairs <= 0
            or _suite_metric_passes(speed, metric="matched_pairs", threshold=float(min_suite_matched_pairs))
        ),
        "openvla_suite_baseline_successes": (
            min_suite_baseline_successes <= 0
            or _suite_metric_passes(
                speed,
                metric="baseline_successes",
                threshold=float(min_suite_baseline_successes),
            )
        ),
        "openvla_suite_speedup": (
            min_suite_speedup is None
            or _suite_metric_passes(speed, metric="speedup", threshold=float(min_suite_speedup))
        ),
        "openvla_speedup": _as_float(speed, "speedup") >= min_speedup,
        "openvla_success_drop": _as_float(speed, "success_drop_abs", 1.0) <= max_success_drop,
        "openvla_baseline_success_regressions": (
            _as_int(speed, "baseline_success_regressions", 10**9) <= max_baseline_success_regressions
        ),
        "openvla_matched_steps": bool(
            gate
            and thresholds.get("require_matched_steps") is True
            and _as_int(speed, "step_mismatch_count", 10**9) == 0
        ),
        "openvla_spec_stats_present": (
            not require_strict_spec
            or bool(gate and _as_int(quality, "candidate_rows_missing_spec_stats", 10**9) == 0)
        ),
        "openvla_strict_spec_thresholds": not require_strict_spec or strict_thresholds,
        "openvla_unverified_action_shortcuts": (
            not require_strict_spec
            or bool(gate and _as_int(quality, "unverified_action_shortcuts", 10**9) == 0)
        ),
        "openvla_unverified_draft_tokens": (
            not require_strict_spec
            or bool(gate and _as_int(quality, "unverified_draft_tokens", 10**9) == 0)
        ),
        "openvla_fast_draft_calls": (
            not require_strict_spec
            or bool(gate and _as_int(quality, "fast_draft_calls", 10**9) == 0)
        ),
        "openvla_chunk_buffer_hits": (
            not require_strict_spec
            or bool(gate and _as_int(quality, "chunk_buffer_hits", 10**9) == 0)
        ),
        "openvla_chunk_buffered_actions": (
            not require_strict_spec
            or bool(gate and _as_int(quality, "chunk_buffered_actions", 10**9) == 0)
        ),
        "openvla_relaxed_group_accepts": (
            not require_strict_spec
            or bool(gate and _as_int(quality, "relaxed_group_accepts", 10**9) == 0)
        ),
        "openvla_tree_depth": (
            not require_strict_spec
            or bool(gate and _as_int(quality, "max_tree_depth_used", 10**9) <= 1)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_evidence": missing_evidence_from_checks(checks),
        "candidate_mode": str(speed.get("candidate_mode", "")) if isinstance(speed, dict) else "",
        "speed": speed,
        "quality": quality,
        "thresholds": thresholds,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    pi0fast_gate = load_json(args.pi0fast_gate)
    pi0fast_manifest = load_json(args.pi0fast_manifest) if args.pi0fast_manifest is not None else None
    openvla_gate_path = getattr(args, "openvla_gate", None)
    openvla_gate = load_json(openvla_gate_path) if openvla_gate_path is not None else None
    synthetic_gate = load_json(args.synthetic_gate) if args.synthetic_gate is not None else None
    pi0fast = audit_pi0fast_gate(
        pi0fast_gate,
        pi0fast_manifest,
        min_pairs=args.min_pairs,
        min_unique_tasks=args.min_unique_tasks,
        min_speedup=args.min_speedup,
        min_baseline_successes=args.min_baseline_successes,
        min_suite_matched_pairs=args.min_suite_matched_pairs,
        min_suite_baseline_successes=args.min_suite_baseline_successes,
        min_suite_speedup=args.min_suite_speedup,
        max_success_drop=args.max_success_drop,
        max_baseline_success_regressions=args.max_baseline_success_regressions,
        min_validation_episodes=args.min_validation_episodes,
        max_action_diff=args.max_action_diff,
        early_stop_mode=args.early_stop_mode,
        require_early_stop_reference=args.require_early_stop_reference,
        expected_policy_kind=args.expected_policy_kind,
        require_manifest=args.pi0fast_manifest is not None,
        require_pattern_source_coverage=args.require_pattern_source_coverage,
        require_pattern_heldout_evidence=args.require_pattern_heldout_evidence,
    )
    openvla = audit_openvla_gate(
        openvla_gate,
        min_pairs=args.min_pairs,
        min_unique_tasks=args.min_unique_tasks,
        min_speedup=args.min_speedup,
        min_baseline_successes=args.min_baseline_successes,
        min_suite_matched_pairs=args.min_suite_matched_pairs,
        min_suite_baseline_successes=args.min_suite_baseline_successes,
        min_suite_speedup=args.min_suite_speedup,
        max_success_drop=args.max_success_drop,
        max_baseline_success_regressions=args.max_baseline_success_regressions,
        require_strict_spec=args.require_openvla_strict_spec,
    )
    synthetic = audit_synthetic_gate(
        synthetic_gate,
        required=args.require_synthetic,
        min_tasks=args.synthetic_min_tasks,
        min_speedup=args.synthetic_min_speedup,
        max_accuracy_drop=args.synthetic_max_accuracy_drop,
    )
    if openvla_gate_path is None:
        checks = {
            "pi0fast": pi0fast["passed"],
            "synthetic": synthetic["passed"],
        }
        missing_evidence = [*pi0fast["missing_evidence"], *synthetic["missing_evidence"]]
    else:
        real_robotics_passed = bool(pi0fast["passed"] or openvla["passed"])
        checks = {
            "real_robotics": real_robotics_passed,
            "synthetic": synthetic["passed"],
        }
        missing_evidence = [] if real_robotics_passed else [*pi0fast["missing_evidence"], *openvla["missing_evidence"]]
        missing_evidence.extend(synthetic["missing_evidence"])
    return {
        "objective_audit_passed": all(checks.values()),
        "checks": checks,
        "missing_evidence": missing_evidence,
        "thresholds": {
            "min_pairs": args.min_pairs,
            "min_unique_tasks": args.min_unique_tasks,
            "min_speedup": args.min_speedup,
            "min_baseline_successes": args.min_baseline_successes,
            "min_suite_matched_pairs": args.min_suite_matched_pairs,
            "min_suite_baseline_successes": args.min_suite_baseline_successes,
            "min_suite_speedup": args.min_suite_speedup,
            "max_success_drop": args.max_success_drop,
            "max_baseline_success_regressions": args.max_baseline_success_regressions,
            "min_validation_episodes": args.min_validation_episodes,
            "max_action_diff": args.max_action_diff,
            "early_stop_mode": args.early_stop_mode,
            "require_early_stop_reference": args.require_early_stop_reference,
            "expected_policy_kind": args.expected_policy_kind,
            "pi0fast_manifest": str(args.pi0fast_manifest) if args.pi0fast_manifest is not None else None,
            "require_pattern_source_coverage": args.require_pattern_source_coverage,
            "require_pattern_heldout_evidence": args.require_pattern_heldout_evidence,
            "require_openvla_strict_spec": args.require_openvla_strict_spec,
            "require_synthetic": args.require_synthetic,
        },
        "pi0fast": pi0fast,
        "openvla": openvla,
        "synthetic": synthetic,
    }


def format_markdown(audit: dict[str, Any]) -> str:
    status = "PASS" if audit["objective_audit_passed"] else "FAIL"
    pi0 = audit["pi0fast"]
    openvla = audit.get("openvla") or {}
    speed = pi0.get("speed") or {}
    openvla_speed = openvla.get("speed") or {}
    exact = pi0.get("exact_validation") or {}
    syn = audit["synthetic"].get("summary") or {}
    use_openvla = bool(openvla.get("passed")) and not bool(pi0.get("passed"))
    lines = [
        f"Robotics speculative decoding objective audit: {status}",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    if use_openvla:
        lines.extend(
            [
                f"| OpenVLA candidate | {openvla.get('candidate_mode', '')} |",
                f"| OpenVLA matched evals | {openvla_speed.get('matched_pairs', 0)} |",
                f"| OpenVLA success drop | {_as_float(openvla_speed, 'success_drop_abs', 1.0):.2%} |",
                f"| OpenVLA speedup | {_as_float(openvla_speed, 'speedup'):.2f}x |",
            ]
        )
    else:
        lines.extend(
            [
                f"| PI0-FAST candidate | {pi0.get('candidate_mode', '')} |",
                f"| PI0-FAST matched evals | {speed.get('matched_pairs', 0)} |",
                f"| PI0-FAST success drop | {_as_float(speed, 'success_drop_abs', 1.0):.2%} |",
                f"| PI0-FAST speedup | {_as_float(speed, 'speedup'):.2f}x |",
                f"| PI0-FAST validation rows | {exact.get('episodes', 0)} |",
                f"| PI0-FAST max action diff | {_as_float(exact, 'max_action_diff', 1.0):.6g} |",
            ]
        )
    lines.extend(
        [
            f"| synthetic tasks | {syn.get('tasks', 0)} |",
            f"| synthetic exact | {syn.get('all_exact', False)} |",
            f"| synthetic speedup | {_as_float(syn, 'speedup'):.2f}x |",
        ]
    )
    lines.extend(["", "Checks:"])
    if use_openvla:
        group_names = ("openvla", "synthetic")
    else:
        group_names = ("pi0fast", "synthetic")
    for group_name in group_names:
        for name, passed in audit[group_name]["checks"].items():
            lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    if audit.get("missing_evidence"):
        lines.extend(["", "Missing evidence:"])
        for item in audit["missing_evidence"]:
            lines.append(f"- {item}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit robotics speculative decoding result artifacts.")
    parser.add_argument("--pi0fast-gate", type=Path, default=Path("outputs/pi0fast_target_eos_120/gate.json"))
    parser.add_argument("--pi0fast-manifest", type=Path, default=None)
    parser.add_argument("--openvla-gate", type=Path, default=None)
    parser.add_argument("--synthetic-gate", type=Path, default=Path("outputs/robotics_spec_synthetic/gate.json"))
    parser.add_argument("--min-pairs", type=int, default=120)
    parser.add_argument("--min-unique-tasks", type=int, default=0)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    parser.add_argument("--min-baseline-successes", type=int, default=1)
    parser.add_argument("--min-suite-matched-pairs", type=int, default=0)
    parser.add_argument("--min-suite-baseline-successes", type=int, default=0)
    parser.add_argument("--min-suite-speedup", type=float, default=None)
    parser.add_argument("--max-success-drop", type=float, default=0.0)
    parser.add_argument("--max-baseline-success-regressions", type=int, default=0)
    parser.add_argument("--min-validation-episodes", type=int, default=120)
    parser.add_argument("--max-action-diff", type=float, default=0.0)
    parser.add_argument("--early-stop-mode", default="target_eos")
    parser.add_argument("--require-early-stop-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--expected-policy-kind",
        default=None,
        help="Require PI0-FAST gate metadata.policy_kind to match, e.g. pi05 for PI0.5 runs.",
    )
    parser.add_argument(
        "--require-pattern-source-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require the PI0-FAST run manifest to include positive required source coverage for a pattern sweep row.",
    )
    parser.add_argument(
        "--require-pattern-heldout-evidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require a pattern sweep manifest to show heldout metrics were enforced when selecting the row.",
    )
    parser.add_argument(
        "--require-openvla-strict-spec",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require OpenVLA/SpecVLA gate artifacts to enforce and report zero unverified shortcut counters.",
    )
    parser.add_argument("--require-synthetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-min-tasks", type=int, default=120)
    parser.add_argument("--synthetic-min-speedup", type=float, default=2.0)
    parser.add_argument("--synthetic-max-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(args)
    text = format_markdown(audit) if args.markdown else json.dumps(audit, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(text)
    return 0 if audit["objective_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
