#!/usr/bin/env python3
"""Render a compact result card from the final robotics spec audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_audit(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_trace_stat(name: str, stat: dict[str, Any]) -> str:
    rows = _as_int(stat, "rows")
    missing = _as_int(stat, "missing_rows")
    total = f", total=`{_as_float(stat, 'total'):.4g}`" if "total" in stat else ""
    return (
        f"`{name}`: min=`{_as_float(stat, 'min'):.4g}`, "
        f"avg=`{_as_float(stat, 'avg'):.4g}`, "
        f"max=`{_as_float(stat, 'max'):.4g}`, "
        f"missing=`{missing}/{rows}`"
        f"{total}"
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_source_counts(selection: dict[str, Any]) -> str:
    sources = selection.get("required_source_coverage")
    counts = _as_dict(selection.get("required_source_counts"))
    if not isinstance(sources, list) or not sources:
        return ""
    parts = []
    for source in sources:
        name = str(source)
        parts.append(f"`{name}`=`{_as_int(counts, name)}`")
    return ", ".join(parts)


def _format_heldout_selection(selection: dict[str, Any]) -> str:
    selected = _as_dict(selection.get("selected"))
    heldout_tasks = _as_int(selected, "heldout_task_count")
    if heldout_tasks <= 0:
        return ""
    parts = [f"tasks=`{heldout_tasks}`"]
    if "heldout_modeled_speedup" in selected:
        parts.append(f"modeled_speedup=`{_as_float(selected, 'heldout_modeled_speedup'):.2f}x`")
    if "heldout_target_forward_reduction" in selected:
        parts.append(f"forward_reduction=`{_as_float(selected, 'heldout_target_forward_reduction'):.2f}x`")
    metrics = selection.get("effective_min_heldout_metric")
    if isinstance(metrics, list) and metrics:
        parts.append(f"metric_thresholds=`{len(metrics)}`")
    return ", ".join(parts)


def render_result_card(audit: dict[str, Any]) -> str:
    pi0 = audit.get("pi0fast") or {}
    openvla = audit.get("openvla") or {}
    use_openvla = bool(openvla.get("passed")) and not bool(pi0.get("passed"))
    platform = "OpenVLA/SpecVLA" if use_openvla else "PI0-FAST/LIBERO"
    speed = (openvla.get("speed") if use_openvla else pi0.get("speed")) or {}
    metadata = {} if use_openvla else _as_dict(pi0.get("metadata"))
    manifest = {} if use_openvla else _as_dict(pi0.get("manifest"))
    manifest_metadata = _as_dict(manifest.get("eval_metadata"))
    pattern_sweep_selection = _as_dict(manifest.get("pattern_sweep_selection"))
    exact = pi0.get("exact_validation") or {}
    extra_exact = pi0.get("extra_exact_validations") or {}
    reference = pi0.get("reference") or {}
    candidate_trace_stats = {} if use_openvla else (pi0.get("candidate_trace_stats") or {})
    openvla_quality = _as_dict(openvla.get("quality"))
    synthetic = (audit.get("synthetic") or {}).get("summary") or {}
    missing_evidence = audit.get("missing_evidence") or []
    passed = bool(audit.get("objective_audit_passed"))
    status = "PASS" if passed else "FAIL"
    candidate = str((openvla if use_openvla else pi0).get("candidate_mode") or speed.get("candidate_mode") or "")
    baseline_successes = _as_int(speed, "baseline_successes")
    candidate_successes = _as_int(speed, "candidate_successes")
    matched_pairs = _as_int(speed, "matched_pairs")
    success_text = (
        f"{candidate_successes}/{matched_pairs}"
        if candidate_successes or matched_pairs
        else "see gate"
    )
    baseline_text = (
        f"{baseline_successes}/{matched_pairs}"
        if baseline_successes or matched_pairs
        else "see gate"
    )

    lines = [
        f"# Robotics Speculative Decoding Result: {status}",
        "",
        "| Requirement | Evidence |",
        "| --- | --- |",
        f"| Candidate | `{candidate}` |",
        f"| 100+ real evals | `{matched_pairs}` matched {platform} evals |",
        f"| Baseline success | `{baseline_text}` |",
        f"| Candidate success | `{success_text}` |",
        f"| Accuracy drop | `{_as_float(speed, 'success_drop_abs', 1.0):.2%}` |",
        f"| Speedup vs baseline | `{_as_float(speed, 'speedup'):.2f}x` |",
        f"| Baseline-success regressions | `{_as_int(speed, 'baseline_success_regressions')}` |",
        f"| Synthetic draft-verify check | `{_as_int(synthetic, 'tasks')}` tasks, exact=`{bool(synthetic.get('all_exact'))}`, speedup=`{_as_float(synthetic, 'speedup'):.2f}x` |",
    ]
    policy_kind = str(metadata.get("policy_kind") or manifest_metadata.get("policy_kind") or "")
    if policy_kind:
        lines.append(f"| Policy kind | `{policy_kind}` |")
    if use_openvla:
        lines.append("| Primary exact validation | `not required for OpenVLA matched success gate` |")
        if openvla_quality:
            lines.append(
                "| OpenVLA strict spec quality | "
                f"missing spec_stats=`{_as_int(openvla_quality, 'candidate_rows_missing_spec_stats')}`, "
                f"unverified shortcuts=`{_as_int(openvla_quality, 'unverified_action_shortcuts')}`, "
                f"unverified draft tokens=`{_as_int(openvla_quality, 'unverified_draft_tokens')}`, "
                f"fast draft calls=`{_as_int(openvla_quality, 'fast_draft_calls')}`, "
                f"chunk buffer hits=`{_as_int(openvla_quality, 'chunk_buffer_hits')}`, "
                f"relaxed accepts=`{_as_int(openvla_quality, 'relaxed_group_accepts')}`, "
                f"max tree depth=`{_as_int(openvla_quality, 'max_tree_depth_used')}` |"
            )
    else:
        lines.append(
            f"| Primary exact validation | `{exact.get('validation_mode', 'unknown')}`: `{_as_int(exact, 'episodes')}` rows, max diff `{_as_float(exact, 'max_action_diff', 1.0):.6g}` |"
        )
        source_counts = _format_source_counts(pattern_sweep_selection)
        if source_counts:
            lines.append(f"| Pattern source coverage | {source_counts} |")
        heldout_selection = _format_heldout_selection(pattern_sweep_selection)
        if heldout_selection:
            lines.append(f"| Pattern heldout evidence | {heldout_selection} |")
    for mode, extra in sorted(extra_exact.items()):
        lines.append(
            f"| Extra exact validation | `{mode}`: `{_as_int(extra, 'episodes')}` rows, max diff `{_as_float(extra, 'max_action_diff', 1.0):.6g}` |"
        )
    if reference:
        lines.append(
            f"| Speedup vs early-stop reference | `{_as_float(reference, 'speedup'):.2f}x` vs `{reference.get('baseline_mode')}` |"
        )
        lines.append(
            f"| Early-stop reference | compared against `{reference.get('baseline_mode')}`: "
            f"speedup=`{_as_float(reference, 'speedup'):.2f}x`, "
            f"drop=`{_as_float(reference, 'success_drop_abs', 1.0):.2%}`, "
            f"regressions=`{_as_int(reference, 'baseline_success_regressions')}` |"
        )
    for name, stat in sorted(candidate_trace_stats.items()):
        if isinstance(stat, dict):
            lines.append(f"| Candidate trace stat | {_format_trace_stat(name, stat)} |")
    lines.extend(
        [
            "",
            "Only use this result card for the final claim when the header says `PASS`.",
        ]
    )
    if missing_evidence:
        lines.extend(["", "Missing evidence:"])
        for item in missing_evidence:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render final robotics speculative decoding result card.")
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Render a failed audit card and exit 0. By default failed audits exit 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = load_audit(args.audit_json)
    text = render_result_card(audit)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if audit.get("objective_audit_passed") or args.allow_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
