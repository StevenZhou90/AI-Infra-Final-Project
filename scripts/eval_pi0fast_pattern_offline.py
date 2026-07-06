#!/usr/bin/env python3
"""Offline acceptance scan for PI0-FAST checkpoint-free pattern speculation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_pattern_drafter import (  # noqa: E402
    PatternDraftConfig,
    PatternFastTokenDrafter,
    evaluate_pattern_drafter,
    parse_source_priority,
)


def parse_stop_tokens(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def normalize_stop_token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        return parse_stop_tokens(value)
    return tuple(int(token) for token in value)


def infer_stop_token_ids(records: list[SimpleNamespace]) -> tuple[int, ...]:
    counts: dict[tuple[int, ...], int] = {}
    for record in records:
        stop_tokens = tuple(int(token) for token in getattr(record, "stop_token_ids", ()))
        if not stop_tokens:
            continue
        counts[stop_tokens] = counts.get(stop_tokens, 0) + 1
    if not counts:
        return ()
    return max(counts, key=lambda tokens: (counts[tokens], -len(tokens), tokens))


def load_lightweight_trace_records(data_dir: str | Path) -> list[SimpleNamespace]:
    data_path = Path(data_dir)
    shard_files = sorted(data_path.glob("**/shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.pt files found in {data_path}")
    records: list[SimpleNamespace] = []
    for shard in shard_files:
        rows = torch.load(shard, map_location="cpu", weights_only=False)
        for idx, row in enumerate(rows):
            task = row.get("task", row.get("suite"))
            if task is None and shard.parent != data_path:
                task = shard.parent.name
            records.append(
                SimpleNamespace(
                    token_ids=row["token_ids"].to(dtype=torch.long),
                    task=None if task is None else str(task),
                    task_id=int(row.get("task_id", -1)),
                    seed=int(row.get("seed", -1)),
                    trace_id=str(row.get("trace_id", f"{shard.stem}:{idx}")),
                    stop_token_ids=normalize_stop_token_ids(
                        row.get("stop_token_ids", row.get("action_end_token_id"))
                    ),
                )
            )
    return records


def _record_task_key(record: Any) -> int | str:
    task_id = int(record.task_id)
    task = getattr(record, "task", None)
    if task is None:
        trace_id = getattr(record, "trace_id", None)
        if isinstance(trace_id, str) and ":task" in trace_id:
            task = trace_id.split(":task", 1)[0]
    if task:
        return f"{task}:{task_id}"
    return task_id


def record_task_key(record: Any) -> int | str:
    return _record_task_key(record)


def record_suite_key(record: Any) -> str | None:
    task = getattr(record, "task", None)
    if task:
        return str(task)
    trace_id = getattr(record, "trace_id", None)
    if isinstance(trace_id, str) and ":task" in trace_id:
        prefix = trace_id.split(":task", 1)[0]
        return prefix or None
    return None


def _task_key_sort_value(task_key: int | str) -> tuple[int, int | str]:
    if isinstance(task_key, int):
        return (0, task_key)
    return (1, str(task_key))


def _grouped_fraction_indices(
    records: list[SimpleNamespace],
    *,
    groups: list[Any],
    record_group: Any,
    val_fraction: float,
    seed: int,
) -> list[int]:
    if not groups:
        return []
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    group_count = _fraction_group_count(len(shuffled), val_fraction)
    selected_groups = set(shuffled[:group_count])
    return sorted(idx for idx, record in enumerate(records) if record_group(record) in selected_groups)


def _fraction_group_count(group_count: int, val_fraction: float) -> int:
    if group_count <= 0:
        return 0
    return max(1, min(group_count, int(round(group_count * float(val_fraction)))))


def _suite_stratified_task_indices(
    records: list[SimpleNamespace],
    *,
    val_fraction: float,
    seed: int,
) -> list[int] | None:
    groups_by_suite: dict[str, set[int | str]] = {}
    for record in records:
        suite = record_suite_key(record)
        if suite is None:
            suite = "__unknown_suite__"
        groups_by_suite.setdefault(str(suite), set()).add(_record_task_key(record))
    if len(groups_by_suite) <= 1:
        return None

    selected_groups: set[int | str] = set()
    for suite, suite_groups in sorted(groups_by_suite.items()):
        shuffled = sorted(suite_groups, key=_task_key_sort_value)
        random.Random(f"{seed}:{suite}").shuffle(shuffled)
        group_count = _fraction_group_count(len(shuffled), val_fraction)
        selected_groups.update(shuffled[:group_count])
    return sorted(idx for idx, record in enumerate(records) if _record_task_key(record) in selected_groups)


def split_eval_indices(
    records: list[SimpleNamespace],
    *,
    split: str,
    val_fraction: float,
    seed: int,
    heldout_task_id: int | None,
    heldout_seed: int | None = None,
) -> list[int]:
    if split == "all":
        return list(range(len(records)))
    if len(records) < 2:
        raise ValueError("Need at least two traces for split evaluation")
    if split == "task":
        if heldout_task_id is None:
            groups = sorted({_record_task_key(record) for record in records}, key=_task_key_sort_value)
            indices = _suite_stratified_task_indices(records, val_fraction=val_fraction, seed=seed)
            if indices is None:
                indices = _grouped_fraction_indices(
                    records,
                    groups=groups,
                    record_group=_record_task_key,
                    val_fraction=val_fraction,
                    seed=seed,
                )
            task_label = ",".join(str(group) for group in groups)
        else:
            indices = [idx for idx, record in enumerate(records) if record.task_id == heldout_task_id]
            task_label = heldout_task_id
        if not indices:
            raise ValueError(f"No traces found for heldout task {task_label}")
        return indices
    if split == "seed":
        if heldout_seed is None:
            seeds = sorted({record.seed for record in records})
            indices = _grouped_fraction_indices(
                records,
                groups=seeds,
                record_group=lambda record: record.seed,
                val_fraction=val_fraction,
                seed=seed,
            )
            seed_label = ",".join(str(group) for group in seeds)
        else:
            indices = [idx for idx, record in enumerate(records) if record.seed == heldout_seed]
            seed_label = heldout_seed
        if not indices:
            raise ValueError(f"No traces found for heldout seed {seed_label}")
        return indices
    if split == "task_seed":
        if heldout_task_id is None and heldout_seed is None:
            groups = sorted(
                {(_record_task_key(record), record.seed) for record in records},
                key=lambda item: (_task_key_sort_value(item[0]), int(item[1])),
            )
            indices = _grouped_fraction_indices(
                records,
                groups=groups,
                record_group=lambda record: (_record_task_key(record), record.seed),
                val_fraction=val_fraction,
                seed=seed,
            )
            task_label = ",".join(f"{task}:{seed_value}" for task, seed_value in groups)
            heldout_seed = "*"
        elif heldout_task_id is None or heldout_seed is None:
            raise ValueError("--heldout-task-id and --heldout-seed must both be set for split=task_seed")
        else:
            indices = [
                idx
                for idx, record in enumerate(records)
                if record.task_id == heldout_task_id and record.seed == heldout_seed
            ]
            task_label = heldout_task_id
        if not indices:
            raise ValueError(f"No traces found for heldout task/seed ({task_label}, {heldout_seed})")
        return indices
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * float(val_fraction))))
    return sorted(indices[:val_count])


def format_markdown(summary: dict) -> str:
    return "\n".join(
        [
            "PI0-FAST pattern drafter offline scan",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| traces | {summary['traces']} |",
            f"| tokens | {summary['tokens']} |",
            f"| lookahead | {summary['lookahead']} |",
            f"| target forward reduction | {summary['target_forward_reduction']:.2f}x |",
            f"| modeled speedup | {summary['modeled_speedup']:.2f}x |",
            f"| acceptance rate | {summary['acceptance_rate']:.2%} |",
            f"| full block reuses | {summary['full_block_reuses']} |",
            f"| bonus tokens | {summary['bonus_tokens']} |",
            f"| tree width | {summary.get('tree_width', 1)} |",
            f"| mean tree candidates | {summary.get('mean_tree_candidates', 0.0):.1f} |",
            f"| tree anchor verifies | {summary.get('tree_anchor_verifies', 0)} |",
            f"| tree anchor accepted tokens | {summary.get('tree_anchor_accepted_tokens', 0)} |",
            f"| mean lookahead | {summary['mean_lookahead']:.1f} |",
            f"| history reset | {summary.get('history_reset', 'task_seed')} |",
            f"| history resets | {summary.get('history_resets', 0)} |",
            f"| source priority | {','.join(summary.get('config', {}).get('source_priority', []))} |",
            f"| drafted tokens | {summary['drafted_tokens']} |",
            f"| accepted tokens | {summary['accepted_tokens']} |",
            f"| previous-chunk drafted tokens | {summary.get('previous_chunk_position_drafted_tokens', 0)} |",
            f"| previous-chunk accepted tokens | {summary.get('previous_chunk_position_accepted_tokens', 0)} |",
            f"| previous-chunk acceptance rate | {summary.get('previous_chunk_position_acceptance_rate', 0.0):.2%} |",
            f"| chunk-prefix drafted tokens | {summary.get('chunk_prefix_retrieval_drafted_tokens', 0)} |",
            f"| chunk-prefix accepted tokens | {summary.get('chunk_prefix_retrieval_accepted_tokens', 0)} |",
            f"| chunk-prefix acceptance rate | {summary.get('chunk_prefix_retrieval_acceptance_rate', 0.0):.2%} |",
            f"| action-token-neighborhood drafted tokens | {summary.get('action_token_neighborhood_drafted_tokens', 0)} |",
            f"| action-token-neighborhood accepted tokens | {summary.get('action_token_neighborhood_accepted_tokens', 0)} |",
            f"| action-token-neighborhood acceptance rate | {summary.get('action_token_neighborhood_acceptance_rate', 0.0):.2%} |",
            f"| position-mode drafted tokens | {summary.get('position_mode_histogram_drafted_tokens', 0)} |",
            f"| position-mode accepted tokens | {summary.get('position_mode_histogram_accepted_tokens', 0)} |",
            f"| position-mode acceptance rate | {summary.get('position_mode_histogram_acceptance_rate', 0.0):.2%} |",
            f"| global-position drafted tokens | {summary.get('global_position_mode_drafted_tokens', 0)} |",
            f"| global-position accepted tokens | {summary.get('global_position_mode_accepted_tokens', 0)} |",
            f"| global-position acceptance rate | {summary.get('global_position_mode_acceptance_rate', 0.0):.2%} |",
            f"| action-dimension-mode drafted tokens | {summary.get('action_dimension_mode_drafted_tokens', 0)} |",
            f"| action-dimension-mode accepted tokens | {summary.get('action_dimension_mode_accepted_tokens', 0)} |",
            f"| action-dimension-mode acceptance rate | {summary.get('action_dimension_mode_acceptance_rate', 0.0):.2%} |",
            f"| hold-action drafted tokens | {summary.get('hold_action_token_drafted_tokens', 0)} |",
            f"| hold-action accepted tokens | {summary.get('hold_action_token_accepted_tokens', 0)} |",
            f"| hold-action acceptance rate | {summary.get('hold_action_token_acceptance_rate', 0.0):.2%} |",
            f"| ngram drafted tokens | {summary.get('ngram_continuation_drafted_tokens', 0)} |",
            f"| ngram accepted tokens | {summary.get('ngram_continuation_accepted_tokens', 0)} |",
            f"| ngram acceptance rate | {summary.get('ngram_continuation_acceptance_rate', 0.0):.2%} |",
            f"| source agreement drafted tokens | {summary.get('source_agreement_drafted_tokens', 0)} |",
            f"| source agreement accepted tokens | {summary.get('source_agreement_accepted_tokens', 0)} |",
            f"| source agreement acceptance rate | {summary.get('source_agreement_acceptance_rate', 0.0):.2%} |",
            f"| action-context-tree drafted tokens | {summary.get('action_context_tree_drafted_tokens', 0)} |",
            f"| action-context-tree accepted tokens | {summary.get('action_context_tree_accepted_tokens', 0)} |",
            f"| action-context-tree acceptance rate | {summary.get('action_context_tree_acceptance_rate', 0.0):.2%} |",
            f"| action-transition drafted tokens | {summary.get('action_transition_histogram_drafted_tokens', 0)} |",
            f"| action-transition accepted tokens | {summary.get('action_transition_histogram_accepted_tokens', 0)} |",
            f"| action-transition acceptance rate | {summary.get('action_transition_histogram_acceptance_rate', 0.0):.2%} |",
            f"| action-vector-transition drafted tokens | {summary.get('action_vector_transition_drafted_tokens', 0)} |",
            f"| action-vector-transition accepted tokens | {summary.get('action_vector_transition_accepted_tokens', 0)} |",
            f"| action-vector-transition acceptance rate | {summary.get('action_vector_transition_acceptance_rate', 0.0):.2%} |",
            f"| action-delta drafted tokens | {summary.get('action_delta_histogram_drafted_tokens', 0)} |",
            f"| action-delta accepted tokens | {summary.get('action_delta_histogram_accepted_tokens', 0)} |",
            f"| action-delta acceptance rate | {summary.get('action_delta_histogram_acceptance_rate', 0.0):.2%} |",
            f"| action-delta-ngram drafted tokens | {summary.get('action_delta_ngram_drafted_tokens', 0)} |",
            f"| action-delta-ngram accepted tokens | {summary.get('action_delta_ngram_accepted_tokens', 0)} |",
            f"| action-delta-ngram acceptance rate | {summary.get('action_delta_ngram_acceptance_rate', 0.0):.2%} |",
            f"| action-repeat-vector drafted tokens | {summary.get('action_repeat_vector_drafted_tokens', 0)} |",
            f"| action-repeat-vector accepted tokens | {summary.get('action_repeat_vector_accepted_tokens', 0)} |",
            f"| action-repeat-vector acceptance rate | {summary.get('action_repeat_vector_acceptance_rate', 0.0):.2%} |",
            f"| chunk-length-stop drafted tokens | {summary.get('chunk_length_stop_drafted_tokens', 0)} |",
            f"| chunk-length-stop accepted tokens | {summary.get('chunk_length_stop_accepted_tokens', 0)} |",
            f"| chunk-length-stop acceptance rate | {summary.get('chunk_length_stop_acceptance_rate', 0.0):.2%} |",
            f"| chunk-delta-template drafted tokens | {summary.get('chunk_delta_template_drafted_tokens', 0)} |",
            f"| chunk-delta-template accepted tokens | {summary.get('chunk_delta_template_accepted_tokens', 0)} |",
            f"| chunk-delta-template acceptance rate | {summary.get('chunk_delta_template_acceptance_rate', 0.0):.2%} |",
            f"| chunk-position-delta drafted tokens | {summary.get('chunk_position_delta_drafted_tokens', 0)} |",
            f"| chunk-position-delta accepted tokens | {summary.get('chunk_position_delta_accepted_tokens', 0)} |",
            f"| chunk-position-delta acceptance rate | {summary.get('chunk_position_delta_acceptance_rate', 0.0):.2%} |",
            f"| source cooldown events | {summary.get('source_cooldown_events', 0)} |",
            f"| source cooldown skipped sources | {summary.get('source_cooldown_skipped_sources', 0)} |",
            f"| source acceptance-bias events | {summary.get('source_acceptance_bias_events', 0)} |",
            f"| source acceptance-bias reorders | {summary.get('source_acceptance_bias_reorders', 0)} |",
            f"| rejected blocks | {summary['rejected_blocks']} |",
            f"| draft misses | {summary['draft_misses']} |",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pattern_sd drafts on PI0-FAST trace shards.")
    parser.add_argument("--data-dir", required=True, help="Directory containing shard_*.pt trace files.")
    parser.add_argument("--split", choices=["trace", "task", "seed", "task_seed", "all"], default="all")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction for trace splits, or group fraction for task/seed/task_seed splits.",
    )
    parser.add_argument("--heldout-task-id", type=int, default=None)
    parser.add_argument("--heldout-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--max-period", type=int, default=16)
    parser.add_argument("--min-period-repeats", type=int, default=2)
    parser.add_argument("--repeat-token-min-run", type=int, default=3)
    parser.add_argument("--linear-action-extrapolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--second-order-action-extrapolation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--second-order-max-accel",
        type=int,
        default=8,
        help="Maximum per-dimension token acceleration for second-order extrapolation; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-action-trend-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from a short per-action-dimension regression trend over verified tokens.",
    )
    parser.add_argument("--pattern-action-trend-history", type=int, default=4)
    parser.add_argument("--pattern-action-trend-top-k", type=int, default=3)
    parser.add_argument(
        "--pattern-action-trend-max-abs",
        type=int,
        default=12,
        help="Maximum absolute projected same-dimension token step for trend proposals; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-action-prefix-lookup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft later action dimensions from prior action vectors with the same verified intra-action prefix.",
    )
    parser.add_argument("--pattern-action-prefix-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-prefix-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-prefix-min-prefix", type=int, default=1)
    parser.add_argument("--pattern-action-prefix-max-mismatches", type=int, default=0)
    parser.add_argument(
        "--pattern-action-vector-transition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from prior full action-vector transitions with matching previous vector and current prefix.",
    )
    parser.add_argument("--pattern-action-vector-transition-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-vector-transition-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-vector-transition-min-count", type=int, default=1)
    parser.add_argument("--pattern-action-vector-transition-max-prev-delta", type=int, default=0)
    parser.add_argument("--pattern-action-vector-transition-max-prefix-delta", type=int, default=0)
    parser.add_argument(
        "--pattern-action-repeat-vector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from repeated full action vectors when the current partial vector still matches.",
    )
    parser.add_argument("--pattern-action-repeat-min-repeats", type=int, default=2)
    parser.add_argument("--pattern-action-repeat-max-delta", type=int, default=0)
    parser.add_argument(
        "--pattern-chunk-length-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft the FAST stop token when recent verified chunks ended at this prefix length.",
    )
    parser.add_argument("--pattern-chunk-length-stop-history-size", type=int, default=4)
    parser.add_argument("--pattern-chunk-length-stop-min-count", type=int, default=2)
    parser.add_argument(
        "--pattern-action-transition-histogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from per-action-dimension transition histograms in verified tokens and recent chunks.",
    )
    parser.add_argument(
        "--pattern-action-context-tree",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from same-action-dimension context trees over verified tokens and recent chunks.",
    )
    parser.add_argument("--pattern-action-context-tree-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-context-tree-max-context", type=int, default=3)
    parser.add_argument("--pattern-action-context-tree-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-context-tree-min-count", type=int, default=1)
    parser.add_argument("--pattern-action-transition-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-transition-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-transition-min-count", type=int, default=1)
    parser.add_argument(
        "--pattern-action-delta-histogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from frequent recent same-action-dimension token deltas.",
    )
    parser.add_argument("--pattern-action-delta-history", type=int, default=8)
    parser.add_argument("--pattern-action-delta-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-delta-min-count", type=int, default=1)
    parser.add_argument(
        "--pattern-action-delta-max-abs",
        type=int,
        default=12,
        help="Maximum absolute token delta for action-delta histogram proposals; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-action-delta-ngram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft by matching same-action-dimension token-delta n-grams in verified tokens and recent chunks.",
    )
    parser.add_argument("--pattern-action-delta-ngram-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-delta-ngram-min-context", type=int, default=1)
    parser.add_argument("--pattern-action-delta-ngram-max-context", type=int, default=4)
    parser.add_argument("--pattern-action-delta-ngram-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-delta-ngram-min-count", type=int, default=1)
    parser.add_argument(
        "--pattern-action-delta-ngram-max-abs",
        type=int,
        default=12,
        help="Maximum absolute token delta for action-delta n-gram proposals; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-chunk-position-delta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft per-position velocity-template deltas from recent verified chunks.",
    )
    parser.add_argument("--pattern-chunk-position-delta-history-size", type=int, default=4)
    parser.add_argument("--pattern-chunk-position-delta-top-k", type=int, default=3)
    parser.add_argument("--pattern-chunk-position-delta-min-count", type=int, default=1)
    parser.add_argument(
        "--pattern-chunk-position-delta-max-abs",
        type=int,
        default=12,
        help="Maximum absolute token delta for chunk-position delta proposals; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-chunk-delta-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from recent verified chunks whose same-dimension delta prefix matches the current chunk.",
    )
    parser.add_argument("--pattern-chunk-delta-template-history-size", type=int, default=4)
    parser.add_argument("--pattern-chunk-delta-template-top-k", type=int, default=3)
    parser.add_argument("--pattern-chunk-delta-template-min-prefix-deltas", type=int, default=1)
    parser.add_argument(
        "--pattern-chunk-delta-template-max-delta-mismatch",
        type=int,
        default=0,
        help="Allowed total absolute mismatch in matched delta prefixes; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-chunk-delta-template-max-abs",
        type=int,
        default=12,
        help="Maximum absolute token delta for chunk-delta template proposals; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-previous-chunk-position",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft token i from token i of recent verified chunks from the same task before local extrapolation.",
    )
    parser.add_argument("--pattern-previous-chunk-history-size", type=int, default=1)
    parser.add_argument(
        "--pattern-chunk-prefix-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from recent verified chunks whose prefix mostly matches the current partial action chunk.",
    )
    parser.add_argument("--pattern-chunk-prefix-history-size", type=int, default=4)
    parser.add_argument("--pattern-chunk-prefix-top-k", type=int, default=3)
    parser.add_argument("--pattern-chunk-prefix-min-matches", type=int, default=2)
    parser.add_argument("--pattern-chunk-prefix-max-mismatches", type=int, default=1)
    parser.add_argument(
        "--pattern-action-token-neighborhood",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft small token neighborhoods around smooth action extrapolations for exact tree verification.",
    )
    parser.add_argument("--pattern-action-token-neighborhood-radius", type=int, default=1)
    parser.add_argument("--pattern-action-token-neighborhood-top-k", type=int, default=3)
    parser.add_argument(
        "--pattern-position-mode-histogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft the per-position mode token from recent verified chunks that share the current prefix.",
    )
    parser.add_argument("--pattern-position-mode-history-size", type=int, default=4)
    parser.add_argument("--pattern-position-mode-top-k", type=int, default=3)
    parser.add_argument("--pattern-position-mode-min-count", type=int, default=2)
    parser.add_argument(
        "--pattern-global-position-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft per-position modal tokens from recent verified chunks without requiring the current prefix to match.",
    )
    parser.add_argument("--pattern-global-position-history-size", type=int, default=16)
    parser.add_argument("--pattern-global-position-top-k", type=int, default=3)
    parser.add_argument("--pattern-global-position-min-count", type=int, default=3)
    parser.add_argument(
        "--pattern-action-dimension-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft the most common verified token for the current action dimension.",
    )
    parser.add_argument("--pattern-action-dimension-mode-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-dimension-mode-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-dimension-mode-min-count", type=int, default=2)
    parser.add_argument(
        "--pattern-hold-action-token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft the token from the same action dimension in the previous action step.",
    )
    parser.add_argument(
        "--pattern-ngram-continuation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from prompt-lookup suffix matches in verified tokens and recent verified chunks.",
    )
    parser.add_argument("--pattern-ngram-min-context", type=int, default=3)
    parser.add_argument("--pattern-ngram-max-context", type=int, default=16)
    parser.add_argument("--pattern-ngram-history-size", type=int, default=4)
    parser.add_argument(
        "--pattern-min-source-agreement",
        type=int,
        default=1,
        help="Prefer a pattern token only when at least this many configured sources propose it; 1 disables agreement labeling.",
    )
    parser.add_argument(
        "--pattern-source-priority",
        default="default",
        help="Named source priority mode or comma-separated source names for pattern drafts.",
    )
    parser.add_argument(
        "--pattern-source-cooldown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily skip a source after verified rejections in the same speculative decode.",
    )
    parser.add_argument("--pattern-source-cooldown-after", type=int, default=1)
    parser.add_argument("--pattern-source-cooldown-steps", type=int, default=1)
    parser.add_argument(
        "--pattern-source-acceptance-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reorder cheap sources by recent verified source acceptance within each decode.",
    )
    parser.add_argument("--pattern-source-acceptance-bias-history-size", type=int, default=32)
    parser.add_argument("--pattern-source-acceptance-bias-min-observations", type=int, default=2)
    parser.add_argument("--reuse-full-blocks", action="store_true")
    parser.add_argument("--emit-bonus-token", action="store_true")
    parser.add_argument("--dynamic-lookahead", action="store_true")
    parser.add_argument("--min-lookahead", type=int, default=1)
    parser.add_argument("--lookahead-growth", type=int, default=1)
    parser.add_argument("--lookahead-shrink", type=int, default=4)
    parser.add_argument(
        "--history-reset",
        choices=["none", "task", "task_seed"],
        default="task_seed",
        help="When previous-chunk history resets during offline simulation. task_seed matches per-episode runner resets.",
    )
    parser.add_argument(
        "--tree-width",
        type=int,
        default=1,
        help="Offline-only exact tree verification candidate count; 1 matches the runnable chain verifier.",
    )
    parser.add_argument("--tree-branch-width", type=int, default=4)
    parser.add_argument("--dynamic-tree-width", action="store_true")
    parser.add_argument("--min-tree-width", type=int, default=1)
    parser.add_argument("--tree-width-growth", type=int, default=1)
    parser.add_argument("--tree-width-shrink", type=int, default=1)
    parser.add_argument(
        "--tree-anchor-target-token",
        action="store_true",
        help="On a tree first-token miss, anchor on the target greedy token and verify drafted continuations after it.",
    )
    parser.add_argument(
        "--tree-anchor-target-continuation",
        action="store_true",
        help="Always anchor on the target greedy token and verify drafted continuations after it.",
    )
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--stop-token-ids", default="")
    parser.add_argument("--target-forward-ms", type=float, default=1.0)
    parser.add_argument("--draft-token-ms", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_lightweight_trace_records(args.data_dir)
    stop_token_ids = parse_stop_tokens(args.stop_token_ids) or infer_stop_token_ids(records)
    indices = split_eval_indices(
        records,
        split=args.split,
        val_fraction=args.val_fraction,
        seed=args.seed,
        heldout_task_id=args.heldout_task_id,
        heldout_seed=args.heldout_seed,
    )
    selected = [records[idx] for idx in indices]
    second_order_max_accel = None if args.second_order_max_accel < 0 else args.second_order_max_accel
    action_trend_max_abs = None if args.pattern_action_trend_max_abs < 0 else args.pattern_action_trend_max_abs
    action_delta_max_abs = None if args.pattern_action_delta_max_abs < 0 else args.pattern_action_delta_max_abs
    action_delta_ngram_max_abs = (
        None if args.pattern_action_delta_ngram_max_abs < 0 else args.pattern_action_delta_ngram_max_abs
    )
    chunk_position_delta_max_abs = (
        None if args.pattern_chunk_position_delta_max_abs < 0 else args.pattern_chunk_position_delta_max_abs
    )
    chunk_delta_template_max_delta_mismatch = (
        None
        if args.pattern_chunk_delta_template_max_delta_mismatch < 0
        else args.pattern_chunk_delta_template_max_delta_mismatch
    )
    chunk_delta_template_max_abs = (
        None if args.pattern_chunk_delta_template_max_abs < 0 else args.pattern_chunk_delta_template_max_abs
    )
    drafter = PatternFastTokenDrafter(
        PatternDraftConfig(
            lookahead=args.lookahead,
            action_dim=args.action_dim,
            max_period=args.max_period,
            min_period_repeats=args.min_period_repeats,
            repeat_token_min_run=args.repeat_token_min_run,
            enable_linear_action_extrapolation=args.linear_action_extrapolation,
            enable_second_order_action_extrapolation=args.second_order_action_extrapolation,
            second_order_max_accel=second_order_max_accel,
            enable_action_trend_regression=args.pattern_action_trend_regression,
            action_trend_history=args.pattern_action_trend_history,
            action_trend_top_k=args.pattern_action_trend_top_k,
            action_trend_max_abs=action_trend_max_abs,
            enable_action_prefix_lookup=args.pattern_action_prefix_lookup,
            action_prefix_history_size=args.pattern_action_prefix_history_size,
            action_prefix_top_k=args.pattern_action_prefix_top_k,
            action_prefix_min_prefix=args.pattern_action_prefix_min_prefix,
            action_prefix_max_mismatches=args.pattern_action_prefix_max_mismatches,
            enable_action_vector_transition=args.pattern_action_vector_transition,
            action_vector_transition_history_size=args.pattern_action_vector_transition_history_size,
            action_vector_transition_top_k=args.pattern_action_vector_transition_top_k,
            action_vector_transition_min_count=args.pattern_action_vector_transition_min_count,
            action_vector_transition_max_prev_delta=args.pattern_action_vector_transition_max_prev_delta,
            action_vector_transition_max_prefix_delta=args.pattern_action_vector_transition_max_prefix_delta,
            enable_action_repeat_vector=args.pattern_action_repeat_vector,
            action_repeat_min_repeats=args.pattern_action_repeat_min_repeats,
            action_repeat_max_delta=args.pattern_action_repeat_max_delta,
            enable_chunk_length_stop=args.pattern_chunk_length_stop,
            chunk_length_stop_history_size=args.pattern_chunk_length_stop_history_size,
            chunk_length_stop_min_count=args.pattern_chunk_length_stop_min_count,
            enable_action_context_tree=args.pattern_action_context_tree,
            action_context_tree_history_size=args.pattern_action_context_tree_history_size,
            action_context_tree_max_context=args.pattern_action_context_tree_max_context,
            action_context_tree_top_k=args.pattern_action_context_tree_top_k,
            action_context_tree_min_count=args.pattern_action_context_tree_min_count,
            enable_action_transition_histogram=args.pattern_action_transition_histogram,
            action_transition_history_size=args.pattern_action_transition_history_size,
            action_transition_top_k=args.pattern_action_transition_top_k,
            action_transition_min_count=args.pattern_action_transition_min_count,
            enable_action_delta_histogram=args.pattern_action_delta_histogram,
            action_delta_history=args.pattern_action_delta_history,
            action_delta_top_k=args.pattern_action_delta_top_k,
            action_delta_min_count=args.pattern_action_delta_min_count,
            action_delta_max_abs=action_delta_max_abs,
            enable_action_delta_ngram=args.pattern_action_delta_ngram,
            action_delta_ngram_history_size=args.pattern_action_delta_ngram_history_size,
            action_delta_ngram_min_context=args.pattern_action_delta_ngram_min_context,
            action_delta_ngram_max_context=args.pattern_action_delta_ngram_max_context,
            action_delta_ngram_top_k=args.pattern_action_delta_ngram_top_k,
            action_delta_ngram_min_count=args.pattern_action_delta_ngram_min_count,
            action_delta_ngram_max_abs=action_delta_ngram_max_abs,
            enable_chunk_position_delta=args.pattern_chunk_position_delta,
            chunk_position_delta_history_size=args.pattern_chunk_position_delta_history_size,
            chunk_position_delta_top_k=args.pattern_chunk_position_delta_top_k,
            chunk_position_delta_min_count=args.pattern_chunk_position_delta_min_count,
            chunk_position_delta_max_abs=chunk_position_delta_max_abs,
            enable_chunk_delta_template=args.pattern_chunk_delta_template,
            chunk_delta_template_history_size=args.pattern_chunk_delta_template_history_size,
            chunk_delta_template_top_k=args.pattern_chunk_delta_template_top_k,
            chunk_delta_template_min_prefix_deltas=args.pattern_chunk_delta_template_min_prefix_deltas,
            chunk_delta_template_max_delta_mismatch=chunk_delta_template_max_delta_mismatch,
            chunk_delta_template_max_abs=chunk_delta_template_max_abs,
            enable_previous_chunk_position=args.pattern_previous_chunk_position,
            previous_chunk_history_size=args.pattern_previous_chunk_history_size,
            enable_chunk_prefix_retrieval=args.pattern_chunk_prefix_retrieval,
            chunk_prefix_history_size=args.pattern_chunk_prefix_history_size,
            chunk_prefix_top_k=args.pattern_chunk_prefix_top_k,
            chunk_prefix_min_matches=args.pattern_chunk_prefix_min_matches,
            chunk_prefix_max_mismatches=args.pattern_chunk_prefix_max_mismatches,
            enable_action_token_neighborhood=args.pattern_action_token_neighborhood,
            action_token_neighborhood_radius=args.pattern_action_token_neighborhood_radius,
            action_token_neighborhood_top_k=args.pattern_action_token_neighborhood_top_k,
            enable_position_mode_histogram=args.pattern_position_mode_histogram,
            position_mode_history_size=args.pattern_position_mode_history_size,
            position_mode_top_k=args.pattern_position_mode_top_k,
            position_mode_min_count=args.pattern_position_mode_min_count,
            enable_global_position_mode=args.pattern_global_position_mode,
            global_position_history_size=args.pattern_global_position_history_size,
            global_position_top_k=args.pattern_global_position_top_k,
            global_position_min_count=args.pattern_global_position_min_count,
            enable_action_dimension_mode=args.pattern_action_dimension_mode,
            action_dimension_mode_history_size=args.pattern_action_dimension_mode_history_size,
            action_dimension_mode_top_k=args.pattern_action_dimension_mode_top_k,
            action_dimension_mode_min_count=args.pattern_action_dimension_mode_min_count,
            enable_hold_action_token=args.pattern_hold_action_token,
            enable_ngram_continuation=args.pattern_ngram_continuation,
            ngram_min_context=args.pattern_ngram_min_context,
            ngram_max_context=args.pattern_ngram_max_context,
            ngram_history_size=args.pattern_ngram_history_size,
            min_source_agreement=args.pattern_min_source_agreement,
            source_priority=parse_source_priority(args.pattern_source_priority),
            enable_source_cooldown=args.pattern_source_cooldown,
            source_cooldown_after=args.pattern_source_cooldown_after,
            source_cooldown_steps=args.pattern_source_cooldown_steps,
            enable_source_acceptance_bias=args.pattern_source_acceptance_bias,
            source_acceptance_bias_history_size=args.pattern_source_acceptance_bias_history_size,
            source_acceptance_bias_min_observations=args.pattern_source_acceptance_bias_min_observations,
            vocab_size=args.vocab_size,
            stop_token_ids=stop_token_ids,
        )
    )
    summary = evaluate_pattern_drafter(
        drafter,
        selected,
        lookahead=args.lookahead,
        reuse_full_blocks=args.reuse_full_blocks,
        emit_bonus_token=args.emit_bonus_token,
        dynamic_lookahead=args.dynamic_lookahead,
        min_lookahead=args.min_lookahead,
        lookahead_growth=args.lookahead_growth,
        lookahead_shrink=args.lookahead_shrink,
        history_reset=args.history_reset,
        tree_width=args.tree_width,
        tree_branch_width=args.tree_branch_width,
        dynamic_tree_width=args.dynamic_tree_width,
        min_tree_width=args.min_tree_width,
        tree_width_growth=args.tree_width_growth,
        tree_width_shrink=args.tree_width_shrink,
        tree_anchor_target_token=args.tree_anchor_target_token,
        tree_anchor_target_continuation=args.tree_anchor_target_continuation,
        target_forward_ms=args.target_forward_ms,
        draft_token_ms=args.draft_token_ms,
    )
    summary["config"] = {
        "data_dir": args.data_dir,
        "split": args.split,
        "eval_trace_indices": indices,
        "action_dim": args.action_dim,
        "max_period": args.max_period,
        "min_period_repeats": args.min_period_repeats,
        "repeat_token_min_run": args.repeat_token_min_run,
        "linear_action_extrapolation": args.linear_action_extrapolation,
        "second_order_action_extrapolation": args.second_order_action_extrapolation,
        "second_order_max_accel": second_order_max_accel,
        "action_trend_regression": args.pattern_action_trend_regression,
        "action_trend_history": args.pattern_action_trend_history,
        "action_trend_top_k": args.pattern_action_trend_top_k,
        "action_trend_max_abs": action_trend_max_abs,
        "action_prefix_lookup": args.pattern_action_prefix_lookup,
        "action_prefix_history_size": args.pattern_action_prefix_history_size,
        "action_prefix_top_k": args.pattern_action_prefix_top_k,
        "action_prefix_min_prefix": args.pattern_action_prefix_min_prefix,
        "action_prefix_max_mismatches": args.pattern_action_prefix_max_mismatches,
        "action_vector_transition": args.pattern_action_vector_transition,
        "action_vector_transition_history_size": args.pattern_action_vector_transition_history_size,
        "action_vector_transition_top_k": args.pattern_action_vector_transition_top_k,
        "action_vector_transition_min_count": args.pattern_action_vector_transition_min_count,
        "action_vector_transition_max_prev_delta": args.pattern_action_vector_transition_max_prev_delta,
        "action_vector_transition_max_prefix_delta": args.pattern_action_vector_transition_max_prefix_delta,
        "action_repeat_vector": args.pattern_action_repeat_vector,
        "action_repeat_min_repeats": args.pattern_action_repeat_min_repeats,
        "action_repeat_max_delta": args.pattern_action_repeat_max_delta,
        "chunk_length_stop": args.pattern_chunk_length_stop,
        "chunk_length_stop_history_size": args.pattern_chunk_length_stop_history_size,
        "chunk_length_stop_min_count": args.pattern_chunk_length_stop_min_count,
        "action_context_tree": args.pattern_action_context_tree,
        "action_context_tree_history_size": args.pattern_action_context_tree_history_size,
        "action_context_tree_max_context": args.pattern_action_context_tree_max_context,
        "action_context_tree_top_k": args.pattern_action_context_tree_top_k,
        "action_context_tree_min_count": args.pattern_action_context_tree_min_count,
        "action_transition_histogram": args.pattern_action_transition_histogram,
        "action_transition_history_size": args.pattern_action_transition_history_size,
        "action_transition_top_k": args.pattern_action_transition_top_k,
        "action_transition_min_count": args.pattern_action_transition_min_count,
        "action_delta_histogram": args.pattern_action_delta_histogram,
        "action_delta_history": args.pattern_action_delta_history,
        "action_delta_top_k": args.pattern_action_delta_top_k,
        "action_delta_min_count": args.pattern_action_delta_min_count,
        "action_delta_max_abs": action_delta_max_abs,
        "action_delta_ngram": args.pattern_action_delta_ngram,
        "action_delta_ngram_history_size": args.pattern_action_delta_ngram_history_size,
        "action_delta_ngram_min_context": args.pattern_action_delta_ngram_min_context,
        "action_delta_ngram_max_context": args.pattern_action_delta_ngram_max_context,
        "action_delta_ngram_top_k": args.pattern_action_delta_ngram_top_k,
        "action_delta_ngram_min_count": args.pattern_action_delta_ngram_min_count,
        "action_delta_ngram_max_abs": action_delta_ngram_max_abs,
        "chunk_position_delta": args.pattern_chunk_position_delta,
        "chunk_position_delta_history_size": args.pattern_chunk_position_delta_history_size,
        "chunk_position_delta_top_k": args.pattern_chunk_position_delta_top_k,
        "chunk_position_delta_min_count": args.pattern_chunk_position_delta_min_count,
        "chunk_position_delta_max_abs": chunk_position_delta_max_abs,
        "chunk_delta_template": args.pattern_chunk_delta_template,
        "chunk_delta_template_history_size": args.pattern_chunk_delta_template_history_size,
        "chunk_delta_template_top_k": args.pattern_chunk_delta_template_top_k,
        "chunk_delta_template_min_prefix_deltas": args.pattern_chunk_delta_template_min_prefix_deltas,
        "chunk_delta_template_max_delta_mismatch": chunk_delta_template_max_delta_mismatch,
        "chunk_delta_template_max_abs": chunk_delta_template_max_abs,
        "previous_chunk_position": args.pattern_previous_chunk_position,
        "previous_chunk_history_size": args.pattern_previous_chunk_history_size,
        "chunk_prefix_retrieval": args.pattern_chunk_prefix_retrieval,
        "chunk_prefix_history_size": args.pattern_chunk_prefix_history_size,
        "chunk_prefix_top_k": args.pattern_chunk_prefix_top_k,
        "chunk_prefix_min_matches": args.pattern_chunk_prefix_min_matches,
        "chunk_prefix_max_mismatches": args.pattern_chunk_prefix_max_mismatches,
        "action_token_neighborhood": args.pattern_action_token_neighborhood,
        "action_token_neighborhood_radius": args.pattern_action_token_neighborhood_radius,
        "action_token_neighborhood_top_k": args.pattern_action_token_neighborhood_top_k,
        "position_mode_histogram": args.pattern_position_mode_histogram,
        "position_mode_history_size": args.pattern_position_mode_history_size,
        "position_mode_top_k": args.pattern_position_mode_top_k,
        "position_mode_min_count": args.pattern_position_mode_min_count,
        "global_position_mode": args.pattern_global_position_mode,
        "global_position_history_size": args.pattern_global_position_history_size,
        "global_position_top_k": args.pattern_global_position_top_k,
        "global_position_min_count": args.pattern_global_position_min_count,
        "action_dimension_mode": args.pattern_action_dimension_mode,
        "action_dimension_mode_history_size": args.pattern_action_dimension_mode_history_size,
        "action_dimension_mode_top_k": args.pattern_action_dimension_mode_top_k,
        "action_dimension_mode_min_count": args.pattern_action_dimension_mode_min_count,
        "hold_action_token": args.pattern_hold_action_token,
        "ngram_continuation": args.pattern_ngram_continuation,
        "ngram_min_context": args.pattern_ngram_min_context,
        "ngram_max_context": args.pattern_ngram_max_context,
        "ngram_history_size": args.pattern_ngram_history_size,
        "min_source_agreement": args.pattern_min_source_agreement,
        "source_priority": list(parse_source_priority(args.pattern_source_priority)),
        "source_cooldown": args.pattern_source_cooldown,
        "source_cooldown_after": args.pattern_source_cooldown_after,
        "source_cooldown_steps": args.pattern_source_cooldown_steps,
        "source_acceptance_bias": args.pattern_source_acceptance_bias,
        "source_acceptance_bias_history_size": args.pattern_source_acceptance_bias_history_size,
        "source_acceptance_bias_min_observations": args.pattern_source_acceptance_bias_min_observations,
        "reuse_full_blocks": args.reuse_full_blocks,
        "emit_bonus_token": args.emit_bonus_token,
        "dynamic_lookahead": args.dynamic_lookahead,
        "min_lookahead": args.min_lookahead,
        "lookahead_growth": args.lookahead_growth,
        "lookahead_shrink": args.lookahead_shrink,
        "history_reset": args.history_reset,
        "tree_width": args.tree_width,
        "tree_branch_width": args.tree_branch_width,
        "dynamic_tree_width": args.dynamic_tree_width,
        "min_tree_width": args.min_tree_width,
        "tree_width_growth": args.tree_width_growth,
        "tree_width_shrink": args.tree_width_shrink,
        "tree_anchor_target_token": args.tree_anchor_target_token,
        "tree_anchor_target_continuation": args.tree_anchor_target_continuation,
        "vocab_size": args.vocab_size,
        "stop_token_ids": list(stop_token_ids),
        "target_forward_ms": args.target_forward_ms,
        "draft_token_ms": args.draft_token_ms,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(format_markdown(summary) if args.markdown else json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
