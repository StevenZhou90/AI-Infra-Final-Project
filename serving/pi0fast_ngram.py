"""N-gram FAST-token drafter for PI0-FAST speculative decoding experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from serving.pi0fast_eagle import PI0FastTraceRecord


@dataclass
class NgramDraftConfig:
    max_context: int = 8
    min_count: int = 1
    lookahead: int = 4
    stop_token_ids: tuple[int, ...] = ()


def trim_at_stop_token(tokens: list[int], stop_token_ids: tuple[int, ...]) -> list[int]:
    if not stop_token_ids:
        return tokens
    stop = set(int(token) for token in stop_token_ids)
    for idx, token in enumerate(tokens):
        if int(token) in stop:
            return tokens[: idx + 1]
    return tokens


class NgramFastTokenDrafter:
    """Backoff n-gram drafter over generated FAST token ids.

    The drafter is deliberately simple and cheap. At a position, it tries the
    longest available token suffix, emits the most common next token, appends
    that prediction to the context, and repeats up to ``lookahead``.
    """

    def __init__(self, config: NgramDraftConfig) -> None:
        self.config = config
        self.tables: list[dict[tuple[int, ...], Counter[int]]] = [
            defaultdict(Counter) for _ in range(config.max_context + 1)
        ]

    def fit(self, traces: Iterable[PI0FastTraceRecord]) -> None:
        for trace in traces:
            tokens = [int(t) for t in trace.token_ids.tolist()]
            tokens = trim_at_stop_token(tokens, self.config.stop_token_ids)
            for pos, token in enumerate(tokens):
                for n in range(self.config.max_context + 1):
                    if pos < n:
                        continue
                    context = tuple(tokens[pos - n : pos])
                    self.tables[n][context][token] += 1

    def draft(self, prefix_tokens: list[int], lookahead: int | None = None) -> list[int]:
        draft: list[int] = []
        context_tokens = list(prefix_tokens)
        steps = self.config.lookahead if lookahead is None else lookahead
        for _ in range(steps):
            next_token = self._next_token(context_tokens)
            if next_token is None:
                break
            draft.append(next_token)
            context_tokens.append(next_token)
        return draft

    def _next_token(self, context_tokens: list[int]) -> int | None:
        max_n = min(self.config.max_context, len(context_tokens))
        for n in range(max_n, -1, -1):
            context = tuple(context_tokens[-n:]) if n else ()
            counts = self.tables[n].get(context)
            if not counts:
                continue
            token, count = counts.most_common(1)[0]
            if count >= self.config.min_count:
                return int(token)
        return None


def exact_prefix_acceptance(draft: list[int], target: list[int]) -> int:
    accepted = 0
    for draft_token, target_token in zip(draft, target):
        if int(draft_token) != int(target_token):
            break
        accepted += 1
    return accepted


def evaluate_ngram_drafter(
    drafter: NgramFastTokenDrafter,
    traces: Iterable[PI0FastTraceRecord],
    *,
    lookahead: int,
) -> dict:
    accepted_counts: list[int] = []
    drafted_counts: list[int] = []
    miss_count = 0
    per_task: dict[int, list[int]] = {}

    for trace in traces:
        tokens = [int(t) for t in trace.token_ids.tolist()]
        tokens = trim_at_stop_token(tokens, drafter.config.stop_token_ids)
        for pos in range(len(tokens) - 1):
            max_k = min(lookahead, len(tokens) - pos - 1)
            draft = drafter.draft(tokens[: pos + 1], lookahead=max_k)
            target = tokens[pos + 1 : pos + 1 + max_k]
            if not draft:
                miss_count += 1
            accepted = exact_prefix_acceptance(draft, target)
            accepted_counts.append(accepted)
            drafted_counts.append(len(draft))
            per_task.setdefault(trace.task_id, []).append(accepted)

    if not accepted_counts:
        raise ValueError("No token positions evaluated")
    total = len(accepted_counts)
    mean_accept = sum(accepted_counts) / total
    p_ge_1 = sum(a >= 1 for a in accepted_counts) / total
    p_ge_2 = sum(a >= 2 for a in accepted_counts) / total
    p_full = sum(a >= lookahead for a in accepted_counts) / total
    mean_drafted = sum(drafted_counts) / total
    metrics = {
        "positions": total,
        "lookahead": lookahead,
        "mean_spec_accept": mean_accept,
        "p_accept_ge_1": p_ge_1,
        "p_accept_ge_2": p_ge_2,
        "p_accept_full": p_full,
        "mean_drafted": mean_drafted,
        "draft_miss_rate": miss_count / total,
        "per_task": {},
    }
    for task_id, values in sorted(per_task.items()):
        task_total = len(values)
        metrics["per_task"][str(task_id)] = {
            "positions": task_total,
            "mean_spec_accept": sum(values) / task_total,
            "p_accept_ge_2": sum(a >= 2 for a in values) / task_total,
        }
    return metrics
