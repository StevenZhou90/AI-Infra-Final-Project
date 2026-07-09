"""Checkpoint-free FAST-token drafter for PI0-FAST pattern speculation.

Robot action token streams are much narrower than open-ended text: adjacent
actions often repeat gripper values, move with near-constant token velocity, or
cycle through the same action dimensions.  This drafter turns those priors into
cheap token proposals; the target PI0-FAST model must still verify every token
before it is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_SOURCE_PRIORITY = (
    "previous_chunk_position",
    "chunk_length_stop",
    "chunk_prefix_retrieval",
    "position_mode_histogram",
    "global_position_mode",
    "ngram_continuation",
    "second_order_action_extrapolation",
    "action_trend_regression",
    "action_prefix_lookup",
    "action_vector_suffix_lookup",
    "action_vector_transition",
    "action_repeat_vector",
    "action_token_neighborhood",
    "action_context_tree",
    "action_transition_histogram",
    "action_delta_histogram",
    "action_delta_ngram",
    "chunk_delta_template",
    "chunk_position_delta",
    "action_dimension_mode",
    "hold_action_token",
    "linear_action_extrapolation",
    "periodic_tail",
    "repeat_token",
)
SOURCE_PRIORITY_MODES = {
    "default": DEFAULT_SOURCE_PRIORITY,
    "previous_chunk_first": DEFAULT_SOURCE_PRIORITY,
    "lookup_first": (
        "chunk_length_stop",
        "ngram_continuation",
        "chunk_prefix_retrieval",
        "previous_chunk_position",
        "position_mode_histogram",
        "global_position_mode",
        "second_order_action_extrapolation",
        "action_trend_regression",
        "action_prefix_lookup",
        "action_vector_suffix_lookup",
        "action_vector_transition",
        "action_repeat_vector",
        "action_token_neighborhood",
        "action_context_tree",
        "action_transition_histogram",
        "action_delta_histogram",
        "action_delta_ngram",
        "chunk_delta_template",
        "chunk_position_delta",
        "action_dimension_mode",
        "hold_action_token",
        "linear_action_extrapolation",
        "periodic_tail",
        "repeat_token",
    ),
    "smooth_first": (
        "chunk_length_stop",
        "second_order_action_extrapolation",
        "action_trend_regression",
        "action_prefix_lookup",
        "action_vector_suffix_lookup",
        "action_vector_transition",
        "action_repeat_vector",
        "action_token_neighborhood",
        "action_context_tree",
        "action_transition_histogram",
        "action_delta_histogram",
        "action_delta_ngram",
        "chunk_delta_template",
        "chunk_position_delta",
        "action_dimension_mode",
        "linear_action_extrapolation",
        "chunk_prefix_retrieval",
        "hold_action_token",
        "periodic_tail",
        "repeat_token",
        "ngram_continuation",
        "position_mode_histogram",
        "global_position_mode",
        "previous_chunk_position",
    ),
    "repeat_first": (
        "chunk_length_stop",
        "periodic_tail",
        "repeat_token",
        "linear_action_extrapolation",
        "action_repeat_vector",
        "action_trend_regression",
        "action_prefix_lookup",
        "action_vector_suffix_lookup",
        "action_vector_transition",
        "action_token_neighborhood",
        "action_context_tree",
        "action_transition_histogram",
        "action_delta_histogram",
        "action_delta_ngram",
        "chunk_delta_template",
        "chunk_position_delta",
        "action_dimension_mode",
        "chunk_prefix_retrieval",
        "hold_action_token",
        "second_order_action_extrapolation",
        "ngram_continuation",
        "position_mode_histogram",
        "global_position_mode",
        "previous_chunk_position",
    ),
}
VALID_SOURCE_NAMES = frozenset(DEFAULT_SOURCE_PRIORITY)


def parse_source_priority(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_SOURCE_PRIORITY
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return DEFAULT_SOURCE_PRIORITY
        mode = SOURCE_PRIORITY_MODES.get(normalized)
        if mode is not None:
            return mode
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in value if str(part).strip()]
    if not parts:
        return DEFAULT_SOURCE_PRIORITY
    unknown = [part for part in parts if part not in VALID_SOURCE_NAMES]
    if unknown:
        raise ValueError(
            "Unknown pattern source priority values: "
            + ", ".join(unknown)
            + ". Valid values: "
            + ", ".join(sorted(VALID_SOURCE_NAMES | set(SOURCE_PRIORITY_MODES)))
        )
    return tuple(parts)


@dataclass(frozen=True)
class PatternDraftConfig:
    lookahead: int = 8
    action_dim: int = 7
    max_period: int = 16
    min_period_repeats: int = 2
    repeat_token_min_run: int = 3
    enable_linear_action_extrapolation: bool = True
    enable_second_order_action_extrapolation: bool = False
    second_order_max_accel: int | None = 8
    enable_action_trend_regression: bool = False
    action_trend_history: int = 4
    action_trend_top_k: int = 3
    action_trend_max_abs: int | None = 12
    enable_action_prefix_lookup: bool = False
    action_prefix_history_size: int = 4
    action_prefix_top_k: int = 3
    action_prefix_min_prefix: int = 1
    action_prefix_max_mismatches: int = 0
    enable_action_vector_suffix_lookup: bool = False
    action_vector_suffix_history_size: int = 4
    action_vector_suffix_top_k: int = 3
    action_vector_suffix_min_prefix: int = 1
    action_vector_suffix_min_count: int = 1
    action_vector_suffix_max_prefix_delta: int = 0
    enable_action_vector_transition: bool = False
    action_vector_transition_history_size: int = 4
    action_vector_transition_top_k: int = 3
    action_vector_transition_min_count: int = 1
    action_vector_transition_max_prev_delta: int = 0
    action_vector_transition_max_prefix_delta: int = 0
    enable_action_repeat_vector: bool = False
    action_repeat_min_repeats: int = 2
    action_repeat_max_delta: int = 0
    enable_action_token_neighborhood: bool = False
    action_token_neighborhood_radius: int = 1
    action_token_neighborhood_top_k: int = 3
    enable_action_context_tree: bool = False
    action_context_tree_history_size: int = 4
    action_context_tree_max_context: int = 3
    action_context_tree_top_k: int = 3
    action_context_tree_min_count: int = 1
    enable_action_transition_histogram: bool = False
    action_transition_history_size: int = 4
    action_transition_top_k: int = 3
    action_transition_min_count: int = 1
    enable_action_delta_histogram: bool = False
    action_delta_history: int = 8
    action_delta_top_k: int = 3
    action_delta_min_count: int = 1
    action_delta_max_abs: int | None = 12
    enable_action_delta_ngram: bool = False
    action_delta_ngram_history_size: int = 4
    action_delta_ngram_min_context: int = 1
    action_delta_ngram_max_context: int = 4
    action_delta_ngram_top_k: int = 3
    action_delta_ngram_min_count: int = 1
    action_delta_ngram_max_abs: int | None = 12
    enable_chunk_delta_template: bool = False
    chunk_delta_template_history_size: int = 4
    chunk_delta_template_top_k: int = 3
    chunk_delta_template_min_prefix_deltas: int = 1
    chunk_delta_template_max_delta_mismatch: int | None = 0
    chunk_delta_template_max_abs: int | None = 12
    enable_chunk_position_delta: bool = False
    chunk_position_delta_history_size: int = 4
    chunk_position_delta_top_k: int = 3
    chunk_position_delta_min_count: int = 1
    chunk_position_delta_max_abs: int | None = 12
    enable_action_dimension_mode: bool = False
    action_dimension_mode_history_size: int = 4
    action_dimension_mode_top_k: int = 3
    action_dimension_mode_min_count: int = 2
    enable_hold_action_token: bool = False
    enable_previous_chunk_position: bool = False
    previous_chunk_history_size: int = 1
    enable_chunk_prefix_retrieval: bool = False
    chunk_prefix_history_size: int = 4
    chunk_prefix_top_k: int = 3
    chunk_prefix_min_matches: int = 2
    chunk_prefix_max_mismatches: int = 1
    enable_chunk_length_stop: bool = False
    chunk_length_stop_history_size: int = 4
    chunk_length_stop_min_count: int = 2
    enable_position_mode_histogram: bool = False
    position_mode_history_size: int = 4
    position_mode_top_k: int = 3
    position_mode_min_count: int = 2
    enable_global_position_mode: bool = False
    global_position_history_size: int = 16
    global_position_top_k: int = 3
    global_position_min_count: int = 3
    enable_ngram_continuation: bool = False
    ngram_min_context: int = 3
    ngram_max_context: int = 16
    ngram_history_size: int = 4
    min_source_agreement: int = 1
    source_priority: tuple[str, ...] = DEFAULT_SOURCE_PRIORITY
    enable_source_cooldown: bool = False
    source_cooldown_after: int = 1
    source_cooldown_steps: int = 1
    enable_source_acceptance_bias: bool = False
    source_acceptance_bias_history_size: int = 32
    source_acceptance_bias_min_observations: int = 2
    vocab_size: int | None = None
    stop_token_ids: tuple[int, ...] = ()


class PatternFastTokenDrafter:
    """Draft future FAST tokens from local robot-action regularity."""

    def __init__(self, config: PatternDraftConfig) -> None:
        self.config = config
        self._stop = {int(token) for token in config.stop_token_ids}
        self._previous_chunks: list[list[int]] = []
        self._last_draft_sources: list[str] = []
        self._last_many_sources: list[list[str]] = []
        self._priority = parse_source_priority(config.source_priority)
        self._source_reject_streaks: dict[str, int] = {}
        self._source_cooldowns: dict[str, int] = {}
        self._source_cooldown_events = 0
        self._source_cooldown_skipped_sources = 0
        self._source_acceptance_history: dict[str, list[int]] = {}
        self._source_acceptance_bias_events = 0
        self._source_acceptance_bias_reorders = 0

    def reset_history(self) -> None:
        self._previous_chunks.clear()
        self.reset_source_feedback()

    def reset_source_feedback(self) -> None:
        self._source_reject_streaks.clear()
        self._source_cooldowns.clear()
        self._source_cooldown_events = 0
        self._source_cooldown_skipped_sources = 0
        self._source_acceptance_history.clear()
        self._source_acceptance_bias_events = 0
        self._source_acceptance_bias_reorders = 0

    def source_cooldown_stats(self) -> dict[str, int]:
        return {
            "source_cooldown_events": int(self._source_cooldown_events),
            "source_cooldown_skipped_sources": int(self._source_cooldown_skipped_sources),
            "source_acceptance_bias_events": int(self._source_acceptance_bias_events),
            "source_acceptance_bias_reorders": int(self._source_acceptance_bias_reorders),
        }

    def record_source_miss(self) -> None:
        self._age_source_cooldowns(set())

    def record_source_feedback(self, sources: Iterable[str], accepted: int) -> None:
        source_list = [str(source) for source in sources]
        accepted = max(0, min(int(accepted), len(source_list)))
        if self.config.enable_source_acceptance_bias:
            self._record_source_acceptance_feedback(source_list, accepted)
        if not self.config.enable_source_cooldown:
            return
        old_cooling = {source for source, remaining in self._source_cooldowns.items() if int(remaining) > 0}
        for source in source_list[:accepted]:
            self._source_reject_streaks[source] = 0
            self._source_cooldowns.pop(source, None)
        newly_cooled: set[str] = set()
        if accepted < len(source_list):
            rejected_source = source_list[accepted]
            streak = self._source_reject_streaks.get(rejected_source, 0) + 1
            threshold = max(int(self.config.source_cooldown_after), 1)
            if streak >= threshold:
                self._source_cooldowns[rejected_source] = max(int(self.config.source_cooldown_steps), 1)
                self._source_reject_streaks[rejected_source] = 0
                self._source_cooldown_events += 1
                newly_cooled.add(rejected_source)
            else:
                self._source_reject_streaks[rejected_source] = streak
        self._age_source_cooldowns(newly_cooled, only=old_cooling)

    def _record_source_acceptance_feedback(self, source_list: list[str], accepted: int) -> None:
        if not source_list:
            return
        history_size = max(int(self.config.source_acceptance_bias_history_size), 1)
        observed = len(source_list) if accepted >= len(source_list) else accepted + 1
        for idx, source in enumerate(source_list[:observed]):
            if not source:
                continue
            history = self._source_acceptance_history.setdefault(source, [])
            history.append(1 if idx < accepted else 0)
            self._source_acceptance_bias_events += 1
            if len(history) > history_size:
                del history[: len(history) - history_size]

    def _source_acceptance_score(self, source: str) -> float:
        history = self._source_acceptance_history.get(source, [])
        min_observations = max(int(self.config.source_acceptance_bias_min_observations), 1)
        if len(history) < min_observations:
            return 0.5
        return float(sum(history)) / float(len(history))

    def _age_source_cooldowns(self, newly_cooled: set[str], *, only: set[str] | None = None) -> None:
        if not self.config.enable_source_cooldown:
            return
        candidates = set(self._source_cooldowns) if only is None else set(only)
        for source in candidates:
            if source in newly_cooled:
                continue
            remaining = self._source_cooldowns.get(source)
            if remaining is None:
                continue
            remaining = int(remaining) - 1
            if remaining <= 0:
                self._source_cooldowns.pop(source, None)
            else:
                self._source_cooldowns[source] = remaining

    def observe(self, token_ids: Iterable[int]) -> None:
        if (
            not self.config.enable_previous_chunk_position
            and not self.config.enable_chunk_prefix_retrieval
            and not self.config.enable_chunk_length_stop
            and not self.config.enable_action_token_neighborhood
            and not self.config.enable_position_mode_histogram
            and not self.config.enable_global_position_mode
            and not self.config.enable_ngram_continuation
            and not self.config.enable_action_context_tree
            and not self.config.enable_action_transition_histogram
            and not self.config.enable_action_delta_histogram
            and not self.config.enable_action_delta_ngram
            and not self.config.enable_chunk_delta_template
            and not self.config.enable_chunk_position_delta
            and not self.config.enable_action_dimension_mode
            and not self.config.enable_action_prefix_lookup
            and not self.config.enable_action_vector_suffix_lookup
            and not self.config.enable_action_vector_transition
            and not self.config.enable_action_repeat_vector
        ):
            return
        tokens = [int(token) for token in token_ids]
        if not tokens:
            return
        history_sizes: list[int] = []
        if self.config.enable_previous_chunk_position:
            history_sizes.append(int(self.config.previous_chunk_history_size))
        if self.config.enable_action_token_neighborhood:
            history_sizes.append(int(self.config.previous_chunk_history_size))
        if self.config.enable_chunk_prefix_retrieval:
            history_sizes.append(int(self.config.chunk_prefix_history_size))
        if self.config.enable_chunk_length_stop:
            history_sizes.append(int(self.config.chunk_length_stop_history_size))
        if self.config.enable_position_mode_histogram:
            history_sizes.append(int(self.config.position_mode_history_size))
        if self.config.enable_global_position_mode:
            history_sizes.append(int(self.config.global_position_history_size))
        if self.config.enable_ngram_continuation:
            history_sizes.append(int(self.config.ngram_history_size))
        if self.config.enable_action_context_tree:
            history_sizes.append(int(self.config.action_context_tree_history_size))
        if self.config.enable_action_transition_histogram:
            history_sizes.append(int(self.config.action_transition_history_size))
        if self.config.enable_action_delta_histogram:
            history_sizes.append(int(self.config.action_delta_history))
        if self.config.enable_action_delta_ngram:
            history_sizes.append(int(self.config.action_delta_ngram_history_size))
        if self.config.enable_action_prefix_lookup:
            history_sizes.append(int(self.config.action_prefix_history_size))
        if self.config.enable_action_vector_suffix_lookup:
            history_sizes.append(int(self.config.action_vector_suffix_history_size))
        if self.config.enable_action_vector_transition:
            history_sizes.append(int(self.config.action_vector_transition_history_size))
        if self.config.enable_chunk_delta_template:
            history_sizes.append(int(self.config.chunk_delta_template_history_size))
        if self.config.enable_chunk_position_delta:
            history_sizes.append(int(self.config.chunk_position_delta_history_size))
        if self.config.enable_action_dimension_mode:
            history_sizes.append(int(self.config.action_dimension_mode_history_size))
        history_size = max([0, *history_sizes])
        if history_size <= 0:
            return
        self._previous_chunks.insert(0, tokens)
        del self._previous_chunks[history_size:]

    def draft(self, prefix_tokens: list[int], lookahead: int | None = None) -> list[int]:
        sim = [int(token) for token in prefix_tokens]
        self._last_draft_sources = []
        self._last_many_sources = []
        if sim and sim[-1] in self._stop:
            return []

        steps = int(self.config.lookahead if lookahead is None else lookahead)
        out: list[int] = []
        sources: list[str] = []
        for _ in range(max(steps, 0)):
            token, source = self._next_token_with_source(sim)
            if token is None:
                break
            out.append(token)
            sources.append(source)
            sim.append(token)
            if token in self._stop:
                break
        self._last_draft_sources = sources
        return out

    def _next_token(self, tokens: list[int]) -> int | None:
        token, _source = self._next_token_with_source(tokens)
        return token

    def _next_token_with_source(self, tokens: list[int]) -> tuple[int | None, str]:
        agreement = self._source_agreement_candidates(tokens)
        if agreement:
            return agreement[0], "source_agreement"
        for source in self._source_priority():
            token = self._token_from_source(source, tokens)
            if token is not None:
                return token, source
        return None, ""

    def last_draft_sources(self) -> list[str]:
        return list(self._last_draft_sources)

    def last_many_sources(self) -> list[list[str]]:
        return [list(row) for row in self._last_many_sources]

    def draft_many(
        self,
        prefix_tokens: list[int],
        lookahead: int | None = None,
        *,
        max_candidates: int = 4,
        branch_width: int = 4,
    ) -> list[list[int]]:
        """Return a small beam of pattern drafts for exact tree verification."""

        sim = [int(token) for token in prefix_tokens]
        if sim and sim[-1] in self._stop:
            return []
        steps = int(self.config.lookahead if lookahead is None else lookahead)
        if steps <= 0 or max_candidates <= 0:
            return []

        self._last_draft_sources = []
        self._last_many_sources = []
        beams: list[tuple[list[int], list[int], list[str], int]] = [(sim, [], [], 0)]
        for _ in range(steps):
            expanded: list[tuple[list[int], list[int], list[str], int]] = []
            for state, draft, sources, score in beams:
                next_tokens = self._candidate_next_tokens(state, branch_width=max(int(branch_width), 1))
                if not next_tokens:
                    if draft:
                        expanded.append((state, draft, sources, score + 100))
                    continue
                for token, token_score, token_source in next_tokens:
                    next_state = [*state, int(token)]
                    next_draft = [*draft, int(token)]
                    next_sources = [*sources, token_source]
                    expanded.append((next_state, next_draft, next_sources, score + int(token_score)))
            if not expanded:
                break
            expanded.sort(key=lambda item: (item[3], len(item[1]), item[1]))
            beams = expanded[: max(int(max_candidates), 1)]
            if all(draft and draft[-1] in self._stop for _state, draft, _sources, _score in beams):
                break

        seen: set[tuple[int, ...]] = set()
        out: list[list[int]] = []
        out_sources: list[list[str]] = []
        for _state, draft, sources, _score in beams:
            if not draft:
                continue
            key = tuple(int(token) for token in draft)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(key))
            out_sources.append(list(sources))
            if len(out) >= int(max_candidates):
                break
        self._last_many_sources = out_sources
        return out

    def _candidate_next_tokens(self, tokens: list[int], *, branch_width: int) -> list[tuple[int, int, str]]:
        candidates: list[tuple[int, int, str]] = []

        def add(token: int | None, score: int, source: str) -> None:
            if token is None:
                return
            token = int(token)
            if all(existing != token for existing, _score, _source in candidates):
                candidates.append((token, score, source))

        for idx, token in enumerate(self._source_agreement_candidates(tokens)):
            add(token, -100 + idx, "source_agreement")
        for source_rank, source in enumerate(self._source_priority()):
            for idx, token in enumerate(self._candidate_tokens_from_source(source, tokens)):
                add(token, source_rank * 10 + idx, source)
        if not tokens:
            return candidates[: max(int(branch_width), 1)]
        return candidates[: max(int(branch_width), 1)]

    def _source_priority(self) -> tuple[str, ...]:
        active: list[str] = []
        for source in self._priority:
            if self.config.enable_source_cooldown and int(self._source_cooldowns.get(source, 0)) > 0:
                self._source_cooldown_skipped_sources += 1
                continue
            active.append(source)
        if self.config.enable_source_acceptance_bias and len(active) > 1:
            original = list(active)
            active.sort(key=lambda source: (-self._source_acceptance_score(source), original.index(source)))
            if active != original:
                self._source_acceptance_bias_reorders += 1
        return tuple(active)

    def _candidate_tokens_from_source(self, source: str, tokens: list[int]) -> list[int]:
        if source == "previous_chunk_position":
            return self._previous_chunk_position_candidates(tokens)
        if source == "chunk_prefix_retrieval":
            return self._chunk_prefix_retrieval_candidates(tokens)
        if source == "chunk_length_stop":
            return self._chunk_length_stop_candidates(tokens)
        if source == "position_mode_histogram":
            return self._position_mode_histogram_candidates(tokens)
        if source == "global_position_mode":
            return self._global_position_mode_candidates(tokens)
        if source == "ngram_continuation":
            return self._ngram_continuation_candidates(tokens)
        if source == "action_trend_regression":
            return self._action_trend_regression_candidates(tokens)
        if source == "action_prefix_lookup":
            return self._action_prefix_lookup_candidates(tokens)
        if source == "action_vector_suffix_lookup":
            return self._action_vector_suffix_lookup_candidates(tokens)
        if source == "action_vector_transition":
            return self._action_vector_transition_candidates(tokens)
        if source == "action_repeat_vector":
            return self._action_repeat_vector_candidates(tokens)
        if source == "action_token_neighborhood":
            return self._action_token_neighborhood_candidates(tokens)
        if source == "action_context_tree":
            return self._action_context_tree_candidates(tokens)
        if source == "action_transition_histogram":
            return self._action_transition_histogram_candidates(tokens)
        if source == "action_delta_histogram":
            return self._action_delta_histogram_candidates(tokens)
        if source == "action_delta_ngram":
            return self._action_delta_ngram_candidates(tokens)
        if source == "chunk_delta_template":
            return self._chunk_delta_template_candidates(tokens)
        if source == "chunk_position_delta":
            return self._chunk_position_delta_candidates(tokens)
        if source == "action_dimension_mode":
            return self._action_dimension_mode_candidates(tokens)
        token = self._token_from_source(source, tokens)
        return [] if token is None else [token]

    def _token_from_source(self, source: str, tokens: list[int]) -> int | None:
        if source == "previous_chunk_position":
            return self._previous_chunk_position_token(tokens)
        if source == "chunk_prefix_retrieval":
            return self._chunk_prefix_retrieval_token(tokens)
        if source == "chunk_length_stop":
            return self._chunk_length_stop_token(tokens)
        if source == "position_mode_histogram":
            return self._position_mode_histogram_token(tokens)
        if source == "global_position_mode":
            return self._global_position_mode_token(tokens)
        if source == "ngram_continuation":
            return self._ngram_continuation_token(tokens)
        if source == "second_order_action_extrapolation":
            return self._second_order_action_token(tokens) if self.config.enable_second_order_action_extrapolation else None
        if source == "action_trend_regression":
            return self._action_trend_regression_token(tokens)
        if source == "action_prefix_lookup":
            return self._action_prefix_lookup_token(tokens)
        if source == "action_vector_suffix_lookup":
            return self._action_vector_suffix_lookup_token(tokens)
        if source == "action_vector_transition":
            return self._action_vector_transition_token(tokens)
        if source == "action_repeat_vector":
            return self._action_repeat_vector_token(tokens)
        if source == "action_token_neighborhood":
            return self._action_token_neighborhood_token(tokens)
        if source == "action_context_tree":
            return self._action_context_tree_token(tokens)
        if source == "action_transition_histogram":
            return self._action_transition_histogram_token(tokens)
        if source == "action_delta_histogram":
            return self._action_delta_histogram_token(tokens)
        if source == "action_delta_ngram":
            return self._action_delta_ngram_token(tokens)
        if source == "chunk_delta_template":
            return self._chunk_delta_template_token(tokens)
        if source == "chunk_position_delta":
            return self._chunk_position_delta_token(tokens)
        if source == "action_dimension_mode":
            return self._action_dimension_mode_token(tokens)
        if source == "linear_action_extrapolation":
            return self._linear_action_token(tokens) if self.config.enable_linear_action_extrapolation else None
        if source == "periodic_tail":
            return self._periodic_token(tokens)
        if source == "repeat_token":
            return self._repeat_token(tokens) if tokens else None
        if source == "hold_action_token":
            return self._hold_action_token(tokens) if self.config.enable_hold_action_token else None
        return None

    def _source_agreement_candidates(self, tokens: list[int]) -> list[int]:
        min_sources = max(int(self.config.min_source_agreement), 1)
        if min_sources <= 1:
            return []
        token_votes: dict[int, tuple[int, int]] = {}
        for source_rank, source in enumerate(self._source_priority()):
            source_tokens: set[int] = set()
            for token in self._candidate_tokens_from_source(source, tokens):
                source_tokens.add(int(token))
            for token in source_tokens:
                count, best_rank = token_votes.get(token, (0, source_rank))
                token_votes[token] = (count + 1, min(best_rank, source_rank))
        agreed = [
            (token, count, best_rank)
            for token, (count, best_rank) in token_votes.items()
            if count >= min_sources
        ]
        agreed.sort(key=lambda item: (-item[1], item[2], item[0]))
        return [token for token, _count, _rank in agreed]

    def _previous_chunk_position_candidates(self, tokens: list[int], *, require_enabled: bool = True) -> list[int]:
        if require_enabled and not self.config.enable_previous_chunk_position:
            return []
        pos = len(tokens)
        candidates: list[int] = []
        for chunk in self._previous_chunks:
            if pos >= len(chunk):
                continue
            if any(int(chunk[idx]) != int(tokens[idx]) for idx in range(pos)):
                continue
            token = int(chunk[pos])
            if any(existing == token for existing in candidates):
                continue
            candidates.append(token)
        return candidates

    def _previous_chunk_position_token(self, tokens: list[int]) -> int | None:
        candidates = self._previous_chunk_position_candidates(tokens)
        return candidates[0] if candidates else None

    def _chunk_prefix_retrieval_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_chunk_prefix_retrieval:
            return []
        pos = len(tokens)
        if pos <= 0:
            return []
        min_matches = max(int(self.config.chunk_prefix_min_matches), 0)
        max_mismatches = max(int(self.config.chunk_prefix_max_mismatches), 0)
        top_k = max(int(self.config.chunk_prefix_top_k), 1)
        best: dict[int, tuple[int, int, int, int]] = {}
        for chunk_idx, chunk in enumerate(self._previous_chunks):
            if pos >= len(chunk):
                continue
            matches = 0
            mismatches = 0
            trailing_matches = 0
            for idx in range(pos):
                if int(chunk[idx]) == int(tokens[idx]):
                    matches += 1
                else:
                    mismatches += 1
            if mismatches > max_mismatches:
                continue
            if mismatches > 0 and matches < min_matches:
                continue
            for idx in range(pos - 1, -1, -1):
                if int(chunk[idx]) != int(tokens[idx]):
                    break
                trailing_matches += 1
            token = int(chunk[pos])
            if token in self._stop:
                continue
            score = (mismatches, -trailing_matches, chunk_idx, token)
            if token not in best or score < best[token]:
                best[token] = score
        candidates = list(best)
        candidates.sort(key=lambda token: best[token])
        return candidates[:top_k]

    def _chunk_prefix_retrieval_token(self, tokens: list[int]) -> int | None:
        candidates = self._chunk_prefix_retrieval_candidates(tokens)
        return candidates[0] if candidates else None

    def _chunk_length_stop_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_chunk_length_stop or not self._stop:
            return []
        if not tokens or int(tokens[-1]) in self._stop:
            return []
        pos = len(tokens)
        min_count = max(int(self.config.chunk_length_stop_min_count), 1)
        history_size = max(int(self.config.chunk_length_stop_history_size), 1)
        counts: dict[int, int] = {}
        best_recency: dict[int, int] = {}
        for chunk_idx, chunk in enumerate(self._previous_chunks[:history_size]):
            if pos >= len(chunk):
                continue
            first_stop_idx: int | None = None
            for idx, token in enumerate(chunk):
                if int(token) in self._stop:
                    first_stop_idx = idx
                    break
            if first_stop_idx != pos:
                continue
            token = int(chunk[pos])
            if token not in self._stop:
                continue
            counts[token] = counts.get(token, 0) + 1
            best_recency[token] = min(best_recency.get(token, chunk_idx), chunk_idx)
        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(key=lambda token: (-counts[token], best_recency[token], token))
        return candidates

    def _chunk_length_stop_token(self, tokens: list[int]) -> int | None:
        candidates = self._chunk_length_stop_candidates(tokens)
        return candidates[0] if candidates else None

    def _position_mode_histogram_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_position_mode_histogram:
            return []
        pos = len(tokens)
        min_count = max(int(self.config.position_mode_min_count), 1)
        top_k = max(int(self.config.position_mode_top_k), 1)
        counts: dict[int, int] = {}
        best_recency: dict[int, int] = {}
        for chunk_idx, chunk in enumerate(self._previous_chunks):
            if pos >= len(chunk):
                continue
            if any(int(chunk[idx]) != int(tokens[idx]) for idx in range(pos)):
                continue
            token = int(chunk[pos])
            if token in self._stop:
                continue
            counts[token] = counts.get(token, 0) + 1
            best_recency[token] = min(best_recency.get(token, chunk_idx), chunk_idx)
        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(key=lambda token: (-counts[token], best_recency[token], int(token)))
        return candidates[:top_k]

    def _position_mode_histogram_token(self, tokens: list[int]) -> int | None:
        candidates = self._position_mode_histogram_candidates(tokens)
        return candidates[0] if candidates else None

    def _global_position_mode_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_global_position_mode:
            return []
        pos = len(tokens)
        min_count = max(int(self.config.global_position_min_count), 1)
        top_k = max(int(self.config.global_position_top_k), 1)
        history_size = max(int(self.config.global_position_history_size), 1)
        counts: dict[int, int] = {}
        best_recency: dict[int, int] = {}
        for chunk_idx, chunk in enumerate(self._previous_chunks[:history_size]):
            if pos >= len(chunk):
                continue
            token = int(chunk[pos])
            if token in self._stop:
                continue
            counts[token] = counts.get(token, 0) + 1
            best_recency[token] = min(best_recency.get(token, chunk_idx), chunk_idx)
        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(key=lambda token: (-counts[token], best_recency[token], int(token)))
        return candidates[:top_k]

    def _global_position_mode_token(self, tokens: list[int]) -> int | None:
        candidates = self._global_position_mode_candidates(tokens)
        return candidates[0] if candidates else None

    def _ngram_continuation_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_ngram_continuation or not tokens:
            return []
        min_context = max(int(self.config.ngram_min_context), 1)
        max_context = min(max(int(self.config.ngram_max_context), min_context), len(tokens))
        if max_context < min_context:
            return []

        def add_candidate(candidates: list[int], token: int) -> None:
            token = int(token)
            if any(existing == token for existing in candidates):
                return
            candidates.append(token)

        def scan_sequence(sequence: list[int], suffix: list[int], candidates: list[int]) -> None:
            suffix_len = len(suffix)
            if len(sequence) <= suffix_len:
                return
            last_start = len(sequence) - suffix_len - 1
            for start in range(last_start, -1, -1):
                if sequence[start : start + suffix_len] == suffix:
                    add_candidate(candidates, sequence[start + suffix_len])

        for context_len in range(max_context, min_context - 1, -1):
            suffix = tokens[-context_len:]
            candidates: list[int] = []
            scan_sequence(tokens, suffix, candidates)
            for chunk in self._previous_chunks:
                scan_sequence(chunk, suffix, candidates)
            if candidates:
                return candidates
        return []

    def _ngram_continuation_token(self, tokens: list[int]) -> int | None:
        candidates = self._ngram_continuation_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_trend_regression_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_trend_regression:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0 or len(tokens) < 2 * action_dim:
            return []
        dim = len(tokens) % action_dim
        values: list[int] = []
        idx = dim
        while idx < len(tokens):
            token = int(tokens[idx])
            if token in self._stop:
                values = []
            else:
                values.append(token)
            idx += action_dim
        history = max(int(self.config.action_trend_history), 2)
        values = values[-history:]
        if len(values) < 2:
            return []
        n = len(values)
        mean_x = (n - 1) / 2.0
        mean_y = sum(values) / float(n)
        denom = sum((idx - mean_x) ** 2 for idx in range(n))
        if denom <= 0.0:
            return []
        slope = sum((idx - mean_x) * (float(value) - mean_y) for idx, value in enumerate(values)) / denom
        predicted = mean_y + slope * (n - mean_x)
        center = int(round(predicted))
        previous = int(values[-1])
        max_abs = self.config.action_trend_max_abs
        top_k = max(int(self.config.action_trend_top_k), 1)

        offsets = [0]
        delta = 1
        while len(offsets) < top_k * 2 + 1:
            offsets.extend([-delta, delta])
            delta += 1

        candidates: list[int] = []
        for offset in offsets:
            token = self._clip_token(center + int(offset), tokens)
            if token in self._stop:
                continue
            if max_abs is not None and abs(int(token) - previous) > int(max_abs):
                continue
            if any(existing == token for existing in candidates):
                continue
            candidates.append(token)
            if len(candidates) >= top_k:
                break
        return candidates

    def _action_trend_regression_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_trend_regression_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_prefix_lookup_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_prefix_lookup:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0:
            return []
        dim = len(tokens) % action_dim
        min_prefix = max(int(self.config.action_prefix_min_prefix), 1)
        if dim < min_prefix:
            return []
        prefix = [int(token) for token in tokens[-dim:]]
        if any(token in self._stop for token in prefix):
            return []
        max_mismatches = max(int(self.config.action_prefix_max_mismatches), 0)
        top_k = max(int(self.config.action_prefix_top_k), 1)
        history_size = max(int(self.config.action_prefix_history_size), 0)
        previous_same_dim = int(tokens[-action_dim]) if len(tokens) >= action_dim else None
        counts: dict[int, int] = {}
        best_score: dict[int, tuple[int, int, int, int, int]] = {}

        def scan_sequence(sequence: list[int], recency_bias: int) -> None:
            if len(sequence) <= dim:
                return
            max_start = len(sequence) - dim - 1
            if max_start < 0:
                return
            start = (max_start // action_dim) * action_dim
            local_rank = 0
            while start >= 0:
                mismatches = 0
                valid = True
                for offset, expected in enumerate(prefix):
                    actual = int(sequence[start + offset])
                    if actual in self._stop:
                        valid = False
                        break
                    if actual != expected:
                        mismatches += 1
                        if mismatches > max_mismatches:
                            valid = False
                            break
                token = int(sequence[start + dim])
                if valid and token not in self._stop:
                    matches = dim - mismatches
                    if matches >= min_prefix:
                        counts[token] = counts.get(token, 0) + 1
                        distance = abs(token - previous_same_dim) if previous_same_dim is not None else 0
                        score = (mismatches, -matches, recency_bias + local_rank, distance, token)
                        if token not in best_score or score < best_score[token]:
                            best_score[token] = score
                start -= action_dim
                local_rank += 1

        scan_sequence(tokens, 0)
        recency_bias = len(tokens) + action_dim
        for chunk_idx, chunk in enumerate(self._previous_chunks[:history_size]):
            scan_sequence(chunk, recency_bias + chunk_idx * (len(chunk) + action_dim))
        candidates = list(counts)
        candidates.sort(key=lambda token: (-counts[token], best_score[token]))
        return candidates[:top_k]

    def _action_prefix_lookup_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_prefix_lookup_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_vector_suffix_lookup_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_vector_suffix_lookup:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0:
            return []
        dim = len(tokens) % action_dim
        min_prefix = max(int(self.config.action_vector_suffix_min_prefix), 1)
        if dim < min_prefix or dim <= 0:
            return []
        prefix = [int(token) for token in tokens[-dim:]]
        if any(token in self._stop for token in prefix):
            return []
        top_k = max(int(self.config.action_vector_suffix_top_k), 1)
        min_count = max(int(self.config.action_vector_suffix_min_count), 1)
        max_prefix_delta = int(self.config.action_vector_suffix_max_prefix_delta)
        history_size = max(int(self.config.action_vector_suffix_history_size), 0)
        counts: dict[int, int] = {}
        best_score: dict[int, tuple[int, int, int, int]] = {}

        def prefix_distance(candidate_prefix: list[int]) -> int | None:
            if len(candidate_prefix) != len(prefix):
                return None
            total = 0
            for actual, expected in zip(candidate_prefix, prefix):
                if int(actual) in self._stop:
                    return None
                delta = abs(int(actual) - int(expected))
                if max_prefix_delta >= 0 and delta > max_prefix_delta:
                    return None
                total += delta
            return total

        def scan_sequence(sequence: list[int], recency_bias: int) -> None:
            if len(sequence) <= dim:
                return
            last_start = ((len(sequence) - dim - 1) // action_dim) * action_dim
            local_rank = 0
            for start in range(last_start, -1, -action_dim):
                candidate_prefix = [int(token) for token in sequence[start : start + dim]]
                distance = prefix_distance(candidate_prefix)
                if distance is None:
                    local_rank += 1
                    continue
                token = int(sequence[start + dim])
                if token in self._stop:
                    local_rank += 1
                    continue
                counts[token] = counts.get(token, 0) + 1
                score = (distance, recency_bias + local_rank, abs(token - prefix[-1]), token)
                if token not in best_score or score < best_score[token]:
                    best_score[token] = score
                local_rank += 1

        scan_sequence(tokens, 0)
        recency_bias = len(tokens) + action_dim
        for chunk_idx, chunk in enumerate(self._previous_chunks[:history_size]):
            scan_sequence(chunk, recency_bias + chunk_idx * (len(chunk) + action_dim))
        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(key=lambda token: (-counts[token], best_score[token]))
        return candidates[:top_k]

    def _action_vector_suffix_lookup_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_vector_suffix_lookup_candidates(tokens)
        return candidates[0] if candidates else None

    @staticmethod
    def _vector_delta_ok(left: list[int], right: list[int], max_delta: int) -> bool:
        if len(left) != len(right):
            return False
        if int(max_delta) < 0:
            return True
        return all(abs(int(a) - int(b)) <= int(max_delta) for a, b in zip(left, right))

    def _action_vector_transition_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_vector_transition:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0:
            return []
        dim = len(tokens) % action_dim
        previous_start = len(tokens) - dim - action_dim
        if previous_start < 0:
            return []
        previous_vector = [int(token) for token in tokens[previous_start : previous_start + action_dim]]
        if len(previous_vector) != action_dim or any(token in self._stop for token in previous_vector):
            return []
        current_prefix = [int(token) for token in tokens[-dim:]] if dim else []
        if any(token in self._stop for token in current_prefix):
            return []

        top_k = max(int(self.config.action_vector_transition_top_k), 1)
        min_count = max(int(self.config.action_vector_transition_min_count), 1)
        history_size = max(int(self.config.action_vector_transition_history_size), 0)
        max_prev_delta = int(self.config.action_vector_transition_max_prev_delta)
        max_prefix_delta = int(self.config.action_vector_transition_max_prefix_delta)
        counts: dict[int, int] = {}
        best_score: dict[int, tuple[int, int, int, int]] = {}

        def vector_distance(left: list[int], right: list[int], max_delta: int) -> int:
            if int(max_delta) < 0:
                return 0
            return sum(abs(int(a) - int(b)) for a, b in zip(left, right))

        def scan_sequence(sequence: list[int], recency_bias: int) -> None:
            if len(sequence) < action_dim * 2:
                return
            last_start = ((len(sequence) - action_dim * 2) // action_dim) * action_dim
            local_rank = 0
            for start in range(last_start, -1, -action_dim):
                older = [int(token) for token in sequence[start : start + action_dim]]
                newer = [int(token) for token in sequence[start + action_dim : start + action_dim * 2]]
                if (
                    len(older) != action_dim
                    or len(newer) != action_dim
                    or any(token in self._stop for token in older)
                    or any(token in self._stop for token in newer)
                ):
                    local_rank += 1
                    continue
                if not self._vector_delta_ok(older, previous_vector, max_prev_delta):
                    local_rank += 1
                    continue
                if dim and not self._vector_delta_ok(newer[:dim], current_prefix, max_prefix_delta):
                    local_rank += 1
                    continue
                token = int(newer[dim])
                if token in self._stop:
                    local_rank += 1
                    continue
                counts[token] = counts.get(token, 0) + 1
                score = (
                    vector_distance(older, previous_vector, max_prev_delta),
                    vector_distance(newer[:dim], current_prefix, max_prefix_delta) if dim else 0,
                    recency_bias + local_rank,
                    token,
                )
                if token not in best_score or score < best_score[token]:
                    best_score[token] = score
                local_rank += 1

        scan_sequence(tokens, 0)
        recency_bias = len(tokens) + action_dim
        for chunk_idx, chunk in enumerate(self._previous_chunks[:history_size]):
            scan_sequence(chunk, recency_bias + chunk_idx * (len(chunk) + action_dim))

        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(key=lambda token: (-counts[token], best_score[token]))
        return candidates[:top_k]

    def _action_vector_transition_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_vector_transition_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_repeat_vector_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_repeat_vector:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0:
            return []
        dim = len(tokens) % action_dim
        previous_start = len(tokens) - dim - action_dim
        if previous_start < 0:
            return []
        previous = [int(token) for token in tokens[previous_start : previous_start + action_dim]]
        if len(previous) != action_dim or any(token in self._stop for token in previous):
            return []
        max_delta = max(int(self.config.action_repeat_max_delta), 0)
        if dim > 0:
            current_prefix = [int(token) for token in tokens[-dim:]]
            for actual, expected in zip(current_prefix, previous[:dim]):
                if actual in self._stop or abs(int(actual) - int(expected)) > max_delta:
                    return []
        min_repeats = max(int(self.config.action_repeat_min_repeats), 1)
        for repeat_idx in range(1, min_repeats):
            start = previous_start - repeat_idx * action_dim
            if start < 0:
                return []
            older = [int(token) for token in tokens[start : start + action_dim]]
            if len(older) != action_dim or any(token in self._stop for token in older):
                return []
            for older_token, previous_token in zip(older, previous):
                if abs(int(older_token) - int(previous_token)) > max_delta:
                    return []
        token = int(previous[dim])
        if token in self._stop:
            return []
        return [token]

    def _action_repeat_vector_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_repeat_vector_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_token_neighborhood_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_token_neighborhood:
            return []
        centers: list[int] = []

        def add_center(token: int | None) -> None:
            if token is None:
                return
            token = int(token)
            if token in self._stop:
                return
            if any(existing == token for existing in centers):
                return
            centers.append(token)

        if self.config.enable_second_order_action_extrapolation:
            add_center(self._second_order_action_token(tokens))
        add_center(self._linear_action_token(tokens))
        for token in self._previous_chunk_position_candidates(tokens, require_enabled=False):
            add_center(token)
        for source in self._priority:
            if self.config.enable_source_cooldown and int(self._source_cooldowns.get(source, 0)) > 0:
                continue
            if source in {
                "action_token_neighborhood",
                "second_order_action_extrapolation",
                "linear_action_extrapolation",
            }:
                continue
            for token in self._candidate_tokens_from_source(source, tokens):
                add_center(token)
        if not centers:
            return []
        radius = max(int(self.config.action_token_neighborhood_radius), 0)
        top_k = max(int(self.config.action_token_neighborhood_top_k), 1)
        offsets = [0]
        for delta in range(1, radius + 1):
            offsets.extend([-delta, delta])
        candidates: list[int] = []
        for offset in offsets:
            for center in centers:
                token = self._clip_token(int(center) + int(offset), tokens)
                if token in self._stop:
                    continue
                if any(existing == token for existing in candidates):
                    continue
                candidates.append(token)
                if len(candidates) >= top_k:
                    break
            if len(candidates) >= top_k:
                break
        return candidates

    def _action_token_neighborhood_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_token_neighborhood_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_context_tree_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_context_tree:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0 or len(tokens) < action_dim:
            return []
        dim = len(tokens) % action_dim
        max_available_context = len(tokens) // action_dim
        max_context = min(max(int(self.config.action_context_tree_max_context), 1), max_available_context)
        top_k = max(int(self.config.action_context_tree_top_k), 1)
        min_count = max(int(self.config.action_context_tree_min_count), 1)
        if max_context <= 0:
            return []

        def suffix_for(context_len: int) -> list[int]:
            return [int(tokens[len(tokens) - action_dim * offset]) for offset in range(context_len, 0, -1)]

        def scan_sequence(
            sequence: list[int],
            context: list[int],
            recency_bias: int,
            counts: dict[int, int],
            best_recency: dict[int, int],
        ) -> None:
            context_len = len(context)
            if len(sequence) <= dim or context_len <= 0:
                return
            last_idx = dim + ((len(sequence) - dim - 1) // action_dim) * action_dim
            local_rank = 0
            for idx in range(last_idx, dim - 1, -action_dim):
                if idx - context_len * action_dim < 0:
                    local_rank += 1
                    continue
                ok = True
                for context_idx, expected in enumerate(context):
                    prior_idx = idx - (context_len - context_idx) * action_dim
                    actual = int(sequence[prior_idx])
                    if actual in self._stop or actual != int(expected):
                        ok = False
                        break
                token = int(sequence[idx])
                if ok and token not in self._stop:
                    counts[token] = counts.get(token, 0) + 1
                    best_recency[token] = min(
                        best_recency.get(token, recency_bias + local_rank),
                        recency_bias + local_rank,
                    )
                local_rank += 1

        for context_len in range(max_context, 0, -1):
            context = suffix_for(context_len)
            if any(token in self._stop for token in context):
                continue
            counts: dict[int, int] = {}
            best_recency: dict[int, int] = {}
            scan_sequence(tokens, context, 0, counts, best_recency)
            recency_bias = len(tokens) + action_dim
            for chunk_idx, chunk in enumerate(self._previous_chunks):
                scan_sequence(chunk, context, recency_bias + chunk_idx * (len(chunk) + action_dim), counts, best_recency)
            candidates = [token for token, count in counts.items() if count >= min_count]
            if not candidates:
                continue
            previous = int(tokens[-action_dim])
            candidates.sort(key=lambda token: (-counts[token], best_recency[token], abs(int(token) - previous), int(token)))
            return candidates[:top_k]
        return []

    def _action_context_tree_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_context_tree_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_transition_histogram_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_transition_histogram:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0 or len(tokens) < action_dim:
            return []
        dim = len(tokens) % action_dim
        previous = int(tokens[-action_dim])
        if previous in self._stop:
            return []
        min_count = max(int(self.config.action_transition_min_count), 1)
        top_k = max(int(self.config.action_transition_top_k), 1)
        counts: dict[int, int] = {}
        best_recency: dict[int, int] = {}

        def scan_sequence(sequence: list[int], recency_bias: int) -> None:
            if len(sequence) <= dim + action_dim:
                return
            local_rank = 0
            last_start = dim + ((len(sequence) - dim - action_dim - 1) // action_dim) * action_dim
            for idx in range(last_start, dim - 1, -action_dim):
                older = int(sequence[idx])
                newer = int(sequence[idx + action_dim])
                if older in self._stop or newer in self._stop:
                    local_rank += 1
                    continue
                if older != previous:
                    local_rank += 1
                    continue
                if newer in self._stop:
                    local_rank += 1
                    continue
                counts[newer] = counts.get(newer, 0) + 1
                best_recency[newer] = min(best_recency.get(newer, recency_bias + local_rank), recency_bias + local_rank)
                local_rank += 1

        scan_sequence(tokens, 0)
        recency_bias = len(tokens) + action_dim
        for chunk_idx, chunk in enumerate(self._previous_chunks):
            scan_sequence(chunk, recency_bias + chunk_idx * (len(chunk) + action_dim))
        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(key=lambda token: (-counts[token], best_recency[token], int(token)))
        return candidates[:top_k]

    def _action_transition_histogram_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_transition_histogram_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_delta_histogram_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_delta_histogram:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0 or len(tokens) < action_dim:
            return []
        dim = len(tokens) % action_dim
        previous = int(tokens[-action_dim])
        if previous in self._stop:
            return []
        max_pairs = max(int(self.config.action_delta_history), 1)
        max_abs = self.config.action_delta_max_abs
        top_k = max(int(self.config.action_delta_top_k), 1)
        min_count = max(int(self.config.action_delta_min_count), 1)
        counts: dict[int, int] = {}
        best_recency: dict[int, int] = {}
        pairs = 0

        def scan_sequence(sequence: list[int], recency_bias: int) -> None:
            nonlocal pairs
            if len(sequence) <= dim + action_dim:
                return
            local_rank = 0
            newer_index = dim + ((len(sequence) - dim - 1) // action_dim) * action_dim
            while pairs < max_pairs and newer_index - action_dim >= 0:
                newer = int(sequence[newer_index])
                older = int(sequence[newer_index - action_dim])
                if newer in self._stop or older in self._stop:
                    break
                delta = newer - older
                if max_abs is None or abs(delta) <= int(max_abs):
                    token = self._clip_token(previous + delta, tokens)
                    if token not in self._stop:
                        counts[token] = counts.get(token, 0) + 1
                        recency = recency_bias + local_rank
                        best_recency[token] = min(best_recency.get(token, recency), recency)
                newer_index -= action_dim
                local_rank += 1
                pairs += 1

        scan_sequence(tokens, 0)
        recency_bias = len(tokens) + action_dim
        for chunk_idx, chunk in enumerate(self._previous_chunks):
            if pairs >= max_pairs:
                break
            scan_sequence(chunk, recency_bias + chunk_idx * (len(chunk) + action_dim))
        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(key=lambda token: (-counts[token], best_recency[token], abs(int(token) - previous), int(token)))
        return candidates[:top_k]

    def _action_delta_histogram_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_delta_histogram_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_delta_ngram_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_delta_ngram:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0 or len(tokens) < action_dim:
            return []
        dim = len(tokens) % action_dim
        previous = int(tokens[-action_dim])
        if previous in self._stop:
            return []
        max_abs = self.config.action_delta_ngram_max_abs
        min_context = max(int(self.config.action_delta_ngram_min_context), 0)
        max_context = max(int(self.config.action_delta_ngram_max_context), min_context)
        top_k = max(int(self.config.action_delta_ngram_top_k), 1)
        min_count = max(int(self.config.action_delta_ngram_min_count), 1)
        history_size = max(int(self.config.action_delta_ngram_history_size), 1)

        def delta_runs(sequence: list[int]) -> list[list[int]]:
            values: list[int] = []
            idx = dim
            while idx < len(sequence):
                values.append(int(sequence[idx]))
                idx += action_dim
            runs: list[list[int]] = []
            current: list[int] = []
            for older, newer in zip(values, values[1:]):
                if older in self._stop or newer in self._stop:
                    if current:
                        runs.append(current)
                    current = []
                    continue
                delta = newer - older
                if max_abs is not None and abs(delta) > int(max_abs):
                    if current:
                        runs.append(current)
                    current = []
                    continue
                current.append(delta)
            if current:
                runs.append(current)
            return runs

        current_runs = delta_runs(tokens)
        current_deltas = current_runs[-1] if current_runs else []
        max_context = min(max_context, len(current_deltas))
        if max_context < min_context:
            return []
        sequences = [(tokens, 0), *[(chunk, chunk_idx + 1) for chunk_idx, chunk in enumerate(self._previous_chunks[:history_size])]]
        for context_len in range(max_context, min_context - 1, -1):
            context = current_deltas[-context_len:] if context_len else []
            counts: dict[int, int] = {}
            best_recency: dict[int, int] = {}
            best_delta_distance: dict[int, int] = {}
            for sequence, sequence_rank in sequences:
                recency_bias = sequence_rank * (len(sequence) + action_dim)
                for run in delta_runs(sequence):
                    if len(run) <= context_len:
                        continue
                    local_rank = 0
                    last_start = len(run) - context_len - 1
                    for start in range(last_start, -1, -1):
                        if context_len and run[start : start + context_len] != context:
                            local_rank += 1
                            continue
                        next_delta = int(run[start + context_len])
                        token = self._clip_token(previous + next_delta, tokens)
                        if token in self._stop:
                            local_rank += 1
                            continue
                        counts[token] = counts.get(token, 0) + 1
                        recency = recency_bias + local_rank
                        best_recency[token] = min(best_recency.get(token, recency), recency)
                        best_delta_distance[token] = min(
                            best_delta_distance.get(token, abs(next_delta)),
                            abs(next_delta),
                        )
                        local_rank += 1
            candidates = [token for token, count in counts.items() if count >= min_count]
            if not candidates:
                continue
            candidates.sort(
                key=lambda token: (
                    -counts[token],
                    best_recency[token],
                    best_delta_distance[token],
                    int(token),
                )
            )
            return candidates[:top_k]
        return []

    def _action_delta_ngram_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_delta_ngram_candidates(tokens)
        return candidates[0] if candidates else None

    def _chunk_delta_template_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_chunk_delta_template:
            return []
        action_dim = int(self.config.action_dim)
        pos = len(tokens)
        if action_dim <= 0 or pos < action_dim:
            return []
        previous = int(tokens[-action_dim])
        if previous in self._stop:
            return []
        max_abs = self.config.chunk_delta_template_max_abs
        top_k = max(int(self.config.chunk_delta_template_top_k), 1)
        history_size = max(int(self.config.chunk_delta_template_history_size), 1)
        min_prefix_deltas = max(int(self.config.chunk_delta_template_min_prefix_deltas), 0)
        max_delta_mismatch = self.config.chunk_delta_template_max_delta_mismatch
        best: dict[int, tuple[int, int, int, int, int]] = {}
        for chunk_idx, chunk in enumerate(self._previous_chunks[:history_size]):
            if pos >= len(chunk) or pos - action_dim < 0:
                continue
            older = int(chunk[pos - action_dim])
            newer = int(chunk[pos])
            if older in self._stop or newer in self._stop:
                continue
            delta = newer - older
            if max_abs is not None and abs(delta) > int(max_abs):
                continue
            prefix_deltas = 0
            mismatch = 0
            valid_prefix = True
            for idx in range(action_dim, pos):
                current_newer = int(tokens[idx])
                current_older = int(tokens[idx - action_dim])
                if idx >= len(chunk):
                    valid_prefix = False
                    break
                chunk_newer = int(chunk[idx])
                chunk_older = int(chunk[idx - action_dim])
                if (
                    current_newer in self._stop
                    or current_older in self._stop
                    or chunk_newer in self._stop
                    or chunk_older in self._stop
                ):
                    valid_prefix = False
                    break
                current_delta = current_newer - current_older
                chunk_delta = chunk_newer - chunk_older
                mismatch += abs(current_delta - chunk_delta)
                prefix_deltas += 1
                if max_delta_mismatch is not None and mismatch > int(max_delta_mismatch):
                    valid_prefix = False
                    break
            if not valid_prefix or prefix_deltas < min_prefix_deltas:
                continue
            token = self._clip_token(previous + delta, tokens)
            if token in self._stop:
                continue
            score = (mismatch, -prefix_deltas, chunk_idx, abs(delta), int(token))
            if token not in best or score < best[token]:
                best[token] = score
        candidates = list(best)
        candidates.sort(key=lambda token: best[token])
        return candidates[:top_k]

    def _chunk_delta_template_token(self, tokens: list[int]) -> int | None:
        candidates = self._chunk_delta_template_candidates(tokens)
        return candidates[0] if candidates else None

    def _chunk_position_delta_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_chunk_position_delta:
            return []
        action_dim = int(self.config.action_dim)
        pos = len(tokens)
        if action_dim <= 0 or pos < action_dim:
            return []
        previous = int(tokens[-action_dim])
        if previous in self._stop:
            return []
        max_abs = self.config.chunk_position_delta_max_abs
        top_k = max(int(self.config.chunk_position_delta_top_k), 1)
        min_count = max(int(self.config.chunk_position_delta_min_count), 1)
        history_size = max(int(self.config.chunk_position_delta_history_size), 1)
        counts: dict[int, int] = {}
        best_score: dict[int, tuple[int, int, int, int]] = {}
        for chunk_idx, chunk in enumerate(self._previous_chunks[:history_size]):
            if pos >= len(chunk) or pos - action_dim < 0:
                continue
            older = int(chunk[pos - action_dim])
            newer = int(chunk[pos])
            if older in self._stop or newer in self._stop:
                continue
            delta = newer - older
            if max_abs is not None and abs(delta) > int(max_abs):
                continue
            token = self._clip_token(previous + delta, tokens)
            if token in self._stop:
                continue
            counts[token] = counts.get(token, 0) + 1
            score = (chunk_idx, abs(delta), int(delta), int(token))
            if token not in best_score or score < best_score[token]:
                best_score[token] = score
        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(key=lambda token: (-counts[token], best_score[token]))
        return candidates[:top_k]

    def _chunk_position_delta_token(self, tokens: list[int]) -> int | None:
        candidates = self._chunk_position_delta_candidates(tokens)
        return candidates[0] if candidates else None

    def _action_dimension_mode_candidates(self, tokens: list[int]) -> list[int]:
        if not self.config.enable_action_dimension_mode:
            return []
        action_dim = int(self.config.action_dim)
        if action_dim <= 0:
            return []
        dim = len(tokens) % action_dim
        min_count = max(int(self.config.action_dimension_mode_min_count), 1)
        top_k = max(int(self.config.action_dimension_mode_top_k), 1)
        previous = int(tokens[-action_dim]) if len(tokens) >= action_dim else None
        counts: dict[int, int] = {}
        best_recency: dict[int, int] = {}

        def scan_sequence(sequence: list[int], recency_bias: int) -> None:
            if len(sequence) <= dim:
                return
            local_rank = 0
            last_idx = dim + ((len(sequence) - dim - 1) // action_dim) * action_dim
            for idx in range(last_idx, dim - 1, -action_dim):
                token = int(sequence[idx])
                if token in self._stop:
                    local_rank += 1
                    continue
                counts[token] = counts.get(token, 0) + 1
                best_recency[token] = min(best_recency.get(token, recency_bias + local_rank), recency_bias + local_rank)
                local_rank += 1

        scan_sequence(tokens, 0)
        recency_bias = len(tokens) + action_dim
        for chunk_idx, chunk in enumerate(self._previous_chunks):
            scan_sequence(chunk, recency_bias + chunk_idx * (len(chunk) + action_dim))
        candidates = [token for token, count in counts.items() if count >= min_count]
        candidates.sort(
            key=lambda token: (
                -counts[token],
                best_recency[token],
                abs(int(token) - previous) if previous is not None else 0,
                int(token),
            )
        )
        return candidates[:top_k]

    def _action_dimension_mode_token(self, tokens: list[int]) -> int | None:
        candidates = self._action_dimension_mode_candidates(tokens)
        return candidates[0] if candidates else None

    def _linear_action_token(self, tokens: list[int]) -> int | None:
        action_dim = int(self.config.action_dim)
        if action_dim <= 0 or len(tokens) < 2 * action_dim:
            return None
        prev = int(tokens[-action_dim])
        prev2 = int(tokens[-2 * action_dim])
        if prev in self._stop or prev2 in self._stop:
            return None
        return self._clip_token(prev + (prev - prev2), tokens)

    def _second_order_action_token(self, tokens: list[int]) -> int | None:
        action_dim = int(self.config.action_dim)
        if action_dim <= 0 or len(tokens) < 3 * action_dim:
            return None
        current = int(tokens[-action_dim])
        previous = int(tokens[-2 * action_dim])
        previous2 = int(tokens[-3 * action_dim])
        if current in self._stop or previous in self._stop or previous2 in self._stop:
            return None
        velocity = current - previous
        previous_velocity = previous - previous2
        accel = velocity - previous_velocity
        max_accel = self.config.second_order_max_accel
        if max_accel is not None and abs(accel) > int(max_accel):
            return None
        return self._clip_token(current + velocity + accel, tokens)

    def _periodic_token(self, tokens: list[int]) -> int | None:
        repeats = max(int(self.config.min_period_repeats), 2)
        max_period = min(int(self.config.max_period), len(tokens) // repeats)
        for period in range(1, max_period + 1):
            tail = tokens[-period:]
            if any(token in self._stop for token in tail):
                continue
            ok = True
            for rep in range(2, repeats + 1):
                start = len(tokens) - period * rep
                end = start + period
                if start < 0 or tokens[start:end] != tail:
                    ok = False
                    break
            if ok:
                return int(tail[0])
        return None

    def _repeat_token(self, tokens: list[int]) -> int | None:
        token = int(tokens[-1])
        if token in self._stop:
            return None
        run = 0
        for value in reversed(tokens):
            if int(value) != token:
                break
            run += 1
        if run >= int(self.config.repeat_token_min_run):
            return token
        return None

    def _hold_action_token(self, tokens: list[int]) -> int | None:
        action_dim = int(self.config.action_dim)
        if action_dim <= 0 or len(tokens) < action_dim:
            return None
        token = int(tokens[-action_dim])
        if token in self._stop:
            return None
        return token

    def _clip_token(self, token: int, observed_tokens: list[int]) -> int:
        if self.config.vocab_size is not None:
            return max(0, min(int(self.config.vocab_size) - 1, int(token)))
        return max(min(observed_tokens), min(max(observed_tokens), int(token)))


@dataclass(frozen=True)
class PatternSpecTraceResult:
    tokens: int
    target_forwards: int
    drafted_tokens: int
    accepted_tokens: int
    rejected_blocks: int
    draft_misses: int
    full_block_reuses: int = 0
    bonus_tokens: int = 0
    deferred_correction_tokens: int = 0
    tree_candidates: int = 0
    tree_verifies: int = 0
    tree_anchor_candidates: int = 0
    tree_anchor_verifies: int = 0
    tree_anchor_accepted_tokens: int = 0
    tree_first_token_checks: int = 0
    tree_first_token_misses: int = 0
    min_tree_width: int = 0
    max_tree_width: int = 0
    mean_tree_width: float = 0.0
    second_order_action_extrapolation_drafted_tokens: int = 0
    second_order_action_extrapolation_accepted_tokens: int = 0
    previous_chunk_position_drafted_tokens: int = 0
    previous_chunk_position_accepted_tokens: int = 0
    chunk_prefix_retrieval_drafted_tokens: int = 0
    chunk_prefix_retrieval_accepted_tokens: int = 0
    chunk_length_stop_drafted_tokens: int = 0
    chunk_length_stop_accepted_tokens: int = 0
    position_mode_histogram_drafted_tokens: int = 0
    position_mode_histogram_accepted_tokens: int = 0
    global_position_mode_drafted_tokens: int = 0
    global_position_mode_accepted_tokens: int = 0
    ngram_continuation_drafted_tokens: int = 0
    ngram_continuation_accepted_tokens: int = 0
    source_agreement_drafted_tokens: int = 0
    source_agreement_accepted_tokens: int = 0
    action_trend_regression_drafted_tokens: int = 0
    action_trend_regression_accepted_tokens: int = 0
    action_prefix_lookup_drafted_tokens: int = 0
    action_prefix_lookup_accepted_tokens: int = 0
    action_vector_suffix_lookup_drafted_tokens: int = 0
    action_vector_suffix_lookup_accepted_tokens: int = 0
    action_vector_transition_drafted_tokens: int = 0
    action_vector_transition_accepted_tokens: int = 0
    action_repeat_vector_drafted_tokens: int = 0
    action_repeat_vector_accepted_tokens: int = 0
    action_token_neighborhood_drafted_tokens: int = 0
    action_token_neighborhood_accepted_tokens: int = 0
    action_transition_histogram_drafted_tokens: int = 0
    action_transition_histogram_accepted_tokens: int = 0
    action_context_tree_drafted_tokens: int = 0
    action_context_tree_accepted_tokens: int = 0
    action_delta_histogram_drafted_tokens: int = 0
    action_delta_histogram_accepted_tokens: int = 0
    action_delta_ngram_drafted_tokens: int = 0
    action_delta_ngram_accepted_tokens: int = 0
    chunk_delta_template_drafted_tokens: int = 0
    chunk_delta_template_accepted_tokens: int = 0
    chunk_position_delta_drafted_tokens: int = 0
    chunk_position_delta_accepted_tokens: int = 0
    action_dimension_mode_drafted_tokens: int = 0
    action_dimension_mode_accepted_tokens: int = 0
    hold_action_token_drafted_tokens: int = 0
    hold_action_token_accepted_tokens: int = 0
    source_cooldown_events: int = 0
    source_cooldown_skipped_sources: int = 0
    source_acceptance_bias_events: int = 0
    source_acceptance_bias_reorders: int = 0
    min_lookahead: int = 0
    max_lookahead: int = 0
    mean_lookahead: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.drafted_tokens, 1)

    @property
    def target_forward_reduction(self) -> float:
        return self.tokens / max(self.target_forwards, 1)

    @property
    def tree_first_token_miss_rate(self) -> float:
        return self.tree_first_token_misses / max(self.tree_first_token_checks, 1)


def trim_at_stop_token(tokens: list[int], stop_token_ids: tuple[int, ...]) -> list[int]:
    if not stop_token_ids:
        return tokens
    stop = {int(token) for token in stop_token_ids}
    for idx, token in enumerate(tokens):
        if int(token) in stop:
            return tokens[: idx + 1]
    return tokens


def exact_prefix_acceptance(draft: list[int], target: list[int]) -> int:
    accepted = 0
    for draft_token, target_token in zip(draft, target):
        if int(draft_token) != int(target_token):
            break
        accepted += 1
    return accepted


def _source_count(sources: list[str], source: str, limit: int | None = None) -> int:
    values = sources if limit is None else sources[: max(int(limit), 0)]
    return sum(1 for value in values if value == source)


def _reset_source_feedback(drafter: Any) -> None:
    if hasattr(drafter, "reset_source_feedback"):
        drafter.reset_source_feedback()


def _record_source_miss(drafter: Any) -> None:
    if hasattr(drafter, "record_source_miss"):
        drafter.record_source_miss()


def _record_source_feedback(drafter: Any, sources: list[str], accepted: int) -> None:
    if hasattr(drafter, "record_source_feedback"):
        drafter.record_source_feedback(sources, accepted)


def _source_cooldown_stats(drafter: Any) -> dict[str, int]:
    if hasattr(drafter, "source_cooldown_stats"):
        return dict(drafter.source_cooldown_stats())
    return {
        "source_cooldown_events": 0,
        "source_cooldown_skipped_sources": 0,
        "source_acceptance_bias_events": 0,
        "source_acceptance_bias_reorders": 0,
    }


def _trace_task_namespace(trace: Any) -> str | None:
    for attr in ("task", "suite", "suite_name", "task_name"):
        value = getattr(trace, attr, None)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    trace_id = getattr(trace, "trace_id", None)
    if isinstance(trace_id, str) and ":task" in trace_id:
        prefix = trace_id.split(":task", 1)[0]
        if prefix:
            return prefix
    return None


def _trace_task_key(trace: Any) -> int | str:
    task_id = int(trace.task_id)
    namespace = _trace_task_namespace(trace)
    if namespace is None:
        return task_id
    return f"{namespace}:{task_id}"


def _task_key_sort_value(task_key: int | str) -> tuple[int, int | str]:
    if isinstance(task_key, int):
        return (0, task_key)
    return (1, str(task_key))


def _history_key(trace: Any, history_reset: str) -> tuple[Any, ...]:
    if history_reset == "none":
        return ()
    if history_reset == "task":
        return (_trace_task_key(trace),)
    if history_reset == "task_seed":
        return (_trace_task_key(trace), int(trace.seed))
    raise ValueError("history_reset must be one of: none, task, task_seed")


def simulate_exact_pattern_spec_decode(
    target_tokens: Iterable[int],
    drafter: PatternFastTokenDrafter,
    *,
    lookahead: int,
    reuse_full_blocks: bool = False,
    emit_bonus_token: bool = False,
    defer_correction_token: bool = False,
    dynamic_lookahead: bool = False,
    min_lookahead: int = 1,
    lookahead_growth: int = 1,
    lookahead_shrink: int = 4,
) -> PatternSpecTraceResult:
    """Simulate exact draft/verify decode over a known target token stream."""

    _reset_source_feedback(drafter)
    target = [int(token) for token in target_tokens]
    generated: list[int] = []
    target_forwards = 0
    drafted_tokens = 0
    accepted_tokens = 0
    rejected_blocks = 0
    draft_misses = 0
    full_block_reuses = 0
    bonus_tokens = 0
    deferred_correction_tokens = 0
    pending_unprocessed_token = False
    second_order_action_extrapolation_drafted_tokens = 0
    second_order_action_extrapolation_accepted_tokens = 0
    previous_chunk_position_drafted_tokens = 0
    previous_chunk_position_accepted_tokens = 0
    chunk_prefix_retrieval_drafted_tokens = 0
    chunk_prefix_retrieval_accepted_tokens = 0
    chunk_length_stop_drafted_tokens = 0
    chunk_length_stop_accepted_tokens = 0
    position_mode_histogram_drafted_tokens = 0
    position_mode_histogram_accepted_tokens = 0
    global_position_mode_drafted_tokens = 0
    global_position_mode_accepted_tokens = 0
    ngram_continuation_drafted_tokens = 0
    ngram_continuation_accepted_tokens = 0
    source_agreement_drafted_tokens = 0
    source_agreement_accepted_tokens = 0
    action_trend_regression_drafted_tokens = 0
    action_trend_regression_accepted_tokens = 0
    action_prefix_lookup_drafted_tokens = 0
    action_prefix_lookup_accepted_tokens = 0
    action_vector_suffix_lookup_drafted_tokens = 0
    action_vector_suffix_lookup_accepted_tokens = 0
    action_vector_transition_drafted_tokens = 0
    action_vector_transition_accepted_tokens = 0
    action_repeat_vector_drafted_tokens = 0
    action_repeat_vector_accepted_tokens = 0
    action_token_neighborhood_drafted_tokens = 0
    action_token_neighborhood_accepted_tokens = 0
    action_context_tree_drafted_tokens = 0
    action_context_tree_accepted_tokens = 0
    action_transition_histogram_drafted_tokens = 0
    action_transition_histogram_accepted_tokens = 0
    action_delta_histogram_drafted_tokens = 0
    action_delta_histogram_accepted_tokens = 0
    action_delta_ngram_drafted_tokens = 0
    action_delta_ngram_accepted_tokens = 0
    chunk_delta_template_drafted_tokens = 0
    chunk_delta_template_accepted_tokens = 0
    chunk_position_delta_drafted_tokens = 0
    chunk_position_delta_accepted_tokens = 0
    action_dimension_mode_drafted_tokens = 0
    action_dimension_mode_accepted_tokens = 0
    hold_action_token_drafted_tokens = 0
    hold_action_token_accepted_tokens = 0
    current_lookahead = max(1, int(lookahead))
    min_dynamic_lookahead = max(1, min(int(min_lookahead), current_lookahead))
    lookahead_values: list[int] = []

    while len(generated) < len(target):
        remaining = len(target) - len(generated)
        block_lookahead = current_lookahead if dynamic_lookahead else int(lookahead)
        block_lookahead = max(1, min(int(block_lookahead), remaining))
        lookahead_values.append(block_lookahead)
        draft = drafter.draft(generated, lookahead=block_lookahead)
        if not draft:
            draft_misses += 1
            target_forwards += 1
            if pending_unprocessed_token:
                pending_unprocessed_token = False
            else:
                generated.append(target[len(generated)])
            _record_source_miss(drafter)
            if dynamic_lookahead:
                current_lookahead = max(min_dynamic_lookahead, current_lookahead - max(int(lookahead_shrink), 1))
            continue

        target_forwards += 1
        pending_unprocessed_token = False
        drafted_tokens += len(draft)
        sources = drafter.last_draft_sources() if hasattr(drafter, "last_draft_sources") else []
        second_order_action_extrapolation_drafted_tokens += _source_count(
            sources,
            "second_order_action_extrapolation",
        )
        previous_chunk_position_drafted_tokens += _source_count(sources, "previous_chunk_position")
        chunk_prefix_retrieval_drafted_tokens += _source_count(sources, "chunk_prefix_retrieval")
        chunk_length_stop_drafted_tokens += _source_count(sources, "chunk_length_stop")
        position_mode_histogram_drafted_tokens += _source_count(sources, "position_mode_histogram")
        global_position_mode_drafted_tokens += _source_count(sources, "global_position_mode")
        ngram_continuation_drafted_tokens += _source_count(sources, "ngram_continuation")
        source_agreement_drafted_tokens += _source_count(sources, "source_agreement")
        action_trend_regression_drafted_tokens += _source_count(sources, "action_trend_regression")
        action_prefix_lookup_drafted_tokens += _source_count(sources, "action_prefix_lookup")
        action_vector_suffix_lookup_drafted_tokens += _source_count(sources, "action_vector_suffix_lookup")
        action_vector_transition_drafted_tokens += _source_count(sources, "action_vector_transition")
        action_repeat_vector_drafted_tokens += _source_count(sources, "action_repeat_vector")
        action_token_neighborhood_drafted_tokens += _source_count(sources, "action_token_neighborhood")
        action_context_tree_drafted_tokens += _source_count(sources, "action_context_tree")
        action_transition_histogram_drafted_tokens += _source_count(sources, "action_transition_histogram")
        action_delta_histogram_drafted_tokens += _source_count(sources, "action_delta_histogram")
        action_delta_ngram_drafted_tokens += _source_count(sources, "action_delta_ngram")
        chunk_delta_template_drafted_tokens += _source_count(sources, "chunk_delta_template")
        chunk_position_delta_drafted_tokens += _source_count(sources, "chunk_position_delta")
        action_dimension_mode_drafted_tokens += _source_count(sources, "action_dimension_mode")
        hold_action_token_drafted_tokens += _source_count(sources, "hold_action_token")
        accepted = exact_prefix_acceptance(draft, target[len(generated) : len(generated) + len(draft)])
        generated.extend(draft[:accepted])
        accepted_tokens += accepted
        _record_source_feedback(drafter, sources, accepted)
        second_order_action_extrapolation_accepted_tokens += _source_count(
            sources,
            "second_order_action_extrapolation",
            accepted,
        )
        previous_chunk_position_accepted_tokens += _source_count(sources, "previous_chunk_position", accepted)
        chunk_prefix_retrieval_accepted_tokens += _source_count(sources, "chunk_prefix_retrieval", accepted)
        chunk_length_stop_accepted_tokens += _source_count(sources, "chunk_length_stop", accepted)
        position_mode_histogram_accepted_tokens += _source_count(sources, "position_mode_histogram", accepted)
        global_position_mode_accepted_tokens += _source_count(sources, "global_position_mode", accepted)
        ngram_continuation_accepted_tokens += _source_count(sources, "ngram_continuation", accepted)
        source_agreement_accepted_tokens += _source_count(sources, "source_agreement", accepted)
        action_trend_regression_accepted_tokens += _source_count(
            sources,
            "action_trend_regression",
            accepted,
        )
        action_prefix_lookup_accepted_tokens += _source_count(sources, "action_prefix_lookup", accepted)
        action_vector_suffix_lookup_accepted_tokens += _source_count(
            sources,
            "action_vector_suffix_lookup",
            accepted,
        )
        action_vector_transition_accepted_tokens += _source_count(
            sources,
            "action_vector_transition",
            accepted,
        )
        action_repeat_vector_accepted_tokens += _source_count(sources, "action_repeat_vector", accepted)
        action_token_neighborhood_accepted_tokens += _source_count(sources, "action_token_neighborhood", accepted)
        action_context_tree_accepted_tokens += _source_count(sources, "action_context_tree", accepted)
        action_transition_histogram_accepted_tokens += _source_count(
            sources,
            "action_transition_histogram",
            accepted,
        )
        action_delta_histogram_accepted_tokens += _source_count(sources, "action_delta_histogram", accepted)
        action_delta_ngram_accepted_tokens += _source_count(sources, "action_delta_ngram", accepted)
        chunk_delta_template_accepted_tokens += _source_count(sources, "chunk_delta_template", accepted)
        chunk_position_delta_accepted_tokens += _source_count(sources, "chunk_position_delta", accepted)
        action_dimension_mode_accepted_tokens += _source_count(sources, "action_dimension_mode", accepted)
        hold_action_token_accepted_tokens += _source_count(sources, "hold_action_token", accepted)
        if dynamic_lookahead:
            if accepted == len(draft):
                current_lookahead = min(int(lookahead), current_lookahead + max(int(lookahead_growth), 1))
            else:
                current_lookahead = max(min_dynamic_lookahead, current_lookahead - max(int(lookahead_shrink), 1))
        if len(generated) >= len(target):
            continue
        if reuse_full_blocks and accepted == len(draft):
            full_block_reuses += 1
            if emit_bonus_token and len(generated) < len(target):
                generated.append(target[len(generated)])
                bonus_tokens += 1
                if defer_correction_token:
                    pending_unprocessed_token = True
            continue

        if accepted < len(draft):
            rejected_blocks += 1
            if defer_correction_token:
                generated.append(target[len(generated)])
                pending_unprocessed_token = True
                deferred_correction_tokens += 1
                continue
        # The verifier supplies the first rejected token, or the bonus target
        # token after a fully accepted block, before the next speculative block.
        target_forwards += 1
        generated.append(target[len(generated)])

    cooldown_stats = _source_cooldown_stats(drafter)
    return PatternSpecTraceResult(
        tokens=len(target),
        target_forwards=target_forwards,
        drafted_tokens=drafted_tokens,
        accepted_tokens=accepted_tokens,
        rejected_blocks=rejected_blocks,
        draft_misses=draft_misses,
        full_block_reuses=full_block_reuses,
        bonus_tokens=bonus_tokens,
        deferred_correction_tokens=deferred_correction_tokens,
        second_order_action_extrapolation_drafted_tokens=second_order_action_extrapolation_drafted_tokens,
        second_order_action_extrapolation_accepted_tokens=second_order_action_extrapolation_accepted_tokens,
        previous_chunk_position_drafted_tokens=previous_chunk_position_drafted_tokens,
        previous_chunk_position_accepted_tokens=previous_chunk_position_accepted_tokens,
        chunk_prefix_retrieval_drafted_tokens=chunk_prefix_retrieval_drafted_tokens,
        chunk_prefix_retrieval_accepted_tokens=chunk_prefix_retrieval_accepted_tokens,
        chunk_length_stop_drafted_tokens=chunk_length_stop_drafted_tokens,
        chunk_length_stop_accepted_tokens=chunk_length_stop_accepted_tokens,
        position_mode_histogram_drafted_tokens=position_mode_histogram_drafted_tokens,
        position_mode_histogram_accepted_tokens=position_mode_histogram_accepted_tokens,
        global_position_mode_drafted_tokens=global_position_mode_drafted_tokens,
        global_position_mode_accepted_tokens=global_position_mode_accepted_tokens,
        ngram_continuation_drafted_tokens=ngram_continuation_drafted_tokens,
        ngram_continuation_accepted_tokens=ngram_continuation_accepted_tokens,
        source_agreement_drafted_tokens=source_agreement_drafted_tokens,
        source_agreement_accepted_tokens=source_agreement_accepted_tokens,
        action_trend_regression_drafted_tokens=action_trend_regression_drafted_tokens,
        action_trend_regression_accepted_tokens=action_trend_regression_accepted_tokens,
        action_prefix_lookup_drafted_tokens=action_prefix_lookup_drafted_tokens,
        action_prefix_lookup_accepted_tokens=action_prefix_lookup_accepted_tokens,
        action_vector_suffix_lookup_drafted_tokens=action_vector_suffix_lookup_drafted_tokens,
        action_vector_suffix_lookup_accepted_tokens=action_vector_suffix_lookup_accepted_tokens,
        action_vector_transition_drafted_tokens=action_vector_transition_drafted_tokens,
        action_vector_transition_accepted_tokens=action_vector_transition_accepted_tokens,
        action_repeat_vector_drafted_tokens=action_repeat_vector_drafted_tokens,
        action_repeat_vector_accepted_tokens=action_repeat_vector_accepted_tokens,
        action_token_neighborhood_drafted_tokens=action_token_neighborhood_drafted_tokens,
        action_token_neighborhood_accepted_tokens=action_token_neighborhood_accepted_tokens,
        action_context_tree_drafted_tokens=action_context_tree_drafted_tokens,
        action_context_tree_accepted_tokens=action_context_tree_accepted_tokens,
        action_transition_histogram_drafted_tokens=action_transition_histogram_drafted_tokens,
        action_transition_histogram_accepted_tokens=action_transition_histogram_accepted_tokens,
        action_delta_histogram_drafted_tokens=action_delta_histogram_drafted_tokens,
        action_delta_histogram_accepted_tokens=action_delta_histogram_accepted_tokens,
        action_delta_ngram_drafted_tokens=action_delta_ngram_drafted_tokens,
        action_delta_ngram_accepted_tokens=action_delta_ngram_accepted_tokens,
        chunk_delta_template_drafted_tokens=chunk_delta_template_drafted_tokens,
        chunk_delta_template_accepted_tokens=chunk_delta_template_accepted_tokens,
        chunk_position_delta_drafted_tokens=chunk_position_delta_drafted_tokens,
        chunk_position_delta_accepted_tokens=chunk_position_delta_accepted_tokens,
        action_dimension_mode_drafted_tokens=action_dimension_mode_drafted_tokens,
        action_dimension_mode_accepted_tokens=action_dimension_mode_accepted_tokens,
        hold_action_token_drafted_tokens=hold_action_token_drafted_tokens,
        hold_action_token_accepted_tokens=hold_action_token_accepted_tokens,
        source_cooldown_events=int(cooldown_stats.get("source_cooldown_events", 0)),
        source_cooldown_skipped_sources=int(cooldown_stats.get("source_cooldown_skipped_sources", 0)),
        source_acceptance_bias_events=int(cooldown_stats.get("source_acceptance_bias_events", 0)),
        source_acceptance_bias_reorders=int(cooldown_stats.get("source_acceptance_bias_reorders", 0)),
        min_lookahead=min(lookahead_values) if lookahead_values else 0,
        max_lookahead=max(lookahead_values) if lookahead_values else 0,
        mean_lookahead=sum(lookahead_values) / len(lookahead_values) if lookahead_values else 0.0,
    )


def simulate_exact_pattern_tree_spec_decode(
    target_tokens: Iterable[int],
    drafter: PatternFastTokenDrafter,
    *,
    lookahead: int,
    tree_width: int = 4,
    tree_branch_width: int = 4,
    dynamic_tree_width: bool = False,
    min_tree_width: int = 1,
    tree_width_growth: int = 1,
    tree_width_shrink: int = 1,
    tree_anchor_target_token: bool = False,
    tree_anchor_target_continuation: bool = False,
    reuse_full_blocks: bool = False,
    emit_bonus_token: bool = False,
    defer_correction_token: bool = False,
    dynamic_lookahead: bool = False,
    min_lookahead: int = 1,
    lookahead_growth: int = 1,
    lookahead_shrink: int = 4,
) -> PatternSpecTraceResult:
    """Simulate exact greedy tree verification over known target tokens."""

    _reset_source_feedback(drafter)
    target = [int(token) for token in target_tokens]
    generated: list[int] = []
    target_forwards = 0
    drafted_tokens = 0
    accepted_tokens = 0
    rejected_blocks = 0
    draft_misses = 0
    full_block_reuses = 0
    bonus_tokens = 0
    deferred_correction_tokens = 0
    pending_unprocessed_token = False
    tree_candidates = 0
    tree_verifies = 0
    tree_anchor_candidates = 0
    tree_anchor_verifies = 0
    tree_anchor_accepted_tokens = 0
    tree_first_token_checks = 0
    tree_first_token_misses = 0
    second_order_action_extrapolation_drafted_tokens = 0
    second_order_action_extrapolation_accepted_tokens = 0
    previous_chunk_position_drafted_tokens = 0
    previous_chunk_position_accepted_tokens = 0
    chunk_prefix_retrieval_drafted_tokens = 0
    chunk_prefix_retrieval_accepted_tokens = 0
    chunk_length_stop_drafted_tokens = 0
    chunk_length_stop_accepted_tokens = 0
    position_mode_histogram_drafted_tokens = 0
    position_mode_histogram_accepted_tokens = 0
    global_position_mode_drafted_tokens = 0
    global_position_mode_accepted_tokens = 0
    ngram_continuation_drafted_tokens = 0
    ngram_continuation_accepted_tokens = 0
    source_agreement_drafted_tokens = 0
    source_agreement_accepted_tokens = 0
    action_trend_regression_drafted_tokens = 0
    action_trend_regression_accepted_tokens = 0
    action_prefix_lookup_drafted_tokens = 0
    action_prefix_lookup_accepted_tokens = 0
    action_vector_suffix_lookup_drafted_tokens = 0
    action_vector_suffix_lookup_accepted_tokens = 0
    action_vector_transition_drafted_tokens = 0
    action_vector_transition_accepted_tokens = 0
    action_repeat_vector_drafted_tokens = 0
    action_repeat_vector_accepted_tokens = 0
    action_token_neighborhood_drafted_tokens = 0
    action_token_neighborhood_accepted_tokens = 0
    action_context_tree_drafted_tokens = 0
    action_context_tree_accepted_tokens = 0
    action_transition_histogram_drafted_tokens = 0
    action_transition_histogram_accepted_tokens = 0
    action_delta_histogram_drafted_tokens = 0
    action_delta_histogram_accepted_tokens = 0
    action_delta_ngram_drafted_tokens = 0
    action_delta_ngram_accepted_tokens = 0
    chunk_delta_template_drafted_tokens = 0
    chunk_delta_template_accepted_tokens = 0
    chunk_position_delta_drafted_tokens = 0
    chunk_position_delta_accepted_tokens = 0
    action_dimension_mode_drafted_tokens = 0
    action_dimension_mode_accepted_tokens = 0
    hold_action_token_drafted_tokens = 0
    hold_action_token_accepted_tokens = 0
    current_lookahead = max(1, int(lookahead))
    max_tree_width = max(1, int(tree_width))
    current_tree_width = max(1, min(max_tree_width, int(min_tree_width) if dynamic_tree_width else max_tree_width))
    min_dynamic_lookahead = max(1, min(int(min_lookahead), current_lookahead))
    lookahead_values: list[int] = []
    tree_width_values: list[int] = []

    while len(generated) < len(target):
        remaining = len(target) - len(generated)
        block_lookahead = current_lookahead if dynamic_lookahead else int(lookahead)
        block_lookahead = max(1, min(int(block_lookahead), remaining))
        lookahead_values.append(block_lookahead)
        block_tree_width = current_tree_width if dynamic_tree_width else max_tree_width
        block_tree_width = max(1, min(max_tree_width, int(block_tree_width)))
        tree_width_values.append(block_tree_width)
        if block_tree_width > 1:
            target_next = target[len(generated)]
            force_anchor = bool(tree_anchor_target_continuation and remaining > 1 and block_lookahead > 1)
            if force_anchor:
                candidates = [[target_next + 1]]
                candidate_sources = [[]]
            else:
                candidates = drafter.draft_many(
                    generated,
                    lookahead=block_lookahead,
                    max_candidates=block_tree_width,
                    branch_width=tree_branch_width,
                )
                candidate_sources = drafter.last_many_sources() if hasattr(drafter, "last_many_sources") else []
            if candidates:
                if not force_anchor:
                    tree_first_token_checks += 1
                matching_indices = [
                    idx for idx, candidate in enumerate(candidates) if candidate and int(candidate[0]) == target_next
                ]
                first_token_miss = not matching_indices
                if first_token_miss and not force_anchor:
                    tree_first_token_misses += 1
                if tree_anchor_target_token or force_anchor:
                    if matching_indices and not force_anchor:
                        candidates = [candidates[idx] for idx in matching_indices]
                        candidate_sources = [
                            candidate_sources[idx] for idx in matching_indices if idx < len(candidate_sources)
                        ]
                    elif remaining > 1 and block_lookahead > 1:
                        if candidate_sources:
                            _record_source_feedback(drafter, candidate_sources[0], 0)
                        anchor_prefix = [*generated, target_next]
                        anchor_lookahead = max(1, min(block_lookahead - 1, remaining - 1))
                        future_candidates = drafter.draft_many(
                            anchor_prefix,
                            lookahead=anchor_lookahead,
                            max_candidates=block_tree_width,
                            branch_width=tree_branch_width,
                        )
                        future_sources = (
                            drafter.last_many_sources() if hasattr(drafter, "last_many_sources") else []
                        )
                        if future_candidates:
                            target_forwards += 1
                            tree_verifies += 1
                            tree_candidates += len(future_candidates)
                            tree_anchor_verifies += 1
                            tree_anchor_candidates += len(future_candidates)
                            generated.append(target_next)
                            drafted_tokens += sum(len(candidate) for candidate in future_candidates)
                            second_order_action_extrapolation_drafted_tokens += sum(
                                _source_count(sources, "second_order_action_extrapolation")
                                for sources in future_sources
                            )
                            previous_chunk_position_drafted_tokens += sum(
                                _source_count(sources, "previous_chunk_position") for sources in future_sources
                            )
                            chunk_prefix_retrieval_drafted_tokens += sum(
                                _source_count(sources, "chunk_prefix_retrieval") for sources in future_sources
                            )
                            chunk_length_stop_drafted_tokens += sum(
                                _source_count(sources, "chunk_length_stop") for sources in future_sources
                            )
                            position_mode_histogram_drafted_tokens += sum(
                                _source_count(sources, "position_mode_histogram") for sources in future_sources
                            )
                            global_position_mode_drafted_tokens += sum(
                                _source_count(sources, "global_position_mode") for sources in future_sources
                            )
                            ngram_continuation_drafted_tokens += sum(
                                _source_count(sources, "ngram_continuation") for sources in future_sources
                            )
                            source_agreement_drafted_tokens += sum(
                                _source_count(sources, "source_agreement") for sources in future_sources
                            )
                            action_trend_regression_drafted_tokens += sum(
                                _source_count(sources, "action_trend_regression") for sources in future_sources
                            )
                            action_prefix_lookup_drafted_tokens += sum(
                                _source_count(sources, "action_prefix_lookup") for sources in future_sources
                            )
                            action_vector_suffix_lookup_drafted_tokens += sum(
                                _source_count(sources, "action_vector_suffix_lookup") for sources in future_sources
                            )
                            action_vector_transition_drafted_tokens += sum(
                                _source_count(sources, "action_vector_transition") for sources in future_sources
                            )
                            action_repeat_vector_drafted_tokens += sum(
                                _source_count(sources, "action_repeat_vector") for sources in future_sources
                            )
                            action_token_neighborhood_drafted_tokens += sum(
                                _source_count(sources, "action_token_neighborhood") for sources in future_sources
                            )
                            action_context_tree_drafted_tokens += sum(
                                _source_count(sources, "action_context_tree") for sources in future_sources
                            )
                            action_transition_histogram_drafted_tokens += sum(
                                _source_count(sources, "action_transition_histogram") for sources in future_sources
                            )
                            action_delta_histogram_drafted_tokens += sum(
                                _source_count(sources, "action_delta_histogram") for sources in future_sources
                            )
                            action_delta_ngram_drafted_tokens += sum(
                                _source_count(sources, "action_delta_ngram") for sources in future_sources
                            )
                            chunk_delta_template_drafted_tokens += sum(
                                _source_count(sources, "chunk_delta_template") for sources in future_sources
                            )
                            chunk_position_delta_drafted_tokens += sum(
                                _source_count(sources, "chunk_position_delta") for sources in future_sources
                            )
                            action_dimension_mode_drafted_tokens += sum(
                                _source_count(sources, "action_dimension_mode") for sources in future_sources
                            )
                            hold_action_token_drafted_tokens += sum(
                                _source_count(sources, "hold_action_token") for sources in future_sources
                            )
                            target_suffix = target[len(generated) :]
                            best_candidate = future_candidates[0]
                            best_accepted = -1
                            best_idx = 0
                            for candidate_idx, candidate in enumerate(future_candidates):
                                accepted = exact_prefix_acceptance(candidate, target_suffix[: len(candidate)])
                                if accepted > best_accepted:
                                    best_candidate = candidate
                                    best_accepted = accepted
                                    best_idx = candidate_idx
                                    if accepted == len(candidate):
                                        break

                            accepted = max(best_accepted, 0)
                            generated.extend(best_candidate[:accepted])
                            accepted_tokens += accepted
                            tree_anchor_accepted_tokens += accepted
                            best_sources = future_sources[best_idx] if best_idx < len(future_sources) else []
                            _record_source_feedback(drafter, best_sources, accepted)
                            if best_idx < len(future_sources):
                                second_order_action_extrapolation_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "second_order_action_extrapolation",
                                    accepted,
                                )
                                previous_chunk_position_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "previous_chunk_position",
                                    accepted,
                                )
                                chunk_prefix_retrieval_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "chunk_prefix_retrieval",
                                    accepted,
                                )
                                chunk_length_stop_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "chunk_length_stop",
                                    accepted,
                                )
                                position_mode_histogram_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "position_mode_histogram",
                                    accepted,
                                )
                                global_position_mode_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "global_position_mode",
                                    accepted,
                                )
                                ngram_continuation_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "ngram_continuation",
                                    accepted,
                                )
                                source_agreement_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "source_agreement",
                                    accepted,
                                )
                                action_trend_regression_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_trend_regression",
                                    accepted,
                                )
                                action_prefix_lookup_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_prefix_lookup",
                                    accepted,
                                )
                                action_vector_suffix_lookup_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_vector_suffix_lookup",
                                    accepted,
                                )
                                action_vector_transition_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_vector_transition",
                                    accepted,
                                )
                                action_repeat_vector_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_repeat_vector",
                                    accepted,
                                )
                                action_token_neighborhood_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_token_neighborhood",
                                    accepted,
                                )
                                action_context_tree_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_context_tree",
                                    accepted,
                                )
                                action_transition_histogram_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_transition_histogram",
                                    accepted,
                                )
                                action_delta_histogram_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_delta_histogram",
                                    accepted,
                                )
                                action_delta_ngram_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_delta_ngram",
                                    accepted,
                                )
                                chunk_delta_template_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "chunk_delta_template",
                                    accepted,
                                )
                                chunk_position_delta_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "chunk_position_delta",
                                    accepted,
                                )
                                action_dimension_mode_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "action_dimension_mode",
                                    accepted,
                                )
                                hold_action_token_accepted_tokens += _source_count(
                                    future_sources[best_idx],
                                    "hold_action_token",
                                    accepted,
                                )
                            if dynamic_lookahead:
                                if accepted == len(best_candidate):
                                    current_lookahead = min(int(lookahead), current_lookahead + max(int(lookahead_growth), 1))
                                else:
                                    current_lookahead = max(
                                        min_dynamic_lookahead,
                                        current_lookahead - max(int(lookahead_shrink), 1),
                                    )
                            if dynamic_tree_width:
                                if accepted == len(best_candidate):
                                    current_tree_width = min(
                                        max_tree_width,
                                        current_tree_width + max(int(tree_width_growth), 1),
                                    )
                                else:
                                    current_tree_width = max(
                                        1,
                                        current_tree_width - max(int(tree_width_shrink), 1),
                                    )
                            if len(generated) >= len(target):
                                continue
                            if reuse_full_blocks and accepted == len(best_candidate):
                                full_block_reuses += 1
                                if emit_bonus_token and len(generated) < len(target):
                                    generated.append(target[len(generated)])
                                    bonus_tokens += 1
                                    if defer_correction_token:
                                        pending_unprocessed_token = True
                                continue

                            if accepted < len(best_candidate):
                                rejected_blocks += 1
                                if defer_correction_token:
                                    generated.append(target[len(generated)])
                                    pending_unprocessed_token = True
                                    deferred_correction_tokens += 1
                                    continue
                            target_forwards += 1
                            generated.append(target[len(generated)])
                            continue
                        candidates = []
                        candidate_sources = []
                    else:
                        candidates = []
                        candidate_sources = []
        else:
            draft = drafter.draft(generated, lookahead=block_lookahead)
            candidates = [draft] if draft else []
            candidate_sources = [drafter.last_draft_sources()] if draft and hasattr(drafter, "last_draft_sources") else []
        if not candidates:
            draft_misses += 1
            target_forwards += 1
            if pending_unprocessed_token:
                pending_unprocessed_token = False
            else:
                generated.append(target[len(generated)])
            _record_source_miss(drafter)
            if dynamic_tree_width:
                current_tree_width = max(1, min(max_tree_width, current_tree_width - max(int(tree_width_shrink), 1)))
            if dynamic_lookahead:
                current_lookahead = max(min_dynamic_lookahead, current_lookahead - max(int(lookahead_shrink), 1))
            continue

        target_forwards += 1
        pending_unprocessed_token = False
        if block_tree_width > 1:
            tree_verifies += 1
            tree_candidates += len(candidates)
        drafted_tokens += sum(len(candidate) for candidate in candidates)
        second_order_action_extrapolation_drafted_tokens += sum(
            _source_count(sources, "second_order_action_extrapolation") for sources in candidate_sources
        )
        previous_chunk_position_drafted_tokens += sum(
            _source_count(sources, "previous_chunk_position") for sources in candidate_sources
        )
        chunk_prefix_retrieval_drafted_tokens += sum(
            _source_count(sources, "chunk_prefix_retrieval") for sources in candidate_sources
        )
        chunk_length_stop_drafted_tokens += sum(
            _source_count(sources, "chunk_length_stop") for sources in candidate_sources
        )
        position_mode_histogram_drafted_tokens += sum(
            _source_count(sources, "position_mode_histogram") for sources in candidate_sources
        )
        global_position_mode_drafted_tokens += sum(
            _source_count(sources, "global_position_mode") for sources in candidate_sources
        )
        ngram_continuation_drafted_tokens += sum(
            _source_count(sources, "ngram_continuation") for sources in candidate_sources
        )
        source_agreement_drafted_tokens += sum(
            _source_count(sources, "source_agreement") for sources in candidate_sources
        )
        action_trend_regression_drafted_tokens += sum(
            _source_count(sources, "action_trend_regression") for sources in candidate_sources
        )
        action_prefix_lookup_drafted_tokens += sum(
            _source_count(sources, "action_prefix_lookup") for sources in candidate_sources
        )
        action_vector_suffix_lookup_drafted_tokens += sum(
            _source_count(sources, "action_vector_suffix_lookup") for sources in candidate_sources
        )
        action_vector_transition_drafted_tokens += sum(
            _source_count(sources, "action_vector_transition") for sources in candidate_sources
        )
        action_repeat_vector_drafted_tokens += sum(
            _source_count(sources, "action_repeat_vector") for sources in candidate_sources
        )
        action_token_neighborhood_drafted_tokens += sum(
            _source_count(sources, "action_token_neighborhood") for sources in candidate_sources
        )
        action_context_tree_drafted_tokens += sum(
            _source_count(sources, "action_context_tree") for sources in candidate_sources
        )
        action_transition_histogram_drafted_tokens += sum(
            _source_count(sources, "action_transition_histogram") for sources in candidate_sources
        )
        action_delta_histogram_drafted_tokens += sum(
            _source_count(sources, "action_delta_histogram") for sources in candidate_sources
        )
        action_delta_ngram_drafted_tokens += sum(
            _source_count(sources, "action_delta_ngram") for sources in candidate_sources
        )
        chunk_delta_template_drafted_tokens += sum(
            _source_count(sources, "chunk_delta_template") for sources in candidate_sources
        )
        chunk_position_delta_drafted_tokens += sum(
            _source_count(sources, "chunk_position_delta") for sources in candidate_sources
        )
        action_dimension_mode_drafted_tokens += sum(
            _source_count(sources, "action_dimension_mode") for sources in candidate_sources
        )
        hold_action_token_drafted_tokens += sum(
            _source_count(sources, "hold_action_token") for sources in candidate_sources
        )
        target_suffix = target[len(generated) :]
        best_candidate = candidates[0]
        best_accepted = -1
        best_idx = 0
        for candidate_idx, candidate in enumerate(candidates):
            accepted = exact_prefix_acceptance(candidate, target_suffix[: len(candidate)])
            if accepted > best_accepted:
                best_candidate = candidate
                best_accepted = accepted
                best_idx = candidate_idx
                if accepted == len(candidate):
                    break

        accepted = max(best_accepted, 0)
        generated.extend(best_candidate[:accepted])
        accepted_tokens += accepted
        best_sources = candidate_sources[best_idx] if best_idx < len(candidate_sources) else []
        _record_source_feedback(drafter, best_sources, accepted)
        if best_idx < len(candidate_sources):
            second_order_action_extrapolation_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "second_order_action_extrapolation",
                accepted,
            )
            previous_chunk_position_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "previous_chunk_position",
                accepted,
            )
            chunk_prefix_retrieval_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "chunk_prefix_retrieval",
                accepted,
            )
            chunk_length_stop_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "chunk_length_stop",
                accepted,
            )
            position_mode_histogram_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "position_mode_histogram",
                accepted,
            )
            global_position_mode_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "global_position_mode",
                accepted,
            )
            ngram_continuation_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "ngram_continuation",
                accepted,
            )
            source_agreement_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "source_agreement",
                accepted,
            )
            action_trend_regression_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_trend_regression",
                accepted,
            )
            action_prefix_lookup_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_prefix_lookup",
                accepted,
            )
            action_vector_suffix_lookup_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_vector_suffix_lookup",
                accepted,
            )
            action_vector_transition_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_vector_transition",
                accepted,
            )
            action_repeat_vector_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_repeat_vector",
                accepted,
            )
            action_token_neighborhood_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_token_neighborhood",
                accepted,
            )
            action_context_tree_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_context_tree",
                accepted,
            )
            action_transition_histogram_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_transition_histogram",
                accepted,
            )
            action_delta_histogram_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_delta_histogram",
                accepted,
            )
            action_delta_ngram_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_delta_ngram",
                accepted,
            )
            chunk_delta_template_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "chunk_delta_template",
                accepted,
            )
            chunk_position_delta_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "chunk_position_delta",
                accepted,
            )
            action_dimension_mode_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "action_dimension_mode",
                accepted,
            )
            hold_action_token_accepted_tokens += _source_count(
                candidate_sources[best_idx],
                "hold_action_token",
                accepted,
            )
        if dynamic_lookahead:
            if accepted == len(best_candidate):
                current_lookahead = min(int(lookahead), current_lookahead + max(int(lookahead_growth), 1))
            else:
                current_lookahead = max(min_dynamic_lookahead, current_lookahead - max(int(lookahead_shrink), 1))
        if dynamic_tree_width:
            if accepted == len(best_candidate):
                current_tree_width = min(max_tree_width, current_tree_width + max(int(tree_width_growth), 1))
            else:
                current_tree_width = max(1, current_tree_width - max(int(tree_width_shrink), 1))
        if len(generated) >= len(target):
            continue
        if reuse_full_blocks and accepted == len(best_candidate):
            full_block_reuses += 1
            if emit_bonus_token and len(generated) < len(target):
                generated.append(target[len(generated)])
                bonus_tokens += 1
                if defer_correction_token:
                    pending_unprocessed_token = True
            continue

        if accepted < len(best_candidate):
            rejected_blocks += 1
            if defer_correction_token:
                generated.append(target[len(generated)])
                pending_unprocessed_token = True
                deferred_correction_tokens += 1
                continue
        target_forwards += 1
        generated.append(target[len(generated)])

    cooldown_stats = _source_cooldown_stats(drafter)
    return PatternSpecTraceResult(
        tokens=len(target),
        target_forwards=target_forwards,
        drafted_tokens=drafted_tokens,
        accepted_tokens=accepted_tokens,
        rejected_blocks=rejected_blocks,
        draft_misses=draft_misses,
        full_block_reuses=full_block_reuses,
        bonus_tokens=bonus_tokens,
        deferred_correction_tokens=deferred_correction_tokens,
        tree_candidates=tree_candidates,
        tree_verifies=tree_verifies,
        tree_anchor_candidates=tree_anchor_candidates,
        tree_anchor_verifies=tree_anchor_verifies,
        tree_anchor_accepted_tokens=tree_anchor_accepted_tokens,
        tree_first_token_checks=tree_first_token_checks,
        tree_first_token_misses=tree_first_token_misses,
        min_tree_width=min(tree_width_values) if tree_width_values else 0,
        max_tree_width=max(tree_width_values) if tree_width_values else 0,
        mean_tree_width=sum(tree_width_values) / len(tree_width_values) if tree_width_values else 0.0,
        second_order_action_extrapolation_drafted_tokens=second_order_action_extrapolation_drafted_tokens,
        second_order_action_extrapolation_accepted_tokens=second_order_action_extrapolation_accepted_tokens,
        previous_chunk_position_drafted_tokens=previous_chunk_position_drafted_tokens,
        previous_chunk_position_accepted_tokens=previous_chunk_position_accepted_tokens,
        chunk_prefix_retrieval_drafted_tokens=chunk_prefix_retrieval_drafted_tokens,
        chunk_prefix_retrieval_accepted_tokens=chunk_prefix_retrieval_accepted_tokens,
        chunk_length_stop_drafted_tokens=chunk_length_stop_drafted_tokens,
        chunk_length_stop_accepted_tokens=chunk_length_stop_accepted_tokens,
        position_mode_histogram_drafted_tokens=position_mode_histogram_drafted_tokens,
        position_mode_histogram_accepted_tokens=position_mode_histogram_accepted_tokens,
        global_position_mode_drafted_tokens=global_position_mode_drafted_tokens,
        global_position_mode_accepted_tokens=global_position_mode_accepted_tokens,
        ngram_continuation_drafted_tokens=ngram_continuation_drafted_tokens,
        ngram_continuation_accepted_tokens=ngram_continuation_accepted_tokens,
        source_agreement_drafted_tokens=source_agreement_drafted_tokens,
        source_agreement_accepted_tokens=source_agreement_accepted_tokens,
        action_trend_regression_drafted_tokens=action_trend_regression_drafted_tokens,
        action_trend_regression_accepted_tokens=action_trend_regression_accepted_tokens,
        action_prefix_lookup_drafted_tokens=action_prefix_lookup_drafted_tokens,
        action_prefix_lookup_accepted_tokens=action_prefix_lookup_accepted_tokens,
        action_vector_suffix_lookup_drafted_tokens=action_vector_suffix_lookup_drafted_tokens,
        action_vector_suffix_lookup_accepted_tokens=action_vector_suffix_lookup_accepted_tokens,
        action_vector_transition_drafted_tokens=action_vector_transition_drafted_tokens,
        action_vector_transition_accepted_tokens=action_vector_transition_accepted_tokens,
        action_repeat_vector_drafted_tokens=action_repeat_vector_drafted_tokens,
        action_repeat_vector_accepted_tokens=action_repeat_vector_accepted_tokens,
        action_token_neighborhood_drafted_tokens=action_token_neighborhood_drafted_tokens,
        action_token_neighborhood_accepted_tokens=action_token_neighborhood_accepted_tokens,
        action_context_tree_drafted_tokens=action_context_tree_drafted_tokens,
        action_context_tree_accepted_tokens=action_context_tree_accepted_tokens,
        action_transition_histogram_drafted_tokens=action_transition_histogram_drafted_tokens,
        action_transition_histogram_accepted_tokens=action_transition_histogram_accepted_tokens,
        action_delta_histogram_drafted_tokens=action_delta_histogram_drafted_tokens,
        action_delta_histogram_accepted_tokens=action_delta_histogram_accepted_tokens,
        action_delta_ngram_drafted_tokens=action_delta_ngram_drafted_tokens,
        action_delta_ngram_accepted_tokens=action_delta_ngram_accepted_tokens,
        chunk_delta_template_drafted_tokens=chunk_delta_template_drafted_tokens,
        chunk_delta_template_accepted_tokens=chunk_delta_template_accepted_tokens,
        chunk_position_delta_drafted_tokens=chunk_position_delta_drafted_tokens,
        chunk_position_delta_accepted_tokens=chunk_position_delta_accepted_tokens,
        action_dimension_mode_drafted_tokens=action_dimension_mode_drafted_tokens,
        action_dimension_mode_accepted_tokens=action_dimension_mode_accepted_tokens,
        hold_action_token_drafted_tokens=hold_action_token_drafted_tokens,
        hold_action_token_accepted_tokens=hold_action_token_accepted_tokens,
        source_cooldown_events=int(cooldown_stats.get("source_cooldown_events", 0)),
        source_cooldown_skipped_sources=int(cooldown_stats.get("source_cooldown_skipped_sources", 0)),
        source_acceptance_bias_events=int(cooldown_stats.get("source_acceptance_bias_events", 0)),
        source_acceptance_bias_reorders=int(cooldown_stats.get("source_acceptance_bias_reorders", 0)),
        min_lookahead=min(lookahead_values) if lookahead_values else 0,
        max_lookahead=max(lookahead_values) if lookahead_values else 0,
        mean_lookahead=sum(lookahead_values) / len(lookahead_values) if lookahead_values else 0.0,
    )


def evaluate_pattern_drafter(
    drafter: PatternFastTokenDrafter,
    traces: Iterable[Any],
    *,
    lookahead: int,
    reuse_full_blocks: bool = False,
    emit_bonus_token: bool = False,
    dynamic_lookahead: bool = False,
    min_lookahead: int = 1,
    lookahead_growth: int = 1,
    lookahead_shrink: int = 4,
    tree_width: int = 1,
    tree_branch_width: int = 4,
    dynamic_tree_width: bool = False,
    min_tree_width: int = 1,
    tree_width_growth: int = 1,
    tree_width_shrink: int = 1,
    tree_anchor_target_token: bool = False,
    tree_anchor_target_continuation: bool = False,
    target_forward_ms: float = 1.0,
    draft_token_ms: float = 0.0,
    defer_correction_token: bool = False,
    history_reset: str = "task_seed",
) -> dict:
    per_trace: list[dict] = []
    total_tokens = 0
    total_target_forwards = 0
    total_drafted = 0
    total_accepted = 0
    total_rejected = 0
    total_misses = 0
    total_full_block_reuses = 0
    total_bonus_tokens = 0
    total_deferred_correction_tokens = 0
    total_tree_candidates = 0
    total_tree_verifies = 0
    total_tree_anchor_candidates = 0
    total_tree_anchor_verifies = 0
    total_tree_anchor_accepted = 0
    total_tree_first_token_checks = 0
    total_tree_first_token_misses = 0
    tree_width_means: list[float] = []
    total_second_order_action_extrapolation_drafted = 0
    total_second_order_action_extrapolation_accepted = 0
    total_previous_chunk_position_drafted = 0
    total_previous_chunk_position_accepted = 0
    total_chunk_prefix_retrieval_drafted = 0
    total_chunk_prefix_retrieval_accepted = 0
    total_chunk_length_stop_drafted = 0
    total_chunk_length_stop_accepted = 0
    total_position_mode_histogram_drafted = 0
    total_position_mode_histogram_accepted = 0
    total_global_position_mode_drafted = 0
    total_global_position_mode_accepted = 0
    total_ngram_continuation_drafted = 0
    total_ngram_continuation_accepted = 0
    total_source_agreement_drafted = 0
    total_source_agreement_accepted = 0
    total_action_trend_regression_drafted = 0
    total_action_trend_regression_accepted = 0
    total_action_prefix_lookup_drafted = 0
    total_action_prefix_lookup_accepted = 0
    total_action_vector_suffix_lookup_drafted = 0
    total_action_vector_suffix_lookup_accepted = 0
    total_action_vector_transition_drafted = 0
    total_action_vector_transition_accepted = 0
    total_action_repeat_vector_drafted = 0
    total_action_repeat_vector_accepted = 0
    total_action_token_neighborhood_drafted = 0
    total_action_token_neighborhood_accepted = 0
    total_action_context_tree_drafted = 0
    total_action_context_tree_accepted = 0
    total_action_transition_histogram_drafted = 0
    total_action_transition_histogram_accepted = 0
    total_action_delta_histogram_drafted = 0
    total_action_delta_histogram_accepted = 0
    total_action_delta_ngram_drafted = 0
    total_action_delta_ngram_accepted = 0
    total_chunk_delta_template_drafted = 0
    total_chunk_delta_template_accepted = 0
    total_chunk_position_delta_drafted = 0
    total_chunk_position_delta_accepted = 0
    total_action_dimension_mode_drafted = 0
    total_action_dimension_mode_accepted = 0
    total_hold_action_token_drafted = 0
    total_hold_action_token_accepted = 0
    total_source_cooldown_events = 0
    total_source_cooldown_skipped_sources = 0
    total_source_acceptance_bias_events = 0
    total_source_acceptance_bias_reorders = 0
    lookahead_means: list[float] = []
    per_task: dict[int | str, list[PatternSpecTraceResult]] = {}
    last_history_key: tuple[Any, ...] | None = None
    history_resets = 0

    for trace in traces:
        history_key = _history_key(trace, history_reset)
        if last_history_key is not None and history_key != last_history_key and hasattr(drafter, "reset_history"):
            drafter.reset_history()
            history_resets += 1
        last_history_key = history_key
        tokens = [int(token) for token in trace.token_ids.tolist()]
        tokens = trim_at_stop_token(tokens, drafter.config.stop_token_ids)
        if not tokens:
            continue
        if int(tree_width) > 1:
            result = simulate_exact_pattern_tree_spec_decode(
                tokens,
                drafter,
                lookahead=lookahead,
                tree_width=tree_width,
                tree_branch_width=tree_branch_width,
                dynamic_tree_width=dynamic_tree_width,
                min_tree_width=min_tree_width,
                tree_width_growth=tree_width_growth,
                tree_width_shrink=tree_width_shrink,
                tree_anchor_target_token=tree_anchor_target_token,
                tree_anchor_target_continuation=tree_anchor_target_continuation,
                reuse_full_blocks=reuse_full_blocks,
                emit_bonus_token=emit_bonus_token,
                defer_correction_token=defer_correction_token,
                dynamic_lookahead=dynamic_lookahead,
                min_lookahead=min_lookahead,
                lookahead_growth=lookahead_growth,
                lookahead_shrink=lookahead_shrink,
            )
        else:
            result = simulate_exact_pattern_spec_decode(
                tokens,
                drafter,
                lookahead=lookahead,
                reuse_full_blocks=reuse_full_blocks,
                emit_bonus_token=emit_bonus_token,
                defer_correction_token=defer_correction_token,
                dynamic_lookahead=dynamic_lookahead,
                min_lookahead=min_lookahead,
                lookahead_growth=lookahead_growth,
                lookahead_shrink=lookahead_shrink,
            )
        total_tokens += result.tokens
        total_target_forwards += result.target_forwards
        total_drafted += result.drafted_tokens
        total_accepted += result.accepted_tokens
        total_rejected += result.rejected_blocks
        total_misses += result.draft_misses
        total_full_block_reuses += result.full_block_reuses
        total_bonus_tokens += result.bonus_tokens
        total_deferred_correction_tokens += result.deferred_correction_tokens
        total_tree_candidates += result.tree_candidates
        total_tree_verifies += result.tree_verifies
        total_tree_anchor_candidates += result.tree_anchor_candidates
        total_tree_anchor_verifies += result.tree_anchor_verifies
        total_tree_anchor_accepted += result.tree_anchor_accepted_tokens
        total_tree_first_token_checks += result.tree_first_token_checks
        total_tree_first_token_misses += result.tree_first_token_misses
        if result.mean_tree_width:
            tree_width_means.append(result.mean_tree_width)
        total_second_order_action_extrapolation_drafted += (
            result.second_order_action_extrapolation_drafted_tokens
        )
        total_second_order_action_extrapolation_accepted += (
            result.second_order_action_extrapolation_accepted_tokens
        )
        total_previous_chunk_position_drafted += result.previous_chunk_position_drafted_tokens
        total_previous_chunk_position_accepted += result.previous_chunk_position_accepted_tokens
        total_chunk_prefix_retrieval_drafted += result.chunk_prefix_retrieval_drafted_tokens
        total_chunk_prefix_retrieval_accepted += result.chunk_prefix_retrieval_accepted_tokens
        total_chunk_length_stop_drafted += result.chunk_length_stop_drafted_tokens
        total_chunk_length_stop_accepted += result.chunk_length_stop_accepted_tokens
        total_position_mode_histogram_drafted += result.position_mode_histogram_drafted_tokens
        total_position_mode_histogram_accepted += result.position_mode_histogram_accepted_tokens
        total_global_position_mode_drafted += result.global_position_mode_drafted_tokens
        total_global_position_mode_accepted += result.global_position_mode_accepted_tokens
        total_ngram_continuation_drafted += result.ngram_continuation_drafted_tokens
        total_ngram_continuation_accepted += result.ngram_continuation_accepted_tokens
        total_source_agreement_drafted += result.source_agreement_drafted_tokens
        total_source_agreement_accepted += result.source_agreement_accepted_tokens
        total_action_trend_regression_drafted += result.action_trend_regression_drafted_tokens
        total_action_trend_regression_accepted += result.action_trend_regression_accepted_tokens
        total_action_prefix_lookup_drafted += result.action_prefix_lookup_drafted_tokens
        total_action_prefix_lookup_accepted += result.action_prefix_lookup_accepted_tokens
        total_action_vector_suffix_lookup_drafted += result.action_vector_suffix_lookup_drafted_tokens
        total_action_vector_suffix_lookup_accepted += result.action_vector_suffix_lookup_accepted_tokens
        total_action_vector_transition_drafted += result.action_vector_transition_drafted_tokens
        total_action_vector_transition_accepted += result.action_vector_transition_accepted_tokens
        total_action_repeat_vector_drafted += result.action_repeat_vector_drafted_tokens
        total_action_repeat_vector_accepted += result.action_repeat_vector_accepted_tokens
        total_action_token_neighborhood_drafted += result.action_token_neighborhood_drafted_tokens
        total_action_token_neighborhood_accepted += result.action_token_neighborhood_accepted_tokens
        total_action_context_tree_drafted += result.action_context_tree_drafted_tokens
        total_action_context_tree_accepted += result.action_context_tree_accepted_tokens
        total_action_transition_histogram_drafted += result.action_transition_histogram_drafted_tokens
        total_action_transition_histogram_accepted += result.action_transition_histogram_accepted_tokens
        total_action_delta_histogram_drafted += result.action_delta_histogram_drafted_tokens
        total_action_delta_histogram_accepted += result.action_delta_histogram_accepted_tokens
        total_action_delta_ngram_drafted += result.action_delta_ngram_drafted_tokens
        total_action_delta_ngram_accepted += result.action_delta_ngram_accepted_tokens
        total_chunk_delta_template_drafted += result.chunk_delta_template_drafted_tokens
        total_chunk_delta_template_accepted += result.chunk_delta_template_accepted_tokens
        total_chunk_position_delta_drafted += result.chunk_position_delta_drafted_tokens
        total_chunk_position_delta_accepted += result.chunk_position_delta_accepted_tokens
        total_action_dimension_mode_drafted += result.action_dimension_mode_drafted_tokens
        total_action_dimension_mode_accepted += result.action_dimension_mode_accepted_tokens
        total_hold_action_token_drafted += result.hold_action_token_drafted_tokens
        total_hold_action_token_accepted += result.hold_action_token_accepted_tokens
        total_source_cooldown_events += result.source_cooldown_events
        total_source_cooldown_skipped_sources += result.source_cooldown_skipped_sources
        total_source_acceptance_bias_events += result.source_acceptance_bias_events
        total_source_acceptance_bias_reorders += result.source_acceptance_bias_reorders
        lookahead_means.append(result.mean_lookahead)
        task_key = _trace_task_key(trace)
        per_task.setdefault(task_key, []).append(result)
        per_trace.append(
            {
                "trace_id": trace.trace_id,
                "task_id": trace.task_id,
                "task": _trace_task_namespace(trace),
                "task_key": str(task_key),
                "seed": trace.seed,
                "tokens": result.tokens,
                "target_forwards": result.target_forwards,
                "target_forward_reduction": result.target_forward_reduction,
                "drafted_tokens": result.drafted_tokens,
                "accepted_tokens": result.accepted_tokens,
                "acceptance_rate": result.acceptance_rate,
                "rejected_blocks": result.rejected_blocks,
                "draft_misses": result.draft_misses,
                "full_block_reuses": result.full_block_reuses,
                "bonus_tokens": result.bonus_tokens,
                "deferred_correction_tokens": result.deferred_correction_tokens,
                "tree_candidates": result.tree_candidates,
                "tree_verifies": result.tree_verifies,
                "tree_anchor_candidates": result.tree_anchor_candidates,
                "tree_anchor_verifies": result.tree_anchor_verifies,
                "tree_anchor_accepted_tokens": result.tree_anchor_accepted_tokens,
                "tree_first_token_checks": result.tree_first_token_checks,
                "tree_first_token_misses": result.tree_first_token_misses,
                "mean_tree_width": result.mean_tree_width,
                "second_order_action_extrapolation_drafted_tokens": (
                    result.second_order_action_extrapolation_drafted_tokens
                ),
                "second_order_action_extrapolation_accepted_tokens": (
                    result.second_order_action_extrapolation_accepted_tokens
                ),
                "previous_chunk_position_drafted_tokens": result.previous_chunk_position_drafted_tokens,
                "previous_chunk_position_accepted_tokens": result.previous_chunk_position_accepted_tokens,
                "chunk_prefix_retrieval_drafted_tokens": result.chunk_prefix_retrieval_drafted_tokens,
                "chunk_prefix_retrieval_accepted_tokens": result.chunk_prefix_retrieval_accepted_tokens,
                "chunk_length_stop_drafted_tokens": result.chunk_length_stop_drafted_tokens,
                "chunk_length_stop_accepted_tokens": result.chunk_length_stop_accepted_tokens,
                "position_mode_histogram_drafted_tokens": result.position_mode_histogram_drafted_tokens,
                "position_mode_histogram_accepted_tokens": result.position_mode_histogram_accepted_tokens,
                "global_position_mode_drafted_tokens": result.global_position_mode_drafted_tokens,
                "global_position_mode_accepted_tokens": result.global_position_mode_accepted_tokens,
                "ngram_continuation_drafted_tokens": result.ngram_continuation_drafted_tokens,
                "ngram_continuation_accepted_tokens": result.ngram_continuation_accepted_tokens,
                "source_agreement_drafted_tokens": result.source_agreement_drafted_tokens,
                "source_agreement_accepted_tokens": result.source_agreement_accepted_tokens,
                "action_trend_regression_drafted_tokens": result.action_trend_regression_drafted_tokens,
                "action_trend_regression_accepted_tokens": result.action_trend_regression_accepted_tokens,
                "action_prefix_lookup_drafted_tokens": result.action_prefix_lookup_drafted_tokens,
                "action_prefix_lookup_accepted_tokens": result.action_prefix_lookup_accepted_tokens,
                "action_vector_suffix_lookup_drafted_tokens": (
                    result.action_vector_suffix_lookup_drafted_tokens
                ),
                "action_vector_suffix_lookup_accepted_tokens": (
                    result.action_vector_suffix_lookup_accepted_tokens
                ),
                "action_vector_transition_drafted_tokens": result.action_vector_transition_drafted_tokens,
                "action_vector_transition_accepted_tokens": result.action_vector_transition_accepted_tokens,
                "action_repeat_vector_drafted_tokens": result.action_repeat_vector_drafted_tokens,
                "action_repeat_vector_accepted_tokens": result.action_repeat_vector_accepted_tokens,
                "action_token_neighborhood_drafted_tokens": result.action_token_neighborhood_drafted_tokens,
                "action_token_neighborhood_accepted_tokens": result.action_token_neighborhood_accepted_tokens,
                "action_context_tree_drafted_tokens": result.action_context_tree_drafted_tokens,
                "action_context_tree_accepted_tokens": result.action_context_tree_accepted_tokens,
                "action_transition_histogram_drafted_tokens": result.action_transition_histogram_drafted_tokens,
                "action_transition_histogram_accepted_tokens": result.action_transition_histogram_accepted_tokens,
                "action_delta_histogram_drafted_tokens": result.action_delta_histogram_drafted_tokens,
                "action_delta_histogram_accepted_tokens": result.action_delta_histogram_accepted_tokens,
                "action_delta_ngram_drafted_tokens": result.action_delta_ngram_drafted_tokens,
                "action_delta_ngram_accepted_tokens": result.action_delta_ngram_accepted_tokens,
                "chunk_delta_template_drafted_tokens": result.chunk_delta_template_drafted_tokens,
                "chunk_delta_template_accepted_tokens": result.chunk_delta_template_accepted_tokens,
                "chunk_position_delta_drafted_tokens": result.chunk_position_delta_drafted_tokens,
                "chunk_position_delta_accepted_tokens": result.chunk_position_delta_accepted_tokens,
                "action_dimension_mode_drafted_tokens": result.action_dimension_mode_drafted_tokens,
                "action_dimension_mode_accepted_tokens": result.action_dimension_mode_accepted_tokens,
                "hold_action_token_drafted_tokens": result.hold_action_token_drafted_tokens,
                "hold_action_token_accepted_tokens": result.hold_action_token_accepted_tokens,
                "source_cooldown_events": result.source_cooldown_events,
                "source_cooldown_skipped_sources": result.source_cooldown_skipped_sources,
                "source_acceptance_bias_events": result.source_acceptance_bias_events,
                "source_acceptance_bias_reorders": result.source_acceptance_bias_reorders,
                "mean_lookahead": result.mean_lookahead,
            }
        )
        drafter.observe(tokens)

    if not per_trace:
        raise ValueError("No trace tokens evaluated")
    baseline_ms = total_tokens * float(target_forward_ms)
    spec_ms = total_target_forwards * float(target_forward_ms) + total_drafted * float(draft_token_ms)
    metrics = {
        "traces": len(per_trace),
        "lookahead": int(lookahead),
        "reuse_full_blocks": bool(reuse_full_blocks),
        "emit_bonus_token": bool(emit_bonus_token),
        "defer_correction_token": bool(defer_correction_token),
        "dynamic_lookahead": bool(dynamic_lookahead),
        "min_lookahead": int(min_lookahead),
        "lookahead_growth": int(lookahead_growth),
        "lookahead_shrink": int(lookahead_shrink),
        "history_reset": history_reset,
        "history_resets": history_resets,
        "tree_width": int(tree_width),
        "tree_branch_width": int(tree_branch_width),
        "dynamic_tree_width": bool(dynamic_tree_width),
        "min_tree_width": int(min_tree_width),
        "tree_width_growth": int(tree_width_growth),
        "tree_width_shrink": int(tree_width_shrink),
        "tree_anchor_target_token": bool(tree_anchor_target_token),
        "tree_anchor_target_continuation": bool(tree_anchor_target_continuation),
        "tokens": total_tokens,
        "target_forwards": total_target_forwards,
        "target_forward_reduction": total_tokens / max(total_target_forwards, 1),
        "drafted_tokens": total_drafted,
        "accepted_tokens": total_accepted,
        "acceptance_rate": total_accepted / max(total_drafted, 1),
        "rejected_blocks": total_rejected,
        "draft_misses": total_misses,
        "full_block_reuses": total_full_block_reuses,
        "bonus_tokens": total_bonus_tokens,
        "deferred_correction_tokens": total_deferred_correction_tokens,
        "tree_candidates": total_tree_candidates,
        "tree_verifies": total_tree_verifies,
        "tree_anchor_candidates": total_tree_anchor_candidates,
        "tree_anchor_verifies": total_tree_anchor_verifies,
        "tree_anchor_accepted_tokens": total_tree_anchor_accepted,
        "tree_anchor_acceptance_rate": total_tree_anchor_accepted / max(total_tree_anchor_candidates, 1),
        "tree_first_token_checks": total_tree_first_token_checks,
        "tree_first_token_misses": total_tree_first_token_misses,
        "tree_first_token_miss_rate": total_tree_first_token_misses / max(total_tree_first_token_checks, 1),
        "mean_tree_candidates": total_tree_candidates / max(total_tree_verifies, 1),
        "mean_tree_width": sum(tree_width_means) / len(tree_width_means) if tree_width_means else 0.0,
        "second_order_action_extrapolation_drafted_tokens": total_second_order_action_extrapolation_drafted,
        "second_order_action_extrapolation_accepted_tokens": total_second_order_action_extrapolation_accepted,
        "second_order_action_extrapolation_acceptance_rate": (
            total_second_order_action_extrapolation_accepted
            / max(total_second_order_action_extrapolation_drafted, 1)
        ),
        "previous_chunk_position_drafted_tokens": total_previous_chunk_position_drafted,
        "previous_chunk_position_accepted_tokens": total_previous_chunk_position_accepted,
        "previous_chunk_position_acceptance_rate": total_previous_chunk_position_accepted
        / max(total_previous_chunk_position_drafted, 1),
        "chunk_prefix_retrieval_drafted_tokens": total_chunk_prefix_retrieval_drafted,
        "chunk_prefix_retrieval_accepted_tokens": total_chunk_prefix_retrieval_accepted,
        "chunk_prefix_retrieval_acceptance_rate": total_chunk_prefix_retrieval_accepted
        / max(total_chunk_prefix_retrieval_drafted, 1),
        "chunk_length_stop_drafted_tokens": total_chunk_length_stop_drafted,
        "chunk_length_stop_accepted_tokens": total_chunk_length_stop_accepted,
        "chunk_length_stop_acceptance_rate": total_chunk_length_stop_accepted
        / max(total_chunk_length_stop_drafted, 1),
        "position_mode_histogram_drafted_tokens": total_position_mode_histogram_drafted,
        "position_mode_histogram_accepted_tokens": total_position_mode_histogram_accepted,
        "position_mode_histogram_acceptance_rate": total_position_mode_histogram_accepted
        / max(total_position_mode_histogram_drafted, 1),
        "global_position_mode_drafted_tokens": total_global_position_mode_drafted,
        "global_position_mode_accepted_tokens": total_global_position_mode_accepted,
        "global_position_mode_acceptance_rate": total_global_position_mode_accepted
        / max(total_global_position_mode_drafted, 1),
        "ngram_continuation_drafted_tokens": total_ngram_continuation_drafted,
        "ngram_continuation_accepted_tokens": total_ngram_continuation_accepted,
        "ngram_continuation_acceptance_rate": total_ngram_continuation_accepted
        / max(total_ngram_continuation_drafted, 1),
        "source_agreement_drafted_tokens": total_source_agreement_drafted,
        "source_agreement_accepted_tokens": total_source_agreement_accepted,
        "source_agreement_acceptance_rate": total_source_agreement_accepted
        / max(total_source_agreement_drafted, 1),
        "action_trend_regression_drafted_tokens": total_action_trend_regression_drafted,
        "action_trend_regression_accepted_tokens": total_action_trend_regression_accepted,
        "action_trend_regression_acceptance_rate": total_action_trend_regression_accepted
        / max(total_action_trend_regression_drafted, 1),
        "action_prefix_lookup_drafted_tokens": total_action_prefix_lookup_drafted,
        "action_prefix_lookup_accepted_tokens": total_action_prefix_lookup_accepted,
        "action_prefix_lookup_acceptance_rate": total_action_prefix_lookup_accepted
        / max(total_action_prefix_lookup_drafted, 1),
        "action_vector_suffix_lookup_drafted_tokens": total_action_vector_suffix_lookup_drafted,
        "action_vector_suffix_lookup_accepted_tokens": total_action_vector_suffix_lookup_accepted,
        "action_vector_suffix_lookup_acceptance_rate": total_action_vector_suffix_lookup_accepted
        / max(total_action_vector_suffix_lookup_drafted, 1),
        "action_vector_transition_drafted_tokens": total_action_vector_transition_drafted,
        "action_vector_transition_accepted_tokens": total_action_vector_transition_accepted,
        "action_vector_transition_acceptance_rate": total_action_vector_transition_accepted
        / max(total_action_vector_transition_drafted, 1),
        "action_repeat_vector_drafted_tokens": total_action_repeat_vector_drafted,
        "action_repeat_vector_accepted_tokens": total_action_repeat_vector_accepted,
        "action_repeat_vector_acceptance_rate": total_action_repeat_vector_accepted
        / max(total_action_repeat_vector_drafted, 1),
        "action_token_neighborhood_drafted_tokens": total_action_token_neighborhood_drafted,
        "action_token_neighborhood_accepted_tokens": total_action_token_neighborhood_accepted,
        "action_token_neighborhood_acceptance_rate": total_action_token_neighborhood_accepted
        / max(total_action_token_neighborhood_drafted, 1),
        "action_context_tree_drafted_tokens": total_action_context_tree_drafted,
        "action_context_tree_accepted_tokens": total_action_context_tree_accepted,
        "action_context_tree_acceptance_rate": total_action_context_tree_accepted
        / max(total_action_context_tree_drafted, 1),
        "action_transition_histogram_drafted_tokens": total_action_transition_histogram_drafted,
        "action_transition_histogram_accepted_tokens": total_action_transition_histogram_accepted,
        "action_transition_histogram_acceptance_rate": total_action_transition_histogram_accepted
        / max(total_action_transition_histogram_drafted, 1),
        "action_delta_histogram_drafted_tokens": total_action_delta_histogram_drafted,
        "action_delta_histogram_accepted_tokens": total_action_delta_histogram_accepted,
        "action_delta_histogram_acceptance_rate": total_action_delta_histogram_accepted
        / max(total_action_delta_histogram_drafted, 1),
        "action_delta_ngram_drafted_tokens": total_action_delta_ngram_drafted,
        "action_delta_ngram_accepted_tokens": total_action_delta_ngram_accepted,
        "action_delta_ngram_acceptance_rate": total_action_delta_ngram_accepted
        / max(total_action_delta_ngram_drafted, 1),
        "chunk_delta_template_drafted_tokens": total_chunk_delta_template_drafted,
        "chunk_delta_template_accepted_tokens": total_chunk_delta_template_accepted,
        "chunk_delta_template_acceptance_rate": total_chunk_delta_template_accepted
        / max(total_chunk_delta_template_drafted, 1),
        "chunk_position_delta_drafted_tokens": total_chunk_position_delta_drafted,
        "chunk_position_delta_accepted_tokens": total_chunk_position_delta_accepted,
        "chunk_position_delta_acceptance_rate": total_chunk_position_delta_accepted
        / max(total_chunk_position_delta_drafted, 1),
        "action_dimension_mode_drafted_tokens": total_action_dimension_mode_drafted,
        "action_dimension_mode_accepted_tokens": total_action_dimension_mode_accepted,
        "action_dimension_mode_acceptance_rate": total_action_dimension_mode_accepted
        / max(total_action_dimension_mode_drafted, 1),
        "hold_action_token_drafted_tokens": total_hold_action_token_drafted,
        "hold_action_token_accepted_tokens": total_hold_action_token_accepted,
        "hold_action_token_acceptance_rate": total_hold_action_token_accepted
        / max(total_hold_action_token_drafted, 1),
        "source_cooldown_events": total_source_cooldown_events,
        "source_cooldown_skipped_sources": total_source_cooldown_skipped_sources,
        "source_acceptance_bias_events": total_source_acceptance_bias_events,
        "source_acceptance_bias_reorders": total_source_acceptance_bias_reorders,
        "mean_lookahead": sum(lookahead_means) / len(lookahead_means),
        "modeled_baseline_ms": baseline_ms,
        "modeled_spec_ms": spec_ms,
        "modeled_speedup": baseline_ms / spec_ms if spec_ms else 0.0,
        "per_task": {},
        "per_trace": per_trace,
    }
    for task_id, values in sorted(per_task.items(), key=lambda item: _task_key_sort_value(item[0])):
        task_tokens = sum(row.tokens for row in values)
        task_forwards = sum(row.target_forwards for row in values)
        task_drafted = sum(row.drafted_tokens for row in values)
        task_accepted = sum(row.accepted_tokens for row in values)
        task_misses = sum(row.draft_misses for row in values)
        task_full_block_reuses = sum(row.full_block_reuses for row in values)
        task_bonus_tokens = sum(row.bonus_tokens for row in values)
        task_deferred_correction_tokens = sum(row.deferred_correction_tokens for row in values)
        task_tree_candidates = sum(row.tree_candidates for row in values)
        task_tree_verifies = sum(row.tree_verifies for row in values)
        task_tree_anchor_candidates = sum(row.tree_anchor_candidates for row in values)
        task_tree_anchor_verifies = sum(row.tree_anchor_verifies for row in values)
        task_tree_anchor_accepted = sum(row.tree_anchor_accepted_tokens for row in values)
        task_tree_first_token_checks = sum(row.tree_first_token_checks for row in values)
        task_tree_first_token_misses = sum(row.tree_first_token_misses for row in values)
        task_second_order_action_extrapolation_drafted = sum(
            row.second_order_action_extrapolation_drafted_tokens for row in values
        )
        task_second_order_action_extrapolation_accepted = sum(
            row.second_order_action_extrapolation_accepted_tokens for row in values
        )
        task_previous_chunk_position_drafted = sum(row.previous_chunk_position_drafted_tokens for row in values)
        task_previous_chunk_position_accepted = sum(row.previous_chunk_position_accepted_tokens for row in values)
        task_chunk_prefix_retrieval_drafted = sum(row.chunk_prefix_retrieval_drafted_tokens for row in values)
        task_chunk_prefix_retrieval_accepted = sum(row.chunk_prefix_retrieval_accepted_tokens for row in values)
        task_chunk_length_stop_drafted = sum(row.chunk_length_stop_drafted_tokens for row in values)
        task_chunk_length_stop_accepted = sum(row.chunk_length_stop_accepted_tokens for row in values)
        task_position_mode_histogram_drafted = sum(row.position_mode_histogram_drafted_tokens for row in values)
        task_position_mode_histogram_accepted = sum(row.position_mode_histogram_accepted_tokens for row in values)
        task_global_position_mode_drafted = sum(row.global_position_mode_drafted_tokens for row in values)
        task_global_position_mode_accepted = sum(row.global_position_mode_accepted_tokens for row in values)
        task_ngram_continuation_drafted = sum(row.ngram_continuation_drafted_tokens for row in values)
        task_ngram_continuation_accepted = sum(row.ngram_continuation_accepted_tokens for row in values)
        task_source_agreement_drafted = sum(row.source_agreement_drafted_tokens for row in values)
        task_source_agreement_accepted = sum(row.source_agreement_accepted_tokens for row in values)
        task_action_trend_regression_drafted = sum(row.action_trend_regression_drafted_tokens for row in values)
        task_action_trend_regression_accepted = sum(row.action_trend_regression_accepted_tokens for row in values)
        task_action_prefix_lookup_drafted = sum(row.action_prefix_lookup_drafted_tokens for row in values)
        task_action_prefix_lookup_accepted = sum(row.action_prefix_lookup_accepted_tokens for row in values)
        task_action_vector_suffix_lookup_drafted = sum(
            row.action_vector_suffix_lookup_drafted_tokens for row in values
        )
        task_action_vector_suffix_lookup_accepted = sum(
            row.action_vector_suffix_lookup_accepted_tokens for row in values
        )
        task_action_vector_transition_drafted = sum(row.action_vector_transition_drafted_tokens for row in values)
        task_action_vector_transition_accepted = sum(row.action_vector_transition_accepted_tokens for row in values)
        task_action_repeat_vector_drafted = sum(row.action_repeat_vector_drafted_tokens for row in values)
        task_action_repeat_vector_accepted = sum(row.action_repeat_vector_accepted_tokens for row in values)
        task_action_token_neighborhood_drafted = sum(row.action_token_neighborhood_drafted_tokens for row in values)
        task_action_token_neighborhood_accepted = sum(row.action_token_neighborhood_accepted_tokens for row in values)
        task_action_context_tree_drafted = sum(row.action_context_tree_drafted_tokens for row in values)
        task_action_context_tree_accepted = sum(row.action_context_tree_accepted_tokens for row in values)
        task_action_transition_histogram_drafted = sum(
            row.action_transition_histogram_drafted_tokens for row in values
        )
        task_action_transition_histogram_accepted = sum(
            row.action_transition_histogram_accepted_tokens for row in values
        )
        task_action_delta_histogram_drafted = sum(row.action_delta_histogram_drafted_tokens for row in values)
        task_action_delta_histogram_accepted = sum(row.action_delta_histogram_accepted_tokens for row in values)
        task_action_delta_ngram_drafted = sum(row.action_delta_ngram_drafted_tokens for row in values)
        task_action_delta_ngram_accepted = sum(row.action_delta_ngram_accepted_tokens for row in values)
        task_chunk_delta_template_drafted = sum(row.chunk_delta_template_drafted_tokens for row in values)
        task_chunk_delta_template_accepted = sum(row.chunk_delta_template_accepted_tokens for row in values)
        task_chunk_position_delta_drafted = sum(row.chunk_position_delta_drafted_tokens for row in values)
        task_chunk_position_delta_accepted = sum(row.chunk_position_delta_accepted_tokens for row in values)
        task_action_dimension_mode_drafted = sum(row.action_dimension_mode_drafted_tokens for row in values)
        task_action_dimension_mode_accepted = sum(row.action_dimension_mode_accepted_tokens for row in values)
        task_hold_action_token_drafted = sum(row.hold_action_token_drafted_tokens for row in values)
        task_hold_action_token_accepted = sum(row.hold_action_token_accepted_tokens for row in values)
        task_source_cooldown_events = sum(row.source_cooldown_events for row in values)
        task_source_cooldown_skipped_sources = sum(row.source_cooldown_skipped_sources for row in values)
        task_source_acceptance_bias_events = sum(row.source_acceptance_bias_events for row in values)
        task_source_acceptance_bias_reorders = sum(row.source_acceptance_bias_reorders for row in values)
        metrics["per_task"][str(task_id)] = {
            "traces": len(values),
            "tokens": task_tokens,
            "target_forwards": task_forwards,
            "target_forward_reduction": task_tokens / max(task_forwards, 1),
            "acceptance_rate": task_accepted / max(task_drafted, 1),
            "draft_miss_rate": task_misses / max(len(values), 1),
            "full_block_reuses": task_full_block_reuses,
            "bonus_tokens": task_bonus_tokens,
            "deferred_correction_tokens": task_deferred_correction_tokens,
            "tree_candidates": task_tree_candidates,
            "tree_verifies": task_tree_verifies,
            "tree_anchor_candidates": task_tree_anchor_candidates,
            "tree_anchor_verifies": task_tree_anchor_verifies,
            "tree_anchor_accepted_tokens": task_tree_anchor_accepted,
            "tree_anchor_acceptance_rate": task_tree_anchor_accepted / max(task_tree_anchor_candidates, 1),
            "tree_first_token_checks": task_tree_first_token_checks,
            "tree_first_token_misses": task_tree_first_token_misses,
            "tree_first_token_miss_rate": task_tree_first_token_misses / max(task_tree_first_token_checks, 1),
            "mean_tree_candidates": task_tree_candidates / max(task_tree_verifies, 1),
            "mean_tree_width": sum(row.mean_tree_width for row in values) / max(len(values), 1),
            "second_order_action_extrapolation_drafted_tokens": (
                task_second_order_action_extrapolation_drafted
            ),
            "second_order_action_extrapolation_accepted_tokens": (
                task_second_order_action_extrapolation_accepted
            ),
            "second_order_action_extrapolation_acceptance_rate": (
                task_second_order_action_extrapolation_accepted
                / max(task_second_order_action_extrapolation_drafted, 1)
            ),
            "previous_chunk_position_drafted_tokens": task_previous_chunk_position_drafted,
            "previous_chunk_position_accepted_tokens": task_previous_chunk_position_accepted,
            "previous_chunk_position_acceptance_rate": task_previous_chunk_position_accepted
            / max(task_previous_chunk_position_drafted, 1),
            "chunk_prefix_retrieval_drafted_tokens": task_chunk_prefix_retrieval_drafted,
            "chunk_prefix_retrieval_accepted_tokens": task_chunk_prefix_retrieval_accepted,
            "chunk_prefix_retrieval_acceptance_rate": task_chunk_prefix_retrieval_accepted
            / max(task_chunk_prefix_retrieval_drafted, 1),
            "chunk_length_stop_drafted_tokens": task_chunk_length_stop_drafted,
            "chunk_length_stop_accepted_tokens": task_chunk_length_stop_accepted,
            "chunk_length_stop_acceptance_rate": task_chunk_length_stop_accepted
            / max(task_chunk_length_stop_drafted, 1),
            "position_mode_histogram_drafted_tokens": task_position_mode_histogram_drafted,
            "position_mode_histogram_accepted_tokens": task_position_mode_histogram_accepted,
            "position_mode_histogram_acceptance_rate": task_position_mode_histogram_accepted
            / max(task_position_mode_histogram_drafted, 1),
            "global_position_mode_drafted_tokens": task_global_position_mode_drafted,
            "global_position_mode_accepted_tokens": task_global_position_mode_accepted,
            "global_position_mode_acceptance_rate": task_global_position_mode_accepted
            / max(task_global_position_mode_drafted, 1),
            "ngram_continuation_drafted_tokens": task_ngram_continuation_drafted,
            "ngram_continuation_accepted_tokens": task_ngram_continuation_accepted,
            "ngram_continuation_acceptance_rate": task_ngram_continuation_accepted
            / max(task_ngram_continuation_drafted, 1),
            "source_agreement_drafted_tokens": task_source_agreement_drafted,
            "source_agreement_accepted_tokens": task_source_agreement_accepted,
            "source_agreement_acceptance_rate": task_source_agreement_accepted
            / max(task_source_agreement_drafted, 1),
            "action_trend_regression_drafted_tokens": task_action_trend_regression_drafted,
            "action_trend_regression_accepted_tokens": task_action_trend_regression_accepted,
            "action_trend_regression_acceptance_rate": task_action_trend_regression_accepted
            / max(task_action_trend_regression_drafted, 1),
            "action_prefix_lookup_drafted_tokens": task_action_prefix_lookup_drafted,
            "action_prefix_lookup_accepted_tokens": task_action_prefix_lookup_accepted,
            "action_prefix_lookup_acceptance_rate": task_action_prefix_lookup_accepted
            / max(task_action_prefix_lookup_drafted, 1),
            "action_vector_suffix_lookup_drafted_tokens": task_action_vector_suffix_lookup_drafted,
            "action_vector_suffix_lookup_accepted_tokens": task_action_vector_suffix_lookup_accepted,
            "action_vector_suffix_lookup_acceptance_rate": task_action_vector_suffix_lookup_accepted
            / max(task_action_vector_suffix_lookup_drafted, 1),
            "action_vector_transition_drafted_tokens": task_action_vector_transition_drafted,
            "action_vector_transition_accepted_tokens": task_action_vector_transition_accepted,
            "action_vector_transition_acceptance_rate": task_action_vector_transition_accepted
            / max(task_action_vector_transition_drafted, 1),
            "action_repeat_vector_drafted_tokens": task_action_repeat_vector_drafted,
            "action_repeat_vector_accepted_tokens": task_action_repeat_vector_accepted,
            "action_repeat_vector_acceptance_rate": task_action_repeat_vector_accepted
            / max(task_action_repeat_vector_drafted, 1),
            "action_token_neighborhood_drafted_tokens": task_action_token_neighborhood_drafted,
            "action_token_neighborhood_accepted_tokens": task_action_token_neighborhood_accepted,
            "action_token_neighborhood_acceptance_rate": task_action_token_neighborhood_accepted
            / max(task_action_token_neighborhood_drafted, 1),
            "action_context_tree_drafted_tokens": task_action_context_tree_drafted,
            "action_context_tree_accepted_tokens": task_action_context_tree_accepted,
            "action_context_tree_acceptance_rate": task_action_context_tree_accepted
            / max(task_action_context_tree_drafted, 1),
            "action_transition_histogram_drafted_tokens": task_action_transition_histogram_drafted,
            "action_transition_histogram_accepted_tokens": task_action_transition_histogram_accepted,
            "action_transition_histogram_acceptance_rate": task_action_transition_histogram_accepted
            / max(task_action_transition_histogram_drafted, 1),
            "action_delta_histogram_drafted_tokens": task_action_delta_histogram_drafted,
            "action_delta_histogram_accepted_tokens": task_action_delta_histogram_accepted,
            "action_delta_histogram_acceptance_rate": task_action_delta_histogram_accepted
            / max(task_action_delta_histogram_drafted, 1),
            "action_delta_ngram_drafted_tokens": task_action_delta_ngram_drafted,
            "action_delta_ngram_accepted_tokens": task_action_delta_ngram_accepted,
            "action_delta_ngram_acceptance_rate": task_action_delta_ngram_accepted
            / max(task_action_delta_ngram_drafted, 1),
            "chunk_delta_template_drafted_tokens": task_chunk_delta_template_drafted,
            "chunk_delta_template_accepted_tokens": task_chunk_delta_template_accepted,
            "chunk_delta_template_acceptance_rate": task_chunk_delta_template_accepted
            / max(task_chunk_delta_template_drafted, 1),
            "chunk_position_delta_drafted_tokens": task_chunk_position_delta_drafted,
            "chunk_position_delta_accepted_tokens": task_chunk_position_delta_accepted,
            "chunk_position_delta_acceptance_rate": task_chunk_position_delta_accepted
            / max(task_chunk_position_delta_drafted, 1),
            "action_dimension_mode_drafted_tokens": task_action_dimension_mode_drafted,
            "action_dimension_mode_accepted_tokens": task_action_dimension_mode_accepted,
            "action_dimension_mode_acceptance_rate": task_action_dimension_mode_accepted
            / max(task_action_dimension_mode_drafted, 1),
            "hold_action_token_drafted_tokens": task_hold_action_token_drafted,
            "hold_action_token_accepted_tokens": task_hold_action_token_accepted,
            "hold_action_token_acceptance_rate": task_hold_action_token_accepted
            / max(task_hold_action_token_drafted, 1),
            "source_cooldown_events": task_source_cooldown_events,
            "source_cooldown_skipped_sources": task_source_cooldown_skipped_sources,
            "source_acceptance_bias_events": task_source_acceptance_bias_events,
            "source_acceptance_bias_reorders": task_source_acceptance_bias_reorders,
        }
    task_rows = list(metrics["per_task"].values())
    task_forward_reductions = [float(row["target_forward_reduction"]) for row in task_rows]
    task_acceptance_rates = [float(row["acceptance_rate"]) for row in task_rows]
    task_trace_counts = [int(row["traces"]) for row in task_rows]
    task_tree_anchor_accepted = [float(row["tree_anchor_accepted_tokens"]) for row in task_rows]
    task_second_order_action_extrapolation_drafted = [
        float(row["second_order_action_extrapolation_drafted_tokens"]) for row in task_rows
    ]
    task_second_order_action_extrapolation_accepted = [
        float(row["second_order_action_extrapolation_accepted_tokens"]) for row in task_rows
    ]
    task_second_order_action_extrapolation_acceptance = [
        float(row["second_order_action_extrapolation_acceptance_rate"]) for row in task_rows
    ]
    task_previous_chunk_position_drafted = [float(row["previous_chunk_position_drafted_tokens"]) for row in task_rows]
    task_previous_chunk_position_accepted = [float(row["previous_chunk_position_accepted_tokens"]) for row in task_rows]
    task_previous_chunk_position_acceptance = [
        float(row["previous_chunk_position_acceptance_rate"]) for row in task_rows
    ]
    task_chunk_prefix_retrieval_drafted = [
        float(row["chunk_prefix_retrieval_drafted_tokens"]) for row in task_rows
    ]
    task_chunk_prefix_retrieval_accepted = [
        float(row["chunk_prefix_retrieval_accepted_tokens"]) for row in task_rows
    ]
    task_chunk_prefix_retrieval_acceptance = [
        float(row["chunk_prefix_retrieval_acceptance_rate"]) for row in task_rows
    ]
    task_chunk_length_stop_drafted = [float(row["chunk_length_stop_drafted_tokens"]) for row in task_rows]
    task_chunk_length_stop_accepted = [float(row["chunk_length_stop_accepted_tokens"]) for row in task_rows]
    task_chunk_length_stop_acceptance = [
        float(row["chunk_length_stop_acceptance_rate"]) for row in task_rows
    ]
    task_position_mode_histogram_drafted = [float(row["position_mode_histogram_drafted_tokens"]) for row in task_rows]
    task_position_mode_histogram_accepted = [float(row["position_mode_histogram_accepted_tokens"]) for row in task_rows]
    task_position_mode_histogram_acceptance = [
        float(row["position_mode_histogram_acceptance_rate"]) for row in task_rows
    ]
    task_global_position_mode_drafted = [float(row["global_position_mode_drafted_tokens"]) for row in task_rows]
    task_global_position_mode_accepted = [float(row["global_position_mode_accepted_tokens"]) for row in task_rows]
    task_global_position_mode_acceptance = [
        float(row["global_position_mode_acceptance_rate"]) for row in task_rows
    ]
    task_ngram_continuation_drafted = [float(row["ngram_continuation_drafted_tokens"]) for row in task_rows]
    task_ngram_continuation_accepted = [float(row["ngram_continuation_accepted_tokens"]) for row in task_rows]
    task_ngram_continuation_acceptance = [float(row["ngram_continuation_acceptance_rate"]) for row in task_rows]
    task_source_agreement_drafted = [float(row["source_agreement_drafted_tokens"]) for row in task_rows]
    task_source_agreement_accepted = [float(row["source_agreement_accepted_tokens"]) for row in task_rows]
    task_source_agreement_acceptance = [float(row["source_agreement_acceptance_rate"]) for row in task_rows]
    task_action_trend_regression_drafted = [
        float(row["action_trend_regression_drafted_tokens"]) for row in task_rows
    ]
    task_action_trend_regression_accepted = [
        float(row["action_trend_regression_accepted_tokens"]) for row in task_rows
    ]
    task_action_trend_regression_acceptance = [
        float(row["action_trend_regression_acceptance_rate"]) for row in task_rows
    ]
    task_action_prefix_lookup_drafted = [
        float(row["action_prefix_lookup_drafted_tokens"]) for row in task_rows
    ]
    task_action_prefix_lookup_accepted = [
        float(row["action_prefix_lookup_accepted_tokens"]) for row in task_rows
    ]
    task_action_prefix_lookup_acceptance = [
        float(row["action_prefix_lookup_acceptance_rate"]) for row in task_rows
    ]
    task_action_vector_suffix_lookup_drafted = [
        float(row["action_vector_suffix_lookup_drafted_tokens"]) for row in task_rows
    ]
    task_action_vector_suffix_lookup_accepted = [
        float(row["action_vector_suffix_lookup_accepted_tokens"]) for row in task_rows
    ]
    task_action_vector_suffix_lookup_acceptance = [
        float(row["action_vector_suffix_lookup_acceptance_rate"]) for row in task_rows
    ]
    task_action_vector_transition_drafted = [
        float(row["action_vector_transition_drafted_tokens"]) for row in task_rows
    ]
    task_action_vector_transition_accepted = [
        float(row["action_vector_transition_accepted_tokens"]) for row in task_rows
    ]
    task_action_vector_transition_acceptance = [
        float(row["action_vector_transition_acceptance_rate"]) for row in task_rows
    ]
    task_action_repeat_vector_drafted = [
        float(row["action_repeat_vector_drafted_tokens"]) for row in task_rows
    ]
    task_action_repeat_vector_accepted = [
        float(row["action_repeat_vector_accepted_tokens"]) for row in task_rows
    ]
    task_action_repeat_vector_acceptance = [
        float(row["action_repeat_vector_acceptance_rate"]) for row in task_rows
    ]
    task_action_token_neighborhood_drafted = [
        float(row["action_token_neighborhood_drafted_tokens"]) for row in task_rows
    ]
    task_action_token_neighborhood_accepted = [
        float(row["action_token_neighborhood_accepted_tokens"]) for row in task_rows
    ]
    task_action_token_neighborhood_acceptance = [
        float(row["action_token_neighborhood_acceptance_rate"]) for row in task_rows
    ]
    task_action_context_tree_drafted = [float(row["action_context_tree_drafted_tokens"]) for row in task_rows]
    task_action_context_tree_accepted = [float(row["action_context_tree_accepted_tokens"]) for row in task_rows]
    task_action_context_tree_acceptance = [float(row["action_context_tree_acceptance_rate"]) for row in task_rows]
    task_action_transition_histogram_drafted = [
        float(row["action_transition_histogram_drafted_tokens"]) for row in task_rows
    ]
    task_action_transition_histogram_accepted = [
        float(row["action_transition_histogram_accepted_tokens"]) for row in task_rows
    ]
    task_action_transition_histogram_acceptance = [
        float(row["action_transition_histogram_acceptance_rate"]) for row in task_rows
    ]
    task_action_delta_histogram_drafted = [float(row["action_delta_histogram_drafted_tokens"]) for row in task_rows]
    task_action_delta_histogram_accepted = [float(row["action_delta_histogram_accepted_tokens"]) for row in task_rows]
    task_action_delta_histogram_acceptance = [
        float(row["action_delta_histogram_acceptance_rate"]) for row in task_rows
    ]
    task_action_delta_ngram_drafted = [float(row["action_delta_ngram_drafted_tokens"]) for row in task_rows]
    task_action_delta_ngram_accepted = [float(row["action_delta_ngram_accepted_tokens"]) for row in task_rows]
    task_action_delta_ngram_acceptance = [
        float(row["action_delta_ngram_acceptance_rate"]) for row in task_rows
    ]
    task_chunk_delta_template_drafted = [float(row["chunk_delta_template_drafted_tokens"]) for row in task_rows]
    task_chunk_delta_template_accepted = [float(row["chunk_delta_template_accepted_tokens"]) for row in task_rows]
    task_chunk_delta_template_acceptance = [
        float(row["chunk_delta_template_acceptance_rate"]) for row in task_rows
    ]
    task_chunk_position_delta_drafted = [float(row["chunk_position_delta_drafted_tokens"]) for row in task_rows]
    task_chunk_position_delta_accepted = [float(row["chunk_position_delta_accepted_tokens"]) for row in task_rows]
    task_chunk_position_delta_acceptance = [
        float(row["chunk_position_delta_acceptance_rate"]) for row in task_rows
    ]
    task_action_dimension_mode_drafted = [float(row["action_dimension_mode_drafted_tokens"]) for row in task_rows]
    task_action_dimension_mode_accepted = [float(row["action_dimension_mode_accepted_tokens"]) for row in task_rows]
    task_action_dimension_mode_acceptance = [
        float(row["action_dimension_mode_acceptance_rate"]) for row in task_rows
    ]
    task_hold_action_token_drafted = [float(row["hold_action_token_drafted_tokens"]) for row in task_rows]
    task_hold_action_token_accepted = [float(row["hold_action_token_accepted_tokens"]) for row in task_rows]
    task_hold_action_token_acceptance = [float(row["hold_action_token_acceptance_rate"]) for row in task_rows]
    task_source_cooldown_events = [float(row["source_cooldown_events"]) for row in task_rows]
    task_source_cooldown_skipped_sources = [float(row["source_cooldown_skipped_sources"]) for row in task_rows]
    task_source_acceptance_bias_events = [float(row["source_acceptance_bias_events"]) for row in task_rows]
    task_source_acceptance_bias_reorders = [float(row["source_acceptance_bias_reorders"]) for row in task_rows]
    metrics.update(
        {
            "task_count": len(task_rows),
            "min_task_target_forward_reduction": min(task_forward_reductions),
            "mean_task_target_forward_reduction": sum(task_forward_reductions) / len(task_forward_reductions),
            "min_task_acceptance_rate": min(task_acceptance_rates),
            "mean_task_acceptance_rate": sum(task_acceptance_rates) / len(task_acceptance_rates),
            "min_task_traces": min(task_trace_counts),
            "min_task_tree_anchor_accepted_tokens": min(task_tree_anchor_accepted),
            "min_task_second_order_action_extrapolation_drafted_tokens": min(
                task_second_order_action_extrapolation_drafted
            ),
            "min_task_second_order_action_extrapolation_accepted_tokens": min(
                task_second_order_action_extrapolation_accepted
            ),
            "min_task_second_order_action_extrapolation_acceptance_rate": min(
                task_second_order_action_extrapolation_acceptance
            ),
            "mean_task_second_order_action_extrapolation_acceptance_rate": sum(
                task_second_order_action_extrapolation_acceptance
            )
            / len(task_second_order_action_extrapolation_acceptance),
            "min_task_previous_chunk_position_drafted_tokens": min(task_previous_chunk_position_drafted),
            "min_task_previous_chunk_position_accepted_tokens": min(task_previous_chunk_position_accepted),
            "min_task_previous_chunk_position_acceptance_rate": min(task_previous_chunk_position_acceptance),
            "mean_task_previous_chunk_position_acceptance_rate": sum(task_previous_chunk_position_acceptance)
            / len(task_previous_chunk_position_acceptance),
            "min_task_chunk_prefix_retrieval_drafted_tokens": min(task_chunk_prefix_retrieval_drafted),
            "min_task_chunk_prefix_retrieval_accepted_tokens": min(task_chunk_prefix_retrieval_accepted),
            "min_task_chunk_prefix_retrieval_acceptance_rate": min(task_chunk_prefix_retrieval_acceptance),
            "mean_task_chunk_prefix_retrieval_acceptance_rate": sum(task_chunk_prefix_retrieval_acceptance)
            / len(task_chunk_prefix_retrieval_acceptance),
            "min_task_chunk_length_stop_drafted_tokens": min(task_chunk_length_stop_drafted),
            "min_task_chunk_length_stop_accepted_tokens": min(task_chunk_length_stop_accepted),
            "min_task_chunk_length_stop_acceptance_rate": min(task_chunk_length_stop_acceptance),
            "mean_task_chunk_length_stop_acceptance_rate": sum(task_chunk_length_stop_acceptance)
            / len(task_chunk_length_stop_acceptance),
            "min_task_position_mode_histogram_drafted_tokens": min(task_position_mode_histogram_drafted),
            "min_task_position_mode_histogram_accepted_tokens": min(task_position_mode_histogram_accepted),
            "min_task_position_mode_histogram_acceptance_rate": min(task_position_mode_histogram_acceptance),
            "mean_task_position_mode_histogram_acceptance_rate": sum(task_position_mode_histogram_acceptance)
            / len(task_position_mode_histogram_acceptance),
            "min_task_global_position_mode_drafted_tokens": min(task_global_position_mode_drafted),
            "min_task_global_position_mode_accepted_tokens": min(task_global_position_mode_accepted),
            "min_task_global_position_mode_acceptance_rate": min(task_global_position_mode_acceptance),
            "mean_task_global_position_mode_acceptance_rate": sum(task_global_position_mode_acceptance)
            / len(task_global_position_mode_acceptance),
            "min_task_ngram_continuation_drafted_tokens": min(task_ngram_continuation_drafted),
            "min_task_ngram_continuation_accepted_tokens": min(task_ngram_continuation_accepted),
            "min_task_ngram_continuation_acceptance_rate": min(task_ngram_continuation_acceptance),
            "mean_task_ngram_continuation_acceptance_rate": sum(task_ngram_continuation_acceptance)
            / len(task_ngram_continuation_acceptance),
            "min_task_source_agreement_drafted_tokens": min(task_source_agreement_drafted),
            "min_task_source_agreement_accepted_tokens": min(task_source_agreement_accepted),
            "min_task_source_agreement_acceptance_rate": min(task_source_agreement_acceptance),
            "mean_task_source_agreement_acceptance_rate": sum(task_source_agreement_acceptance)
            / len(task_source_agreement_acceptance),
            "min_task_action_trend_regression_drafted_tokens": min(task_action_trend_regression_drafted),
            "min_task_action_trend_regression_accepted_tokens": min(task_action_trend_regression_accepted),
            "min_task_action_trend_regression_acceptance_rate": min(task_action_trend_regression_acceptance),
            "mean_task_action_trend_regression_acceptance_rate": sum(task_action_trend_regression_acceptance)
            / len(task_action_trend_regression_acceptance),
            "min_task_action_prefix_lookup_drafted_tokens": min(task_action_prefix_lookup_drafted),
            "min_task_action_prefix_lookup_accepted_tokens": min(task_action_prefix_lookup_accepted),
            "min_task_action_prefix_lookup_acceptance_rate": min(task_action_prefix_lookup_acceptance),
            "mean_task_action_prefix_lookup_acceptance_rate": sum(task_action_prefix_lookup_acceptance)
            / len(task_action_prefix_lookup_acceptance),
            "min_task_action_vector_suffix_lookup_drafted_tokens": min(
                task_action_vector_suffix_lookup_drafted
            ),
            "min_task_action_vector_suffix_lookup_accepted_tokens": min(
                task_action_vector_suffix_lookup_accepted
            ),
            "min_task_action_vector_suffix_lookup_acceptance_rate": min(
                task_action_vector_suffix_lookup_acceptance
            ),
            "mean_task_action_vector_suffix_lookup_acceptance_rate": sum(
                task_action_vector_suffix_lookup_acceptance
            )
            / len(task_action_vector_suffix_lookup_acceptance),
            "min_task_action_vector_transition_drafted_tokens": min(task_action_vector_transition_drafted),
            "min_task_action_vector_transition_accepted_tokens": min(task_action_vector_transition_accepted),
            "min_task_action_vector_transition_acceptance_rate": min(task_action_vector_transition_acceptance),
            "mean_task_action_vector_transition_acceptance_rate": sum(task_action_vector_transition_acceptance)
            / len(task_action_vector_transition_acceptance),
            "min_task_action_repeat_vector_drafted_tokens": min(task_action_repeat_vector_drafted),
            "min_task_action_repeat_vector_accepted_tokens": min(task_action_repeat_vector_accepted),
            "min_task_action_repeat_vector_acceptance_rate": min(task_action_repeat_vector_acceptance),
            "mean_task_action_repeat_vector_acceptance_rate": sum(task_action_repeat_vector_acceptance)
            / len(task_action_repeat_vector_acceptance),
            "min_task_action_token_neighborhood_drafted_tokens": min(task_action_token_neighborhood_drafted),
            "min_task_action_token_neighborhood_accepted_tokens": min(task_action_token_neighborhood_accepted),
            "min_task_action_token_neighborhood_acceptance_rate": min(task_action_token_neighborhood_acceptance),
            "mean_task_action_token_neighborhood_acceptance_rate": sum(task_action_token_neighborhood_acceptance)
            / len(task_action_token_neighborhood_acceptance),
            "min_task_action_context_tree_drafted_tokens": min(task_action_context_tree_drafted),
            "min_task_action_context_tree_accepted_tokens": min(task_action_context_tree_accepted),
            "min_task_action_context_tree_acceptance_rate": min(task_action_context_tree_acceptance),
            "mean_task_action_context_tree_acceptance_rate": sum(task_action_context_tree_acceptance)
            / len(task_action_context_tree_acceptance),
            "min_task_action_transition_histogram_drafted_tokens": min(task_action_transition_histogram_drafted),
            "min_task_action_transition_histogram_accepted_tokens": min(task_action_transition_histogram_accepted),
            "min_task_action_transition_histogram_acceptance_rate": min(task_action_transition_histogram_acceptance),
            "mean_task_action_transition_histogram_acceptance_rate": sum(task_action_transition_histogram_acceptance)
            / len(task_action_transition_histogram_acceptance),
            "min_task_action_delta_histogram_drafted_tokens": min(task_action_delta_histogram_drafted),
            "min_task_action_delta_histogram_accepted_tokens": min(task_action_delta_histogram_accepted),
            "min_task_action_delta_histogram_acceptance_rate": min(task_action_delta_histogram_acceptance),
            "mean_task_action_delta_histogram_acceptance_rate": sum(task_action_delta_histogram_acceptance)
            / len(task_action_delta_histogram_acceptance),
            "min_task_action_delta_ngram_drafted_tokens": min(task_action_delta_ngram_drafted),
            "min_task_action_delta_ngram_accepted_tokens": min(task_action_delta_ngram_accepted),
            "min_task_action_delta_ngram_acceptance_rate": min(task_action_delta_ngram_acceptance),
            "mean_task_action_delta_ngram_acceptance_rate": sum(task_action_delta_ngram_acceptance)
            / len(task_action_delta_ngram_acceptance),
            "min_task_chunk_delta_template_drafted_tokens": min(task_chunk_delta_template_drafted),
            "min_task_chunk_delta_template_accepted_tokens": min(task_chunk_delta_template_accepted),
            "min_task_chunk_delta_template_acceptance_rate": min(task_chunk_delta_template_acceptance),
            "mean_task_chunk_delta_template_acceptance_rate": sum(task_chunk_delta_template_acceptance)
            / len(task_chunk_delta_template_acceptance),
            "min_task_chunk_position_delta_drafted_tokens": min(task_chunk_position_delta_drafted),
            "min_task_chunk_position_delta_accepted_tokens": min(task_chunk_position_delta_accepted),
            "min_task_chunk_position_delta_acceptance_rate": min(task_chunk_position_delta_acceptance),
            "mean_task_chunk_position_delta_acceptance_rate": sum(task_chunk_position_delta_acceptance)
            / len(task_chunk_position_delta_acceptance),
            "min_task_action_dimension_mode_drafted_tokens": min(task_action_dimension_mode_drafted),
            "min_task_action_dimension_mode_accepted_tokens": min(task_action_dimension_mode_accepted),
            "min_task_action_dimension_mode_acceptance_rate": min(task_action_dimension_mode_acceptance),
            "mean_task_action_dimension_mode_acceptance_rate": sum(task_action_dimension_mode_acceptance)
            / len(task_action_dimension_mode_acceptance),
            "min_task_hold_action_token_drafted_tokens": min(task_hold_action_token_drafted),
            "min_task_hold_action_token_accepted_tokens": min(task_hold_action_token_accepted),
            "min_task_hold_action_token_acceptance_rate": min(task_hold_action_token_acceptance),
            "mean_task_hold_action_token_acceptance_rate": sum(task_hold_action_token_acceptance)
            / len(task_hold_action_token_acceptance),
            "min_task_source_cooldown_events": min(task_source_cooldown_events),
            "mean_task_source_cooldown_events": sum(task_source_cooldown_events)
            / len(task_source_cooldown_events),
            "min_task_source_cooldown_skipped_sources": min(task_source_cooldown_skipped_sources),
            "mean_task_source_cooldown_skipped_sources": sum(task_source_cooldown_skipped_sources)
            / len(task_source_cooldown_skipped_sources),
            "min_task_source_acceptance_bias_events": min(task_source_acceptance_bias_events),
            "mean_task_source_acceptance_bias_events": sum(task_source_acceptance_bias_events)
            / len(task_source_acceptance_bias_events),
            "min_task_source_acceptance_bias_reorders": min(task_source_acceptance_bias_reorders),
            "mean_task_source_acceptance_bias_reorders": sum(task_source_acceptance_bias_reorders)
            / len(task_source_acceptance_bias_reorders),
        }
    )
    return metrics
