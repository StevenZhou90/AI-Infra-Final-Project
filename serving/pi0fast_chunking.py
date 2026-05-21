"""Chunk execution helpers for π0-FAST style action-token policies.

The code in this module is intentionally policy-agnostic: it operates on
continuous action chunks and optional FAST token sequences, so it can be tested
without importing LeRobot or downloading π0-FAST weights.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np


def _as_action_chunk(actions: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [horizon, action_dim], got {arr.shape}")
    return arr


@dataclass
class ChunkGuardConfig:
    """Thresholds for robot-aware chunk execution."""

    action_dim: int = 7
    default_execute_window: int = 3
    smooth_execute_window: int = 5
    max_execute_window: int = 6
    min_confidence: float = 0.75
    position_error_threshold: float = 0.035
    rotation_error_threshold: float = 0.25
    action_jump_threshold: float = 0.18
    jerk_threshold: float = 0.22
    gripper_change_threshold: float = 0.5
    action_bound: float = 1.05
    smooth_position_delta: float = 0.045
    smooth_rotation_delta: float = 0.18
    gripper_index: int = 6

    def __post_init__(self) -> None:
        if self.default_execute_window < 1:
            raise ValueError("default_execute_window must be >= 1")
        if self.smooth_execute_window < self.default_execute_window:
            raise ValueError("smooth_execute_window must be >= default_execute_window")
        if self.max_execute_window < self.smooth_execute_window:
            raise ValueError("max_execute_window must be >= smooth_execute_window")


@dataclass
class ChunkDecision:
    accepted: bool
    execute_window: int
    reasons: list[str] = field(default_factory=list)
    smooth_free_space: bool = False


@dataclass
class ChunkExecutionStats:
    chunks_seen: int = 0
    chunks_accepted: int = 0
    actions_enqueued: int = 0
    actions_executed: int = 0
    model_calls: int = 0
    guard_reasons: Counter = field(default_factory=Counter)
    accepted_windows: list[int] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    exact_verifies: int = 0
    exact_drafted_tokens: int = 0
    exact_accepted_tokens: int = 0
    action_max_diffs: list[float] = field(default_factory=list)
    action_mean_diffs: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        avg_ms = float(np.mean(self.inference_ms)) if self.inference_ms else 0.0
        avg_window = float(np.mean(self.accepted_windows)) if self.accepted_windows else 0.0
        avg_tokens = float(np.mean(self.token_counts)) if self.token_counts else 0.0
        return {
            "chunks_seen": self.chunks_seen,
            "chunks_accepted": self.chunks_accepted,
            "chunk_accept_rate": self.chunks_accepted / max(self.chunks_seen, 1),
            "actions_enqueued": self.actions_enqueued,
            "actions_executed": self.actions_executed,
            "model_calls": self.model_calls,
            "model_calls_per_action": self.model_calls / max(self.actions_executed, 1),
            "avg_model_call_ms": avg_ms,
            "avg_accepted_window": avg_window,
            "avg_fast_token_count": avg_tokens,
            "exact_verifies": self.exact_verifies,
            "exact_drafted_tokens": self.exact_drafted_tokens,
            "exact_accepted_tokens": self.exact_accepted_tokens,
            "exact_token_acceptance_rate": self.exact_accepted_tokens / max(self.exact_drafted_tokens, 1),
            "max_action_diff": float(np.max(self.action_max_diffs)) if self.action_max_diffs else 0.0,
            "mean_action_diff": float(np.mean(self.action_mean_diffs)) if self.action_mean_diffs else 0.0,
            "guard_reasons": dict(self.guard_reasons),
        }


class ChunkGuard:
    """Validate and size a receding-horizon execution window."""

    def __init__(self, config: ChunkGuardConfig | None = None) -> None:
        self.config = config or ChunkGuardConfig()
        self.previous_action: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_action = None

    def decide(
        self,
        chunk: np.ndarray | Iterable[Iterable[float]],
        *,
        confidence: float | None = None,
        contact_phase_change: bool = False,
        reference_chunk: np.ndarray | None = None,
        relaxed: bool = False,
    ) -> ChunkDecision:
        cfg = self.config
        actions = _as_action_chunk(chunk)
        if actions.shape[1] < cfg.action_dim:
            raise ValueError(f"Expected at least {cfg.action_dim} action dims, got {actions.shape[1]}")

        reasons: list[str] = []
        if not np.all(np.isfinite(actions)):
            reasons.append("nonfinite_action")
        if np.max(np.abs(actions[:, : cfg.action_dim])) > cfg.action_bound:
            reasons.append("action_bounds")
        if confidence is not None and confidence < cfg.min_confidence:
            reasons.append("low_confidence")
        if contact_phase_change:
            reasons.append("contact_phase_change")
        if self._has_gripper_change(actions):
            reasons.append("gripper_change")
        if self._max_step_delta(actions) > cfg.action_jump_threshold:
            reasons.append("large_action_jump")
        if self._max_jerk(actions) > cfg.jerk_threshold:
            reasons.append("high_jerk")
        if self.previous_action is not None:
            first_delta = float(np.linalg.norm(actions[0, : cfg.action_dim] - self.previous_action[: cfg.action_dim]))
            if first_delta > cfg.action_jump_threshold:
                reasons.append("large_first_action_jump")
        if relaxed and reference_chunk is not None:
            reasons.extend(self._relaxed_reference_failures(actions, reference_chunk))

        accepted = not reasons
        smooth = accepted and self._is_smooth_free_space(actions)
        window = cfg.smooth_execute_window if smooth else cfg.default_execute_window
        window = min(window, cfg.max_execute_window, actions.shape[0])
        return ChunkDecision(accepted=accepted, execute_window=window if accepted else 0, reasons=reasons, smooth_free_space=smooth)

    def mark_executed(self, action: np.ndarray) -> None:
        self.previous_action = np.asarray(action, dtype=np.float32).copy()

    def _has_gripper_change(self, actions: np.ndarray) -> bool:
        gi = self.config.gripper_index
        if gi >= actions.shape[1]:
            return False
        values = actions[:, gi]
        if values.size <= 1:
            return False
        return bool(np.max(np.abs(np.diff(values))) > self.config.gripper_change_threshold)

    def _max_step_delta(self, actions: np.ndarray) -> float:
        if len(actions) <= 1:
            return 0.0
        return float(np.max(np.linalg.norm(np.diff(actions[:, : self.config.action_dim], axis=0), axis=1)))

    def _max_jerk(self, actions: np.ndarray) -> float:
        if len(actions) <= 2:
            return 0.0
        second_diff = np.diff(actions[:, : self.config.action_dim], n=2, axis=0)
        return float(np.max(np.linalg.norm(second_diff, axis=1)))

    def _is_smooth_free_space(self, actions: np.ndarray) -> bool:
        cfg = self.config
        if self._has_gripper_change(actions):
            return False
        if len(actions) <= 1:
            return False
        pos_delta = float(np.max(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)))
        rot_delta = float(np.max(np.linalg.norm(np.diff(actions[:, 3:6], axis=0), axis=1))) if actions.shape[1] >= 6 else 0.0
        return pos_delta <= cfg.smooth_position_delta and rot_delta <= cfg.smooth_rotation_delta

    def _relaxed_reference_failures(self, draft: np.ndarray, reference: np.ndarray) -> list[str]:
        cfg = self.config
        ref = _as_action_chunk(reference)
        horizon = min(len(draft), len(ref))
        failures: list[str] = []
        pos_err = float(np.max(np.linalg.norm(draft[:horizon, :3] - ref[:horizon, :3], axis=1)))
        if pos_err > cfg.position_error_threshold:
            failures.append("position_error")
        if draft.shape[1] >= 6 and ref.shape[1] >= 6:
            rot_err = float(np.max(np.linalg.norm(draft[:horizon, 3:6] - ref[:horizon, 3:6], axis=1)))
            if rot_err > cfg.rotation_error_threshold:
                failures.append("rotation_error")
        gi = cfg.gripper_index
        if gi < draft.shape[1] and gi < ref.shape[1] and not np.allclose(draft[:horizon, gi], ref[:horizon, gi]):
            failures.append("gripper_mismatch")
        return failures


class ChunkExecutionController:
    """Queue accepted chunk prefixes and expose one action per control step."""

    def __init__(self, guard: ChunkGuard | None = None) -> None:
        self.guard = guard or ChunkGuard()
        self.queue: deque[np.ndarray] = deque()
        self.stats = ChunkExecutionStats()

    def reset(self) -> None:
        self.queue.clear()
        self.guard.reset()
        self.stats = ChunkExecutionStats()

    def has_buffered_action(self) -> bool:
        return bool(self.queue)

    def pop(self) -> np.ndarray | None:
        if not self.queue:
            return None
        action = self.queue.popleft()
        self.guard.mark_executed(action)
        self.stats.actions_executed += 1
        return action.copy()

    def offer_chunk(
        self,
        chunk: np.ndarray | Iterable[Iterable[float]],
        *,
        confidence: float | None = None,
        contact_phase_change: bool = False,
        reference_chunk: np.ndarray | None = None,
        relaxed: bool = False,
        inference_ms: float | None = None,
        token_count: int | None = None,
    ) -> ChunkDecision:
        actions = _as_action_chunk(chunk)
        self.stats.chunks_seen += 1
        self.stats.model_calls += 1
        if inference_ms is not None:
            self.stats.inference_ms.append(float(inference_ms))
        if token_count is not None:
            self.stats.token_counts.append(int(token_count))

        decision = self.guard.decide(
            actions,
            confidence=confidence,
            contact_phase_change=contact_phase_change,
            reference_chunk=reference_chunk,
            relaxed=relaxed,
        )
        if not decision.accepted:
            self.stats.guard_reasons.update(decision.reasons)
            return decision

        self.queue.clear()
        for action in actions[: decision.execute_window]:
            self.queue.append(action.copy())
        self.stats.chunks_accepted += 1
        self.stats.actions_enqueued += decision.execute_window
        self.stats.accepted_windows.append(decision.execute_window)
        return decision


def exact_fast_prefix_acceptance(target_token_ids: Iterable[int], draft_token_ids: Iterable[int]) -> int:
    """Return the longest exact greedy prefix accepted by target FAST tokens."""

    accepted = 0
    for target, draft in zip(target_token_ids, draft_token_ids):
        if int(target) != int(draft):
            break
        accepted += 1
    return accepted


@dataclass
class RetrievalDraft:
    token_ids: list[int]
    action_chunk: np.ndarray
    distance: float


class RetrievalChunkDrafter:
    """Nearest-neighbor drafter over previously accepted chunks."""

    def __init__(self, max_entries: int = 512) -> None:
        self.max_entries = max_entries
        self._entries: deque[tuple[np.ndarray, list[int], np.ndarray]] = deque(maxlen=max_entries)

    def add(self, state_key: np.ndarray, token_ids: Iterable[int], action_chunk: np.ndarray) -> None:
        self._entries.append(
            (
                np.asarray(state_key, dtype=np.float32).reshape(-1),
                [int(t) for t in token_ids],
                _as_action_chunk(action_chunk),
            )
        )

    def draft(self, state_key: np.ndarray) -> RetrievalDraft | None:
        if not self._entries:
            return None
        query = np.asarray(state_key, dtype=np.float32).reshape(-1)
        best: tuple[float, list[int], np.ndarray] | None = None
        for key, tokens, chunk in self._entries:
            if key.shape != query.shape:
                continue
            dist = float(np.linalg.norm(query - key))
            if best is None or dist < best[0]:
                best = (dist, tokens, chunk)
        if best is None:
            return None
        return RetrievalDraft(token_ids=list(best[1]), action_chunk=best[2].copy(), distance=best[0])


def dataclass_dict(obj) -> dict:
    """Small helper for JSON-friendly configs in experiment logs."""

    return asdict(obj)
