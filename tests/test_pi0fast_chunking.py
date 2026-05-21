from __future__ import annotations

import numpy as np

from serving.pi0fast_chunking import (
    ChunkExecutionController,
    ChunkGuard,
    ChunkGuardConfig,
    RetrievalChunkDrafter,
    exact_fast_prefix_acceptance,
)


def smooth_chunk(horizon: int = 10) -> np.ndarray:
    actions = np.zeros((horizon, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(0.0, 0.04, horizon)
    actions[:, 1] = np.linspace(0.0, 0.01, horizon)
    actions[:, 6] = 0.0
    return actions


def test_smooth_chunk_gets_smooth_window() -> None:
    guard = ChunkGuard(ChunkGuardConfig(default_execute_window=3, smooth_execute_window=5))
    decision = guard.decide(smooth_chunk(), confidence=1.0)

    assert decision.accepted
    assert decision.smooth_free_space
    assert decision.execute_window == 5
    assert decision.reasons == []


def test_gripper_change_forces_refresh() -> None:
    chunk = smooth_chunk()
    chunk[4:, 6] = 1.0
    guard = ChunkGuard()

    decision = guard.decide(chunk, confidence=1.0)

    assert not decision.accepted
    assert decision.execute_window == 0
    assert "gripper_change" in decision.reasons


def test_large_jump_rejected() -> None:
    chunk = smooth_chunk()
    chunk[5, 0] = 0.8
    guard = ChunkGuard(ChunkGuardConfig(action_jump_threshold=0.2))

    decision = guard.decide(chunk, confidence=1.0)

    assert not decision.accepted
    assert "large_action_jump" in decision.reasons


def test_relaxed_reference_checks_position_rotation_and_gripper() -> None:
    draft = smooth_chunk()
    reference = smooth_chunk()
    reference[:, 0] += 0.2
    reference[:, 6] = 1.0
    guard = ChunkGuard()

    decision = guard.decide(draft, reference_chunk=reference, relaxed=True, confidence=1.0)

    assert not decision.accepted
    assert "position_error" in decision.reasons
    assert "gripper_mismatch" in decision.reasons


def test_controller_enqueues_and_pops_accepted_prefix() -> None:
    controller = ChunkExecutionController(ChunkGuard(ChunkGuardConfig(default_execute_window=3, smooth_execute_window=3)))
    decision = controller.offer_chunk(smooth_chunk(), confidence=1.0, inference_ms=20.0, token_count=12)

    assert decision.accepted
    assert controller.stats.model_calls == 1
    assert controller.stats.actions_enqueued == 3
    assert [controller.pop() is not None for _ in range(3)] == [True, True, True]
    assert controller.pop() is None
    assert controller.stats.actions_executed == 3
    assert controller.stats.summary()["avg_fast_token_count"] == 12.0


def test_exact_token_stats_are_reported() -> None:
    controller = ChunkExecutionController()
    controller.stats.exact_verifies = 2
    controller.stats.exact_drafted_tokens = 10
    controller.stats.exact_accepted_tokens = 7

    summary = controller.stats.summary()

    assert summary["exact_verifies"] == 2
    assert summary["exact_drafted_tokens"] == 10
    assert summary["exact_accepted_tokens"] == 7
    assert summary["exact_token_acceptance_rate"] == 0.7


def test_exact_fast_prefix_acceptance() -> None:
    assert exact_fast_prefix_acceptance([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert exact_fast_prefix_acceptance([1, 2], [1, 2, 3]) == 2
    assert exact_fast_prefix_acceptance([1, 2], [9, 2]) == 0


def test_retrieval_chunk_drafter_returns_nearest_entry() -> None:
    drafter = RetrievalChunkDrafter()
    far_chunk = smooth_chunk()
    near_chunk = smooth_chunk()
    near_chunk[:, 1] += 0.02
    drafter.add(np.array([10.0, 10.0]), [1, 2, 3], far_chunk)
    drafter.add(np.array([0.1, 0.2]), [4, 5, 6], near_chunk)

    draft = drafter.draft(np.array([0.0, 0.0]))

    assert draft is not None
    assert draft.token_ids == [4, 5, 6]
    np.testing.assert_allclose(draft.action_chunk, near_chunk)
