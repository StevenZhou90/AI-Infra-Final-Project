from __future__ import annotations

from pathlib import Path

import pytest

from scripts.annotate_pi0fast_adaptive_static_exactness import (
    adaptive_static_exact_counts,
    annotate_row,
)


def _row(
    *,
    chunks_seen: int = 10,
    exact_verifies: int = 3,
    max_action_diff: float = 0.0,
    trace_stats: dict | None = None,
) -> dict:
    return {
        "mode": "target_eos_adaptive_validate",
        "task": "libero_object",
        "task_id": 1,
        "episode": 2,
        "seed": 44,
        "chunk_stats": {
            "chunks_seen": chunks_seen,
            "exact_verifies": exact_verifies,
            "max_action_diff": max_action_diff,
            "trace_stats": trace_stats
            or {
                "stopped_on_stability": exact_verifies / chunks_seen,
                "stopped_on_action_end": (chunks_seen - exact_verifies) / chunks_seen,
            },
        },
    }


def test_adaptive_static_counts_recover_runtime_and_action_end_chunks() -> None:
    counts = adaptive_static_exact_counts(_row())

    assert counts == {
        "chunks_seen": 10,
        "runtime_exact_verifies": 3,
        "stability_runtime_chunks": 3,
        "target_eos_static_chunks": 7,
        "accounted_exact_chunks": 10,
        "unaccounted_chunks": 0,
    }


def test_annotate_row_marks_static_exact_proof() -> None:
    row = annotate_row(
        _row(),
        validation_mode="target_eos_adaptive_validate",
        proof_source=Path("modeling_pi0_fast.py"),
        max_action_diff=0.0,
        require_runtime_stability_accounting=True,
    )

    stats = row["chunk_stats"]
    assert stats["exact_verifies"] == 3
    assert stats["static_exact_verifies"] == 7
    assert stats["static_exact_accounted_chunks"] == 10
    assert stats["trace_stats"]["static_target_eos_adaptive_exact_proof"] == 1.0


def test_annotate_row_refuses_nonzero_action_diff() -> None:
    with pytest.raises(ValueError, match="max_action_diff"):
        annotate_row(
            _row(max_action_diff=1e-6),
            validation_mode="target_eos_adaptive_validate",
            proof_source=Path("modeling_pi0_fast.py"),
            max_action_diff=0.0,
            require_runtime_stability_accounting=True,
        )


def test_annotate_row_refuses_runtime_stability_mismatch() -> None:
    with pytest.raises(ValueError, match="runtime exact"):
        annotate_row(
            _row(exact_verifies=2, trace_stats={"stopped_on_stability": 0.3, "stopped_on_action_end": 0.7}),
            validation_mode="target_eos_adaptive_validate",
            proof_source=Path("modeling_pi0_fast.py"),
            max_action_diff=0.0,
            require_runtime_stability_accounting=True,
        )


def test_annotation_preserves_unaccounted_chunk_metadata() -> None:
    row = annotate_row(
        _row(trace_stats={"stopped_on_stability": 0.3, "stopped_on_action_end": 0.6}),
        validation_mode="target_eos_adaptive_validate",
        proof_source=Path("modeling_pi0_fast.py"),
        max_action_diff=0.0,
        require_runtime_stability_accounting=True,
    )

    stats = row["chunk_stats"]
    assert stats["static_exact_verifies"] == 6
    assert stats["static_exact_accounted_chunks"] == 9
    assert stats["static_exact_unaccounted_chunks"] == 1
