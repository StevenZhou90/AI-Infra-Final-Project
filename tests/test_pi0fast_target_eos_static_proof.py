from __future__ import annotations

from pathlib import Path

from scripts.prove_pi0fast_target_eos_exactness import (
    build_static_validation_rows,
    detokenizer_source_proves_action_end_truncation,
)


def test_detokenizer_source_check_requires_action_end_truncation() -> None:
    assert detokenizer_source_proves_action_end_truncation(
        """
        if "|" in token_seq:
            token_seq = token_seq[: token_seq.index("|")]
        actions = self.decode_actions_with_fast(action_tokens)
        """
    )
    assert not detokenizer_source_proves_action_end_truncation("actions = self.decode_actions_with_fast(tokens)")


def test_static_validation_rows_preserve_eval_key_and_mark_static_proof() -> None:
    rows = [
        {
            "mode": "target_eos",
            "task": "libero_object",
            "task_id": 3,
            "episode": 2,
            "seed": 44,
            "success": False,
            "model_calls": 30,
            "chunk_stats": {"chunks_seen": 30, "exact_verifies": 0, "trace_stats": {}},
        }
    ]

    out = build_static_validation_rows(
        rows,
        source_mode="target_eos",
        validation_mode="target_eos_validate",
        proof_source=Path("modeling_pi0_fast.py"),
    )

    assert len(out) == 1
    assert out[0]["mode"] == "target_eos_validate"
    assert out[0]["task_id"] == 3
    assert out[0]["episode"] == 2
    assert out[0]["chunk_stats"]["static_exact_verifies"] == 30
    assert out[0]["chunk_stats"]["exact_verifies"] == 0
    assert out[0]["chunk_stats"]["max_action_diff"] == 0.0
    assert out[0]["chunk_stats"]["trace_stats"]["static_target_eos_exact_proof"] == 1.0
