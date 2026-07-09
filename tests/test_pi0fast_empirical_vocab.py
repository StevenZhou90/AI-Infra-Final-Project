from __future__ import annotations

import torch

from scripts.build_pi0fast_empirical_vocab import build_vocab_payload, collect_token_counts, select_token_ids


def test_empirical_vocab_counts_trace_shards_with_mode_filter(tmp_path) -> None:
    rows = [
        {
            "token_ids": torch.tensor([10, 20, 10]),
            "mode": "target_eos",
            "stop_token_ids": [99],
        },
        {
            "token_ids": torch.tensor([30, 40]),
            "mode": "baseline",
        },
        {
            "token_ids": [20, 50],
            "mode": "target_eos",
        },
    ]
    torch.save(rows, tmp_path / "shard_00000.pt")

    counts, metadata = collect_token_counts(
        tmp_path,
        modes={"target_eos"},
        include_stop_tokens=True,
    )

    assert metadata == {"shards": 1, "rows": 2, "total_tokens": 6}
    assert counts[10] == 2
    assert counts[20] == 2
    assert counts[50] == 1
    assert counts[99] == 1
    assert 30 not in counts


def test_empirical_vocab_selects_min_count_and_top_k() -> None:
    counts = {10: 4, 20: 2, 30: 3, 40: 1}

    assert select_token_ids(counts, min_count=2, top_k=2) == [10, 30]
    assert select_token_ids(counts, min_count=3, top_k=0) == [10, 30]


def test_empirical_vocab_payload_records_metadata(tmp_path) -> None:
    rows = [
        {
            "token_ids": torch.tensor([7, 7, 8]),
            "mode": "target_eos_constrained",
        }
    ]
    torch.save(rows, tmp_path / "shard_00000.pt")

    payload = build_vocab_payload(tmp_path, modes={"target_eos_constrained"}, min_count=2)

    assert payload["token_ids"] == [7]
    assert payload["token_count"] == 1
    assert payload["rows"] == 1
    assert payload["mode_filter"] == ["target_eos_constrained"]
    assert payload["counts"] == {"7": 2}
