from __future__ import annotations

import torch

from serving.pi0fast_eagle import (
    CompactTokenMap,
    PI0FastEagleConfig,
    PI0FastEagleHead,
    PI0FastTraceRecord,
    build_compact_token_map,
    evaluate_offline_acceptance,
    make_teacher_forcing_rows,
    split_trace_records,
)


def test_compact_token_map_roundtrip() -> None:
    token_map = CompactTokenMap([42, 7, 42, 99])
    encoded = token_map.encode_tensor(torch.tensor([7, 99, 123, 42]))

    assert encoded.tolist() == [0, 2, -100, 1]
    assert token_map.decode_tensor(torch.tensor([0, 1, 2])).tolist() == [7, 42, 99]


def test_trace_split_has_no_overlap() -> None:
    records = [
        PI0FastTraceRecord(torch.zeros(3, 4), torch.tensor([1, 2, 3]), task_id=i % 2, seed=i, trace_id=str(i))
        for i in range(6)
    ]

    train, val = split_trace_records(records, split="trace", val_fraction=0.33, seed=0)

    assert train
    assert val
    assert not (set(train) & set(val))


def test_teacher_forcing_rows_drop_oov() -> None:
    records = [
        PI0FastTraceRecord(torch.zeros(4, 4), torch.tensor([1, 2, 3, 4]), task_id=0, seed=0, trace_id="a"),
        PI0FastTraceRecord(torch.zeros(4, 4), torch.tensor([1, 9, 3, 4]), task_id=1, seed=1, trace_id="b"),
    ]
    token_map = build_compact_token_map(records, [0])

    hidden, inputs, targets, trace_indices = make_teacher_forcing_rows(records, [1], token_map, drop_oov=True)

    assert hidden.shape == (1, 4)
    assert inputs.tolist() == [2]
    assert targets.tolist() == [3]
    assert trace_indices.tolist() == [1]


def test_eagle_forward_shape_and_reset() -> None:
    model = PI0FastEagleHead(
        PI0FastEagleConfig(hidden_size=8, num_attention_heads=2, num_key_value_heads=2, intermediate_size=16),
        vocab_size=5,
    )

    logits, hidden = model(
        torch.randn(2, 3, 8),
        torch.tensor([[0, 1, 2], [2, 3, 4]]),
        return_hidden=True,
    )
    model.reset_kv()

    assert logits.shape == (2, 3, 5)
    assert hidden.shape == (2, 3, 8)
    assert model.eagle._kv_cache == [None]


def test_offline_acceptance_reports_oov() -> None:
    records = [
        PI0FastTraceRecord(torch.zeros(4, 8), torch.tensor([1, 2, 3, 4]), task_id=0, seed=0, trace_id="train"),
        PI0FastTraceRecord(torch.zeros(4, 8), torch.tensor([1, 2, 9, 4]), task_id=1, seed=1, trace_id="val"),
    ]
    token_map = build_compact_token_map(records, [0])
    model = PI0FastEagleHead(
        PI0FastEagleConfig(hidden_size=8, num_attention_heads=2, num_key_value_heads=2, intermediate_size=16),
        vocab_size=len(token_map),
    )

    metrics = evaluate_offline_acceptance(model, records, [1], token_map, lookahead=2, device="cpu")

    assert metrics["positions"] == 3
    assert metrics["future_oov_rate"] > 0
    assert "1" in metrics["per_task"]
