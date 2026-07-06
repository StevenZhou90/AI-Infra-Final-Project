from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.eval_pi0fast_pattern_offline import load_lightweight_trace_records
from scripts.pi0fast_token_trace_export import TokenTraceSink, parse_token_trace_modes


def test_parse_token_trace_modes() -> None:
    assert parse_token_trace_modes("target_eos, pattern_sd_direct") == {"target_eos", "pattern_sd_direct"}
    assert parse_token_trace_modes("all") is None
    assert parse_token_trace_modes("*") is None
    assert parse_token_trace_modes("") is None


def test_token_trace_sink_writes_lightweight_shards(tmp_path: Path) -> None:
    sink = TokenTraceSink(output_dir=tmp_path, modes={"target_eos"}, max_rows_per_shard=1)
    prediction = SimpleNamespace(token_ids=torch.tensor([[11, 12, 13]]), stats={"action_end_token_id": 99})

    sink.record(
        prediction,
        mode="target_eos",
        task="libero_goal",
        task_id=4,
        episode=2,
        seed=44,
        step=5,
    )
    sink.record(
        prediction,
        mode="baseline",
        task="libero_goal",
        task_id=4,
        episode=2,
        seed=44,
        step=6,
    )
    sink.flush()

    shard_files = sorted(tmp_path.glob("shard_*.pt"))
    assert len(shard_files) == 1
    records = load_lightweight_trace_records(tmp_path)
    assert len(records) == 1
    assert records[0].token_ids.tolist() == [11, 12, 13]
    assert records[0].task_id == 4
    assert records[0].seed == 44
    assert records[0].stop_token_ids == (99,)
    assert "modetarget_eos" in records[0].trace_id
