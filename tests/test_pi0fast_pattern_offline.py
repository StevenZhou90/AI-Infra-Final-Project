from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import torch

from scripts.eval_pi0fast_pattern_offline import infer_stop_token_ids, load_lightweight_trace_records, split_eval_indices
from scripts.sweep_pi0fast_pattern_offline import main as sweep_main


def test_lightweight_trace_loader_reads_token_shards(tmp_path: Path) -> None:
    torch.save(
        [
            {
                "token_ids": torch.tensor([1, 2, 3]),
                "task": "libero_goal",
                "task_id": 0,
                "seed": 11,
                "trace_id": "a",
            },
            {"token_ids": torch.tensor([4, 5, 6]), "task_id": 1, "seed": 12, "trace_id": "b"},
        ],
        tmp_path / "shard_000.pt",
    )

    records = load_lightweight_trace_records(tmp_path)

    assert len(records) == 2
    assert records[0].token_ids.tolist() == [1, 2, 3]
    assert records[0].task == "libero_goal"
    assert records[1].task_id == 1


def test_lightweight_trace_loader_reads_stop_token_metadata(tmp_path: Path) -> None:
    torch.save(
        [
            {"token_ids": torch.tensor([1, 2, 99]), "task_id": 0, "seed": 11, "stop_token_ids": [99]},
            {"token_ids": torch.tensor([4, 5, 99]), "task_id": 1, "seed": 12, "action_end_token_id": 99},
        ],
        tmp_path / "shard_000.pt",
    )

    records = load_lightweight_trace_records(tmp_path)

    assert records[0].stop_token_ids == (99,)
    assert records[1].stop_token_ids == (99,)
    assert infer_stop_token_ids(records) == (99,)


def test_pattern_sweep_cli_infers_stop_token_metadata(tmp_path: Path) -> None:
    torch.save(
        [
            {"token_ids": torch.tensor([10, 20, 30, 99]), "task_id": 0, "seed": 1, "stop_token_ids": [99]},
            {"token_ids": torch.tensor([11, 21, 31, 99]), "task_id": 0, "seed": 1, "stop_token_ids": [99]},
        ],
        tmp_path / "shard_000.pt",
    )
    output = tmp_path / "sweep.json"
    old_argv = sys.argv
    try:
        sys.argv = [
            "sweep_pi0fast_pattern_offline.py",
            "--data-dir",
            str(tmp_path),
            "--lookaheads",
            "2",
            "--action-dims",
            "2",
            "--max-periods",
            "0",
            "--min-period-repeats",
            "2",
            "--repeat-token-min-runs",
            "99",
            "--linear-action-extrapolation",
            "false",
            "--second-order-action-extrapolation",
            "false",
            "--second-order-max-accels",
            "8",
            "--chunk-length-stop",
            "both",
            "--chunk-length-stop-history-sizes",
            "2",
            "--chunk-length-stop-min-counts",
            "1",
            "--source-priority-modes",
            "chunk_length_stop",
            "--reuse-full-blocks",
            "true",
            "--emit-bonus-token",
            "false",
            "--dynamic-lookahead",
            "false",
            "--tree-widths",
            "1",
            "--top-k",
            "2",
            "--output",
            str(output),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            assert sweep_main() == 0
    finally:
        sys.argv = old_argv

    summary = json.loads(output.read_text())
    assert summary["best"]["config"]["stop_token_ids"] == [99]
    assert summary["best"]["config"]["chunk_length_stop"] is True
    assert summary["best"]["chunk_length_stop_accepted_tokens"] > 0


def test_pattern_sweep_cli_records_task_disjoint_heldout_metadata(tmp_path: Path) -> None:
    torch.save(
        [
            {"token_ids": torch.tensor([10, 20, 30, 40]), "task": "libero_goal", "task_id": 0, "seed": 1},
            {"token_ids": torch.tensor([11, 21, 31, 41]), "task": "libero_goal", "task_id": 1, "seed": 1},
            {"token_ids": torch.tensor([12, 22, 32, 42]), "task": "libero_spatial", "task_id": 0, "seed": 1},
            {"token_ids": torch.tensor([13, 23, 33, 43]), "task": "libero_spatial", "task_id": 1, "seed": 1},
        ],
        tmp_path / "shard_000.pt",
    )
    output = tmp_path / "sweep.json"
    old_argv = sys.argv
    try:
        sys.argv = [
            "sweep_pi0fast_pattern_offline.py",
            "--data-dir",
            str(tmp_path),
            "--heldout-split",
            "task",
            "--heldout-val-fraction",
            "0.5",
            "--lookaheads",
            "2",
            "--action-dims",
            "2",
            "--max-periods",
            "0",
            "--min-period-repeats",
            "2",
            "--repeat-token-min-runs",
            "99",
            "--linear-action-extrapolation",
            "false",
            "--second-order-action-extrapolation",
            "false",
            "--second-order-max-accels",
            "8",
            "--reuse-full-blocks",
            "true",
            "--emit-bonus-token",
            "false",
            "--dynamic-lookahead",
            "false",
            "--tree-widths",
            "1",
            "--top-k",
            "1",
            "--max-configs",
            "1",
            "--output",
            str(output),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            assert sweep_main() == 0
    finally:
        sys.argv = old_argv

    selection = json.loads(output.read_text())["selection"]
    assert selection["heldout_split"] == "task"
    assert selection["heldout_task_disjoint"] is True
    assert selection["heldout_task_overlap"] == []
    assert selection["rank_task_count"] == 2
    assert selection["heldout_task_count"] == 2
    assert selection["rank_suite_keys"] == ["libero_goal", "libero_spatial"]
    assert selection["rank_suite_count"] == 2
    assert selection["heldout_suite_keys"] == ["libero_goal", "libero_spatial"]
    assert selection["heldout_suite_count"] == 2


def test_lightweight_trace_loader_reads_nested_suite_shards(tmp_path: Path) -> None:
    suite_dir = tmp_path / "libero_goal"
    suite_dir.mkdir()
    torch.save(
        [{"token_ids": torch.tensor([7, 8, 9]), "task_id": 2, "seed": 13, "trace_id": "nested"}],
        suite_dir / "shard_00000.pt",
    )

    records = load_lightweight_trace_records(tmp_path)

    assert len(records) == 1
    assert records[0].task == "libero_goal"
    assert records[0].trace_id == "nested"
    assert records[0].token_ids.tolist() == [7, 8, 9]


def test_split_eval_indices_can_select_grouped_holdouts() -> None:
    class Record:
        def __init__(self, task_id: int, seed: int) -> None:
            self.task_id = task_id
            self.seed = seed

    records = [Record(0, 7), Record(1, 7), Record(1, 8), Record(2, 8)]

    assert split_eval_indices(records, split="task", val_fraction=0.5, seed=0, heldout_task_id=1) == [1, 2]
    assert split_eval_indices(records, split="seed", val_fraction=0.5, seed=0, heldout_task_id=None, heldout_seed=8) == [
        2,
        3,
    ]
    assert split_eval_indices(
        records,
        split="task_seed",
        val_fraction=0.5,
        seed=0,
        heldout_task_id=1,
        heldout_seed=8,
    ) == [2]
    assert split_eval_indices(records, split="all", val_fraction=0.5, seed=0, heldout_task_id=None) == [0, 1, 2, 3]


def test_split_eval_indices_auto_holdout_uses_suite_scoped_task_key() -> None:
    class Record:
        def __init__(self, task: str, task_id: int, seed: int) -> None:
            self.task = task
            self.task_id = task_id
            self.seed = seed

    records = [
        Record("libero_goal", 0, 7),
        Record("libero_goal", 1, 7),
        Record("libero_spatial", 0, 7),
        Record("libero_spatial", 1, 7),
    ]

    task_indices = split_eval_indices(records, split="task", val_fraction=0.5, seed=0, heldout_task_id=None)
    task_seed_indices = split_eval_indices(
        records,
        split="task_seed",
        val_fraction=0.5,
        seed=0,
        heldout_task_id=None,
        heldout_seed=None,
    )

    assert len(task_indices) == 2
    assert len({(records[idx].task, records[idx].task_id) for idx in task_indices}) == 2
    assert len(task_seed_indices) == 2
    assert len({(records[idx].task, records[idx].task_id, records[idx].seed) for idx in task_seed_indices}) == 2


def test_split_eval_indices_task_holdout_is_suite_stratified() -> None:
    class Record:
        def __init__(self, task: str, task_id: int, seed: int) -> None:
            self.task = task
            self.task_id = task_id
            self.seed = seed

    records = [
        Record(suite, task_id, 7)
        for suite in ("libero_goal", "libero_object", "libero_spatial")
        for task_id in range(5)
    ]

    indices = split_eval_indices(records, split="task", val_fraction=0.2, seed=0, heldout_task_id=None)

    heldout_suites = {records[idx].task for idx in indices}
    assert heldout_suites == {"libero_goal", "libero_object", "libero_spatial"}
    assert len({(records[idx].task, records[idx].task_id) for idx in indices}) == 3


def test_split_eval_indices_auto_group_holdout_uses_validation_fraction() -> None:
    class Record:
        def __init__(self, task_id: int, seed: int) -> None:
            self.task_id = task_id
            self.seed = seed

    records = [Record(0, 7), Record(1, 7), Record(2, 8), Record(3, 8)]

    task_indices = split_eval_indices(records, split="task", val_fraction=0.5, seed=0, heldout_task_id=None)
    seed_indices = split_eval_indices(records, split="seed", val_fraction=0.5, seed=0, heldout_task_id=None)
    task_seed_indices = split_eval_indices(
        records,
        split="task_seed",
        val_fraction=0.5,
        seed=0,
        heldout_task_id=None,
        heldout_seed=None,
    )

    assert len({records[idx].task_id for idx in task_indices}) == 2
    assert len({records[idx].seed for idx in seed_indices}) == 1
    assert len({(records[idx].task_id, records[idx].seed) for idx in task_seed_indices}) == 2


def test_split_eval_indices_rejects_partial_task_seed_selector() -> None:
    class Record:
        def __init__(self, task_id: int, seed: int) -> None:
            self.task_id = task_id
            self.seed = seed

    try:
        split_eval_indices(
            [Record(0, 7), Record(1, 8)],
            split="task_seed",
            val_fraction=0.5,
            seed=0,
            heldout_task_id=1,
            heldout_seed=None,
        )
    except ValueError as exc:
        assert "heldout-task-id" in str(exc)
    else:
        raise AssertionError("expected partial task_seed selector to fail")
