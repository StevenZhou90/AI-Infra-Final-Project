from __future__ import annotations

import torch

from serving.kv_cache_manager import repeat_kv_batch, select_kv_batch


def test_repeat_kv_batch_repeats_tuple_cache_rows() -> None:
    key = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
    value = key + 100
    repeated = repeat_kv_batch(((key, value),), batch_size=3)

    assert repeated[0][0].shape == (3, 2, 3, 4)
    assert repeated[0][1].shape == (3, 2, 3, 4)
    assert torch.equal(repeated[0][0][2], key[0])
    assert torch.equal(repeated[0][1][1], value[0])


def test_select_kv_batch_selects_tuple_cache_row() -> None:
    key = torch.arange(3 * 2 * 3 * 4, dtype=torch.float32).view(3, 2, 3, 4)
    value = key + 100
    selected = select_kv_batch(((key, value),), batch_idx=1)

    assert selected[0][0].shape == (1, 2, 3, 4)
    assert selected[0][1].shape == (1, 2, 3, 4)
    assert torch.equal(selected[0][0][0], key[1])
    assert torch.equal(selected[0][1][0], value[1])
