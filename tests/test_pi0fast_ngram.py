from __future__ import annotations

import torch

from serving.pi0fast_eagle import PI0FastTraceRecord
from serving.pi0fast_ngram import (
    NgramDraftConfig,
    NgramFastTokenDrafter,
    evaluate_ngram_drafter,
    exact_prefix_acceptance,
)


def test_exact_prefix_acceptance() -> None:
    assert exact_prefix_acceptance([1, 2, 3], [1, 2, 9]) == 2
    assert exact_prefix_acceptance([1], [2]) == 0


def test_ngram_drafts_with_backoff() -> None:
    traces = [
        PI0FastTraceRecord(torch.zeros(5, 2), torch.tensor([1, 2, 3, 4, 5]), 0, 0, "a"),
        PI0FastTraceRecord(torch.zeros(5, 2), torch.tensor([1, 2, 3, 4, 6]), 0, 1, "b"),
    ]
    drafter = NgramFastTokenDrafter(NgramDraftConfig(max_context=3, lookahead=3))
    drafter.fit(traces)

    assert drafter.draft([1, 2], lookahead=2)[:2] == [3, 4]
    assert drafter.draft([99], lookahead=1)


def test_ngram_eval_reports_acceptance() -> None:
    train = PI0FastTraceRecord(torch.zeros(6, 2), torch.tensor([1, 2, 3, 1, 2, 3]), 0, 0, "train")
    val = PI0FastTraceRecord(torch.zeros(6, 2), torch.tensor([1, 2, 3, 1, 2, 9]), 1, 1, "val")
    drafter = NgramFastTokenDrafter(NgramDraftConfig(max_context=2, lookahead=2))
    drafter.fit([train])

    metrics = evaluate_ngram_drafter(drafter, [val], lookahead=2)

    assert metrics["positions"] == 5
    assert metrics["mean_spec_accept"] > 0
    assert "1" in metrics["per_task"]
