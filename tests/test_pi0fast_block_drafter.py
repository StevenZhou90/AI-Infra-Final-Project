from __future__ import annotations

import torch

from serving.pi0fast_block_drafter import PI0FastBlockDrafter, PI0FastBlockDrafterConfig
from serving.pi0fast_eagle import CompactTokenMap


def test_block_drafter_forward_and_draft_shapes() -> None:
    config = PI0FastBlockDrafterConfig(
        hidden_dim=8,
        vocab_size=5,
        context_len=4,
        block_len=3,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = PI0FastBlockDrafter(config)
    hidden = torch.randn(2, 8)
    context = torch.tensor([[5, 5, 0, 1], [5, 2, 3, 4]])

    logits = model(hidden, context)
    draft, confidence = model.draft(hidden, context, steps=2)

    assert logits.shape == (2, 3, 5)
    assert draft.shape == (2, 2)
    assert confidence.shape == (2, 2)
    assert bool((confidence >= 0).all())
    assert bool((confidence <= 1).all())


def test_block_drafter_token_map_decode() -> None:
    token_map = CompactTokenMap([101, 205, 303])
    classes = torch.tensor([[0, 2]])

    assert token_map.decode_tensor(classes).tolist() == [[101, 303]]
