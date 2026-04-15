import pytest
import torch
import torch.nn as nn

from server.config import SpecDecodeConfig
from server.spec_decode import SpeculativeDecoder


class _SimpleDrafter(nn.Module):
    def __init__(self, action_dim=6, chunk_size=10):
        super().__init__()
        self.linear = nn.Linear(10, action_dim * chunk_size)
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def forward(self, images, joint_state=None):
        b = images.shape[0]
        x = images.reshape(b, -1)[:, :10]
        if x.shape[1] < 10:
            x = torch.nn.functional.pad(x, (0, 10 - x.shape[1]))
        out = self.linear(x)
        return out.reshape(b, self.chunk_size, self.action_dim)


class _SimpleVerifier(nn.Module):
    def __init__(self, action_dim=6, chunk_size=10):
        super().__init__()
        self.linear = nn.Linear(10, action_dim * chunk_size)
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def forward(self, images, joint_state=None):
        b = images.shape[0]
        x = images.reshape(b, -1)[:, :10]
        if x.shape[1] < 10:
            x = torch.nn.functional.pad(x, (0, 10 - x.shape[1]))
        out = self.linear(x)
        return out.reshape(b, self.chunk_size, self.action_dim)


@pytest.fixture
def spec_decoder():
    config = SpecDecodeConfig(
        enabled=True,
        acceptance_threshold=0.15,
        min_acceptance_rate=0.3,
        drafter_chunk_size=10,
    )
    return SpeculativeDecoder(config)


@pytest.fixture
def models():
    torch.manual_seed(42)
    drafter = _SimpleDrafter()
    verifier = _SimpleVerifier()
    return drafter, verifier


class TestSpeculativeDecoder:
    def test_decode_returns_chunk(self, spec_decoder, models):
        drafter, verifier = models
        images = torch.randn(1, 3, 10, 10)
        joints = torch.zeros(1, 6)

        result = spec_decoder.decode(drafter, verifier, images, joints, chunk_size=10)
        assert result.action_chunk is not None
        assert len(result.action_chunk.steps) == 10
        assert result.total_time_ms > 0

    def test_speculative_flag(self, spec_decoder, models):
        drafter, verifier = models
        images = torch.randn(1, 3, 10, 10)
        joints = torch.zeros(1, 6)

        result = spec_decoder.decode(drafter, verifier, images, joints, chunk_size=10)
        assert result.used_speculative is True

    def test_fallback_when_disabled(self, models):
        config = SpecDecodeConfig(enabled=False, drafter_chunk_size=10)
        decoder = SpeculativeDecoder(config)
        drafter, verifier = models
        images = torch.randn(1, 3, 10, 10)
        joints = torch.zeros(1, 6)

        result = decoder.decode(drafter, verifier, images, joints, chunk_size=10)
        assert result.used_speculative is False
        assert len(result.action_chunk.steps) == 10

    def test_acceptance_rate_tracked(self, spec_decoder, models):
        drafter, verifier = models
        images = torch.randn(1, 3, 10, 10)
        joints = torch.zeros(1, 6)

        for _ in range(5):
            spec_decoder.decode(drafter, verifier, images, joints, chunk_size=10)

        stats = spec_decoder.stats
        assert stats["total_proposed"] == 50
        assert stats["total_accepted"] >= 0

    def test_identical_models_full_acceptance(self):
        """When drafter == verifier, acceptance should be 100%."""
        config = SpecDecodeConfig(enabled=True, acceptance_threshold=0.15, drafter_chunk_size=10)
        decoder = SpeculativeDecoder(config)
        torch.manual_seed(0)
        model = _SimpleDrafter()

        images = torch.randn(1, 3, 10, 10)
        joints = torch.zeros(1, 6)

        result = decoder.decode(model, model, images, joints, chunk_size=10)
        assert result.accepted_length == 10
        assert result.acceptance_rate == 1.0
