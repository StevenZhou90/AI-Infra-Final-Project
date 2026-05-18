from __future__ import annotations

import numpy as np
import torch

from serving.trajectory_draft_head import TinyTrajectoryHead, TrajectoryHeadConfig, bins_to_token_ids, token_ids_to_bins
from serving.trajectory_phase import label_phase
from serving.trajectory_speculative_decoder import RetrievalEntry, TrajectorySpeculativeDecoder


class DummyModel:
    vocab_size = 32000
    bin_centers = np.linspace(-1.0, 1.0, 256, dtype=np.float32)

    def __init__(self) -> None:
        self.generate_calls = 0

    def get_action_stats(self, _key: str) -> dict:
        return {
            "q01": np.full(7, -1.0, dtype=np.float32),
            "q99": np.full(7, 1.0, dtype=np.float32),
            "mask": np.ones(7, dtype=bool),
        }

    def generate(self, input_ids, **_kwargs):
        self.generate_calls += 1
        action = torch.tensor([[31900, 31901, 31902, 31903, 31904, 31905, 31906]], dtype=torch.long)
        return torch.cat([input_ids.cpu(), action], dim=1)


class ConstantHead:
    def __init__(self, bins: np.ndarray, history_size: int = 2, confidence: float = 0.95) -> None:
        self.config = TrajectoryHeadConfig(history_size=history_size)
        self.bins = torch.as_tensor(bins, dtype=torch.long)
        self.confidence = confidence

    def predict(self, history_bins, top_k: int = 3, prefill_hidden=None):
        batch = history_bins.shape[0]
        if self.config.action_horizon > 1:
            top_bins = self.bins.view(1, 1, 7, 1).expand(batch, self.config.action_horizon, 7, top_k).clone()
            top_probs = torch.full((batch, self.config.action_horizon, 7, top_k), 0.01, dtype=torch.float32)
        else:
            top_bins = self.bins.view(1, 7, 1).expand(batch, 7, top_k).clone()
            top_probs = torch.full((batch, 7, top_k), 0.01, dtype=torch.float32)
        top_probs[..., 0] = self.confidence
        max_probs = top_probs[..., 0]
        return top_bins, top_probs, max_probs


def test_action_token_bin_round_trip() -> None:
    bins = np.array([0, 1, 17, 64, 127, 200, 255], dtype=np.int64)
    token_ids = bins_to_token_ids(bins, DummyModel.vocab_size)

    assert np.array_equal(token_ids_to_bins(token_ids, DummyModel.vocab_size).numpy(), bins)


def test_retrieval_candidate_uses_matching_recent_prefix() -> None:
    decoder = TrajectorySpeculativeDecoder(
        model=DummyModel(),
        device="cpu",
        decoder_mode="trajectory-hybrid-spec",
        retrieval_min_confidence=0.1,
        retrieval_context_steps=2,
    )
    prefix = np.array(
        [
            [31900, 31901, 31902, 31903, 31904, 31905, 31906],
            [31890, 31891, 31892, 31893, 31894, 31895, 31896],
        ],
        dtype=np.int64,
    )
    future = np.array([31880, 31881, 31882, 31883, 31884, 31885, 31886], dtype=np.int64)
    decoder.token_history = [prefix[0], prefix[1]]
    decoder.retrieval_entries = [
        RetrievalEntry(task_key="default", timestep_bucket=0, prefix=prefix.copy(), next_tokens=future.copy())
    ]

    draft = decoder._draft_retrieval(max_new_tokens=7)

    assert draft is not None
    draft_ids, bands, _residual_bins, confidence = draft
    assert np.array_equal(draft_ids.numpy()[0], future)
    assert all(int(tok) in band for tok, band in zip(future, bands))
    assert confidence.all()
    assert decoder.stats.retrieval_hits == 1


def test_kinematic_switch_prefers_retrieval_only_for_smooth_segments() -> None:
    decoder = TrajectorySpeculativeDecoder(model=DummyModel(), device="cpu", kinematic_threshold=4.0)
    base = np.array([31800, 31800, 31800, 31800, 31800, 31800, 31800], dtype=np.int64)

    smooth_1 = base.copy()
    smooth_2 = base + 2
    smooth_3 = base + 4
    smooth_2[6] = base[6]
    smooth_3[6] = base[6]
    decoder.token_history = [smooth_1, smooth_2, smooth_3]
    assert decoder._kinematic_retrieval_allowed()

    curved_mid = base + 20
    curved_mid[6] = base[6]
    decoder.token_history = [base, curved_mid, base]
    assert not decoder._kinematic_retrieval_allowed()

    decoder.token_history = [base, base + 2, np.array([31804, 31804, 31804, 31804, 31804, 31804, 31810])]
    assert not decoder._kinematic_retrieval_allowed()


def test_relaxed_group_acceptance_thresholds() -> None:
    decoder = TrajectorySpeculativeDecoder(
        model=DummyModel(),
        device="cpu",
        decoder_mode="trajectory-hybrid-spec",
        relaxed_tolerance=(3.0, 3.0, 0.0),
    )
    prev2 = np.array([100, 100, 100, 200, 200, 200, 300], dtype=np.int64)
    prev1 = np.array([102, 102, 102, 202, 202, 202, 300], dtype=np.int64)
    decoder.token_history = [prev2, prev1]

    assert decoder._relaxed_group_accepts(np.array([104, 105, 104, 204, 205, 204, 300], dtype=np.int64))
    assert not decoder._relaxed_group_accepts(np.array([120, 120, 120, 204, 205, 204, 300], dtype=np.int64))
    assert not decoder._relaxed_group_accepts(np.array([104, 104, 104, 204, 204, 204, 301], dtype=np.int64))


def test_phase_label_smooth_vs_complex() -> None:
    smooth = np.array(
        [
            [100, 100, 100, 120, 120, 120, 200],
            [104, 104, 104, 124, 124, 124, 200],
            [108, 108, 108, 128, 128, 128, 200],
            [112, 112, 112, 132, 132, 132, 200],
        ],
        dtype=np.int64,
    )
    curved = smooth.copy()
    curved[-1, :6] += 30
    gripper = smooth.copy()
    gripper[-1, 6] += 1

    assert label_phase(smooth) == "smooth"
    assert label_phase(curved) == "complex"
    assert label_phase(gripper) == "complex"


def test_two_head_router_selects_phase_head() -> None:
    cfg = TrajectoryHeadConfig(history_size=4, hidden_dim=16, embed_dim=8, num_layers=1)
    smooth_head = TinyTrajectoryHead(cfg)
    complex_head = TinyTrajectoryHead(cfg)
    decoder = TrajectorySpeculativeDecoder(
        model=DummyModel(),
        device="cpu",
        decoder_mode="trajectory-two-head-spec",
        smooth_draft_head=smooth_head,
        complex_draft_head=complex_head,
    )
    smooth_bins = np.array(
        [
            [100, 100, 100, 120, 120, 120, 200],
            [104, 104, 104, 124, 124, 124, 200],
            [108, 108, 108, 128, 128, 128, 200],
            [112, 112, 112, 132, 132, 132, 200],
        ],
        dtype=np.int64,
    )
    decoder.token_history = [bins_to_token_ids(row, DummyModel.vocab_size).numpy() for row in smooth_bins]
    head, phase = decoder._select_phase_head()
    assert phase == "smooth"
    assert head is smooth_head

    complex_bins = smooth_bins.copy()
    complex_bins[-1, :6] += 30
    decoder.token_history = [bins_to_token_ids(row, DummyModel.vocab_size).numpy() for row in complex_bins]
    head, phase = decoder._select_phase_head()
    assert phase == "complex"
    assert head is complex_head


def test_tiny_trajectory_head_direct_chunk_shape() -> None:
    cfg = TrajectoryHeadConfig(history_size=4, hidden_dim=16, embed_dim=8, num_layers=1, action_horizon=4)
    head = TinyTrajectoryHead(cfg)
    history = torch.zeros(2, 4, 7, dtype=torch.long)

    logits = head(history)
    top_bins, top_probs, max_probs = head.predict(history, top_k=3)

    assert logits.shape == (2, 4, 7, 256)
    assert top_bins.shape == (2, 4, 7, 3)
    assert top_probs.shape == (2, 4, 7, 3)
    assert max_probs.shape == (2, 4, 7)


def test_chunk_mode_buffers_after_vla_anchor_and_consumes_without_global_cap() -> None:
    model = DummyModel()
    head = ConstantHead(np.array([101, 102, 103, 104, 105, 106, 93]), history_size=2)
    decoder = TrajectorySpeculativeDecoder(
        model=model,
        device="cpu",
        decoder_mode="trajectory-chunk-spec",
        smooth_draft_head=head,  # type: ignore[arg-type]
        complex_draft_head=head,  # type: ignore[arg-type]
        min_history=1,
        head_threshold=0.2,
        fast_max_draft_calls=0,
        chunk_smooth_len=3,
        chunk_complex_len=3,
        chunk_min_confident_tokens=7,
        chunk_max_token_delta=500.0,
    )
    seed_tokens = np.array([31910, 31911, 31912, 31913, 31914, 31915, 31906], dtype=np.int64)
    decoder.token_history = [seed_tokens.copy(), seed_tokens.copy()]
    decoder.history = [np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)]
    inputs = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    decoder.predict_action(inputs, unnorm_key="dummy", max_new_tokens=7)
    assert model.generate_calls == 1
    assert decoder.stats.chunk_anchor_calls == 1
    assert decoder.stats.chunk_buffered_actions > 0

    decoder.predict_action(inputs, unnorm_key="dummy", max_new_tokens=7)
    assert model.generate_calls == 1
    assert decoder.stats.chunk_buffer_hits == 1


def test_chunk_refresh_guards_gripper_and_large_delta() -> None:
    decoder = TrajectorySpeculativeDecoder(
        model=DummyModel(),
        device="cpu",
        decoder_mode="trajectory-chunk-spec",
        chunk_max_token_delta=5.0,
    )
    decoder.token_history = [np.array([100, 100, 100, 100, 100, 100, 200], dtype=np.int64)]
    decoder._chunk_buffer = [{"token_ids": np.array([101, 101, 101, 101, 101, 101, 201]), "phase": "complex"}]
    assert decoder._chunk_refresh_reason() == "gripper_change"

    decoder._chunk_buffer = [{"token_ids": np.array([140, 140, 140, 140, 140, 140, 200]), "phase": "complex"}]
    assert decoder._chunk_refresh_reason() == "large_delta"


def test_direct_chunk_mode_fills_multiple_actions_from_one_head_call() -> None:
    model = DummyModel()
    head = ConstantHead(np.array([101, 102, 103, 104, 105, 106, 93]), history_size=2)
    head.config.action_horizon = 2
    decoder = TrajectorySpeculativeDecoder(
        model=model,
        device="cpu",
        decoder_mode="trajectory-direct-chunk-spec",
        direct_chunk_head=head,  # type: ignore[arg-type]
        min_history=1,
        head_threshold=0.2,
        chunk_smooth_len=3,
        chunk_complex_len=3,
        chunk_min_confident_tokens=7,
        chunk_max_token_delta=500.0,
    )
    seed_tokens = np.array([31910, 31911, 31912, 31913, 31914, 31915, 31906], dtype=np.int64)
    decoder.token_history = [seed_tokens.copy(), seed_tokens.copy()]
    decoder.history = [np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)]
    inputs = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    decoder.predict_action(inputs, unnorm_key="dummy", max_new_tokens=7)

    assert model.generate_calls == 1
    assert decoder.stats.chunk_buffered_actions == 2
