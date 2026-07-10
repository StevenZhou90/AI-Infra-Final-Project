from __future__ import annotations

import argparse

import pytest
import torch

from scripts.run_pi0fast_chunk_eval import (
    _adaptive_stability_risk_gate_from_args,
    _adapter_action_end_token_id,
    _load_token_id_file,
    _parse_risk_gate_clause_spec,
    _pi05_unsupported_fast_token_modes,
    _predict_prefix_cutoff_chunk,
    _predict_target_eos_chunk,
    parse_checkpoint_stable_checks,
)
from serving.pi0fast_token_hooks import PI0FastGenerationTrace


class _FakeTokenAdapter:
    def __init__(self) -> None:
        self.action_end_calls = 0
        self.cutoff_calls = 0
        self.trace_calls = 0
        self.action_end_kwargs = {}

    def predict_action_chunk_action_end(self, batch, **kwargs):
        self.action_end_calls += 1
        self.action_end_kwargs = dict(kwargs)
        return PI0FastGenerationTrace(
            actions=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            token_ids=torch.tensor([[10, 20, 30]]),
            logits=torch.empty((1, 0, 0)),
            stats={"mode": "action_end_no_logits", "action_end_token_id": 30},
        )

    def predict_action_chunk_with_trace(self, batch, early_stop_action_end=False):
        self.trace_calls += 1
        raise AssertionError("target_eos speed path should not collect logits")

    def predict_action_chunk_prefix_cutoff(self, batch, **kwargs):
        self.cutoff_calls += 1
        assert kwargs["cutoff_tokens"] == 24
        assert kwargs["collect_logits"] is False
        assert kwargs["force_action_prefix"] is False
        return PI0FastGenerationTrace(
            actions=torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
            token_ids=torch.tensor([[10, 20, 30]]),
            logits=torch.empty((1, 0, 0)),
            stats={"mode": "prefix_cutoff_no_logits", "action_end_token_id": 30},
        )


def test_target_eos_chunk_uses_no_logits_action_end_path() -> None:
    adapter = _FakeTokenAdapter()

    prediction = _predict_target_eos_chunk(
        adapter,
        batch={},
        postprocessor=lambda action: action,
        device="cpu",
    )

    assert adapter.action_end_calls == 1
    assert adapter.trace_calls == 0
    assert prediction.token_count == 3
    assert prediction.token_ids.tolist() == [[10, 20, 30]]
    assert prediction.stats["mode"] == "action_end_no_logits"


def test_target_eos_chunk_can_request_action_char_stop() -> None:
    adapter = _FakeTokenAdapter()

    _predict_target_eos_chunk(
        adapter,
        batch={},
        postprocessor=lambda action: action,
        device="cpu",
        stop_on_action_chars=True,
        action_char_min_chars=4,
        action_char_plateau_tokens=2,
        action_char_stable_checks=1,
        action_char_stable_tolerance=0.001,
        action_char_stable_max_chars=16,
        action_char_plateau_reject_eos_restart=True,
        action_char_restart_continue=True,
        action_char_restart_reset=True,
        action_char_restart_reset_low_text_tokens=2,
        action_char_plateau_low_text_tail_tokens=3,
        action_char_strict_target_stop=True,
        action_char_reset_requires_action_end=True,
        action_char_target_confirm_tokens=4,
    )

    assert adapter.action_end_calls == 1
    assert adapter.action_end_kwargs["stop_on_action_chars"] is True
    assert adapter.action_end_kwargs["action_char_min_chars"] == 4
    assert adapter.action_end_kwargs["action_char_plateau_tokens"] == 2
    assert adapter.action_end_kwargs["action_char_stable_checks"] == 1
    assert adapter.action_end_kwargs["action_char_stable_tolerance"] == 0.001
    assert adapter.action_end_kwargs["action_char_stable_max_chars"] == 16
    assert adapter.action_end_kwargs["action_char_plateau_reject_eos_restart"] is True
    assert adapter.action_end_kwargs["action_char_restart_continue"] is True
    assert adapter.action_end_kwargs["action_char_restart_reset"] is True
    assert adapter.action_end_kwargs["action_char_restart_reset_low_text_tokens"] == 2
    assert adapter.action_end_kwargs["action_char_plateau_low_text_tail_tokens"] == 3
    assert adapter.action_end_kwargs["action_char_strict_target_stop"] is True
    assert adapter.action_end_kwargs["action_char_reset_requires_action_end"] is True
    assert adapter.action_end_kwargs["action_char_target_confirm_tokens"] == 4


def test_target_eos_chunk_passes_constrained_extra_token_ids() -> None:
    adapter = _FakeTokenAdapter()

    _predict_target_eos_chunk(
        adapter,
        batch={},
        postprocessor=lambda action: action,
        device="cpu",
        constrained_action_vocab=True,
        constrained_extra_token_ids=[11, 22],
    )

    assert adapter.action_end_calls == 1
    assert adapter.action_end_kwargs["constrained_action_vocab"] is True
    assert adapter.action_end_kwargs["constrained_extra_token_ids"] == [11, 22]


def test_load_token_id_file_accepts_payload_dict(tmp_path) -> None:
    path = tmp_path / "tokens.json"
    path.write_text('{"token_ids": [7, "3", 7]}')

    assert _load_token_id_file(path) == [3, 7]

    empty_path = tmp_path / "empty_tokens.json"
    empty_path.write_text('{"token_ids": []}')
    assert _load_token_id_file(empty_path) == []


def test_parse_checkpoint_stable_checks_accepts_csv_overrides() -> None:
    assert parse_checkpoint_stable_checks("") == {}
    assert parse_checkpoint_stable_checks("152=5, 160 = 4") == {152: 5, 160: 4}


def test_parse_checkpoint_stable_checks_rejects_invalid_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_checkpoint_stable_checks("152")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_checkpoint_stable_checks("0=4")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_checkpoint_stable_checks("152=0")


def test_parse_risk_gate_clause_spec_accepts_csv_and_json() -> None:
    assert _parse_risk_gate_clause_spec(
        "max_checkpoint=160,max_token_count=161,max_logprob_mean=-1.04"
    ) == {
        "max_checkpoint": 160.0,
        "max_token_count": 161.0,
        "max_logprob_mean": -1.04,
    }
    assert _parse_risk_gate_clause_spec('{"min_entropy_mean": 2.7, "max_max_step_delta": 0.3}') == {
        "min_entropy_mean": 2.7,
        "max_max_step_delta": 0.3,
    }


def test_parse_risk_gate_clause_spec_rejects_invalid_entries() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_risk_gate_clause_spec("unknown=1")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_risk_gate_clause_spec("max_checkpoint=late")


def test_adaptive_stability_risk_gate_from_args_omits_disabled_thresholds() -> None:
    args = argparse.Namespace(
        adaptive_stability_risk_min_checkpoint=-1,
        adaptive_stability_risk_max_checkpoint=160,
        adaptive_stability_risk_max_token_count=153,
        adaptive_stability_risk_logprob_mean_max=-1.04,
        adaptive_stability_risk_entropy_mean_min=2.9,
        adaptive_stability_risk_action_abs_min=None,
        adaptive_stability_risk_position_span_max=0.0,
        adaptive_stability_risk_rotation_span_max=0.0,
        adaptive_stability_risk_max_step_delta_max=0.0,
    )

    assert _adaptive_stability_risk_gate_from_args(args) == {
        "max_checkpoint": 160.0,
        "max_token_count": 153.0,
        "max_logprob_mean": -1.04,
        "min_entropy_mean": 2.9,
        "max_position_span": 0.0,
        "max_rotation_span": 0.0,
        "max_max_step_delta": 0.0,
    }


def test_adaptive_stability_risk_gate_from_args_adds_motion_clause() -> None:
    args = argparse.Namespace(
        adaptive_stability_risk_min_checkpoint=-1,
        adaptive_stability_risk_max_checkpoint=192,
        adaptive_stability_risk_max_token_count=193,
        adaptive_stability_risk_logprob_mean_max=-1.04,
        adaptive_stability_risk_entropy_mean_min=2.9,
        adaptive_stability_risk_action_abs_min=None,
        adaptive_stability_risk_position_span_max=0.0,
        adaptive_stability_risk_rotation_span_max=0.0,
        adaptive_stability_risk_max_step_delta_max=0.0,
        adaptive_stability_motion_risk_min_checkpoint=224,
        adaptive_stability_motion_risk_logprob_mean_max=-1.0,
        adaptive_stability_motion_risk_position_span_min=1.5,
        adaptive_stability_motion_risk_rotation_span_min=2.0,
        adaptive_stability_motion_risk_max_step_delta_min=1.5,
        adaptive_stability_risk_extra_clause=[
            {
                "max_checkpoint": 160.0,
                "max_token_count": 161.0,
                "max_logprob_mean": -1.04,
                "min_entropy_mean": 2.7,
                "min_position_span": 0.8,
                "min_rotation_span": 0.7,
                "max_max_step_delta": 0.3,
            }
        ],
    )

    assert _adaptive_stability_risk_gate_from_args(args) == {
        "clauses": [
            {
                "max_checkpoint": 192.0,
                "max_token_count": 193.0,
                "max_logprob_mean": -1.04,
                "min_entropy_mean": 2.9,
                "max_position_span": 0.0,
                "max_rotation_span": 0.0,
                "max_max_step_delta": 0.0,
            },
            {
                "min_checkpoint": 224.0,
                "max_logprob_mean": -1.0,
                "min_position_span": 1.5,
                "min_rotation_span": 2.0,
                "min_max_step_delta": 1.5,
            },
            {
                "max_checkpoint": 160.0,
                "max_token_count": 161.0,
                "max_logprob_mean": -1.04,
                "min_entropy_mean": 2.7,
                "min_position_span": 0.8,
                "min_rotation_span": 0.7,
                "max_max_step_delta": 0.3,
            },
        ]
    }


def test_prefix_cutoff_chunk_uses_no_logits_path_by_default() -> None:
    adapter = _FakeTokenAdapter()

    prediction = _predict_prefix_cutoff_chunk(
        adapter,
        batch={},
        postprocessor=lambda action: action,
        device="cpu",
        cutoff_tokens=24,
    )

    assert adapter.cutoff_calls == 1
    assert prediction.token_count == 3
    assert prediction.token_ids.tolist() == [[10, 20, 30]]
    assert prediction.stats["mode"] == "prefix_cutoff_no_logits"


def test_pi05_rejects_pi0fast_token_decode_modes() -> None:
    assert _pi05_unsupported_fast_token_modes(
        ["baseline", "target_eos", "pattern_sd_direct", "block_sd_validate"]
    ) == ["target_eos", "pattern_sd_direct", "block_sd_validate"]
    assert _pi05_unsupported_fast_token_modes(["baseline", "chunk_m3"]) == []


def test_prefix_cutoff_chunk_can_force_action_prefix() -> None:
    class PrefixAdapter(_FakeTokenAdapter):
        def predict_action_chunk_prefix_cutoff(self, batch, **kwargs):
            self.cutoff_calls += 1
            assert kwargs["force_action_prefix"] is True
            return PI0FastGenerationTrace(
                actions=torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
                token_ids=torch.tensor([[10, 20, 30]]),
                logits=torch.empty((1, 0, 0)),
                stats={"mode": "prefix_cutoff_prefix_no_logits", "action_end_token_id": 30},
            )

    adapter = PrefixAdapter()

    prediction = _predict_prefix_cutoff_chunk(
        adapter,
        batch={},
        postprocessor=lambda action: action,
        device="cpu",
        cutoff_tokens=24,
        force_action_prefix=True,
    )

    assert adapter.cutoff_calls == 1
    assert prediction.stats["mode"] == "prefix_cutoff_prefix_no_logits"


def test_adapter_action_end_token_id_accepts_property_or_method() -> None:
    class PropertyAdapter:
        action_end_token_id = 123

    class MethodAdapter:
        def action_end_token_id(self):
            return 456

    assert _adapter_action_end_token_id(PropertyAdapter()) == 123
    assert _adapter_action_end_token_id(MethodAdapter()) == 456
