from __future__ import annotations

from scripts.benchmark_pi0fast_system_components import pi0fast_recommendation


def test_pi0fast_recommendation_requires_single_request_target() -> None:
    rows = [
        {
            "decode_path": "action_end",
            "errors": [],
            "actions_per_request": 10,
            "single_inference": {
                "mean_ms": 500.0,
                "per_action_mean_ms": 50.0,
            },
            "batch_inference": {
                "8": {
                    "mean_ms": 720.0,
                    "per_request_mean_ms": 90.0,
                    "per_action_mean_ms": 9.0,
                }
            },
        }
    ]

    assert pi0fast_recommendation(rows, target_latency_ms=100.0) is None


def test_pi0fast_recommendation_reports_true_single_request_pass() -> None:
    rows = [
        {
            "decode_path": "action_end",
            "errors": [],
            "actions_per_request": 10,
            "single_inference": {
                "mean_ms": 95.0,
                "per_action_mean_ms": 9.5,
            },
            "batch_inference": {},
        }
    ]

    recommendation = pi0fast_recommendation(rows, target_latency_ms=100.0)

    assert recommendation is not None
    assert recommendation["path"] == "single_request"
    assert recommendation["per_request_mean_ms"] == 95.0
    assert recommendation["meets_target"] is True
