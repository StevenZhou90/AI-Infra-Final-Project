from __future__ import annotations

from unittest.mock import patch

from scripts.benchmark_robotics_spec_decode_synthetic import parse_args
from serving.robotics_spec_benchmark import (
    ConstantVelocityActionDrafter,
    RobotSpecBenchmarkConfig,
    exact_spec_decode,
    generate_narrow_action_tokens,
    run_robotics_spec_benchmark,
)


def test_synthetic_benchmark_cli_defaults_to_120_tasks() -> None:
    with patch("sys.argv", ["benchmark_robotics_spec_decode_synthetic.py"]):
        args = parse_args()

    assert args.num_tasks == 120


def test_exact_spec_decode_matches_target_with_phase_rejections() -> None:
    config = RobotSpecBenchmarkConfig(num_tasks=1, action_steps=30, phase_len=10, lookahead=14)
    target = generate_narrow_action_tokens(config, task_id=0).tolist()
    drafter = ConstantVelocityActionDrafter(action_dim=config.action_dim, vocab_size=config.vocab_size)

    result = exact_spec_decode(target, drafter, lookahead=config.lookahead)

    assert result.output_tokens == target
    assert result.target_forwards < len(target)
    assert result.drafted_tokens > 0
    assert result.accepted_tokens > 0
    assert result.rejected_blocks > 0


def test_robotics_spec_benchmark_hits_zero_drop_and_two_x_on_120_tasks() -> None:
    config = RobotSpecBenchmarkConfig(num_tasks=120, action_steps=36, phase_len=12, lookahead=14)

    summary = run_robotics_spec_benchmark(config)

    assert summary["tasks"] == 120
    assert summary["all_exact"] is True
    assert summary["accuracy_drop_abs"] == 0.0
    assert summary["speedup"] >= 2.0
    assert summary["acceptance_rate"] > 0.5
