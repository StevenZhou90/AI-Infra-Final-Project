"""Synthetic robotics action-token speculative decoding benchmark.

This module is deliberately lightweight: it models robot actions as discrete
tokens with long smooth phases, then applies an exact draft-verify loop.  The
motion-prior drafter is robotics-specific, while the verifier semantics mirror
LLM speculative decoding: generated tokens must exactly match the target
autoregressive stream.

The benchmark is a CI sanity check for the algorithmic idea.  It is not a
substitute for the real LIBERO/PI0-FAST 120-eval gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RobotSpecBenchmarkConfig:
    num_tasks: int = 120
    action_steps: int = 36
    action_dim: int = 7
    vocab_size: int = 256
    lookahead: int = 14
    phase_len: int = 12
    target_forward_ms: float = 1.0
    draft_token_ms: float = 0.02
    seed: int = 7


@dataclass(frozen=True)
class SpecDecodeResult:
    output_tokens: list[int]
    target_forwards: int
    drafted_tokens: int
    accepted_tokens: int
    rejected_blocks: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.drafted_tokens, 1)


def generate_narrow_action_tokens(config: RobotSpecBenchmarkConfig, task_id: int) -> np.ndarray:
    """Generate a deterministic smooth-phase action-token trace.

    Each task follows piecewise-constant velocity in token space.  This is the
    narrow robot-control distribution we want a drafter to exploit: most future
    actions are well predicted by recent motion, while phase changes still
    produce verifier rejections and corrections.
    """

    rng = np.random.default_rng(config.seed + task_id * 9973)
    low = 16
    high = config.vocab_size - 17
    action = rng.integers(low + 24, high - 24, size=config.action_dim, dtype=np.int64)
    velocity = rng.integers(-2, 3, size=config.action_dim, dtype=np.int64)
    velocity[-1] = 0
    rows: list[np.ndarray] = []

    for step in range(config.action_steps):
        if step > 0 and step % config.phase_len == 0:
            velocity = rng.integers(-2, 3, size=config.action_dim, dtype=np.int64)
            velocity[-1] = 0
            # Rare gripper phase changes are intentionally discontinuous.
            if rng.random() < 0.35:
                action[-1] = high if action[-1] < (config.vocab_size // 2) else low
        action = np.clip(action + velocity, 0, config.vocab_size - 1)
        rows.append(action.copy())

    return np.asarray(rows, dtype=np.int64).reshape(-1)


class ConstantVelocityActionDrafter:
    """Draft future action tokens from the last two completed actions."""

    def __init__(self, *, action_dim: int, vocab_size: int) -> None:
        self.action_dim = int(action_dim)
        self.vocab_size = int(vocab_size)

    def draft(self, prefix_tokens: Iterable[int], max_tokens: int) -> list[int]:
        sim = [int(token) for token in prefix_tokens]
        draft: list[int] = []
        for _ in range(max_tokens):
            if len(sim) < 2 * self.action_dim:
                break
            pos = len(sim)
            dim = pos % self.action_dim
            action_idx = pos // self.action_dim
            if action_idx < 2:
                break
            prev = sim[(action_idx - 1) * self.action_dim + dim]
            prev2 = sim[(action_idx - 2) * self.action_dim + dim]
            pred = int(np.clip(prev + (prev - prev2), 0, self.vocab_size - 1))
            draft.append(pred)
            sim.append(pred)
        return draft


def exact_spec_decode(
    target_tokens: Iterable[int],
    drafter: ConstantVelocityActionDrafter,
    *,
    lookahead: int,
) -> SpecDecodeResult:
    """Decode target tokens with exact speculative draft-verify semantics."""

    target = [int(token) for token in target_tokens]
    generated: list[int] = []
    target_forwards = 0
    drafted_tokens = 0
    accepted_tokens = 0
    rejected_blocks = 0

    while len(generated) < len(target):
        remaining = len(target) - len(generated)
        draft = drafter.draft(generated, min(int(lookahead), remaining))
        if not draft:
            target_forwards += 1
            generated.append(target[len(generated)])
            continue

        target_forwards += 1
        drafted_tokens += len(draft)
        accepted = 0
        offset = len(generated)
        for idx, token in enumerate(draft):
            if offset + idx >= len(target) or int(token) != target[offset + idx]:
                break
            accepted += 1

        generated.extend(draft[:accepted])
        accepted_tokens += accepted
        if len(generated) >= len(target):
            continue

        if accepted < len(draft):
            rejected_blocks += 1
            generated.append(target[len(generated)])
        else:
            # Bonus token from the verifier, as in standard speculative decode.
            generated.append(target[len(generated)])

    return SpecDecodeResult(
        output_tokens=generated[: len(target)],
        target_forwards=target_forwards,
        drafted_tokens=drafted_tokens,
        accepted_tokens=accepted_tokens,
        rejected_blocks=rejected_blocks,
    )


def run_robotics_spec_benchmark(config: RobotSpecBenchmarkConfig) -> dict:
    drafter = ConstantVelocityActionDrafter(action_dim=config.action_dim, vocab_size=config.vocab_size)
    per_task: list[dict] = []
    baseline_successes = 0
    spec_successes = 0
    total_baseline_ms = 0.0
    total_spec_ms = 0.0
    total_drafted = 0
    total_accepted = 0
    total_target_forwards = 0
    total_baseline_forwards = 0
    total_rejected_blocks = 0

    for task_id in range(config.num_tasks):
        target = generate_narrow_action_tokens(config, task_id).tolist()
        result = exact_spec_decode(target, drafter, lookahead=config.lookahead)
        exact_match = result.output_tokens == target
        baseline_success = True
        spec_success = exact_match
        baseline_ms = len(target) * config.target_forward_ms
        spec_ms = result.target_forwards * config.target_forward_ms + result.drafted_tokens * config.draft_token_ms

        baseline_successes += int(baseline_success)
        spec_successes += int(spec_success)
        total_baseline_ms += baseline_ms
        total_spec_ms += spec_ms
        total_drafted += result.drafted_tokens
        total_accepted += result.accepted_tokens
        total_target_forwards += result.target_forwards
        total_baseline_forwards += len(target)
        total_rejected_blocks += result.rejected_blocks
        per_task.append(
            {
                "task_id": task_id,
                "tokens": len(target),
                "exact_match": exact_match,
                "baseline_ms": baseline_ms,
                "spec_ms": spec_ms,
                "speedup": baseline_ms / spec_ms if spec_ms else 0.0,
                "target_forwards": result.target_forwards,
                "drafted_tokens": result.drafted_tokens,
                "accepted_tokens": result.accepted_tokens,
                "acceptance_rate": result.acceptance_rate,
                "rejected_blocks": result.rejected_blocks,
            }
        )

    tasks = max(config.num_tasks, 1)
    baseline_success_rate = baseline_successes / tasks
    spec_success_rate = spec_successes / tasks
    speedup = total_baseline_ms / total_spec_ms if total_spec_ms else 0.0
    return {
        "config": asdict(config),
        "tasks": config.num_tasks,
        "baseline_successes": baseline_successes,
        "spec_successes": spec_successes,
        "baseline_success_rate": baseline_success_rate,
        "spec_success_rate": spec_success_rate,
        "accuracy_drop_abs": baseline_success_rate - spec_success_rate,
        "speedup": speedup,
        "baseline_ms": total_baseline_ms,
        "spec_ms": total_spec_ms,
        "baseline_target_forwards": total_baseline_forwards,
        "spec_target_forwards": total_target_forwards,
        "target_forward_reduction": total_baseline_forwards / max(total_target_forwards, 1),
        "drafted_tokens": total_drafted,
        "accepted_tokens": total_accepted,
        "acceptance_rate": total_accepted / max(total_drafted, 1),
        "rejected_blocks": total_rejected_blocks,
        "all_exact": spec_successes == config.num_tasks,
        "per_task": per_task,
    }
