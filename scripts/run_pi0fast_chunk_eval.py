#!/usr/bin/env python3
"""Run π0-FAST chunk-execution experiments in LeRobot LIBERO.

This script is intentionally separate from the OpenVLA/SimplerEnv path. It uses
public LeRobot π0-FAST checkpoints and the LIBERO simulator, then logs matched
baseline/chunk metrics as JSONL plus a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_chunking import (  # noqa: E402
    ChunkExecutionController,
    ChunkGuard,
    ChunkGuardConfig,
    RetrievalChunkDrafter,
    dataclass_dict,
)
from serving.pi0fast_action_gate import ACTION_GATE_FEATURES, action_gate_feature_values, load_action_gate  # noqa: E402
from serving.pi0fast_eagle import load_trace_records  # noqa: E402
from serving.pi0fast_ngram import NgramDraftConfig, NgramFastTokenDrafter  # noqa: E402
from serving.pi0fast_pattern_drafter import PatternDraftConfig, PatternFastTokenDrafter, parse_source_priority  # noqa: E402
from serving.pi0fast_block_drafter import load_block_drafter_checkpoint  # noqa: E402
from serving.pi0fast_block_gate import load_block_gate  # noqa: E402
from serving.pi0fast_cutoff_selector import load_cutoff_selector  # noqa: E402
from serving.pi0fast_medusa import load_medusa_checkpoint  # noqa: E402
from serving.pi0fast_prefix_gate import PREFIX_GATE_FEATURES, action_feature_values, load_prefix_gate  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402
from serving.pi0fast_action_dflash import load_action_dflash_checkpoint  # noqa: E402
from serving.pi0fast_trajectory_head import load_trajectory_tail_checkpoint  # noqa: E402
from scripts.pi0fast_token_trace_export import TokenTraceSink, parse_token_trace_modes  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger("run_pi0fast_chunk_eval")


def _ensure_libero_config(config_path: str | None) -> None:
    """Avoid LIBERO's interactive first-run dataset-path prompt."""

    if config_path:
        os.environ["LIBERO_CONFIG_PATH"] = config_path
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")).expanduser()
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        return

    libero_root = None
    for entry in sys.path:
        candidate = Path(entry) / "libero" / "libero"
        if (candidate / "bddl_files").exists() and (candidate / "init_files").exists():
            libero_root = candidate.resolve()
            break
    if libero_root is None:
        raise RuntimeError("Could not locate installed LIBERO package root before env creation")

    cfg_dir.mkdir(parents=True, exist_ok=True)
    datasets = libero_root.parent / "datasets"
    cfg_file.write_text(
        "\n".join(
            [
                f"assets: {libero_root / 'assets'}",
                f"bddl_files: {libero_root / 'bddl_files'}",
                f"benchmark_root: {libero_root}",
                f"datasets: {datasets}",
                f"init_states: {libero_root / 'init_files'}",
                "",
            ]
        )
    )
    logger.info("Wrote noninteractive LIBERO config to %s", cfg_file)


@dataclass
class EpisodeResult:
    mode: str
    episode: int
    seed: int
    task: str
    task_id: int | None
    success: bool
    reward_sum: float
    steps: int
    wall_s: float
    avg_ms_per_control_step: float
    model_calls: int
    model_calls_per_step: float
    chunk_stats: dict


def _import_lerobot():
    try:
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    except ImportError as exc:
        raise SystemExit(
            "Missing LeRobot π0-FAST/LIBERO dependencies. Install them on this branch, for example:\n"
            '  uv pip install "lerobot[pi,libero] @ git+https://github.com/huggingface/lerobot.git"\n'
            "Then rerun this script."
        ) from exc
    return make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy


def _to_numpy_action(action: Any) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 3:
        action = action[0]
    if action.ndim == 2:
        return action
    if action.ndim == 1:
        return action.reshape(1, -1)
    raise ValueError(f"Unsupported action shape: {action.shape}")


def _extract_success(info: dict[str, Any]) -> bool:
    if "final_info" in info and isinstance(info["final_info"], dict):
        final_info = info["final_info"]
        if "is_success" in final_info:
            value = final_info["is_success"]
            return bool(np.asarray(value).any())
    for key in ("is_success", "success"):
        if key in info:
            return bool(np.asarray(info[key]).any())
    return False


def _call_task(env) -> list[str]:
    try:
        return list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            return list(env.call("task"))
        except (AttributeError, NotImplementedError):
            return [""] * env.num_envs


def _prepare_observation(
    observation: dict[str, Any],
    env,
    env_preprocessor,
    policy_preprocessor,
    preprocess_observation,
) -> dict[str, Any]:
    obs = preprocess_observation(observation)
    obs["task"] = _call_task(env)
    obs = env_preprocessor(obs)
    return policy_preprocessor(obs)


def _state_key(observation: dict[str, Any]) -> np.ndarray:
    state = observation.get("observation.state")
    if state is not None:
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size:
            return arr
    # LIBERO observations can be nested or use raw simulator keys before
    # LeRobot preprocessing. Fall back to a cheap visual-free constant key so
    # retrieval mode remains deterministic even if no proprio key is exposed.
    return np.zeros(1, dtype=np.float32)


def _extend_action_chunk(actions: np.ndarray, total_horizon: int, damping: float = 0.65) -> np.ndarray:
    """Speculate a short receding-horizon tail from the target model's chunk."""

    chunk = _to_numpy_action(actions)
    if total_horizon <= chunk.shape[0] or chunk.shape[0] < 2:
        return chunk
    extended = [row.copy() for row in chunk]
    last = chunk[-1].copy()
    delta = chunk[-1] - chunk[-2]
    if chunk.shape[1] >= 7:
        delta[6] = 0.0
    for _ in range(total_horizon - chunk.shape[0]):
        delta = delta * damping
        last = last + delta
        if chunk.shape[1] >= 7:
            last[6] = chunk[-1, 6]
        extended.append(np.clip(last.copy(), -1.0, 1.0))
    return np.asarray(extended, dtype=np.float32)


def _repeat_tail_action_chunk(actions: np.ndarray, total_horizon: int, period: int = 1) -> np.ndarray:
    """Speculate a tail by repeating the last action pattern instead of extrapolating deltas."""

    chunk = _to_numpy_action(actions)
    if total_horizon <= chunk.shape[0] or chunk.shape[0] == 0:
        return chunk
    period = max(1, min(int(period), chunk.shape[0]))
    tail = chunk[-period:]
    extended = [row.copy() for row in chunk]
    while len(extended) < total_horizon:
        for row in tail:
            if len(extended) >= total_horizon:
                break
            extended.append(row.copy())
    return np.asarray(extended, dtype=np.float32)


def _chunk_is_extension_safe(actions: np.ndarray, guard: ChunkGuard) -> bool:
    chunk = _to_numpy_action(actions)
    cfg = guard.config
    if chunk.shape[0] < 2:
        return False
    if chunk.shape[1] > cfg.gripper_index:
        if np.max(np.abs(np.diff(chunk[:, cfg.gripper_index]))) > cfg.gripper_change_threshold:
            return False
    pos_delta = float(np.max(np.linalg.norm(np.diff(chunk[:, :3], axis=0), axis=1)))
    rot_delta = float(np.max(np.linalg.norm(np.diff(chunk[:, 3:6], axis=0), axis=1))) if chunk.shape[1] >= 6 else 0.0
    return pos_delta <= cfg.smooth_position_delta and rot_delta <= cfg.smooth_rotation_delta


@torch.inference_mode()
def _extend_action_chunk_with_trajectory_head(
    actions: np.ndarray,
    trajectory_head,
    *,
    total_horizon: int,
    device: str,
    guard: ChunkGuard | None = None,
    blend_with_damped: float = 1.0,
    project_smooth: bool = False,
) -> np.ndarray:
    chunk = _to_numpy_action(actions)
    if trajectory_head is None or total_horizon <= chunk.shape[0]:
        return chunk
    tensor = torch.as_tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
    extended = trajectory_head.extend_chunk(tensor, total_horizon=total_horizon)
    out = extended[0].detach().cpu().numpy().astype(np.float32)
    if blend_with_damped < 1.0:
        damped = _extend_action_chunk(chunk, total_horizon)
        alpha = float(np.clip(blend_with_damped, 0.0, 1.0))
        out[chunk.shape[0] :] = (1.0 - alpha) * damped[chunk.shape[0] :] + alpha * out[chunk.shape[0] :]
    if project_smooth and guard is not None:
        out = _project_smooth_extension(out, chunk, guard)
    return out


def _project_smooth_extension(actions: np.ndarray, reference_chunk: np.ndarray, guard: ChunkGuard) -> np.ndarray:
    """Clamp speculative tail dynamics into the guard's smooth-action cone."""

    out = _to_numpy_action(actions).copy()
    ref = _to_numpy_action(reference_chunk)
    cfg = guard.config
    start = min(ref.shape[0], out.shape[0])
    if start >= out.shape[0]:
        return out
    if out.shape[1] > cfg.gripper_index and ref.shape[1] > cfg.gripper_index:
        out[start:, cfg.gripper_index] = ref[-1, cfg.gripper_index]
    prev = out[start - 1].copy()
    for idx in range(start, out.shape[0]):
        cur = out[idx].copy()
        pos_delta = cur[:3] - prev[:3]
        pos_norm = float(np.linalg.norm(pos_delta))
        if pos_norm > cfg.smooth_position_delta:
            cur[:3] = prev[:3] + pos_delta * (cfg.smooth_position_delta / max(pos_norm, 1e-8))
        if out.shape[1] >= 6:
            rot_delta = cur[3:6] - prev[3:6]
            rot_norm = float(np.linalg.norm(rot_delta))
            if rot_norm > cfg.smooth_rotation_delta:
                cur[3:6] = prev[3:6] + rot_delta * (cfg.smooth_rotation_delta / max(rot_norm, 1e-8))
        out[idx] = np.clip(cur, -1.0, 1.0)
        prev = out[idx].copy()
    return out


@dataclass
class PredictionTrace:
    actions: np.ndarray
    elapsed_ms: float
    token_count: int | None = None
    token_ids: torch.Tensor | None = None
    stats: dict[str, Any] | None = None


@torch.inference_mode()
def _predict_action_chunk(
    policy,
    batch: dict[str, Any],
    postprocessor,
    device: str,
    token_adapter: PI0FastTokenLogitAdapter | None = None,
    early_stop_action_end: bool = False,
) -> PredictionTrace:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    token_count: int | None = None
    token_ids: torch.Tensor | None = None
    if token_adapter is not None:
        trace = token_adapter.predict_action_chunk_with_trace(batch, early_stop_action_end=early_stop_action_end)
        raw_action = trace.actions
        token_count = trace.token_count
        token_ids = trace.token_ids.detach()
        stats = trace.stats
    elif hasattr(policy, "predict_action_chunk"):
        raw_action = policy.predict_action_chunk(batch)
        stats = None
    else:
        raw_action = policy.select_action(batch)
        stats = None

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        processed = postprocessor(raw_action)
    except Exception:
        processed = raw_action

    actions = _to_numpy_action(processed)
    if hasattr(policy, "_last_action_tokens"):
        try:
            token_count = int(np.asarray(policy._last_action_tokens).size)
        except Exception:
            token_count = None
    return PredictionTrace(actions=actions, elapsed_ms=elapsed_ms, token_count=token_count, token_ids=token_ids, stats=stats)


@torch.inference_mode()
def _predict_target_eos_chunk(
    token_adapter: PI0FastTokenLogitAdapter,
    batch: dict[str, Any],
    postprocessor,
    device: str,
    *,
    constrained_action_vocab: bool = False,
    constrained_action_vocab_size: int | None = None,
    constrained_text_vocab_size: int | None = None,
    constrained_full_head_margin: float | None = None,
    constrained_force_action_prefix: bool = True,
    constrained_structural_token_radius: int = 512,
    force_action_prefix: bool = False,
) -> PredictionTrace:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if constrained_action_vocab:
        trace = token_adapter.predict_action_chunk_action_end(
            batch,
            constrained_action_vocab=True,
            constrained_action_vocab_size=constrained_action_vocab_size,
            constrained_text_vocab_size=constrained_text_vocab_size,
            constrained_full_head_margin=constrained_full_head_margin,
            constrained_force_action_prefix=constrained_force_action_prefix,
            constrained_structural_token_radius=constrained_structural_token_radius,
            force_action_prefix=force_action_prefix,
        )
    else:
        trace = token_adapter.predict_action_chunk_action_end(
            batch,
            force_action_prefix=force_action_prefix,
        )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        processed = postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return PredictionTrace(
        actions=_to_numpy_action(processed),
        elapsed_ms=elapsed_ms,
        token_count=trace.token_count,
        token_ids=trace.token_ids.detach(),
        stats=trace.stats,
    )


@torch.inference_mode()
def _predict_adaptive_prefix_chunk(
    token_adapter: PI0FastTokenLogitAdapter,
    batch: dict[str, Any],
    postprocessor,
    device: str,
    checkpoints: list[int],
    stable_tolerance: float,
    stable_checks: int,
    early_stable_checks: int | None,
    early_max_stable_checkpoint: int | None,
    skip_unproductive_checks: bool,
    skip_unproductive_after_checkpoint: int,
    continue_to_action_end_on_unstable: bool,
    prefix_gate,
    prefix_gate_threshold: float,
) -> PredictionTrace:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trace = token_adapter.predict_action_chunk_adaptive_prefix_cutoff(
        batch,
        checkpoints=checkpoints,
        stable_tolerance=stable_tolerance,
        stable_checks=stable_checks,
        early_stable_checks=early_stable_checks,
        early_max_stable_checkpoint=early_max_stable_checkpoint,
        skip_unproductive_checks=skip_unproductive_checks,
        skip_unproductive_after_checkpoint=skip_unproductive_after_checkpoint,
        continue_to_action_end_on_unstable=continue_to_action_end_on_unstable,
        prefix_gate=prefix_gate,
        prefix_gate_threshold=prefix_gate_threshold,
        early_stop_action_end=True,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        processed = postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return PredictionTrace(
        actions=_to_numpy_action(processed),
        elapsed_ms=elapsed_ms,
        token_count=trace.token_count,
        token_ids=trace.token_ids.detach(),
        stats={**(trace.stats or {}), "_logits": trace.logits.detach()},
    )


@torch.inference_mode()
def _predict_prefix_cutoff_chunk(
    token_adapter: PI0FastTokenLogitAdapter,
    batch: dict[str, Any],
    postprocessor,
    device: str,
    cutoff_tokens: int,
    collect_logits: bool = False,
    force_action_prefix: bool = False,
) -> PredictionTrace:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trace = token_adapter.predict_action_chunk_prefix_cutoff(
        batch,
        cutoff_tokens=cutoff_tokens,
        early_stop_action_end=True,
        collect_logits=collect_logits,
        force_action_prefix=force_action_prefix,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        processed = postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return PredictionTrace(
        actions=_to_numpy_action(processed),
        elapsed_ms=elapsed_ms,
        token_count=trace.token_count,
        token_ids=trace.token_ids.detach(),
        stats=trace.stats,
    )


@torch.inference_mode()
def _predict_ngram_spec_chunk(
    token_adapter: PI0FastTokenLogitAdapter,
    batch: dict[str, Any],
    postprocessor,
    device: str,
    ngram_drafter: NgramFastTokenDrafter,
    lookahead: int,
    reuse_full_blocks: bool,
    min_verify_margin: float = 0.0,
    emit_bonus_token: bool = False,
    verify_from_scratch: bool = False,
    replay_accepted_cache: bool = False,
    resync_accepted_cache: bool = False,
    diagnose_verify_alignment: bool = False,
    dynamic_lookahead: bool = False,
    min_lookahead: int = 1,
    lookahead_growth: int = 1,
    lookahead_shrink: int = 4,
    tree_width: int = 1,
    tree_branch_width: int = 4,
    dynamic_tree_width: bool = False,
    min_tree_width: int = 1,
    tree_width_growth: int = 1,
    tree_width_shrink: int = 1,
    tree_anchor_target_token: bool = False,
    tree_anchor_target_continuation: bool = False,
    early_stop_action_end: bool = True,
) -> PredictionTrace:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trace = token_adapter.predict_action_chunk_ngram_speculative(
        batch,
        drafter=ngram_drafter,
        lookahead=lookahead,
        reuse_full_blocks=reuse_full_blocks,
        min_verify_margin=min_verify_margin,
        verify_from_scratch=verify_from_scratch,
        emit_bonus_token=emit_bonus_token,
        replay_accepted_cache=replay_accepted_cache,
        resync_accepted_cache=resync_accepted_cache,
        diagnose_verify_alignment=diagnose_verify_alignment,
        dynamic_lookahead=dynamic_lookahead,
        min_lookahead=min_lookahead,
        lookahead_growth=lookahead_growth,
        lookahead_shrink=lookahead_shrink,
        tree_width=tree_width,
        tree_branch_width=tree_branch_width,
        dynamic_tree_width=dynamic_tree_width,
        min_tree_width=min_tree_width,
        tree_width_growth=tree_width_growth,
        tree_width_shrink=tree_width_shrink,
        tree_anchor_target_token=tree_anchor_target_token,
        tree_anchor_target_continuation=tree_anchor_target_continuation,
        early_stop_action_end=early_stop_action_end,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        processed = postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return PredictionTrace(
        actions=_to_numpy_action(processed),
        elapsed_ms=elapsed_ms,
        token_count=trace.token_count,
        token_ids=trace.token_ids.detach(),
        stats=trace.stats,
    )


@torch.inference_mode()
def _predict_block_spec_chunk(
    token_adapter: PI0FastTokenLogitAdapter,
    batch: dict[str, Any],
    postprocessor,
    device: str,
    block_drafter,
    block_token_map,
    block_gate,
    *,
    lookahead: int,
    min_draft_confidence: float,
    min_verify_confidence: float,
    min_verify_margin: float,
    block_gate_threshold: float,
    max_future_accept: int | None,
    min_future_accept: int,
    min_spec_position: int,
    reject_cooldown_steps: int,
    reject_cooldown_after: int,
    spec_fallback_cooldown_steps: int,
    spec_fallback_cooldown_after: int,
    allow_unknown_context: bool,
    repeat_token_draft: bool,
    repeat_token_min_run: int,
    repeat_pattern_draft: bool,
    repeat_pattern_max_period: int,
    repeat_pattern_min_position: int,
    pattern_only: bool,
    unverified_pattern_tail: bool,
    unverified_pattern_eos: bool,
    full_block_only: bool,
    accept_partial_blocks: bool,
    refine_steps: int,
    verify_from_scratch: bool,
    resync_accepted_cache: bool,
    draft_after_known_token: bool,
    max_decoding_steps: int | None = None,
    force_action_end: bool = False,
    early_stop_action_end: bool,
) -> PredictionTrace:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trace = token_adapter.predict_action_chunk_block_speculative(
        batch,
        block_drafter=block_drafter,
        token_map=block_token_map,
        block_gate=block_gate,
        lookahead=lookahead,
        min_draft_confidence=min_draft_confidence,
        min_verify_confidence=min_verify_confidence,
        min_verify_margin=min_verify_margin,
        block_gate_threshold=block_gate_threshold,
        max_future_accept=max_future_accept,
        min_future_accept=min_future_accept,
        min_spec_position=min_spec_position,
        reject_cooldown_steps=reject_cooldown_steps,
        reject_cooldown_after=reject_cooldown_after,
        spec_fallback_cooldown_steps=spec_fallback_cooldown_steps,
        spec_fallback_cooldown_after=spec_fallback_cooldown_after,
        allow_unknown_context=allow_unknown_context,
        repeat_token_draft=repeat_token_draft,
        repeat_token_min_run=repeat_token_min_run,
        repeat_pattern_draft=repeat_pattern_draft,
        repeat_pattern_max_period=repeat_pattern_max_period,
        repeat_pattern_min_position=repeat_pattern_min_position,
        pattern_only=pattern_only,
        unverified_pattern_tail=unverified_pattern_tail,
        unverified_pattern_eos=unverified_pattern_eos,
        full_block_only=full_block_only,
        accept_partial_blocks=accept_partial_blocks,
        refine_steps=refine_steps,
        verify_from_scratch=verify_from_scratch,
        resync_accepted_cache=resync_accepted_cache,
        draft_after_known_token=draft_after_known_token,
        max_decoding_steps=max_decoding_steps,
        force_action_end=force_action_end,
        early_stop_action_end=early_stop_action_end,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        processed = postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return PredictionTrace(
        actions=_to_numpy_action(processed),
        elapsed_ms=elapsed_ms,
        token_count=trace.token_count,
        token_ids=trace.token_ids.detach(),
        stats=trace.stats,
    )


def _ngram_spec_is_safe(
    prediction: PredictionTrace,
    *,
    min_acceptance: float,
    min_tokens_per_forward: float,
    max_fallback_rate: float,
) -> bool:
    stats = prediction.stats or {}
    acceptance = float(stats.get("acceptance_rate", 0.0))
    tokens_per_forward = float(stats.get("tokens_per_target_forward", 0.0))
    fallback = float(stats.get("fallback_forwards", 0.0))
    target = float(stats.get("target_forwards", 1.0))
    fallback_rate = fallback / max(target, 1.0)
    return (
        acceptance >= min_acceptance
        and tokens_per_forward >= min_tokens_per_forward
        and fallback_rate <= max_fallback_rate
    )


@torch.inference_mode()
def _predict_medusa_spec_chunk(
    token_adapter: PI0FastTokenLogitAdapter,
    batch: dict[str, Any],
    postprocessor,
    device: str,
    medusa_head,
    medusa_token_map,
    *,
    lookahead: int,
    min_draft_confidence: float,
    min_verify_confidence: float,
    min_spec_position: int,
    accept_partial_blocks: bool,
    replay_accepted_cache: bool,
    resync_accepted_cache: bool,
    verify_from_scratch: bool,
    early_stop_action_end: bool,
) -> PredictionTrace:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trace = token_adapter.predict_action_chunk_medusa_speculative(
        batch,
        medusa_head=medusa_head,
        token_map=medusa_token_map,
        lookahead=lookahead,
        min_draft_confidence=min_draft_confidence,
        min_verify_confidence=min_verify_confidence,
        min_spec_position=min_spec_position,
        accept_partial_blocks=accept_partial_blocks,
        replay_accepted_cache=replay_accepted_cache,
        resync_accepted_cache=resync_accepted_cache,
        verify_from_scratch=verify_from_scratch,
        early_stop_action_end=early_stop_action_end,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        processed = postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return PredictionTrace(
        actions=_to_numpy_action(processed),
        elapsed_ms=elapsed_ms,
        token_count=trace.token_count,
        token_ids=trace.token_ids.detach(),
        stats=trace.stats,
    )


@torch.inference_mode()
def _select_queued_action(policy, batch: dict[str, Any], postprocessor, device: str) -> PredictionTrace:
    """Mirror LeRobot eval's select_action path, including the policy's action queue."""

    queue = getattr(policy, "_action_queue", None)
    will_refresh = queue is None or len(queue) == 0
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    raw_action = policy.select_action(batch)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        processed = postprocessor(raw_action)
    except Exception:
        processed = raw_action

    actions = _to_numpy_action(processed)
    return PredictionTrace(
        actions=actions,
        elapsed_ms=elapsed_ms,
        token_count=None,
        token_ids=None,
    ), will_refresh


def _env_step(env, action: np.ndarray, env_postprocessor) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    action_t = torch.as_tensor(action.reshape(1, -1), dtype=torch.float32)
    try:
        transition = env_postprocessor({"action": action_t})
        action_np = transition["action"].detach().cpu().numpy()
    except Exception:
        action_np = action_t.numpy()

    observation, reward, terminated, truncated, info = env.step(action_np)
    reward_scalar = float(np.asarray(reward).reshape(-1)[0])
    terminated_b = bool(np.asarray(terminated).reshape(-1)[0])
    truncated_b = bool(np.asarray(truncated).reshape(-1)[0])
    return observation, reward_scalar, terminated_b, truncated_b, info


def _prediction_action_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    horizon = min(a.shape[0], b.shape[0])
    dim = min(a.shape[1], b.shape[1]) if a.ndim == 2 and b.ndim == 2 else 0
    if horizon == 0 or dim == 0:
        return float("inf"), float("inf")
    delta = np.abs(a[:horizon, :dim] - b[:horizon, :dim])
    return float(np.max(delta)), float(np.mean(delta))


def _prediction_token_diff_stats(prediction: PredictionTrace, target: PredictionTrace, prefix: str) -> dict[str, float]:
    if prediction.token_ids is None or target.token_ids is None:
        return {
            f"{prefix}_token_comparison_available": 0.0,
            f"{prefix}_token_exact": 0.0,
        }
    pred = prediction.token_ids.detach().flatten().cpu().tolist()
    tgt = target.token_ids.detach().flatten().cpu().tolist()
    shared = min(len(pred), len(tgt))
    mismatch_pos = -1
    for idx in range(shared):
        if int(pred[idx]) != int(tgt[idx]):
            mismatch_pos = idx
            break
    if mismatch_pos < 0 and len(pred) != len(tgt):
        mismatch_pos = shared
    pred_token = pred[mismatch_pos] if 0 <= mismatch_pos < len(pred) else -1
    tgt_token = tgt[mismatch_pos] if 0 <= mismatch_pos < len(tgt) else -1
    exact = float(mismatch_pos < 0)
    return {
        f"{prefix}_token_comparison_available": 1.0,
        f"{prefix}_token_exact": exact,
        f"{prefix}_token_first_mismatch_pos": float(mismatch_pos),
        f"{prefix}_token_pred_at_mismatch": float(pred_token),
        f"{prefix}_token_target_at_mismatch": float(tgt_token),
        f"{prefix}_token_pred_len": float(len(pred)),
        f"{prefix}_token_target_len": float(len(tgt)),
    }


def _adapter_action_end_token_id(token_adapter: PI0FastTokenLogitAdapter) -> int:
    value = getattr(token_adapter, "action_end_token_id")
    return int(value() if callable(value) else value)


@torch.no_grad()
def _prefix_gate_probability_for_prediction(
    prediction: PredictionTrace,
    *,
    cutoff_tokens: int,
    prefix_gate,
    device: str,
) -> float:
    if prefix_gate is None or prediction.token_ids is None:
        return 0.0
    token_ids = prediction.token_ids.to(device)
    logits = None
    if prediction.stats is not None:
        logits = prediction.stats.get("_logits")
    token_stats = {"logprob_mean": 0.0, "logprob_min": 0.0, "entropy_mean": 0.0, "entropy_max": 0.0}
    if isinstance(logits, torch.Tensor) and logits.numel():
        logits = logits.to(device).float()
        steps = min(logits.shape[1], token_ids.shape[1])
        if steps:
            ids = token_ids[:, :steps].to(device)
            log_probs = F.log_softmax(logits[:, :steps, :], dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            probs = torch.softmax(logits[:, :steps, :], dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            token_stats = {
                "logprob_mean": float(log_probs.mean().item()),
                "logprob_min": float(log_probs.min().item()),
                "entropy_mean": float(entropy.mean().item()),
                "entropy_max": float(entropy.max().item()),
            }
    forced_eos = float(not isinstance(logits, torch.Tensor) or token_ids.shape[1] > logits.shape[1])
    token_count = int(token_ids.shape[1])
    row = {
        "cutoff_norm": cutoff_tokens / 256.0,
        "token_count_norm": token_count / 256.0,
        "forced_eos": forced_eos,
        **token_stats,
        **action_feature_values(prediction.actions),
    }
    features = torch.tensor(
        [[float(row.get(name, 0.0)) for name in PREFIX_GATE_FEATURES]],
        dtype=torch.float32,
        device=device,
    )
    return float(prefix_gate.probability(features).item())


@torch.no_grad()
def _action_gate_probability_for_prediction(
    prediction: PredictionTrace,
    *,
    action_gate,
    threshold: float,
    device: str,
    max_decoding_steps: int,
) -> tuple[bool, float]:
    if action_gate is None:
        return True, 1.0
    features_dict = action_gate_feature_values(
        prediction.actions,
        token_count=prediction.token_count,
        stats=prediction.stats,
        max_decoding_steps=max_decoding_steps,
    )
    features = torch.tensor(
        [[float(features_dict.get(name, 0.0)) for name in ACTION_GATE_FEATURES]],
        dtype=torch.float32,
        device=device,
    )
    expected_dim = int(getattr(getattr(action_gate, "config", None), "input_dim", features.shape[1]))
    if features.shape[1] > expected_dim:
        features = features[:, :expected_dim]
    elif features.shape[1] < expected_dim:
        features = torch.nn.functional.pad(features, (0, expected_dim - features.shape[1]))
    probability = float(action_gate.probability(features).item())
    return probability >= threshold, probability


@torch.no_grad()
def _stable_block_prefix_prediction(
    token_adapter: PI0FastTokenLogitAdapter,
    prediction: PredictionTrace,
    postprocessor,
    *,
    cutoffs: list[int],
    stable_tolerance: float,
    stable_checks: int,
) -> tuple[PredictionTrace | None, dict[str, Any]]:
    if prediction.token_ids is None or not cutoffs:
        return None, {"stable_prefix_selected": 0.0, "stable_prefix_reason": "missing_tokens"}
    token_ids = prediction.token_ids
    action_end = _adapter_action_end_token_id(token_adapter)
    previous_actions: np.ndarray | None = None
    stable_count = 0
    checked: list[dict[str, Any]] = []
    required = max(int(stable_checks), 1)
    for cutoff in sorted({int(value) for value in cutoffs if int(value) > 0}):
        keep = min(cutoff, int(token_ids.shape[1]))
        prefix = token_ids[:, :keep]
        forced_end = int(prefix.shape[1] == 0 or int(prefix[0, -1].item()) != action_end)
        if forced_end:
            eos = torch.tensor([[action_end]], dtype=prefix.dtype, device=prefix.device)
            prefix = torch.cat([prefix, eos], dim=1)
        try:
            decoded = token_adapter._detokenize_generated_actions(prefix)
            try:
                processed = postprocessor(decoded)
            except Exception:
                processed = decoded
            actions = _to_numpy_action(processed)
        except Exception as exc:  # noqa: BLE001
            checked.append({"cutoff": cutoff, "error": repr(exc)})
            previous_actions = None
            stable_count = 0
            continue
        max_delta = None
        if previous_actions is not None:
            max_delta, _mean_delta = _prediction_action_diff(actions, previous_actions)
            if max_delta <= stable_tolerance:
                stable_count += 1
            else:
                stable_count = 0
        previous_actions = actions
        checked.append(
            {
                "cutoff": cutoff,
                "tokens": int(prefix.shape[1]),
                "forced_end": forced_end,
                "max_delta": max_delta,
                "stable_count": stable_count,
            }
        )
        if stable_count >= required:
            stats = {
                **(prediction.stats or {}),
                "stable_prefix_selected": 1.0,
                "stable_prefix_cutoff": cutoff,
                "stable_prefix_tokens": int(prefix.shape[1]),
                "stable_prefix_checks": checked,
            }
            return (
                PredictionTrace(
                    actions=actions,
                    elapsed_ms=prediction.elapsed_ms,
                    token_count=int(prefix.shape[1]),
                    token_ids=prefix.detach(),
                    stats=stats,
                ),
                stats,
            )
    return (
        None,
        {
            **(prediction.stats or {}),
            "stable_prefix_selected": 0.0,
            "stable_prefix_checks": checked,
        },
    )


def run_episode(
    *,
    mode: str,
    episode: int,
    seed: int,
    task: str,
    task_id: int | None,
    max_steps: int,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    policy_preprocessor,
    policy_postprocessor,
    preprocess_observation,
    controller: ChunkExecutionController,
    drafter: RetrievalChunkDrafter | None,
    token_adapter: PI0FastTokenLogitAdapter | None,
    ngram_drafter: NgramFastTokenDrafter | None,
    pattern_drafter: PatternFastTokenDrafter | None,
    block_drafter,
    block_token_map,
    block_gate,
    block_action_gate,
    block_action_gate_threshold: float,
    cutoff_selector,
    cutoff_selector_risk: float,
    medusa_head,
    medusa_token_map,
    medusa_lookahead: int,
    medusa_min_draft_confidence: float,
    medusa_min_verify_confidence: float,
    medusa_min_spec_position: int,
    medusa_accept_partial_blocks: bool,
    medusa_replay_accepted_cache: bool,
    medusa_resync_accepted_cache: bool,
    medusa_verify_from_scratch: bool,
    block_lookahead: int,
    block_min_draft_confidence: float,
    block_min_verify_confidence: float,
    block_min_verify_margin: float,
    block_gate_threshold: float,
    block_max_future_accept: int | None,
    block_min_future_accept: int,
    block_min_spec_position: int,
    block_reject_cooldown_steps: int,
    block_reject_cooldown_after: int,
    block_spec_fallback_cooldown_steps: int,
    block_spec_fallback_cooldown_after: int,
    block_allow_unknown_context: bool,
    block_repeat_token_draft: bool,
    block_repeat_token_min_run: int,
    block_repeat_pattern_draft: bool,
    block_repeat_pattern_max_period: int,
    block_repeat_pattern_min_position: int,
    block_pattern_only: bool,
    block_unverified_pattern_tail: bool,
    block_unverified_pattern_eos: bool,
    block_full_block_only: bool,
    block_accept_partial_blocks: bool,
    block_refine_steps: int,
    block_verify_from_scratch: bool,
    block_resync_accepted_cache: bool,
    block_draft_after_known_token: bool,
    ngram_lookahead: int,
    ngram_reuse_full_blocks: bool,
    pattern_lookahead: int,
    pattern_reuse_full_blocks: bool,
    pattern_min_verify_margin: float,
    pattern_emit_bonus_token: bool,
    pattern_verify_from_scratch: bool,
    pattern_replay_accepted_cache: bool,
    pattern_resync_accepted_cache: bool,
    pattern_diagnose_verify_alignment: bool,
    pattern_dynamic_lookahead: bool,
    pattern_min_lookahead: int,
    pattern_lookahead_growth: int,
    pattern_lookahead_shrink: int,
    pattern_tree_width: int,
    pattern_tree_branch_width: int,
    pattern_dynamic_tree_width: bool,
    pattern_min_tree_width: int,
    pattern_tree_width_growth: int,
    pattern_tree_width_shrink: int,
    pattern_tree_anchor_target_token: bool,
    pattern_tree_anchor_target_continuation: bool,
    trajectory_head,
    trajectory_tail_blend: float,
    trajectory_project_smooth: bool,
    action_corrector,
    action_corrector_refine_steps: int,
    action_corrector_min_confidence: float,
    action_corrector_project_smooth: bool,
    action_corrector_preserve_prefix_actions: int,
    action_corrector_blend: float,
    guarded_min_acceptance: float,
    guarded_min_tokens_per_forward: float,
    guarded_max_fallback_rate: float,
    adaptive_prefix_checkpoints: list[int],
    adaptive_stable_tolerance: float,
    adaptive_stable_checks: int,
    adaptive_early_stable_checks: int | None,
    adaptive_early_max_stable_checkpoint: int | None,
    adaptive_skip_unproductive_checks: bool,
    adaptive_skip_unproductive_after_checkpoint: int,
    adaptive_continue_to_action_end_on_unstable: bool,
    adaptive_prefix_gate,
    adaptive_prefix_gate_threshold: float,
    target_eos_constrained_action_vocab_size: int,
    target_eos_constrained_text_vocab_size: int,
    target_eos_constrained_full_head_margin: float | None,
    target_eos_constrained_no_force_prefix: bool,
    target_eos_constrained_structural_token_radius: int,
    token_trace_sink: TokenTraceSink | None,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype | None = None,
) -> EpisodeResult:
    policy.reset()
    controller.reset()
    if pattern_drafter is not None and hasattr(pattern_drafter, "reset_history"):
        pattern_drafter.reset_history()
    observation, _info = env.reset(seed=[seed])
    done = False
    steps = 0
    reward_sum = 0.0
    success = False
    control_step_ms: list[float] = []
    wall_start = time.perf_counter()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp and device.startswith("cuda") and amp_dtype is not None
        else torch.autocast(device_type="cuda")
        if use_amp and device.startswith("cuda")
        else nullcontext()
    )

    while not done and steps < max_steps:
        step_start = time.perf_counter()
        prediction: PredictionTrace | None = None
        buffered = controller.pop() if mode != "baseline" else None
        if buffered is None:
            state_key = _state_key(observation)
            retrieval_draft = drafter.draft(state_key) if drafter is not None else None
            batch = _prepare_observation(observation, env, env_preprocessor, policy_preprocessor, preprocess_observation)

            exact_draft_accepted = False
            if mode.startswith("exact_fast_sd") and token_adapter is not None and retrieval_draft is not None and retrieval_draft.token_ids:
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                verify_t0 = time.perf_counter()
                draft_tokens = torch.as_tensor(retrieval_draft.token_ids, dtype=torch.long)
                with autocast_ctx:
                    verify = token_adapter.verify_draft_tokens(batch, draft_tokens)
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                verify_ms = (time.perf_counter() - verify_t0) * 1000
                controller.stats.exact_verifies += 1
                controller.stats.exact_drafted_tokens += int(verify.draft_token_ids.shape[-1])
                controller.stats.exact_accepted_tokens += int(verify.accepted_prefix)
                exact_draft_accepted = verify.accepted_prefix == verify.draft_token_ids.shape[-1]
                if exact_draft_accepted:
                    controller.offer_chunk(
                        retrieval_draft.action_chunk,
                        confidence=1.0,
                        inference_ms=verify_ms,
                        token_count=int(verify.draft_token_ids.shape[-1]),
                    )
                    action = controller.pop()
                    if action is None:
                        action = retrieval_draft.action_chunk[0]

            if exact_draft_accepted:
                pass
            else:
                if mode == "baseline":
                    with autocast_ctx:
                        prediction, refreshed = _select_queued_action(policy, batch, policy_postprocessor, device)
                    if refreshed:
                        controller.stats.model_calls += 1
                        controller.stats.inference_ms.append(prediction.elapsed_ms)
                    action = prediction.actions[0]
                elif mode.startswith("ngram_sd"):
                    if token_adapter is None or ngram_drafter is None:
                        raise RuntimeError("ngram_sd mode requires token hooks and an n-gram drafter")
                    with autocast_ctx:
                        prediction = _predict_ngram_spec_chunk(
                            token_adapter,
                            batch,
                            policy_postprocessor,
                            device,
                            ngram_drafter,
                            ngram_lookahead,
                            ngram_reuse_full_blocks,
                            0.0,
                            False,
                            early_stop_action_end=True,
                        )
                    controller.stats.record_trace_stats(prediction.stats)
                    guarded_fallback = mode.startswith("ngram_sd_guarded") and not _ngram_spec_is_safe(
                        prediction,
                        min_acceptance=guarded_min_acceptance,
                        min_tokens_per_forward=guarded_min_tokens_per_forward,
                        max_fallback_rate=guarded_max_fallback_rate,
                    )
                    if guarded_fallback:
                        with autocast_ctx:
                            fallback_prediction = _predict_action_chunk(
                                policy,
                                batch,
                                policy_postprocessor,
                                device,
                                token_adapter,
                                early_stop_action_end=True,
                            )
                        prediction = PredictionTrace(
                            actions=fallback_prediction.actions,
                            elapsed_ms=prediction.elapsed_ms + fallback_prediction.elapsed_ms,
                            token_count=fallback_prediction.token_count,
                            token_ids=fallback_prediction.token_ids,
                            stats={
                                **(prediction.stats or {}),
                                "guarded_fallback": True,
                                "fallback_target_ms": fallback_prediction.elapsed_ms,
                            },
                        )
                        controller.stats.guard_reasons.update(["guarded_sd_fallback"])
                    if mode.startswith("ngram_sd_validate"):
                        with autocast_ctx:
                            target_prediction = _predict_action_chunk(
                                policy,
                                batch,
                                policy_postprocessor,
                                device,
                                token_adapter,
                                early_stop_action_end=True,
                            )
                        max_diff, mean_diff = _prediction_action_diff(prediction.actions, target_prediction.actions)
                        controller.stats.exact_verifies += 1
                        controller.stats.action_max_diffs.append(max_diff)
                        controller.stats.action_mean_diffs.append(mean_diff)
                        if max_diff != 0.0:
                            controller.stats.guard_reasons.update(["ngram_sd_action_diff_fallback"])
                            prediction = PredictionTrace(
                                actions=target_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + target_prediction.elapsed_ms,
                                token_count=target_prediction.token_count,
                                token_ids=target_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    "fallback_target_ms": target_prediction.elapsed_ms,
                                    "fallback_action_max_diff": max_diff,
                                },
                            )
                    if mode.startswith("ngram_sd_direct") or mode.startswith("ngram_sd_guarded"):
                        controller.queue.clear()
                        for queued_action in prediction.actions:
                            controller.queue.append(queued_action.copy())
                        controller.stats.chunks_seen += 1
                        controller.stats.chunks_accepted += 1
                        controller.stats.actions_enqueued += int(prediction.actions.shape[0])
                        controller.stats.accepted_windows.append(int(prediction.actions.shape[0]))
                        controller.stats.model_calls += 1
                        controller.stats.inference_ms.append(prediction.elapsed_ms)
                        if prediction.token_count is not None:
                            controller.stats.token_counts.append(prediction.token_count)
                    else:
                        controller.offer_chunk(
                            prediction.actions,
                            confidence=1.0,
                            inference_ms=prediction.elapsed_ms,
                            token_count=prediction.token_count,
                        )
                    action = controller.pop()
                    if action is None:
                        action = prediction.actions[0]
                elif mode.startswith("pattern_sd"):
                    if token_adapter is None or pattern_drafter is None:
                        raise RuntimeError("pattern_sd mode requires token hooks")
                    with autocast_ctx:
                        prediction = _predict_ngram_spec_chunk(
                            token_adapter,
                            batch,
                            policy_postprocessor,
                            device,
                            pattern_drafter,
                            pattern_lookahead,
                            pattern_reuse_full_blocks,
                            pattern_min_verify_margin,
                            pattern_emit_bonus_token,
                            pattern_verify_from_scratch,
                            pattern_replay_accepted_cache,
                            pattern_resync_accepted_cache,
                            pattern_diagnose_verify_alignment,
                            pattern_dynamic_lookahead,
                            pattern_min_lookahead,
                            pattern_lookahead_growth,
                            pattern_lookahead_shrink,
                            pattern_tree_width,
                            pattern_tree_branch_width,
                            pattern_dynamic_tree_width,
                            pattern_min_tree_width,
                            pattern_tree_width_growth,
                            pattern_tree_width_shrink,
                            pattern_tree_anchor_target_token,
                            pattern_tree_anchor_target_continuation,
                            early_stop_action_end=True,
                        )
                    prediction.stats = {**(prediction.stats or {}), "pattern_sd": 1.0}
                    controller.stats.record_trace_stats(prediction.stats)
                    guarded_fallback = mode.startswith("pattern_sd_guarded") and not _ngram_spec_is_safe(
                        prediction,
                        min_acceptance=guarded_min_acceptance,
                        min_tokens_per_forward=guarded_min_tokens_per_forward,
                        max_fallback_rate=guarded_max_fallback_rate,
                    )
                    if guarded_fallback:
                        with autocast_ctx:
                            fallback_prediction = _predict_action_chunk(
                                policy,
                                batch,
                                policy_postprocessor,
                                device,
                                token_adapter,
                                early_stop_action_end=True,
                            )
                        prediction = PredictionTrace(
                            actions=fallback_prediction.actions,
                            elapsed_ms=prediction.elapsed_ms + fallback_prediction.elapsed_ms,
                            token_count=fallback_prediction.token_count,
                            token_ids=fallback_prediction.token_ids,
                            stats={
                                **(prediction.stats or {}),
                                "guarded_fallback": True,
                                "fallback_target_ms": fallback_prediction.elapsed_ms,
                            },
                        )
                        controller.stats.guard_reasons.update(["pattern_sd_guarded_fallback"])
                    if mode.startswith("pattern_sd_validate"):
                        with autocast_ctx:
                            target_prediction = _predict_action_chunk(
                                policy,
                                batch,
                                policy_postprocessor,
                                device,
                                token_adapter,
                                early_stop_action_end=True,
                            )
                        if token_trace_sink is not None:
                            token_trace_sink.record(
                                prediction,
                                mode=f"{mode}_pattern_raw",
                                task=task,
                                task_id=task_id,
                                episode=episode,
                                seed=seed,
                                step=steps,
                            )
                            token_trace_sink.record(
                                target_prediction,
                                mode=f"{mode}_target",
                                task=task,
                                task_id=task_id,
                                episode=episode,
                                seed=seed,
                                step=steps,
                            )
                        max_diff, mean_diff = _prediction_action_diff(prediction.actions, target_prediction.actions)
                        controller.stats.record_trace_stats(
                            _prediction_token_diff_stats(prediction, target_prediction, "pattern")
                        )
                        controller.stats.exact_verifies += 1
                        controller.stats.action_max_diffs.append(max_diff)
                        controller.stats.action_mean_diffs.append(mean_diff)
                        if max_diff != 0.0:
                            controller.stats.guard_reasons.update(["pattern_sd_action_diff_fallback"])
                            prediction = PredictionTrace(
                                actions=target_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + target_prediction.elapsed_ms,
                                token_count=target_prediction.token_count,
                                token_ids=target_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    "fallback_target_ms": target_prediction.elapsed_ms,
                                    "fallback_action_max_diff": max_diff,
                                    },
                                )
                    if prediction.token_ids is not None and hasattr(pattern_drafter, "observe"):
                        pattern_drafter.observe(prediction.token_ids[0].detach().cpu().tolist())
                    if mode.startswith("pattern_sd_direct") or mode.startswith("pattern_sd_guarded"):
                        controller.queue.clear()
                        for queued_action in prediction.actions:
                            controller.queue.append(queued_action.copy())
                        controller.stats.chunks_seen += 1
                        controller.stats.chunks_accepted += 1
                        controller.stats.actions_enqueued += int(prediction.actions.shape[0])
                        controller.stats.accepted_windows.append(int(prediction.actions.shape[0]))
                        controller.stats.model_calls += 1
                        controller.stats.inference_ms.append(prediction.elapsed_ms)
                        if prediction.token_count is not None:
                            controller.stats.token_counts.append(prediction.token_count)
                    else:
                        controller.offer_chunk(
                            prediction.actions,
                            confidence=1.0,
                            inference_ms=prediction.elapsed_ms,
                            token_count=prediction.token_count,
                        )
                    action = controller.pop()
                    if action is None:
                        action = prediction.actions[0]
                elif mode.startswith("medusa_sd"):
                    if token_adapter is None or medusa_head is None or medusa_token_map is None:
                        raise RuntimeError("medusa_sd mode requires token hooks and --medusa-checkpoint")
                    with autocast_ctx:
                        prediction = _predict_medusa_spec_chunk(
                            token_adapter,
                            batch,
                            policy_postprocessor,
                            device,
                            medusa_head,
                            medusa_token_map,
                            lookahead=medusa_lookahead,
                            min_draft_confidence=medusa_min_draft_confidence,
                            min_verify_confidence=medusa_min_verify_confidence,
                            min_spec_position=medusa_min_spec_position,
                            accept_partial_blocks=medusa_accept_partial_blocks,
                            replay_accepted_cache=medusa_replay_accepted_cache,
                            resync_accepted_cache=medusa_resync_accepted_cache,
                            verify_from_scratch=medusa_verify_from_scratch,
                            early_stop_action_end=True,
                        )
                    controller.stats.record_trace_stats(prediction.stats)
                    if mode.startswith("medusa_sd_guarded"):
                        decision = controller.guard.decide(prediction.actions, confidence=1.0)
                        if not decision.accepted:
                            controller.stats.guard_reasons.update(decision.reasons)
                            with autocast_ctx:
                                fallback_prediction = _predict_action_chunk(
                                    policy,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    token_adapter,
                                    early_stop_action_end=True,
                                )
                            prediction = PredictionTrace(
                                actions=fallback_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + fallback_prediction.elapsed_ms,
                                token_count=fallback_prediction.token_count,
                                token_ids=fallback_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    "fallback_target_ms": fallback_prediction.elapsed_ms,
                                    "guarded_fallback": 1.0,
                                },
                            )
                    elif mode.startswith("medusa_sd_project"):
                        prediction = PredictionTrace(
                            actions=_project_smooth_extension(
                                prediction.actions,
                                prediction.actions[:1],
                                controller.guard,
                            ),
                            elapsed_ms=prediction.elapsed_ms,
                            token_count=prediction.token_count,
                            token_ids=prediction.token_ids,
                            stats={**(prediction.stats or {}), "smooth_projected": 1.0},
                        )
                    controller.queue.clear()
                    for queued_action in prediction.actions:
                        controller.queue.append(queued_action.copy())
                    controller.stats.chunks_seen += 1
                    controller.stats.chunks_accepted += 1
                    controller.stats.actions_enqueued += int(prediction.actions.shape[0])
                    controller.stats.accepted_windows.append(int(prediction.actions.shape[0]))
                    controller.stats.model_calls += 1
                    controller.stats.inference_ms.append(prediction.elapsed_ms)
                    if prediction.token_count is not None:
                        controller.stats.token_counts.append(prediction.token_count)
                    action = controller.pop()
                    if action is None:
                        action = prediction.actions[0]
                elif mode.startswith("block_sd"):
                    if token_adapter is None or block_drafter is None or block_token_map is None:
                        raise RuntimeError("block_sd mode requires token hooks and --block-checkpoint")
                    stable_cutoffs: list[int] = []
                    if mode.startswith("block_sd_stable"):
                        stable_cutoffs = [
                            int(value)
                            for value in re.findall(r"\d+", mode.split("block_sd_stable", 1)[1].split("_tol", 1)[0])
                        ]
                    selector_ms = 0.0
                    selector_stats: dict[str, float] = {}
                    if mode.startswith("block_sd_select"):
                        if cutoff_selector is None:
                            raise RuntimeError("block_sd_select modes require --cutoff-selector-checkpoint")
                        if device.startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        selector_t0 = time.perf_counter()
                        with torch.autocast(device_type="cuda", enabled=False) if device.startswith("cuda") else nullcontext():
                            selector_hidden = token_adapter.fast_prefix_hidden(batch).to(device=device, dtype=torch.float32)
                            selected_cutoff, selector_probs = cutoff_selector.choose(
                                selector_hidden,
                                max_risk=cutoff_selector_risk,
                            )
                        if device.startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        selector_ms = (time.perf_counter() - selector_t0) * 1000
                        block_cutoff_tokens = int(selected_cutoff)
                        selector_probs_cpu = selector_probs[0].detach().float().cpu().tolist()
                        selector_stats = {
                            "cutoff_selector_ms": selector_ms,
                            "cutoff_selector_selected": float(block_cutoff_tokens),
                        }
                        for cutoff, prob in zip(cutoff_selector.config.cutoffs, selector_probs_cpu, strict=False):
                            selector_stats[f"cutoff_selector_p_{cutoff}"] = float(prob)
                    else:
                        block_cutoff_tokens = max(stable_cutoffs) if stable_cutoffs else _mode_int_after(mode, "_cutoff", 0)
                    with autocast_ctx:
                        prediction = _predict_block_spec_chunk(
                            token_adapter,
                            batch,
                            policy_postprocessor,
                            device,
                            block_drafter,
                            block_token_map,
                            block_gate,
                            lookahead=block_lookahead,
                            min_draft_confidence=block_min_draft_confidence,
                            min_verify_confidence=block_min_verify_confidence,
                            min_verify_margin=block_min_verify_margin,
                            block_gate_threshold=block_gate_threshold,
                            max_future_accept=block_max_future_accept,
                            min_future_accept=block_min_future_accept,
                            min_spec_position=block_min_spec_position,
                            reject_cooldown_steps=block_reject_cooldown_steps,
                            reject_cooldown_after=block_reject_cooldown_after,
                            spec_fallback_cooldown_steps=block_spec_fallback_cooldown_steps,
                            spec_fallback_cooldown_after=block_spec_fallback_cooldown_after,
                            allow_unknown_context=block_allow_unknown_context,
                            repeat_token_draft=block_repeat_token_draft,
                            repeat_token_min_run=block_repeat_token_min_run,
                            repeat_pattern_draft=block_repeat_pattern_draft,
                            repeat_pattern_max_period=block_repeat_pattern_max_period,
                            repeat_pattern_min_position=block_repeat_pattern_min_position,
                            pattern_only=block_pattern_only,
                            unverified_pattern_tail=block_unverified_pattern_tail,
                            unverified_pattern_eos=block_unverified_pattern_eos,
                            full_block_only=block_full_block_only,
                            accept_partial_blocks=block_accept_partial_blocks,
                            refine_steps=block_refine_steps,
                            verify_from_scratch=block_verify_from_scratch,
                            resync_accepted_cache=block_resync_accepted_cache,
                            draft_after_known_token=block_draft_after_known_token,
                            max_decoding_steps=block_cutoff_tokens or None,
                            force_action_end=block_cutoff_tokens > 0,
                            early_stop_action_end=True,
                        )
                    if selector_ms:
                        prediction.elapsed_ms += selector_ms
                        prediction.stats = {**(prediction.stats or {}), **selector_stats}
                    controller.stats.record_trace_stats(prediction.stats)
                    if mode.startswith("block_sd_stable"):
                        stable_prediction, stable_stats = _stable_block_prefix_prediction(
                            token_adapter,
                            prediction,
                            policy_postprocessor,
                            cutoffs=stable_cutoffs,
                            stable_tolerance=adaptive_stable_tolerance,
                            stable_checks=adaptive_stable_checks,
                        )
                        if stable_prediction is not None:
                            prediction = stable_prediction
                            controller.stats.guard_reasons.update(["block_sd_stable_prefix_accept"])
                        elif "_fallback" in mode:
                            with autocast_ctx:
                                target_prediction = _predict_action_chunk(
                                    policy,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    token_adapter,
                                    early_stop_action_end=True,
                                )
                            controller.stats.guard_reasons.update(["block_sd_stable_prefix_fallback"])
                            prediction = PredictionTrace(
                                actions=target_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + target_prediction.elapsed_ms,
                                token_count=target_prediction.token_count,
                                token_ids=target_prediction.token_ids,
                                stats={
                                    **stable_stats,
                                    "stable_prefix_fallback_ms": target_prediction.elapsed_ms,
                                    "stable_prefix_fallback": 1.0,
                                },
                            )
                        else:
                            prediction.stats = stable_stats
                    if mode.startswith("block_sd_validate"):
                        with autocast_ctx:
                            target_prediction = _predict_action_chunk(
                                policy,
                                batch,
                                policy_postprocessor,
                                device,
                                token_adapter,
                                early_stop_action_end=True,
                            )
                        max_diff, mean_diff = _prediction_action_diff(prediction.actions, target_prediction.actions)
                        controller.stats.exact_verifies += 1
                        controller.stats.action_max_diffs.append(max_diff)
                        controller.stats.action_mean_diffs.append(mean_diff)
                        if max_diff != 0.0:
                            controller.stats.guard_reasons.update(["block_sd_action_diff_fallback"])
                            prediction = PredictionTrace(
                                actions=target_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + target_prediction.elapsed_ms,
                                token_count=target_prediction.token_count,
                                token_ids=target_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    "fallback_target_ms": target_prediction.elapsed_ms,
                                    "fallback_action_max_diff": max_diff,
                                },
                            )
                    if "_gatefull" in mode:
                        if block_cutoff_tokens <= 0:
                            raise RuntimeError("block_sd*_gatefull requires a _cutoffN mode")
                        if adaptive_prefix_gate is None:
                            raise RuntimeError("block_sd*_gatefull requires --adaptive-prefix-gate-checkpoint")
                        gate_prob = _prefix_gate_probability_for_prediction(
                            prediction,
                            cutoff_tokens=block_cutoff_tokens,
                            prefix_gate=adaptive_prefix_gate,
                            device=device,
                        )
                        prediction.stats = {
                            **(prediction.stats or {}),
                            "prefix_gate_probability": gate_prob,
                        }
                        if gate_prob < adaptive_prefix_gate_threshold:
                            with autocast_ctx:
                                target_prediction = _predict_action_chunk(
                                    policy,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    token_adapter,
                                    early_stop_action_end=True,
                                )
                            controller.stats.guard_reasons.update(["block_sd_cutoff_gate_full_refresh"])
                            prediction = PredictionTrace(
                                actions=target_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + target_prediction.elapsed_ms,
                                token_count=target_prediction.token_count,
                                token_ids=target_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    "gate_full_refresh_ms": target_prediction.elapsed_ms,
                                    "gate_full_refresh": 1.0,
                                },
                            )
                    if mode.startswith("block_sd_correct"):
                        if action_corrector is None:
                            raise RuntimeError("block_sd_correct modes require --action-corrector-checkpoint")
                        hidden = (prediction.stats or {}).get("prefix_hidden")
                        hidden_ms = 0.0
                        if hidden is None:
                            if device.startswith("cuda") and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            hidden_t0 = time.perf_counter()
                            with torch.autocast(device_type="cuda", enabled=False) if device.startswith("cuda") else nullcontext():
                                hidden = token_adapter.fast_prefix_hidden(batch)
                            if device.startswith("cuda") and torch.cuda.is_available():
                                torch.cuda.synchronize()
                            hidden_ms = (time.perf_counter() - hidden_t0) * 1000
                        if not torch.is_tensor(hidden):
                            hidden = torch.as_tensor(hidden, dtype=torch.float32)
                        hidden = hidden.to(device=device, dtype=torch.float32)
                        if hidden.ndim == 1:
                            hidden = hidden.unsqueeze(0)
                        elif hidden.ndim > 2:
                            hidden = hidden.reshape(hidden.shape[0], -1)

                        init_np = _to_numpy_action(prediction.actions)
                        horizon = int(action_corrector.config.action_horizon)
                        if init_np.shape[0] < horizon:
                            pad = np.repeat(init_np[-1:, :], horizon - init_np.shape[0], axis=0)
                            init_np = np.concatenate([init_np, pad], axis=0)
                        init_np = init_np[:horizon]
                        init = torch.as_tensor(init_np, dtype=torch.float32, device=device).unsqueeze(0)

                        if device.startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        correct_t0 = time.perf_counter()
                        with torch.autocast(device_type="cuda", enabled=False) if device.startswith("cuda") else nullcontext():
                            corrected, confidence, correct_stats = action_corrector.draft(
                                hidden,
                                refine_steps=action_corrector_refine_steps,
                                init=init,
                            )
                        if device.startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        correction_ms = (time.perf_counter() - correct_t0) * 1000
                        corrected_np = corrected[0].detach().float().cpu().numpy().astype(np.float32)
                        blend = float(np.clip(action_corrector_blend, 0.0, 1.0))
                        if blend < 1.0:
                            corrected_np = init_np + blend * (corrected_np - init_np)
                        preserve = min(max(int(action_corrector_preserve_prefix_actions), 0), corrected_np.shape[0])
                        if preserve:
                            corrected_np[:preserve] = init_np[:preserve]
                        if action_corrector_project_smooth:
                            corrected_np = _project_smooth_extension(corrected_np, prediction.actions, controller.guard)
                        conf_mean = float(confidence.mean().detach().cpu().item())
                        conf_min = float(confidence.min().detach().cpu().item())
                        correct_stats = {
                            key: float(value)
                            for key, value in (correct_stats or {}).items()
                            if isinstance(value, (int, float, bool, np.integer, np.floating))
                        }
                        correct_stats.update(
                            {
                                "action_corrector_ms": correction_ms,
                                "action_corrector_hidden_ms": hidden_ms,
                                "action_corrector_confidence": conf_mean,
                                "action_corrector_min_confidence": conf_min,
                            }
                        )
                        if conf_min < action_corrector_min_confidence and "_fallback" in mode:
                            with autocast_ctx:
                                target_prediction = _predict_action_chunk(
                                    policy,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    token_adapter,
                                    early_stop_action_end=True,
                                )
                            controller.stats.guard_reasons.update(["block_sd_correct_conf_fallback"])
                            prediction = PredictionTrace(
                                actions=target_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + hidden_ms + correction_ms + target_prediction.elapsed_ms,
                                token_count=target_prediction.token_count,
                                token_ids=target_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    **correct_stats,
                                    "fallback_target_ms": target_prediction.elapsed_ms,
                                    "action_corrector_fallback": 1.0,
                                },
                            )
                        else:
                            prediction = PredictionTrace(
                                actions=corrected_np,
                                elapsed_ms=prediction.elapsed_ms + hidden_ms + correction_ms,
                                token_count=prediction.token_count,
                                token_ids=prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    **correct_stats,
                                    "action_corrector_applied": 1.0,
                                },
                            )
                        controller.stats.record_trace_stats(prediction.stats)
                    if mode.startswith("block_sd_action_gate"):
                        gate_ok, gate_probability = _action_gate_probability_for_prediction(
                            prediction,
                            action_gate=block_action_gate,
                            threshold=block_action_gate_threshold,
                            device=device,
                            max_decoding_steps=policy.config.max_decoding_steps,
                        )
                        if prediction.stats is not None:
                            prediction.stats["action_gate_probability"] = gate_probability
                            prediction.stats["action_gate_accept"] = float(gate_ok)
                        if gate_ok:
                            controller.queue.clear()
                            for queued_action in prediction.actions:
                                controller.queue.append(queued_action.copy())
                            controller.stats.chunks_seen += 1
                            controller.stats.chunks_accepted += 1
                            controller.stats.actions_enqueued += int(prediction.actions.shape[0])
                            controller.stats.accepted_windows.append(int(prediction.actions.shape[0]))
                            controller.stats.model_calls += 1
                            controller.stats.inference_ms.append(prediction.elapsed_ms)
                            if prediction.token_count is not None:
                                controller.stats.token_counts.append(prediction.token_count)
                        else:
                            controller.stats.chunks_seen += 1
                            controller.stats.model_calls += 1
                            controller.stats.inference_ms.append(prediction.elapsed_ms)
                            if prediction.token_count is not None:
                                controller.stats.token_counts.append(prediction.token_count)
                            controller.stats.guard_reasons.update(["block_sd_action_gate_reject"])
                            with autocast_ctx:
                                fallback_prediction = _predict_action_chunk(
                                    policy,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    token_adapter,
                                    early_stop_action_end=True,
                                )
                            controller.queue.clear()
                            for queued_action in fallback_prediction.actions:
                                controller.queue.append(queued_action.copy())
                            controller.stats.chunks_seen += 1
                            controller.stats.chunks_accepted += 1
                            controller.stats.actions_enqueued += int(fallback_prediction.actions.shape[0])
                            controller.stats.accepted_windows.append(int(fallback_prediction.actions.shape[0]))
                            controller.stats.model_calls += 1
                            controller.stats.inference_ms.append(fallback_prediction.elapsed_ms)
                            if fallback_prediction.token_count is not None:
                                controller.stats.token_counts.append(fallback_prediction.token_count)
                            prediction = PredictionTrace(
                                actions=fallback_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + fallback_prediction.elapsed_ms,
                                token_count=fallback_prediction.token_count,
                                token_ids=fallback_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    "fallback_target_ms": fallback_prediction.elapsed_ms,
                                    "action_gate_probability": gate_probability,
                                    "action_gate_fallback": 1.0,
                                },
                            )
                    if mode.startswith("block_sd_guarded"):
                        decision = controller.offer_chunk(
                            prediction.actions,
                            confidence=1.0,
                            inference_ms=prediction.elapsed_ms,
                            token_count=prediction.token_count,
                        )
                        if not decision.accepted:
                            controller.stats.guard_reasons.update(["block_sd_guarded_target_fallback"])
                            with autocast_ctx:
                                fallback_prediction = _predict_action_chunk(
                                    policy,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    token_adapter,
                                    early_stop_action_end=True,
                                )
                            controller.offer_chunk(
                                fallback_prediction.actions,
                                confidence=1.0,
                                inference_ms=fallback_prediction.elapsed_ms,
                                token_count=fallback_prediction.token_count,
                            )
                            prediction = PredictionTrace(
                                actions=fallback_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + fallback_prediction.elapsed_ms,
                                token_count=fallback_prediction.token_count,
                                token_ids=fallback_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    "fallback_target_ms": fallback_prediction.elapsed_ms,
                                    "guarded_fallback": 1.0,
                                },
                            )
                    if mode.startswith("block_sd_extend"):
                        suffix = mode.removeprefix("block_sd_extend")
                        total_horizon = _leading_int(suffix, 15)
                        chunk = prediction.actions
                        if _chunk_is_extension_safe(chunk, controller.guard):
                            chunk = _extend_action_chunk(chunk, total_horizon)
                            controller.stats.guard_reasons.update(["block_sd_extend_applied"])
                        else:
                            controller.stats.guard_reasons.update(["block_sd_extend_unsafe"])
                        controller.queue.clear()
                        for queued_action in chunk:
                            controller.queue.append(queued_action.copy())
                        controller.stats.chunks_seen += 1
                        controller.stats.chunks_accepted += 1
                        controller.stats.actions_enqueued += int(chunk.shape[0])
                        controller.stats.accepted_windows.append(int(chunk.shape[0]))
                        controller.stats.model_calls += 1
                        controller.stats.inference_ms.append(prediction.elapsed_ms)
                        if prediction.token_count is not None:
                            controller.stats.token_counts.append(prediction.token_count)
                    elif mode.startswith("block_sd_action_gate") or mode.startswith("block_sd_guarded"):
                        pass
                    elif (
                        mode.startswith("block_sd_direct")
                        or mode.startswith("block_sd_validate")
                        or mode.startswith("block_sd_cutoff")
                        or mode.startswith("block_sd_select")
                        or mode.startswith("block_sd_stable")
                        or mode.startswith("block_sd_correct")
                    ):
                        controller.queue.clear()
                        for queued_action in prediction.actions:
                            controller.queue.append(queued_action.copy())
                        controller.stats.chunks_seen += 1
                        controller.stats.chunks_accepted += 1
                        controller.stats.actions_enqueued += int(prediction.actions.shape[0])
                        controller.stats.accepted_windows.append(int(prediction.actions.shape[0]))
                        controller.stats.model_calls += 1
                        controller.stats.inference_ms.append(prediction.elapsed_ms)
                        if prediction.token_count is not None:
                            controller.stats.token_counts.append(prediction.token_count)
                    else:
                        controller.offer_chunk(
                            prediction.actions,
                            confidence=1.0,
                            inference_ms=prediction.elapsed_ms,
                            token_count=prediction.token_count,
                        )
                    action = controller.pop()
                    if action is None:
                        action = prediction.actions[0]
                elif mode.startswith("ngram_extend"):
                    if token_adapter is None or ngram_drafter is None:
                        raise RuntimeError("ngram_extend mode requires token hooks and an n-gram drafter")
                    with autocast_ctx:
                        prediction = _predict_ngram_spec_chunk(
                            token_adapter,
                            batch,
                            policy_postprocessor,
                            device,
                            ngram_drafter,
                            ngram_lookahead,
                            ngram_reuse_full_blocks,
                            0.0,
                            early_stop_action_end=True,
                        )
                    suffix = mode.removeprefix("ngram_extend")
                    total_horizon = int(suffix) if suffix.isdigit() else 20
                    chunk = prediction.actions
                    if _chunk_is_extension_safe(chunk, controller.guard):
                        chunk = _extend_action_chunk(chunk, total_horizon)
                    controller.queue.clear()
                    for queued_action in chunk:
                        controller.queue.append(queued_action.copy())
                    controller.stats.chunks_seen += 1
                    controller.stats.chunks_accepted += 1
                    controller.stats.actions_enqueued += int(chunk.shape[0])
                    controller.stats.accepted_windows.append(int(chunk.shape[0]))
                    controller.stats.model_calls += 1
                    controller.stats.inference_ms.append(prediction.elapsed_ms)
                    if prediction.token_count is not None:
                        controller.stats.token_counts.append(prediction.token_count)
                    action = controller.pop()
                    if action is None:
                        action = chunk[0]
                elif mode.startswith("ngram_traj_tail"):
                    if token_adapter is None or ngram_drafter is None:
                        raise RuntimeError("ngram_traj_tail mode requires token hooks and an n-gram drafter")
                    if trajectory_head is None:
                        raise RuntimeError("ngram_traj_tail mode requires --trajectory-head-checkpoint")
                    with autocast_ctx:
                        prediction = _predict_ngram_spec_chunk(
                            token_adapter,
                            batch,
                            policy_postprocessor,
                            device,
                            ngram_drafter,
                            ngram_lookahead,
                            ngram_reuse_full_blocks,
                            0.0,
                            early_stop_action_end=True,
                        )
                    suffix = mode.removeprefix("ngram_traj_tail")
                    total_horizon = int(suffix) if suffix.isdigit() else 18
                    chunk = prediction.actions
                    if _chunk_is_extension_safe(chunk, controller.guard):
                        with torch.autocast(device_type="cuda", enabled=False) if device.startswith("cuda") else nullcontext():
                            chunk = _extend_action_chunk_with_trajectory_head(
                                chunk,
                                trajectory_head,
                                total_horizon=total_horizon,
                                device=device,
                                guard=controller.guard,
                                blend_with_damped=trajectory_tail_blend,
                                project_smooth=trajectory_project_smooth,
                            )
                        if not _chunk_is_extension_safe(chunk, controller.guard):
                            chunk = prediction.actions
                    controller.queue.clear()
                    for queued_action in chunk:
                        controller.queue.append(queued_action.copy())
                    controller.stats.chunks_seen += 1
                    controller.stats.chunks_accepted += 1
                    controller.stats.actions_enqueued += int(chunk.shape[0])
                    controller.stats.accepted_windows.append(int(chunk.shape[0]))
                    controller.stats.model_calls += 1
                    controller.stats.inference_ms.append(prediction.elapsed_ms)
                    if prediction.token_count is not None:
                        controller.stats.token_counts.append(prediction.token_count)
                    action = controller.pop()
                    if action is None:
                        action = chunk[0]
                else:
                    if (mode.startswith("target_eos") or mode.startswith("target_cutoff")) and token_adapter is None:
                        raise RuntimeError("target_eos/target_cutoff modes require --enable-fast-token-hooks")
                    warmup_calls = _mode_int_after(mode, "_warmup", 0)
                    refresh_period = _mode_int_after(mode, "_refresh", 0)
                    force_full_refresh = controller.stats.model_calls < warmup_calls
                    if refresh_period > 0 and controller.stats.model_calls >= warmup_calls:
                        force_full_refresh = ((controller.stats.model_calls - warmup_calls) % refresh_period) == 0
                    if force_full_refresh and (mode.startswith("target_cutoff") or mode.startswith("target_eos_adaptive")):
                        with autocast_ctx:
                            prediction = _predict_action_chunk(
                                policy,
                                batch,
                                policy_postprocessor,
                                device,
                                token_adapter,
                                early_stop_action_end=True,
                            )
                        prediction.stats = {
                            **(prediction.stats or {}),
                            "forced_full_refresh": 1.0,
                        }
                    elif mode.startswith("target_cutoff"):
                        suffix = mode.removeprefix("target_cutoff")
                        cutoff_tokens = _leading_int(suffix, 64)
                        with autocast_ctx:
                            prediction = _predict_prefix_cutoff_chunk(
                                token_adapter,
                                batch,
                                policy_postprocessor,
                                device,
                                cutoff_tokens=cutoff_tokens,
                                collect_logits="_gatefull" in mode,
                                force_action_prefix="_prefix" in mode,
                            )
                        if "_gatefull" in mode:
                            if adaptive_prefix_gate is None:
                                raise RuntimeError("target_cutoff*_gatefull requires --adaptive-prefix-gate-checkpoint")
                            gate_prob = _prefix_gate_probability_for_prediction(
                                prediction,
                                cutoff_tokens=cutoff_tokens,
                                prefix_gate=adaptive_prefix_gate,
                                device=device,
                            )
                            prediction.stats = {
                                **(prediction.stats or {}),
                                "prefix_gate_probability": gate_prob,
                            }
                            if gate_prob < adaptive_prefix_gate_threshold:
                                with autocast_ctx:
                                    target_prediction = _predict_action_chunk(
                                        policy,
                                        batch,
                                        policy_postprocessor,
                                        device,
                                        token_adapter,
                                        early_stop_action_end=True,
                                    )
                                controller.stats.guard_reasons.update(["target_cutoff_gate_full_refresh"])
                                prediction = PredictionTrace(
                                    actions=target_prediction.actions,
                                    elapsed_ms=prediction.elapsed_ms + target_prediction.elapsed_ms,
                                    token_count=target_prediction.token_count,
                                    token_ids=target_prediction.token_ids,
                                    stats={
                                        **(prediction.stats or {}),
                                        "gate_full_refresh_ms": target_prediction.elapsed_ms,
                                        "gate_full_refresh": 1.0,
                                    },
                                )
                    elif mode.startswith("target_eos_adaptive"):
                        if not adaptive_prefix_checkpoints:
                            raise RuntimeError("target_eos_adaptive requires --adaptive-prefix-checkpoints")
                        with autocast_ctx:
                            prediction = _predict_adaptive_prefix_chunk(
                                token_adapter,
                                batch,
                                policy_postprocessor,
                                device,
                                checkpoints=adaptive_prefix_checkpoints,
                                stable_tolerance=adaptive_stable_tolerance,
                                stable_checks=adaptive_stable_checks,
                                early_stable_checks=adaptive_early_stable_checks,
                                early_max_stable_checkpoint=adaptive_early_max_stable_checkpoint,
                                skip_unproductive_checks=adaptive_skip_unproductive_checks,
                                skip_unproductive_after_checkpoint=adaptive_skip_unproductive_after_checkpoint,
                                continue_to_action_end_on_unstable=adaptive_continue_to_action_end_on_unstable,
                                prefix_gate=adaptive_prefix_gate,
                                prefix_gate_threshold=adaptive_prefix_gate_threshold,
                            )
                        if mode.startswith("target_eos_adaptive_validate"):
                            with autocast_ctx:
                                target_prediction = _predict_action_chunk(
                                    policy,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    token_adapter,
                                    early_stop_action_end=True,
                                )
                            max_diff, mean_diff = _prediction_action_diff(prediction.actions, target_prediction.actions)
                            controller.stats.exact_verifies += 1
                            controller.stats.action_max_diffs.append(max_diff)
                            controller.stats.action_mean_diffs.append(mean_diff)
                            if max_diff != 0.0:
                                controller.stats.guard_reasons.update(["adaptive_action_diff_fallback"])
                                prediction = PredictionTrace(
                                    actions=target_prediction.actions,
                                    elapsed_ms=prediction.elapsed_ms + target_prediction.elapsed_ms,
                                    token_count=target_prediction.token_count,
                                    token_ids=target_prediction.token_ids,
                                    stats={
                                        **(prediction.stats or {}),
                                        "fallback_target_ms": target_prediction.elapsed_ms,
                                        "fallback_action_max_diff": max_diff,
                                    },
                                )
                    elif mode.startswith("target_eos"):
                        with autocast_ctx:
                            prediction = _predict_target_eos_chunk(
                                token_adapter,
                                batch,
                                policy_postprocessor,
                                device,
                                constrained_action_vocab="constrained" in mode,
                                constrained_action_vocab_size=target_eos_constrained_action_vocab_size,
                                constrained_text_vocab_size=target_eos_constrained_text_vocab_size,
                                constrained_full_head_margin=target_eos_constrained_full_head_margin,
                                constrained_force_action_prefix=not (
                                    target_eos_constrained_no_force_prefix or "_noforce" in mode
                                ),
                                constrained_structural_token_radius=target_eos_constrained_structural_token_radius,
                                force_action_prefix="_prefix" in mode,
                            )
                        controller.stats.record_trace_stats(prediction.stats)
                    else:
                        with autocast_ctx:
                            prediction = _predict_action_chunk(
                                policy,
                                batch,
                                policy_postprocessor,
                                device,
                                token_adapter,
                                early_stop_action_end=mode.startswith("target_eos"),
                            )
                    if mode.startswith("target_cutoff") and "validate" in mode:
                        with autocast_ctx:
                            target_prediction = _predict_action_chunk(
                                policy,
                                batch,
                                policy_postprocessor,
                                device,
                                token_adapter,
                                early_stop_action_end=True,
                            )
                        max_diff, mean_diff = _prediction_action_diff(prediction.actions, target_prediction.actions)
                        controller.stats.exact_verifies += 1
                        controller.stats.action_max_diffs.append(max_diff)
                        controller.stats.action_mean_diffs.append(mean_diff)
                        if max_diff != 0.0:
                            controller.stats.guard_reasons.update(["target_cutoff_action_diff_fallback"])
                            prediction = PredictionTrace(
                                actions=target_prediction.actions,
                                elapsed_ms=prediction.elapsed_ms + target_prediction.elapsed_ms,
                                token_count=target_prediction.token_count,
                                token_ids=target_prediction.token_ids,
                                stats={
                                    **(prediction.stats or {}),
                                    "fallback_target_ms": target_prediction.elapsed_ms,
                                    "fallback_action_max_diff": max_diff,
                                },
                            )
                    if mode.startswith("target_eos") and "validate" in mode:
                        with autocast_ctx:
                            if "constrained" in mode or "_prefix" in mode:
                                full_prediction = _predict_target_eos_chunk(
                                    token_adapter,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    constrained_action_vocab=False,
                                )
                            else:
                                full_prediction = _predict_action_chunk(
                                    policy,
                                    batch,
                                    policy_postprocessor,
                                    device,
                                    token_adapter=None,
                                )
                        horizon = min(len(prediction.actions), len(full_prediction.actions))
                        diff = np.abs(prediction.actions[:horizon] - full_prediction.actions[:horizon])
                        max_diff = float(np.max(diff)) if diff.size else 0.0
                        mean_diff = float(np.mean(diff)) if diff.size else 0.0
                        controller.stats.record_trace_stats(
                            _prediction_token_diff_stats(prediction, full_prediction, "target_eos")
                        )
                        controller.stats.exact_verifies += 1
                        controller.stats.action_max_diffs.append(max_diff)
                        controller.stats.action_mean_diffs.append(mean_diff)
                        if max_diff != 0.0:
                            controller.stats.guard_reasons.update(["target_eos_action_diff"])
                    chunk = prediction.actions
                    relaxed_tail = False
                    chunk_extend_handled = False
                    if mode.startswith("target_eos") or mode.startswith("target_cutoff"):
                        if mode.startswith("target_cutoff") and "_extend" in mode:
                            extend_suffix = mode.split("_extend", 1)[1]
                            total_horizon = _leading_int(extend_suffix, 15)
                            if _chunk_is_extension_safe(chunk, controller.guard):
                                chunk = _extend_action_chunk(chunk, total_horizon)
                                controller.stats.guard_reasons.update(["target_cutoff_extend_applied"])
                            else:
                                controller.stats.guard_reasons.update(["target_cutoff_extend_unsafe"])
                        if "_repeat" in mode:
                            repeat_suffix = mode.split("_repeat", 1)[1]
                            total_horizon = _leading_int(repeat_suffix, 15)
                            period = _mode_int_after(mode, "_period", 1)
                            if _chunk_is_extension_safe(chunk, controller.guard):
                                chunk = _repeat_tail_action_chunk(chunk, total_horizon, period=period)
                                controller.stats.guard_reasons.update(["target_repeat_tail_applied"])
                            else:
                                controller.stats.guard_reasons.update(["target_repeat_tail_unsafe"])
                        controller.queue.clear()
                        for queued_action in chunk:
                            controller.queue.append(queued_action.copy())
                        controller.stats.chunks_seen += 1
                        controller.stats.chunks_accepted += 1
                        controller.stats.actions_enqueued += int(chunk.shape[0])
                        controller.stats.accepted_windows.append(int(chunk.shape[0]))
                        controller.stats.model_calls += 1
                        controller.stats.inference_ms.append(prediction.elapsed_ms)
                        if prediction.token_count is not None:
                            controller.stats.token_counts.append(prediction.token_count)
                        action = controller.pop()
                        if action is None:
                            action = chunk[0]
                        chunk_extend_handled = True
                    elif mode.startswith("chunk_extend"):
                        suffix = mode.removeprefix("chunk_extend")
                        total_horizon = int(suffix) if suffix.isdigit() else 15
                        if _chunk_is_extension_safe(chunk, controller.guard):
                            chunk = _extend_action_chunk(chunk, total_horizon)
                            relaxed_tail = True
                        controller.queue.clear()
                        for queued_action in chunk:
                            controller.queue.append(queued_action.copy())
                        controller.stats.chunks_seen += 1
                        controller.stats.chunks_accepted += 1
                        controller.stats.actions_enqueued += int(chunk.shape[0])
                        controller.stats.accepted_windows.append(int(chunk.shape[0]))
                        controller.stats.model_calls += 1
                        controller.stats.inference_ms.append(prediction.elapsed_ms)
                        if prediction.token_count is not None:
                            controller.stats.token_counts.append(prediction.token_count)
                        action = controller.pop()
                        if action is None:
                            action = chunk[0]
                        chunk_extend_handled = True
                    elif mode.startswith("traj_tail"):
                        if trajectory_head is None:
                            raise RuntimeError("traj_tail mode requires --trajectory-head-checkpoint")
                        suffix = mode.removeprefix("traj_tail")
                        total_horizon = int(suffix) if suffix.isdigit() else 18
                        if _chunk_is_extension_safe(chunk, controller.guard):
                            with torch.autocast(device_type="cuda", enabled=False) if device.startswith("cuda") else nullcontext():
                                chunk = _extend_action_chunk_with_trajectory_head(
                                    chunk,
                                    trajectory_head,
                                    total_horizon=total_horizon,
                                    device=device,
                                    guard=controller.guard,
                                    blend_with_damped=trajectory_tail_blend,
                                    project_smooth=trajectory_project_smooth,
                                )
                            if not _chunk_is_extension_safe(chunk, controller.guard):
                                chunk = prediction.actions
                        controller.queue.clear()
                        for queued_action in chunk:
                            controller.queue.append(queued_action.copy())
                        controller.stats.chunks_seen += 1
                        controller.stats.chunks_accepted += 1
                        controller.stats.actions_enqueued += int(chunk.shape[0])
                        controller.stats.accepted_windows.append(int(chunk.shape[0]))
                        controller.stats.model_calls += 1
                        controller.stats.inference_ms.append(prediction.elapsed_ms)
                        if prediction.token_count is not None:
                            controller.stats.token_counts.append(prediction.token_count)
                        action = controller.pop()
                        if action is None:
                            action = chunk[0]
                        chunk_extend_handled = True
                    if not chunk_extend_handled:
                        confidence = 1.0
                        if drafter is not None:
                            token_list = (
                                prediction.token_ids[0].detach().cpu().tolist()
                                if prediction.token_ids is not None
                                else []
                            )
                            drafter.add(state_key, token_list, chunk)

                        if mode.startswith("relaxed_chunk_retrieval") and retrieval_draft is not None:
                            controller.offer_chunk(
                                retrieval_draft.action_chunk,
                                confidence=confidence,
                                reference_chunk=chunk,
                                relaxed=True,
                                inference_ms=prediction.elapsed_ms,
                                token_count=prediction.token_count,
                            )
                        elif mode.startswith("exact_fast_sd"):
                            controller.offer_chunk(
                                chunk,
                                confidence=confidence,
                                inference_ms=prediction.elapsed_ms,
                                token_count=prediction.token_count,
                            )
                            if token_adapter is None:
                                controller.stats.guard_reasons.update(["exact_fast_token_hooks_unavailable"])
                        else:
                            controller.offer_chunk(
                                chunk,
                                confidence=confidence,
                                reference_chunk=prediction.actions if relaxed_tail else None,
                                relaxed=relaxed_tail,
                                inference_ms=prediction.elapsed_ms,
                                token_count=prediction.token_count,
                            )
                        action = controller.pop()
                        if action is None:
                            action = chunk[0]
        else:
            action = buffered

        if buffered is None and token_trace_sink is not None and prediction is not None:
            token_trace_sink.record(
                prediction,
                mode=mode,
                task=task,
                task_id=task_id,
                episode=episode,
                seed=seed,
                step=steps,
            )

        observation, reward, terminated, truncated, info = _env_step(env, action, env_postprocessor)
        steps += 1
        reward_sum += reward
        success = success or _extract_success(info)
        done = terminated or truncated
        control_step_ms.append((time.perf_counter() - step_start) * 1000)

    wall_s = time.perf_counter() - wall_start
    chunk_summary = controller.stats.summary()
    return EpisodeResult(
        mode=mode,
        episode=episode,
        seed=seed,
        task=task,
        task_id=task_id,
        success=success,
        reward_sum=reward_sum,
        steps=steps,
        wall_s=wall_s,
        avg_ms_per_control_step=float(np.mean(control_step_ms)) if control_step_ms else 0.0,
        model_calls=controller.stats.model_calls,
        model_calls_per_step=controller.stats.model_calls / max(steps, 1),
        chunk_stats=chunk_summary,
    )


def summarize(results: list[EpisodeResult], baseline_mode: str = "baseline") -> dict:
    by_mode: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    summary: dict[str, Any] = {"total_episodes": len(results), "by_mode": {}}
    baseline_ms = None
    baseline_success = None
    if baseline_mode in by_mode:
        base = by_mode[baseline_mode]
        baseline_ms = float(np.mean([r.avg_ms_per_control_step for r in base]))
        baseline_success = float(np.mean([r.success for r in base]))

    for mode, rows in sorted(by_mode.items()):
        avg_ms = float(np.mean([r.avg_ms_per_control_step for r in rows]))
        success_rate = float(np.mean([r.success for r in rows]))
        mode_summary = {
            "episodes": len(rows),
            "successes": int(sum(r.success for r in rows)),
            "success_rate": success_rate,
            "avg_ms_per_control_step": avg_ms,
            "avg_model_calls_per_step": float(np.mean([r.model_calls_per_step for r in rows])),
            "speedup_vs_baseline": (baseline_ms / avg_ms) if baseline_ms and avg_ms else None,
            "success_drop_abs_vs_baseline": (baseline_success - success_rate) if baseline_success is not None else None,
        }
        summary["by_mode"][mode] = mode_summary
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run π0-FAST chunk execution in LIBERO")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--policy-kind", choices=["pi0fast", "pi05"], default="pi0fast")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Optional comma-separated task ids to evaluate in one process. Overrides --task-id.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--episode-ids",
        default=None,
        help="Optional comma-separated episode indices to run. Seeds remain args.seed + episode_index.",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--modes", default="baseline,chunk_m3,chunk_m5_smooth")
    parser.add_argument("--output-dir", default="outputs/pi0fast_chunk")
    parser.add_argument(
        "--token-trace-output-dir",
        type=Path,
        default=None,
        help="Optional directory for lightweight shard_*.pt FAST-token traces used by offline pattern sweeps.",
    )
    parser.add_argument(
        "--token-trace-modes",
        default="target_eos",
        help="Comma-separated modes to export when --token-trace-output-dir is set; use all to export every mode.",
    )
    parser.add_argument(
        "--token-trace-max-rows-per-shard",
        type=int,
        default=512,
        help="Maximum decoded chunks per lightweight token trace shard.",
    )
    parser.add_argument(
        "--summary-baseline-mode",
        default="baseline",
        help="Mode used as denominator for summary speedup and success-drop columns.",
    )
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default=None)
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable model gradient checkpointing after loading for inference latency measurements.",
    )
    parser.add_argument(
        "--target-eos-constrained-action-vocab-size",
        type=int,
        default=8192,
        help="Raw high-token action span for target_eos_constrained modes.",
    )
    parser.add_argument(
        "--target-eos-constrained-text-vocab-size",
        type=int,
        default=8192,
        help="Low-token text/special span for target_eos_constrained modes.",
    )
    parser.add_argument(
        "--target-eos-constrained-full-head-margin",
        type=float,
        default=-1.0,
        help=(
            "If non-negative, target_eos_constrained modes fall back to the full lm_head when the "
            "restricted top-1/top-2 logit margin is below this value."
        ),
    )
    parser.add_argument(
        "--target-eos-constrained-no-force-prefix",
        action="store_true",
        help="Let constrained target_eos choose the Action prefix through the restricted/full-head path.",
    )
    parser.add_argument(
        "--target-eos-constrained-structural-token-radius",
        type=int,
        default=512,
        help="Include token ids within this radius of the PI0-FAST action-end token in constrained modes.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Flow inference steps for PI0.5 policies.",
    )
    parser.add_argument(
        "--enable-fast-token-hooks",
        action="store_true",
        help="Use LeRobot π0-FAST internals to return generated FAST token IDs and logits.",
    )
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--default-window", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--max-window", type=int, default=6)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--position-error-threshold", type=float, default=0.035)
    parser.add_argument("--rotation-error-threshold", type=float, default=0.25)
    parser.add_argument("--action-jump-threshold", type=float, default=0.18)
    parser.add_argument("--jerk-threshold", type=float, default=0.22)
    parser.add_argument("--smooth-position-delta", type=float, default=0.045)
    parser.add_argument("--smooth-rotation-delta", type=float, default=0.18)
    parser.add_argument("--ngram-data-dir", default="outputs/pi0fast_eagle_tasks0_4_20trace_data")
    parser.add_argument("--ngram-train-task-ids", default="0,1,2,3")
    parser.add_argument("--ngram-lookahead", type=int, default=8)
    parser.add_argument("--ngram-max-context", type=int, default=4)
    parser.add_argument("--ngram-min-count", type=int, default=1)
    parser.add_argument(
        "--ngram-stop-token-ids",
        default="",
        help="Optional comma-separated stop token ids for trimming n-gram traces before fitting.",
    )
    parser.add_argument(
        "--ngram-reuse-full-blocks",
        action="store_true",
        help="Reuse verifier KV on fully accepted n-gram draft blocks. Faster but approximate.",
    )
    parser.add_argument("--pattern-lookahead", type=int, default=8)
    parser.add_argument("--pattern-action-dim", type=int, default=7)
    parser.add_argument("--pattern-max-period", type=int, default=16)
    parser.add_argument("--pattern-min-period-repeats", type=int, default=2)
    parser.add_argument("--pattern-repeat-token-min-run", type=int, default=3)
    parser.add_argument("--pattern-linear-action-extrapolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pattern-second-order-action-extrapolation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pattern-second-order-max-accel",
        type=int,
        default=8,
        help="Maximum per-dimension token acceleration for second-order pattern drafts; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-action-trend-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from a short per-action-dimension regression trend over verified FAST tokens.",
    )
    parser.add_argument("--pattern-action-trend-history", type=int, default=4)
    parser.add_argument("--pattern-action-trend-top-k", type=int, default=3)
    parser.add_argument(
        "--pattern-action-trend-max-abs",
        type=int,
        default=12,
        help="Maximum absolute projected same-dimension token step for trend drafts; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-action-prefix-lookup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft later action dimensions from prior action vectors with the same verified intra-action prefix.",
    )
    parser.add_argument("--pattern-action-prefix-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-prefix-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-prefix-min-prefix", type=int, default=1)
    parser.add_argument("--pattern-action-prefix-max-mismatches", type=int, default=0)
    parser.add_argument(
        "--pattern-action-vector-suffix-lookup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Draft later action dimensions from prior full action vectors whose verified "
            "intra-action prefix matches the current prefix."
        ),
    )
    parser.add_argument("--pattern-action-vector-suffix-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-vector-suffix-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-vector-suffix-min-prefix", type=int, default=1)
    parser.add_argument("--pattern-action-vector-suffix-min-count", type=int, default=1)
    parser.add_argument("--pattern-action-vector-suffix-max-prefix-delta", type=int, default=0)
    parser.add_argument(
        "--pattern-action-vector-transition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Draft later action dimensions from prior full action-vector transitions whose previous "
            "vector and current prefix match verified tokens."
        ),
    )
    parser.add_argument("--pattern-action-vector-transition-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-vector-transition-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-vector-transition-min-count", type=int, default=1)
    parser.add_argument("--pattern-action-vector-transition-max-prev-delta", type=int, default=0)
    parser.add_argument("--pattern-action-vector-transition-max-prefix-delta", type=int, default=0)
    parser.add_argument(
        "--pattern-action-repeat-vector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from repeated full action vectors when the current partial vector still matches.",
    )
    parser.add_argument("--pattern-action-repeat-min-repeats", type=int, default=2)
    parser.add_argument("--pattern-action-repeat-max-delta", type=int, default=0)
    parser.add_argument(
        "--pattern-chunk-length-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft the FAST stop token when recent verified chunks ended at this prefix length.",
    )
    parser.add_argument("--pattern-chunk-length-stop-history-size", type=int, default=4)
    parser.add_argument("--pattern-chunk-length-stop-min-count", type=int, default=2)
    parser.add_argument(
        "--pattern-action-transition-histogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from same-action-dimension transition histograms in verified FAST tokens and recent chunks.",
    )
    parser.add_argument(
        "--pattern-action-context-tree",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from same-action-dimension context trees over verified FAST tokens and recent chunks.",
    )
    parser.add_argument("--pattern-action-context-tree-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-context-tree-max-context", type=int, default=3)
    parser.add_argument("--pattern-action-context-tree-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-context-tree-min-count", type=int, default=1)
    parser.add_argument("--pattern-action-transition-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-transition-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-transition-min-count", type=int, default=1)
    parser.add_argument(
        "--pattern-action-delta-histogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from frequent recent same-action-dimension FAST-token deltas.",
    )
    parser.add_argument("--pattern-action-delta-history", type=int, default=8)
    parser.add_argument("--pattern-action-delta-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-delta-min-count", type=int, default=1)
    parser.add_argument(
        "--pattern-action-delta-max-abs",
        type=int,
        default=12,
        help="Maximum absolute token delta for action-delta histogram pattern drafts; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-action-delta-ngram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft by matching same-action-dimension FAST-token delta n-grams in verified tokens and chunks.",
    )
    parser.add_argument("--pattern-action-delta-ngram-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-delta-ngram-min-context", type=int, default=1)
    parser.add_argument("--pattern-action-delta-ngram-max-context", type=int, default=4)
    parser.add_argument("--pattern-action-delta-ngram-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-delta-ngram-min-count", type=int, default=1)
    parser.add_argument(
        "--pattern-action-delta-ngram-max-abs",
        type=int,
        default=12,
        help="Maximum absolute token delta for action-delta n-gram pattern drafts; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-chunk-position-delta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft per-position FAST-token deltas from recent verified action chunks.",
    )
    parser.add_argument("--pattern-chunk-position-delta-history-size", type=int, default=4)
    parser.add_argument("--pattern-chunk-position-delta-top-k", type=int, default=3)
    parser.add_argument("--pattern-chunk-position-delta-min-count", type=int, default=1)
    parser.add_argument(
        "--pattern-chunk-position-delta-max-abs",
        type=int,
        default=12,
        help="Maximum absolute token delta for chunk-position delta pattern drafts; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-chunk-delta-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from recent verified chunks whose same-dimension delta prefix matches the current chunk.",
    )
    parser.add_argument("--pattern-chunk-delta-template-history-size", type=int, default=4)
    parser.add_argument("--pattern-chunk-delta-template-top-k", type=int, default=3)
    parser.add_argument("--pattern-chunk-delta-template-min-prefix-deltas", type=int, default=1)
    parser.add_argument(
        "--pattern-chunk-delta-template-max-delta-mismatch",
        type=int,
        default=0,
        help="Allowed total absolute mismatch in matched delta prefixes; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-chunk-delta-template-max-abs",
        type=int,
        default=12,
        help="Maximum absolute token delta for chunk-delta template pattern drafts; use -1 for no guard.",
    )
    parser.add_argument(
        "--pattern-previous-chunk-position",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft token i from token i of recent verified chunks from the same episode before local pattern rules.",
    )
    parser.add_argument("--pattern-previous-chunk-history-size", type=int, default=1)
    parser.add_argument(
        "--pattern-chunk-prefix-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from recent verified chunks whose prefix mostly matches the current partial action chunk.",
    )
    parser.add_argument("--pattern-chunk-prefix-history-size", type=int, default=4)
    parser.add_argument("--pattern-chunk-prefix-top-k", type=int, default=3)
    parser.add_argument("--pattern-chunk-prefix-min-matches", type=int, default=2)
    parser.add_argument("--pattern-chunk-prefix-max-mismatches", type=int, default=1)
    parser.add_argument(
        "--pattern-action-token-neighborhood",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft small FAST-token neighborhoods around smooth action extrapolations for exact tree verification.",
    )
    parser.add_argument("--pattern-action-token-neighborhood-radius", type=int, default=1)
    parser.add_argument("--pattern-action-token-neighborhood-top-k", type=int, default=3)
    parser.add_argument(
        "--pattern-position-mode-histogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft the per-position mode token from recent verified chunks with the same current prefix.",
    )
    parser.add_argument("--pattern-position-mode-history-size", type=int, default=4)
    parser.add_argument("--pattern-position-mode-top-k", type=int, default=3)
    parser.add_argument("--pattern-position-mode-min-count", type=int, default=2)
    parser.add_argument(
        "--pattern-global-position-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft per-position modal tokens from recent verified chunks without requiring the current prefix to match.",
    )
    parser.add_argument("--pattern-global-position-history-size", type=int, default=16)
    parser.add_argument("--pattern-global-position-top-k", type=int, default=3)
    parser.add_argument("--pattern-global-position-min-count", type=int, default=3)
    parser.add_argument(
        "--pattern-action-dimension-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft the most common verified FAST token for the current action dimension.",
    )
    parser.add_argument("--pattern-action-dimension-mode-history-size", type=int, default=4)
    parser.add_argument("--pattern-action-dimension-mode-top-k", type=int, default=3)
    parser.add_argument("--pattern-action-dimension-mode-min-count", type=int, default=2)
    parser.add_argument(
        "--pattern-hold-action-token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft the token from the same action dimension in the previous action step.",
    )
    parser.add_argument(
        "--pattern-ngram-continuation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft from prompt-lookup suffix matches in verified FAST tokens and recent verified chunks.",
    )
    parser.add_argument("--pattern-ngram-min-context", type=int, default=3)
    parser.add_argument("--pattern-ngram-max-context", type=int, default=16)
    parser.add_argument("--pattern-ngram-history-size", type=int, default=4)
    parser.add_argument(
        "--pattern-min-source-agreement",
        type=int,
        default=1,
        help="Prefer a pattern token only when at least this many configured sources propose it.",
    )
    parser.add_argument(
        "--pattern-source-priority",
        default="default",
        help="Named source priority mode or comma-separated source names for pattern drafts.",
    )
    parser.add_argument(
        "--pattern-source-cooldown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily skip pattern sources after verified rejections within the same decode.",
    )
    parser.add_argument("--pattern-source-cooldown-after", type=int, default=1)
    parser.add_argument("--pattern-source-cooldown-steps", type=int, default=1)
    parser.add_argument(
        "--pattern-source-acceptance-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reorder cheap sources by recent verified source acceptance within each decode.",
    )
    parser.add_argument("--pattern-source-acceptance-bias-history-size", type=int, default=32)
    parser.add_argument("--pattern-source-acceptance-bias-min-observations", type=int, default=2)
    parser.add_argument(
        "--pattern-reuse-full-blocks",
        action="store_true",
        help="Reuse verifier KV on fully accepted checkpoint-free pattern draft blocks.",
    )
    parser.add_argument(
        "--pattern-min-verify-margin",
        type=float,
        default=0.0,
        help=(
            "When >0, pattern_sd only reuses a batched verifier block if every draft token "
            "matches and every verifier top-1/top-2 margin is at least this value."
        ),
    )
    parser.add_argument(
        "--pattern-emit-bonus-token",
        action="store_true",
        help="After a fully accepted pattern_sd block, emit the verifier's next greedy token and process it in the next verify pass.",
    )
    parser.add_argument(
        "--pattern-verify-from-scratch",
        action="store_true",
        help="Verify pattern_sd candidates by replaying the full FAST prefix instead of using the speculative KV cache.",
    )
    parser.add_argument(
        "--pattern-replay-accepted-cache",
        action="store_true",
        help="After accepting pattern_sd draft tokens, replay them through the one-token target cache path.",
    )
    parser.add_argument(
        "--pattern-resync-accepted-cache",
        action="store_true",
        help="After accepting pattern_sd draft tokens, rebuild the target KV cache from the emitted FAST prefix.",
    )
    parser.add_argument(
        "--pattern-diagnose-verify-alignment",
        action="store_true",
        help="Compare pattern_sd batched verifier argmaxes with stepwise teacher-forced target argmaxes.",
    )
    parser.add_argument(
        "--pattern-dynamic-lookahead",
        action="store_true",
        help="Grow or shrink pattern_sd lookahead from recent exact acceptance in the same decode.",
    )
    parser.add_argument("--pattern-min-lookahead", type=int, default=1)
    parser.add_argument("--pattern-lookahead-growth", type=int, default=1)
    parser.add_argument("--pattern-lookahead-shrink", type=int, default=4)
    parser.add_argument(
        "--pattern-tree-width",
        type=int,
        default=1,
        help="Verify multiple pattern candidates in one target pass. 1 keeps the single-chain verifier.",
    )
    parser.add_argument("--pattern-tree-branch-width", type=int, default=4)
    parser.add_argument(
        "--pattern-dynamic-tree-width",
        action="store_true",
        help="Grow or shrink pattern_sd tree candidate count from recent exact acceptance.",
    )
    parser.add_argument("--pattern-min-tree-width", type=int, default=1)
    parser.add_argument("--pattern-tree-width-growth", type=int, default=1)
    parser.add_argument("--pattern-tree-width-shrink", type=int, default=1)
    parser.add_argument(
        "--pattern-tree-anchor-target-token",
        action="store_true",
        help="When every tree row starts with the wrong token, anchor on the target greedy token and verify drafted continuations after it.",
    )
    parser.add_argument(
        "--pattern-tree-anchor-target-continuation",
        action="store_true",
        help="Always anchor on the target greedy token and verify drafted continuations after it for tree pattern_sd.",
    )
    parser.add_argument(
        "--medusa-checkpoint",
        default=None,
        help="Checkpoint from train_pi0fast_medusa_from_traces.py for medusa_sd modes.",
    )
    parser.add_argument("--medusa-lookahead", type=int, default=4)
    parser.add_argument("--medusa-min-draft-confidence", type=float, default=0.0)
    parser.add_argument("--medusa-min-verify-confidence", type=float, default=0.0)
    parser.add_argument("--medusa-min-spec-position", type=int, default=42)
    parser.add_argument("--medusa-accept-partial-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--medusa-replay-accepted-cache", action="store_true")
    parser.add_argument("--medusa-resync-accepted-cache", action="store_true")
    parser.add_argument("--medusa-verify-from-scratch", action="store_true")
    parser.add_argument(
        "--block-checkpoint",
        default=None,
        help="Checkpoint from train_pi0fast_block_drafter.py for block_sd modes.",
    )
    parser.add_argument(
        "--block-gate-checkpoint",
        default=None,
        help="Optional checkpoint from train_pi0fast_block_gate.py for block_sd modes.",
    )
    parser.add_argument("--block-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--block-action-gate-checkpoint",
        default=None,
        help="Optional checkpoint from train_pi0fast_action_gate.py for block_sd_action_gate modes.",
    )
    parser.add_argument("--block-action-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--cutoff-selector-checkpoint",
        default=None,
        help="Checkpoint from train_pi0fast_cutoff_selector.py for block_sd_select modes.",
    )
    parser.add_argument("--cutoff-selector-risk", type=float, default=0.45)
    parser.add_argument("--block-lookahead", type=int, default=5)
    parser.add_argument("--block-min-draft-confidence", type=float, default=0.0)
    parser.add_argument("--block-min-verify-confidence", type=float, default=0.0)
    parser.add_argument("--block-min-verify-margin", type=float, default=0.0)
    parser.add_argument(
        "--block-max-future-accept",
        type=int,
        default=-1,
        help="Maximum future tokens accepted after the known target token; -1 leaves uncapped.",
    )
    parser.add_argument(
        "--block-min-future-accept",
        type=int,
        default=0,
        help="Reject a speculative block unless at least this many future tokens match.",
    )
    parser.add_argument("--block-min-spec-position", type=int, default=42)
    parser.add_argument("--block-reject-cooldown-steps", type=int, default=4)
    parser.add_argument("--block-reject-cooldown-after", type=int, default=2)
    parser.add_argument("--block-spec-fallback-cooldown-steps", type=int, default=0)
    parser.add_argument("--block-spec-fallback-cooldown-after", type=int, default=0)
    parser.add_argument(
        "--block-allow-unknown-context",
        action="store_true",
        help="Pad unknown compact-vocab context tokens instead of disabling a speculative draft attempt.",
    )
    parser.add_argument(
        "--block-repeat-token-draft",
        action="store_true",
        help="Use a zero-cost repeated-token drafter inside block_sd modes.",
    )
    parser.add_argument("--block-repeat-token-min-run", type=int, default=2)
    parser.add_argument(
        "--block-repeat-pattern-draft",
        action="store_true",
        help="Use a zero-cost periodic-pattern drafter inside block_sd modes.",
    )
    parser.add_argument("--block-repeat-pattern-max-period", type=int, default=8)
    parser.add_argument("--block-repeat-pattern-min-position", type=int, default=0)
    parser.add_argument(
        "--block-pattern-only",
        action="store_true",
        help="Only use zero-cost repeat/pattern drafts; skip learned block drafts elsewhere.",
    )
    parser.add_argument(
        "--block-unverified-pattern-tail",
        action="store_true",
        help="When a periodic FAST-token tail is detected, synthesize the rest of the tail without verifier calls.",
    )
    parser.add_argument(
        "--block-unverified-pattern-eos",
        action="store_true",
        help="When a periodic FAST-token tail is detected, force action-end immediately without verifier calls.",
    )
    parser.add_argument(
        "--block-full-block-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accept only complete verified speculative blocks; partial rejects fall back to a normal target step.",
    )
    parser.add_argument("--block-accept-partial-blocks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-refine-steps", type=int, default=1)
    parser.add_argument("--block-verify-from-scratch", action="store_true")
    parser.add_argument("--block-resync-accepted-cache", action="store_true")
    parser.add_argument(
        "--block-draft-after-known-token",
        action="store_true",
        help="Advance the target's known next token before drafting futures, matching block-drafter training state.",
    )
    parser.add_argument("--block-compile-draft", action="store_true")
    parser.add_argument("--guarded-min-acceptance", type=float, default=0.70)
    parser.add_argument("--guarded-min-tokens-per-forward", type=float, default=2.0)
    parser.add_argument("--guarded-max-fallback-rate", type=float, default=0.70)
    parser.add_argument("--adaptive-prefix-checkpoints", default="")
    parser.add_argument("--adaptive-stable-tolerance", type=float, default=0.0)
    parser.add_argument("--adaptive-stable-checks", type=int, default=3)
    parser.add_argument("--adaptive-early-stable-checks", type=int, default=0)
    parser.add_argument("--adaptive-early-max-stable-checkpoint", type=int, default=0)
    parser.add_argument("--adaptive-skip-unproductive-checks", action="store_true")
    parser.add_argument("--adaptive-skip-unproductive-after-checkpoint", type=int, default=0)
    parser.add_argument("--adaptive-continue-to-action-end-on-unstable", action="store_true")
    parser.add_argument("--adaptive-prefix-gate-checkpoint", default=None)
    parser.add_argument("--adaptive-prefix-gate-threshold", type=float, default=0.98)
    parser.add_argument(
        "--trajectory-head-checkpoint",
        default=None,
        help="Checkpoint from train_pi0fast_trajectory_head.py for traj_tail/ngram_traj_tail modes.",
    )
    parser.add_argument(
        "--trajectory-tail-blend",
        type=float,
        default=1.0,
        help="Blend learned tail with damped tail: 1.0=head only, 0.0=damped only.",
    )
    parser.add_argument(
        "--trajectory-project-smooth",
        action="store_true",
        help="Project learned tail step deltas into the guard's smooth-action limits before queueing.",
    )
    parser.add_argument(
        "--action-corrector-checkpoint",
        default=None,
        help="Checkpoint from train_pi0fast_action_corrector.py for block_sd_correct modes.",
    )
    parser.add_argument("--action-corrector-refine-steps", type=int, default=2)
    parser.add_argument("--action-corrector-min-confidence", type=float, default=0.0)
    parser.add_argument("--action-corrector-preserve-prefix-actions", type=int, default=0)
    parser.add_argument("--action-corrector-blend", type=float, default=1.0)
    parser.add_argument(
        "--action-corrector-project-smooth",
        action="store_true",
        help="Project corrected chunks into the guard's smooth-action limits before queueing.",
    )
    return parser.parse_args()


def _parse_ids(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _leading_int(value: str, default: int) -> int:
    digits: list[str] = []
    for char in value:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else default


def _mode_int_after(mode: str, marker: str, default: int = 0) -> int:
    if marker not in mode:
        return default
    return _leading_int(mode.split(marker, 1)[1], default)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()
    if args.policy is None:
        args.policy = "lerobot/pi05_libero_finetuned_v044" if args.policy_kind == "pi05" else "lerobot/pi0fast-libero"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    token_trace_sink = None
    if args.token_trace_output_dir is not None:
        token_trace_sink = TokenTraceSink(
            output_dir=args.token_trace_output_dir,
            modes=parse_token_trace_modes(args.token_trace_modes),
            max_rows_per_shard=args.token_trace_max_rows_per_shard,
        )

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    amp_dtype = getattr(torch, args.amp_dtype) if args.amp_dtype else None
    if args.policy_kind == "pi05" and dtype in (torch.bfloat16, torch.float16):
        args.use_amp = True
        amp_dtype = amp_dtype or dtype
    torch.backends.cuda.matmul.allow_tf32 = True

    selected_task_ids = sorted(_parse_ids(args.task_ids)) if args.task_ids else None
    if selected_task_ids is None and args.task_id is not None:
        selected_task_ids = [args.task_id]
    selected_episode_ids = sorted(_parse_ids(args.episode_ids)) if args.episode_ids else list(range(args.episodes))
    if any(ep < 0 or ep >= args.episodes for ep in selected_episode_ids):
        raise ValueError("--episode-ids entries must be in [0, --episodes)")

    env_kwargs = {"task": args.task, "control_mode": args.control_mode}
    if selected_task_ids is not None:
        env_kwargs["task_ids"] = selected_task_ids
    env_cfg = LiberoEnv(**env_kwargs)

    logger.info("Loading policy %s on %s", args.policy, device)
    if args.policy_kind == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        PolicyClass = PI05Policy
    else:
        PolicyClass = PI0FastPolicy
    policy = PolicyClass.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    if args.disable_gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
        policy.eval()
    if args.num_inference_steps is not None and hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = int(args.num_inference_steps)
    token_adapter = PI0FastTokenLogitAdapter(policy) if args.enable_fast_token_hooks else None
    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    task_ids = selected_task_ids if selected_task_ids is not None else sorted(env_map[args.task])

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    adaptive_prefix_checkpoints = [int(part.strip()) for part in args.adaptive_prefix_checkpoints.split(",") if part.strip()]
    ngram_drafter = None
    block_drafter = None
    block_token_map = None
    block_gate = None
    block_gate_threshold = 0.0
    block_action_gate = None
    block_action_gate_threshold = 0.0
    cutoff_selector = None
    cutoff_selector_summary = None
    block_summary = None
    block_gate_summary = None
    block_action_gate_summary = None
    medusa_head = None
    medusa_token_map = None
    medusa_summary = None
    pattern_drafter = None
    if any(mode.startswith("ngram_sd") or mode.startswith("ngram_extend") or mode.startswith("ngram_traj_tail") for mode in modes):
        if token_adapter is None:
            token_adapter = PI0FastTokenLogitAdapter(policy)
        records = load_trace_records(args.ngram_data_dir)
        train_task_ids = _parse_ids(args.ngram_train_task_ids)
        stop_token_ids = tuple(int(part.strip()) for part in args.ngram_stop_token_ids.split(",") if part.strip())
        if not stop_token_ids:
            stop_token_ids = (_adapter_action_end_token_id(token_adapter),)
        ngram_drafter = NgramFastTokenDrafter(
            NgramDraftConfig(
                max_context=args.ngram_max_context,
                min_count=args.ngram_min_count,
                lookahead=args.ngram_lookahead,
                stop_token_ids=stop_token_ids,
            )
        )
        ngram_drafter.fit(record for record in records if record.task_id in train_task_ids)
    if any(mode.startswith("pattern_sd") for mode in modes):
        if token_adapter is None:
            token_adapter = PI0FastTokenLogitAdapter(policy)
        pattern_second_order_max_accel = (
            None if args.pattern_second_order_max_accel < 0 else args.pattern_second_order_max_accel
        )
        pattern_action_trend_max_abs = (
            None if args.pattern_action_trend_max_abs < 0 else args.pattern_action_trend_max_abs
        )
        pattern_action_delta_max_abs = (
            None if args.pattern_action_delta_max_abs < 0 else args.pattern_action_delta_max_abs
        )
        pattern_action_delta_ngram_max_abs = (
            None if args.pattern_action_delta_ngram_max_abs < 0 else args.pattern_action_delta_ngram_max_abs
        )
        pattern_chunk_position_delta_max_abs = (
            None if args.pattern_chunk_position_delta_max_abs < 0 else args.pattern_chunk_position_delta_max_abs
        )
        pattern_chunk_delta_template_max_delta_mismatch = (
            None
            if args.pattern_chunk_delta_template_max_delta_mismatch < 0
            else args.pattern_chunk_delta_template_max_delta_mismatch
        )
        pattern_chunk_delta_template_max_abs = (
            None if args.pattern_chunk_delta_template_max_abs < 0 else args.pattern_chunk_delta_template_max_abs
        )
        pattern_drafter = PatternFastTokenDrafter(
            PatternDraftConfig(
                lookahead=args.pattern_lookahead,
                action_dim=args.pattern_action_dim,
                max_period=args.pattern_max_period,
                min_period_repeats=args.pattern_min_period_repeats,
                repeat_token_min_run=args.pattern_repeat_token_min_run,
                enable_linear_action_extrapolation=args.pattern_linear_action_extrapolation,
                enable_second_order_action_extrapolation=args.pattern_second_order_action_extrapolation,
                second_order_max_accel=pattern_second_order_max_accel,
                enable_action_trend_regression=args.pattern_action_trend_regression,
                action_trend_history=args.pattern_action_trend_history,
                action_trend_top_k=args.pattern_action_trend_top_k,
                action_trend_max_abs=pattern_action_trend_max_abs,
                enable_action_prefix_lookup=args.pattern_action_prefix_lookup,
                action_prefix_history_size=args.pattern_action_prefix_history_size,
                action_prefix_top_k=args.pattern_action_prefix_top_k,
                action_prefix_min_prefix=args.pattern_action_prefix_min_prefix,
                action_prefix_max_mismatches=args.pattern_action_prefix_max_mismatches,
                enable_action_vector_suffix_lookup=args.pattern_action_vector_suffix_lookup,
                action_vector_suffix_history_size=args.pattern_action_vector_suffix_history_size,
                action_vector_suffix_top_k=args.pattern_action_vector_suffix_top_k,
                action_vector_suffix_min_prefix=args.pattern_action_vector_suffix_min_prefix,
                action_vector_suffix_min_count=args.pattern_action_vector_suffix_min_count,
                action_vector_suffix_max_prefix_delta=args.pattern_action_vector_suffix_max_prefix_delta,
                enable_action_vector_transition=args.pattern_action_vector_transition,
                action_vector_transition_history_size=args.pattern_action_vector_transition_history_size,
                action_vector_transition_top_k=args.pattern_action_vector_transition_top_k,
                action_vector_transition_min_count=args.pattern_action_vector_transition_min_count,
                action_vector_transition_max_prev_delta=args.pattern_action_vector_transition_max_prev_delta,
                action_vector_transition_max_prefix_delta=args.pattern_action_vector_transition_max_prefix_delta,
                enable_action_repeat_vector=args.pattern_action_repeat_vector,
                action_repeat_min_repeats=args.pattern_action_repeat_min_repeats,
                action_repeat_max_delta=args.pattern_action_repeat_max_delta,
                enable_chunk_length_stop=args.pattern_chunk_length_stop,
                chunk_length_stop_history_size=args.pattern_chunk_length_stop_history_size,
                chunk_length_stop_min_count=args.pattern_chunk_length_stop_min_count,
                enable_action_context_tree=args.pattern_action_context_tree,
                action_context_tree_history_size=args.pattern_action_context_tree_history_size,
                action_context_tree_max_context=args.pattern_action_context_tree_max_context,
                action_context_tree_top_k=args.pattern_action_context_tree_top_k,
                action_context_tree_min_count=args.pattern_action_context_tree_min_count,
                enable_action_transition_histogram=args.pattern_action_transition_histogram,
                action_transition_history_size=args.pattern_action_transition_history_size,
                action_transition_top_k=args.pattern_action_transition_top_k,
                action_transition_min_count=args.pattern_action_transition_min_count,
                enable_action_delta_histogram=args.pattern_action_delta_histogram,
                action_delta_history=args.pattern_action_delta_history,
                action_delta_top_k=args.pattern_action_delta_top_k,
                action_delta_min_count=args.pattern_action_delta_min_count,
                action_delta_max_abs=pattern_action_delta_max_abs,
                enable_action_delta_ngram=args.pattern_action_delta_ngram,
                action_delta_ngram_history_size=args.pattern_action_delta_ngram_history_size,
                action_delta_ngram_min_context=args.pattern_action_delta_ngram_min_context,
                action_delta_ngram_max_context=args.pattern_action_delta_ngram_max_context,
                action_delta_ngram_top_k=args.pattern_action_delta_ngram_top_k,
                action_delta_ngram_min_count=args.pattern_action_delta_ngram_min_count,
                action_delta_ngram_max_abs=pattern_action_delta_ngram_max_abs,
                enable_chunk_position_delta=args.pattern_chunk_position_delta,
                chunk_position_delta_history_size=args.pattern_chunk_position_delta_history_size,
                chunk_position_delta_top_k=args.pattern_chunk_position_delta_top_k,
                chunk_position_delta_min_count=args.pattern_chunk_position_delta_min_count,
                chunk_position_delta_max_abs=pattern_chunk_position_delta_max_abs,
                enable_chunk_delta_template=args.pattern_chunk_delta_template,
                chunk_delta_template_history_size=args.pattern_chunk_delta_template_history_size,
                chunk_delta_template_top_k=args.pattern_chunk_delta_template_top_k,
                chunk_delta_template_min_prefix_deltas=args.pattern_chunk_delta_template_min_prefix_deltas,
                chunk_delta_template_max_delta_mismatch=pattern_chunk_delta_template_max_delta_mismatch,
                chunk_delta_template_max_abs=pattern_chunk_delta_template_max_abs,
                enable_previous_chunk_position=args.pattern_previous_chunk_position,
                previous_chunk_history_size=args.pattern_previous_chunk_history_size,
                enable_chunk_prefix_retrieval=args.pattern_chunk_prefix_retrieval,
                chunk_prefix_history_size=args.pattern_chunk_prefix_history_size,
                chunk_prefix_top_k=args.pattern_chunk_prefix_top_k,
                chunk_prefix_min_matches=args.pattern_chunk_prefix_min_matches,
                chunk_prefix_max_mismatches=args.pattern_chunk_prefix_max_mismatches,
                enable_action_token_neighborhood=args.pattern_action_token_neighborhood,
                action_token_neighborhood_radius=args.pattern_action_token_neighborhood_radius,
                action_token_neighborhood_top_k=args.pattern_action_token_neighborhood_top_k,
                enable_position_mode_histogram=args.pattern_position_mode_histogram,
                position_mode_history_size=args.pattern_position_mode_history_size,
                position_mode_top_k=args.pattern_position_mode_top_k,
                position_mode_min_count=args.pattern_position_mode_min_count,
                enable_global_position_mode=args.pattern_global_position_mode,
                global_position_history_size=args.pattern_global_position_history_size,
                global_position_top_k=args.pattern_global_position_top_k,
                global_position_min_count=args.pattern_global_position_min_count,
                enable_action_dimension_mode=args.pattern_action_dimension_mode,
                action_dimension_mode_history_size=args.pattern_action_dimension_mode_history_size,
                action_dimension_mode_top_k=args.pattern_action_dimension_mode_top_k,
                action_dimension_mode_min_count=args.pattern_action_dimension_mode_min_count,
                enable_hold_action_token=args.pattern_hold_action_token,
                enable_ngram_continuation=args.pattern_ngram_continuation,
                ngram_min_context=args.pattern_ngram_min_context,
                ngram_max_context=args.pattern_ngram_max_context,
                ngram_history_size=args.pattern_ngram_history_size,
                min_source_agreement=args.pattern_min_source_agreement,
                source_priority=parse_source_priority(args.pattern_source_priority),
                enable_source_cooldown=args.pattern_source_cooldown,
                source_cooldown_after=args.pattern_source_cooldown_after,
                source_cooldown_steps=args.pattern_source_cooldown_steps,
                enable_source_acceptance_bias=args.pattern_source_acceptance_bias,
                source_acceptance_bias_history_size=args.pattern_source_acceptance_bias_history_size,
                source_acceptance_bias_min_observations=args.pattern_source_acceptance_bias_min_observations,
                vocab_size=int(token_adapter.model._paligemma_tokenizer.vocab_size),
                stop_token_ids=(_adapter_action_end_token_id(token_adapter),),
            )
        )
    if any(mode.startswith("block_sd") for mode in modes):
        if not args.block_checkpoint:
            raise SystemExit("block_sd modes require --block-checkpoint")
        if token_adapter is None:
            token_adapter = PI0FastTokenLogitAdapter(policy)
        block_drafter, block_token_map, block_summary = load_block_drafter_checkpoint(args.block_checkpoint, device=device)
        if args.block_gate_checkpoint:
            block_gate, loaded_gate_threshold, block_gate_summary = load_block_gate(args.block_gate_checkpoint, device=device)
            block_gate_threshold = loaded_gate_threshold if args.block_gate_threshold is None else float(args.block_gate_threshold)
        elif args.block_gate_threshold is not None:
            block_gate_threshold = float(args.block_gate_threshold)
        if args.block_action_gate_checkpoint:
            (
                block_action_gate,
                loaded_action_gate_threshold,
                block_action_gate_summary,
            ) = load_action_gate(args.block_action_gate_checkpoint, device=device)
            block_action_gate_threshold = (
                loaded_action_gate_threshold
                if args.block_action_gate_threshold is None
                else float(args.block_action_gate_threshold)
            )
        elif args.block_action_gate_threshold is not None:
            block_action_gate_threshold = float(args.block_action_gate_threshold)
        if args.block_compile_draft:
            hidden = torch.zeros(
                (1, block_drafter.config.hidden_dim),
                dtype=next(block_drafter.parameters()).dtype,
                device=device,
            )
            context = torch.full(
                (1, block_drafter.config.context_len),
                len(block_token_map),
                dtype=torch.long,
                device=device,
            )
            with torch.inference_mode():
                block_drafter.draft(hidden, context, steps=args.block_lookahead, refine_steps=args.block_refine_steps)
            block_drafter = torch.compile(block_drafter, mode="reduce-overhead")
            with torch.inference_mode():
                block_drafter.draft(hidden, context, steps=args.block_lookahead, refine_steps=args.block_refine_steps)
        logger.info(
            "Loaded block drafter %s lookahead=%d gate=%s threshold=%.3f",
            args.block_checkpoint,
            args.block_lookahead,
            args.block_gate_checkpoint,
            block_gate_threshold,
        )
    if any(mode.startswith("block_sd_select") for mode in modes):
        if not args.cutoff_selector_checkpoint:
            raise SystemExit("block_sd_select modes require --cutoff-selector-checkpoint")
        cutoff_selector, cutoff_selector_summary = load_cutoff_selector(
            args.cutoff_selector_checkpoint,
            map_location=device,
        )
        cutoff_selector = cutoff_selector.to(device=device, dtype=torch.float32).eval()
        logger.info(
            "Loaded cutoff selector %s risk=%.3f cutoffs=%s",
            args.cutoff_selector_checkpoint,
            args.cutoff_selector_risk,
            cutoff_selector.config.cutoffs,
        )
    if any(mode.startswith("medusa_sd") for mode in modes):
        if not args.medusa_checkpoint:
            raise SystemExit("medusa_sd modes require --medusa-checkpoint")
        if token_adapter is None:
            token_adapter = PI0FastTokenLogitAdapter(policy)
        medusa_head, medusa_token_map, medusa_summary = load_medusa_checkpoint(args.medusa_checkpoint, device=device)
        logger.info(
            "Loaded Medusa head %s lookahead=%d",
            args.medusa_checkpoint,
            args.medusa_lookahead,
        )
    trajectory_head = None
    action_corrector = None
    action_corrector_summary = None
    adaptive_prefix_gate = None
    adaptive_prefix_gate_summary = None
    if args.adaptive_prefix_gate_checkpoint:
        adaptive_prefix_gate, gate_threshold, adaptive_prefix_gate_summary = load_prefix_gate(
            args.adaptive_prefix_gate_checkpoint,
            device=device,
        )
        if args.adaptive_prefix_gate_threshold is None:
            args.adaptive_prefix_gate_threshold = gate_threshold
        logger.info(
            "Loaded adaptive prefix gate %s threshold=%.3f",
            args.adaptive_prefix_gate_checkpoint,
            args.adaptive_prefix_gate_threshold,
        )
    if any(mode.startswith("traj_tail") or mode.startswith("ngram_traj_tail") for mode in modes):
        if not args.trajectory_head_checkpoint:
            raise SystemExit("traj_tail and ngram_traj_tail modes require --trajectory-head-checkpoint")
        trajectory_head, _trajectory_extra = load_trajectory_tail_checkpoint(args.trajectory_head_checkpoint, map_location=device)
        trajectory_head = trajectory_head.to(device=device, dtype=torch.float32).eval()
        logger.info(
            "Loaded trajectory head %s input_horizon=%d tail_horizon=%d",
            args.trajectory_head_checkpoint,
            trajectory_head.config.input_horizon,
            trajectory_head.config.tail_horizon,
        )
    if any(mode.startswith("block_sd_correct") for mode in modes):
        if not args.action_corrector_checkpoint:
            raise SystemExit("block_sd_correct modes require --action-corrector-checkpoint")
        action_corrector, action_corrector_summary = load_action_dflash_checkpoint(
            args.action_corrector_checkpoint,
            map_location=device,
        )
        action_corrector = action_corrector.to(device=device, dtype=torch.float32).eval()
        logger.info(
            "Loaded action corrector %s horizon=%d refine_steps=%d",
            args.action_corrector_checkpoint,
            action_corrector.config.action_horizon,
            args.action_corrector_refine_steps,
        )
    guard_cfg = ChunkGuardConfig(
        default_execute_window=args.default_window,
        smooth_execute_window=args.smooth_window,
        max_execute_window=args.max_window,
        min_confidence=args.min_confidence,
        position_error_threshold=args.position_error_threshold,
        rotation_error_threshold=args.rotation_error_threshold,
        action_jump_threshold=args.action_jump_threshold,
        jerk_threshold=args.jerk_threshold,
        smooth_position_delta=args.smooth_position_delta,
        smooth_rotation_delta=args.smooth_rotation_delta,
    )

    all_results: list[EpisodeResult] = []
    for task_id in task_ids:
        env = env_map[args.task][task_id]
        logger.info("=== task=%s task_id=%d ===", args.task, task_id)
        for mode in modes:
            if mode == "chunk_m3":
                cfg = ChunkGuardConfig(**{**dataclass_dict(guard_cfg), "smooth_execute_window": args.default_window})
            else:
                cfg = guard_cfg
            controller = ChunkExecutionController(ChunkGuard(cfg))
            drafter = RetrievalChunkDrafter() if "retrieval" in mode or mode.startswith("exact_fast_sd") else None
            for ep in selected_episode_ids:
                seed = args.seed + ep
                logger.info("=== task_id=%d mode=%s episode=%d seed=%d ===", task_id, mode, ep, seed)
                result = run_episode(
                    mode=mode,
                    episode=ep,
                    seed=seed,
                    task=args.task,
                    task_id=task_id,
                    max_steps=args.steps,
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    policy_preprocessor=policy_preprocessor,
                    policy_postprocessor=policy_postprocessor,
                    preprocess_observation=preprocess_observation,
                    controller=controller,
                    drafter=drafter,
                    token_adapter=token_adapter,
                    ngram_drafter=ngram_drafter,
                    pattern_drafter=pattern_drafter,
                    block_drafter=block_drafter,
                    block_token_map=block_token_map,
                    block_gate=block_gate,
                    block_action_gate=block_action_gate,
                    block_action_gate_threshold=block_action_gate_threshold,
                    cutoff_selector=cutoff_selector,
                    cutoff_selector_risk=args.cutoff_selector_risk,
                    medusa_head=medusa_head,
                    medusa_token_map=medusa_token_map,
                    medusa_lookahead=args.medusa_lookahead,
                    medusa_min_draft_confidence=args.medusa_min_draft_confidence,
                    medusa_min_verify_confidence=args.medusa_min_verify_confidence,
                    medusa_min_spec_position=args.medusa_min_spec_position,
                    medusa_accept_partial_blocks=args.medusa_accept_partial_blocks,
                    medusa_replay_accepted_cache=args.medusa_replay_accepted_cache,
                    medusa_resync_accepted_cache=args.medusa_resync_accepted_cache,
                    medusa_verify_from_scratch=args.medusa_verify_from_scratch,
                    block_lookahead=args.block_lookahead,
                    block_min_draft_confidence=args.block_min_draft_confidence,
                    block_min_verify_confidence=args.block_min_verify_confidence,
                    block_min_verify_margin=args.block_min_verify_margin,
                    block_gate_threshold=block_gate_threshold,
                    block_max_future_accept=None
                    if args.block_max_future_accept < 0
                    else args.block_max_future_accept,
                    block_min_future_accept=args.block_min_future_accept,
                    block_min_spec_position=args.block_min_spec_position,
                    block_reject_cooldown_steps=args.block_reject_cooldown_steps,
                    block_reject_cooldown_after=args.block_reject_cooldown_after,
                    block_spec_fallback_cooldown_steps=args.block_spec_fallback_cooldown_steps,
                    block_spec_fallback_cooldown_after=args.block_spec_fallback_cooldown_after,
                    block_allow_unknown_context=args.block_allow_unknown_context,
                    block_repeat_token_draft=args.block_repeat_token_draft,
                    block_repeat_token_min_run=args.block_repeat_token_min_run,
                    block_repeat_pattern_draft=args.block_repeat_pattern_draft,
                    block_repeat_pattern_max_period=args.block_repeat_pattern_max_period,
                    block_repeat_pattern_min_position=args.block_repeat_pattern_min_position,
                    block_pattern_only=args.block_pattern_only,
                    block_unverified_pattern_tail=args.block_unverified_pattern_tail,
                    block_unverified_pattern_eos=args.block_unverified_pattern_eos,
                    block_full_block_only=args.block_full_block_only,
                    block_accept_partial_blocks=args.block_accept_partial_blocks,
                    block_refine_steps=args.block_refine_steps,
                    block_verify_from_scratch=args.block_verify_from_scratch,
                    block_resync_accepted_cache=args.block_resync_accepted_cache,
                    block_draft_after_known_token=args.block_draft_after_known_token,
                    ngram_lookahead=args.ngram_lookahead,
                    ngram_reuse_full_blocks=args.ngram_reuse_full_blocks,
                    pattern_lookahead=args.pattern_lookahead,
                    pattern_reuse_full_blocks=args.pattern_reuse_full_blocks,
                    pattern_min_verify_margin=args.pattern_min_verify_margin,
                    pattern_emit_bonus_token=args.pattern_emit_bonus_token,
                    pattern_verify_from_scratch=args.pattern_verify_from_scratch,
                    pattern_replay_accepted_cache=args.pattern_replay_accepted_cache,
                    pattern_resync_accepted_cache=args.pattern_resync_accepted_cache,
                    pattern_diagnose_verify_alignment=args.pattern_diagnose_verify_alignment,
                    pattern_dynamic_lookahead=args.pattern_dynamic_lookahead,
                    pattern_min_lookahead=args.pattern_min_lookahead,
                    pattern_lookahead_growth=args.pattern_lookahead_growth,
                    pattern_lookahead_shrink=args.pattern_lookahead_shrink,
                    pattern_tree_width=args.pattern_tree_width,
                    pattern_tree_branch_width=args.pattern_tree_branch_width,
                    pattern_dynamic_tree_width=args.pattern_dynamic_tree_width,
                    pattern_min_tree_width=args.pattern_min_tree_width,
                    pattern_tree_width_growth=args.pattern_tree_width_growth,
                    pattern_tree_width_shrink=args.pattern_tree_width_shrink,
                    pattern_tree_anchor_target_token=args.pattern_tree_anchor_target_token,
                    pattern_tree_anchor_target_continuation=args.pattern_tree_anchor_target_continuation,
                    trajectory_head=trajectory_head,
                    trajectory_tail_blend=args.trajectory_tail_blend,
                    trajectory_project_smooth=args.trajectory_project_smooth,
                    action_corrector=action_corrector,
                    action_corrector_refine_steps=args.action_corrector_refine_steps,
                    action_corrector_min_confidence=args.action_corrector_min_confidence,
                    action_corrector_project_smooth=args.action_corrector_project_smooth,
                    action_corrector_preserve_prefix_actions=args.action_corrector_preserve_prefix_actions,
                    action_corrector_blend=args.action_corrector_blend,
                    guarded_min_acceptance=args.guarded_min_acceptance,
                    guarded_min_tokens_per_forward=args.guarded_min_tokens_per_forward,
                    guarded_max_fallback_rate=args.guarded_max_fallback_rate,
                    adaptive_prefix_checkpoints=adaptive_prefix_checkpoints,
                    adaptive_stable_tolerance=args.adaptive_stable_tolerance,
                    adaptive_stable_checks=args.adaptive_stable_checks,
                    adaptive_early_stable_checks=args.adaptive_early_stable_checks or None,
                    adaptive_early_max_stable_checkpoint=args.adaptive_early_max_stable_checkpoint or None,
                    adaptive_skip_unproductive_checks=args.adaptive_skip_unproductive_checks,
                    adaptive_skip_unproductive_after_checkpoint=args.adaptive_skip_unproductive_after_checkpoint,
                    adaptive_continue_to_action_end_on_unstable=args.adaptive_continue_to_action_end_on_unstable,
                    adaptive_prefix_gate=adaptive_prefix_gate,
                    adaptive_prefix_gate_threshold=args.adaptive_prefix_gate_threshold,
                    target_eos_constrained_action_vocab_size=args.target_eos_constrained_action_vocab_size,
                    target_eos_constrained_text_vocab_size=args.target_eos_constrained_text_vocab_size,
                    target_eos_constrained_full_head_margin=(
                        None
                        if args.target_eos_constrained_full_head_margin < 0
                        else args.target_eos_constrained_full_head_margin
                    ),
                    target_eos_constrained_no_force_prefix=args.target_eos_constrained_no_force_prefix,
                    target_eos_constrained_structural_token_radius=args.target_eos_constrained_structural_token_radius,
                    token_trace_sink=token_trace_sink,
                    device=str(device),
                    use_amp=args.use_amp,
                    amp_dtype=amp_dtype,
                )
                all_results.append(result)
                with metrics_path.open("a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
                logger.info(
                    "result task_id=%d mode=%s success=%s avg_ms=%.1f calls/step=%.3f",
                    task_id,
                    result.mode,
                    result.success,
                    result.avg_ms_per_control_step,
                    result.model_calls_per_step,
                )

    if token_trace_sink is not None:
        token_trace_sink.flush()

    summary = summarize(all_results, baseline_mode=args.summary_baseline_mode)
    summary["config"] = {
        "policy": args.policy,
        "policy_kind": args.policy_kind,
        "dtype": args.dtype,
        "use_amp": args.use_amp,
        "amp_dtype": args.amp_dtype or (args.dtype if amp_dtype is not None else None),
        "disable_gradient_checkpointing": args.disable_gradient_checkpointing,
        "target_eos_constrained_action_vocab_size": args.target_eos_constrained_action_vocab_size,
        "target_eos_constrained_text_vocab_size": args.target_eos_constrained_text_vocab_size,
        "target_eos_constrained_full_head_margin": args.target_eos_constrained_full_head_margin,
        "target_eos_constrained_no_force_prefix": args.target_eos_constrained_no_force_prefix,
        "target_eos_constrained_structural_token_radius": args.target_eos_constrained_structural_token_radius,
        "num_inference_steps": args.num_inference_steps,
        "task": args.task,
        "task_id": args.task_id,
        "task_ids": task_ids,
        "episode_ids": selected_episode_ids,
        "summary_baseline_mode": args.summary_baseline_mode,
        "token_trace_output_dir": None
        if args.token_trace_output_dir is None
        else str(args.token_trace_output_dir),
        "token_trace_modes": None if token_trace_sink is None or token_trace_sink.modes is None else sorted(token_trace_sink.modes),
        "token_trace_rows": 0 if token_trace_sink is None else token_trace_sink.total_rows,
        "token_trace_shards": 0 if token_trace_sink is None else token_trace_sink.shards_written,
        "guard": dataclass_dict(guard_cfg),
        "medusa_checkpoint": args.medusa_checkpoint,
        "medusa_summary": medusa_summary,
        "medusa_lookahead": args.medusa_lookahead,
        "medusa_min_draft_confidence": args.medusa_min_draft_confidence,
        "medusa_min_verify_confidence": args.medusa_min_verify_confidence,
        "medusa_min_spec_position": args.medusa_min_spec_position,
        "medusa_accept_partial_blocks": args.medusa_accept_partial_blocks,
        "medusa_replay_accepted_cache": args.medusa_replay_accepted_cache,
        "medusa_resync_accepted_cache": args.medusa_resync_accepted_cache,
        "medusa_verify_from_scratch": args.medusa_verify_from_scratch,
        "block_checkpoint": args.block_checkpoint,
        "block_gate_checkpoint": args.block_gate_checkpoint,
        "block_gate_threshold": block_gate_threshold,
        "block_action_gate_checkpoint": args.block_action_gate_checkpoint,
        "block_action_gate_threshold": block_action_gate_threshold,
        "cutoff_selector_checkpoint": args.cutoff_selector_checkpoint,
        "cutoff_selector_risk": args.cutoff_selector_risk,
        "cutoff_selector_summary": cutoff_selector_summary,
        "block_summary": block_summary,
        "block_gate_summary": block_gate_summary,
        "block_action_gate_summary": block_action_gate_summary,
        "block_lookahead": args.block_lookahead,
        "block_min_draft_confidence": args.block_min_draft_confidence,
        "block_min_verify_confidence": args.block_min_verify_confidence,
        "block_min_verify_margin": args.block_min_verify_margin,
        "block_max_future_accept": args.block_max_future_accept,
        "block_min_future_accept": args.block_min_future_accept,
        "block_min_spec_position": args.block_min_spec_position,
        "block_reject_cooldown_steps": args.block_reject_cooldown_steps,
        "block_reject_cooldown_after": args.block_reject_cooldown_after,
        "block_spec_fallback_cooldown_steps": args.block_spec_fallback_cooldown_steps,
        "block_spec_fallback_cooldown_after": args.block_spec_fallback_cooldown_after,
        "block_allow_unknown_context": args.block_allow_unknown_context,
        "block_repeat_token_draft": args.block_repeat_token_draft,
        "block_repeat_token_min_run": args.block_repeat_token_min_run,
        "block_repeat_pattern_draft": args.block_repeat_pattern_draft,
        "block_repeat_pattern_max_period": args.block_repeat_pattern_max_period,
        "block_repeat_pattern_min_position": args.block_repeat_pattern_min_position,
        "block_pattern_only": args.block_pattern_only,
        "block_unverified_pattern_tail": args.block_unverified_pattern_tail,
        "block_unverified_pattern_eos": args.block_unverified_pattern_eos,
        "block_full_block_only": args.block_full_block_only,
        "block_accept_partial_blocks": args.block_accept_partial_blocks,
        "block_refine_steps": args.block_refine_steps,
        "block_verify_from_scratch": args.block_verify_from_scratch,
        "block_resync_accepted_cache": args.block_resync_accepted_cache,
        "block_draft_after_known_token": args.block_draft_after_known_token,
        "block_compile_draft": args.block_compile_draft,
        "pattern_lookahead": args.pattern_lookahead,
        "pattern_action_dim": args.pattern_action_dim,
        "pattern_max_period": args.pattern_max_period,
        "pattern_min_period_repeats": args.pattern_min_period_repeats,
        "pattern_repeat_token_min_run": args.pattern_repeat_token_min_run,
        "pattern_linear_action_extrapolation": args.pattern_linear_action_extrapolation,
        "pattern_second_order_action_extrapolation": args.pattern_second_order_action_extrapolation,
        "pattern_second_order_max_accel": None
        if args.pattern_second_order_max_accel < 0
        else args.pattern_second_order_max_accel,
        "pattern_action_trend_regression": args.pattern_action_trend_regression,
        "pattern_action_trend_history": args.pattern_action_trend_history,
        "pattern_action_trend_top_k": args.pattern_action_trend_top_k,
        "pattern_action_trend_max_abs": None
        if args.pattern_action_trend_max_abs < 0
        else args.pattern_action_trend_max_abs,
        "pattern_action_prefix_lookup": args.pattern_action_prefix_lookup,
        "pattern_action_prefix_history_size": args.pattern_action_prefix_history_size,
        "pattern_action_prefix_top_k": args.pattern_action_prefix_top_k,
        "pattern_action_prefix_min_prefix": args.pattern_action_prefix_min_prefix,
        "pattern_action_prefix_max_mismatches": args.pattern_action_prefix_max_mismatches,
        "pattern_action_vector_suffix_lookup": args.pattern_action_vector_suffix_lookup,
        "pattern_action_vector_suffix_history_size": args.pattern_action_vector_suffix_history_size,
        "pattern_action_vector_suffix_top_k": args.pattern_action_vector_suffix_top_k,
        "pattern_action_vector_suffix_min_prefix": args.pattern_action_vector_suffix_min_prefix,
        "pattern_action_vector_suffix_min_count": args.pattern_action_vector_suffix_min_count,
        "pattern_action_vector_suffix_max_prefix_delta": args.pattern_action_vector_suffix_max_prefix_delta,
        "pattern_action_vector_transition": args.pattern_action_vector_transition,
        "pattern_action_vector_transition_history_size": args.pattern_action_vector_transition_history_size,
        "pattern_action_vector_transition_top_k": args.pattern_action_vector_transition_top_k,
        "pattern_action_vector_transition_min_count": args.pattern_action_vector_transition_min_count,
        "pattern_action_vector_transition_max_prev_delta": args.pattern_action_vector_transition_max_prev_delta,
        "pattern_action_vector_transition_max_prefix_delta": args.pattern_action_vector_transition_max_prefix_delta,
        "pattern_action_repeat_vector": args.pattern_action_repeat_vector,
        "pattern_action_repeat_min_repeats": args.pattern_action_repeat_min_repeats,
        "pattern_action_repeat_max_delta": args.pattern_action_repeat_max_delta,
        "pattern_chunk_length_stop": args.pattern_chunk_length_stop,
        "pattern_chunk_length_stop_history_size": args.pattern_chunk_length_stop_history_size,
        "pattern_chunk_length_stop_min_count": args.pattern_chunk_length_stop_min_count,
        "pattern_action_context_tree": args.pattern_action_context_tree,
        "pattern_action_context_tree_history_size": args.pattern_action_context_tree_history_size,
        "pattern_action_context_tree_max_context": args.pattern_action_context_tree_max_context,
        "pattern_action_context_tree_top_k": args.pattern_action_context_tree_top_k,
        "pattern_action_context_tree_min_count": args.pattern_action_context_tree_min_count,
        "pattern_action_transition_histogram": args.pattern_action_transition_histogram,
        "pattern_action_transition_history_size": args.pattern_action_transition_history_size,
        "pattern_action_transition_top_k": args.pattern_action_transition_top_k,
        "pattern_action_transition_min_count": args.pattern_action_transition_min_count,
        "pattern_action_delta_histogram": args.pattern_action_delta_histogram,
        "pattern_action_delta_history": args.pattern_action_delta_history,
        "pattern_action_delta_top_k": args.pattern_action_delta_top_k,
        "pattern_action_delta_min_count": args.pattern_action_delta_min_count,
        "pattern_action_delta_max_abs": None
        if args.pattern_action_delta_max_abs < 0
        else args.pattern_action_delta_max_abs,
        "pattern_action_delta_ngram": args.pattern_action_delta_ngram,
        "pattern_action_delta_ngram_history_size": args.pattern_action_delta_ngram_history_size,
        "pattern_action_delta_ngram_min_context": args.pattern_action_delta_ngram_min_context,
        "pattern_action_delta_ngram_max_context": args.pattern_action_delta_ngram_max_context,
        "pattern_action_delta_ngram_top_k": args.pattern_action_delta_ngram_top_k,
        "pattern_action_delta_ngram_min_count": args.pattern_action_delta_ngram_min_count,
        "pattern_action_delta_ngram_max_abs": None
        if args.pattern_action_delta_ngram_max_abs < 0
        else args.pattern_action_delta_ngram_max_abs,
        "pattern_chunk_position_delta": args.pattern_chunk_position_delta,
        "pattern_chunk_position_delta_history_size": args.pattern_chunk_position_delta_history_size,
        "pattern_chunk_position_delta_top_k": args.pattern_chunk_position_delta_top_k,
        "pattern_chunk_position_delta_min_count": args.pattern_chunk_position_delta_min_count,
        "pattern_chunk_position_delta_max_abs": None
        if args.pattern_chunk_position_delta_max_abs < 0
        else args.pattern_chunk_position_delta_max_abs,
        "pattern_chunk_delta_template": args.pattern_chunk_delta_template,
        "pattern_chunk_delta_template_history_size": args.pattern_chunk_delta_template_history_size,
        "pattern_chunk_delta_template_top_k": args.pattern_chunk_delta_template_top_k,
        "pattern_chunk_delta_template_min_prefix_deltas": args.pattern_chunk_delta_template_min_prefix_deltas,
        "pattern_chunk_delta_template_max_delta_mismatch": None
        if args.pattern_chunk_delta_template_max_delta_mismatch < 0
        else args.pattern_chunk_delta_template_max_delta_mismatch,
        "pattern_chunk_delta_template_max_abs": None
        if args.pattern_chunk_delta_template_max_abs < 0
        else args.pattern_chunk_delta_template_max_abs,
        "pattern_previous_chunk_position": args.pattern_previous_chunk_position,
        "pattern_previous_chunk_history_size": args.pattern_previous_chunk_history_size,
        "pattern_chunk_prefix_retrieval": args.pattern_chunk_prefix_retrieval,
        "pattern_chunk_prefix_history_size": args.pattern_chunk_prefix_history_size,
        "pattern_chunk_prefix_top_k": args.pattern_chunk_prefix_top_k,
        "pattern_chunk_prefix_min_matches": args.pattern_chunk_prefix_min_matches,
        "pattern_chunk_prefix_max_mismatches": args.pattern_chunk_prefix_max_mismatches,
        "pattern_action_token_neighborhood": args.pattern_action_token_neighborhood,
        "pattern_action_token_neighborhood_radius": args.pattern_action_token_neighborhood_radius,
        "pattern_action_token_neighborhood_top_k": args.pattern_action_token_neighborhood_top_k,
        "pattern_position_mode_histogram": args.pattern_position_mode_histogram,
        "pattern_position_mode_history_size": args.pattern_position_mode_history_size,
        "pattern_position_mode_top_k": args.pattern_position_mode_top_k,
        "pattern_position_mode_min_count": args.pattern_position_mode_min_count,
        "pattern_global_position_mode": args.pattern_global_position_mode,
        "pattern_global_position_history_size": args.pattern_global_position_history_size,
        "pattern_global_position_top_k": args.pattern_global_position_top_k,
        "pattern_global_position_min_count": args.pattern_global_position_min_count,
        "pattern_action_dimension_mode": args.pattern_action_dimension_mode,
        "pattern_action_dimension_mode_history_size": args.pattern_action_dimension_mode_history_size,
        "pattern_action_dimension_mode_top_k": args.pattern_action_dimension_mode_top_k,
        "pattern_action_dimension_mode_min_count": args.pattern_action_dimension_mode_min_count,
        "pattern_hold_action_token": args.pattern_hold_action_token,
        "pattern_ngram_continuation": args.pattern_ngram_continuation,
        "pattern_ngram_min_context": args.pattern_ngram_min_context,
        "pattern_ngram_max_context": args.pattern_ngram_max_context,
        "pattern_ngram_history_size": args.pattern_ngram_history_size,
        "pattern_min_source_agreement": args.pattern_min_source_agreement,
        "pattern_source_priority": list(parse_source_priority(args.pattern_source_priority)),
        "pattern_source_cooldown": args.pattern_source_cooldown,
        "pattern_source_cooldown_after": args.pattern_source_cooldown_after,
        "pattern_source_cooldown_steps": args.pattern_source_cooldown_steps,
        "pattern_source_acceptance_bias": args.pattern_source_acceptance_bias,
        "pattern_source_acceptance_bias_history_size": args.pattern_source_acceptance_bias_history_size,
        "pattern_source_acceptance_bias_min_observations": args.pattern_source_acceptance_bias_min_observations,
        "pattern_reuse_full_blocks": args.pattern_reuse_full_blocks,
        "pattern_min_verify_margin": args.pattern_min_verify_margin,
        "pattern_emit_bonus_token": args.pattern_emit_bonus_token,
        "pattern_verify_from_scratch": args.pattern_verify_from_scratch,
        "pattern_replay_accepted_cache": args.pattern_replay_accepted_cache,
        "pattern_resync_accepted_cache": args.pattern_resync_accepted_cache,
        "pattern_diagnose_verify_alignment": args.pattern_diagnose_verify_alignment,
        "pattern_dynamic_lookahead": args.pattern_dynamic_lookahead,
        "pattern_min_lookahead": args.pattern_min_lookahead,
        "pattern_lookahead_growth": args.pattern_lookahead_growth,
        "pattern_lookahead_shrink": args.pattern_lookahead_shrink,
        "pattern_tree_width": args.pattern_tree_width,
        "pattern_tree_branch_width": args.pattern_tree_branch_width,
        "pattern_dynamic_tree_width": args.pattern_dynamic_tree_width,
        "pattern_min_tree_width": args.pattern_min_tree_width,
        "pattern_tree_width_growth": args.pattern_tree_width_growth,
        "pattern_tree_width_shrink": args.pattern_tree_width_shrink,
        "pattern_tree_anchor_target_token": args.pattern_tree_anchor_target_token,
        "pattern_tree_anchor_target_continuation": args.pattern_tree_anchor_target_continuation,
        "adaptive_prefix_checkpoints": adaptive_prefix_checkpoints,
        "adaptive_stable_tolerance": args.adaptive_stable_tolerance,
        "adaptive_stable_checks": args.adaptive_stable_checks,
        "adaptive_early_stable_checks": args.adaptive_early_stable_checks,
        "adaptive_early_max_stable_checkpoint": args.adaptive_early_max_stable_checkpoint,
        "adaptive_skip_unproductive_checks": args.adaptive_skip_unproductive_checks,
        "adaptive_skip_unproductive_after_checkpoint": args.adaptive_skip_unproductive_after_checkpoint,
        "adaptive_continue_to_action_end_on_unstable": args.adaptive_continue_to_action_end_on_unstable,
        "adaptive_prefix_gate_checkpoint": args.adaptive_prefix_gate_checkpoint,
        "adaptive_prefix_gate_threshold": args.adaptive_prefix_gate_threshold,
        "adaptive_prefix_gate_summary": adaptive_prefix_gate_summary,
        "action_corrector_checkpoint": args.action_corrector_checkpoint,
        "action_corrector_summary": action_corrector_summary,
        "action_corrector_refine_steps": args.action_corrector_refine_steps,
        "action_corrector_min_confidence": args.action_corrector_min_confidence,
        "action_corrector_project_smooth": args.action_corrector_project_smooth,
        "action_corrector_preserve_prefix_actions": args.action_corrector_preserve_prefix_actions,
        "action_corrector_blend": args.action_corrector_blend,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    try:
        for env in env_map[args.task].values():
            env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
