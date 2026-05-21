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
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.pi0fast_chunking import (  # noqa: E402
    ChunkExecutionController,
    ChunkGuard,
    ChunkGuardConfig,
    RetrievalChunkDrafter,
    dataclass_dict,
)
from serving.pi0fast_eagle import load_trace_records  # noqa: E402
from serving.pi0fast_ngram import NgramDraftConfig, NgramFastTokenDrafter  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402
from serving.pi0fast_trajectory_head import load_trajectory_tail_checkpoint  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

logger = logging.getLogger("run_pi0fast_chunk_eval")


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
    elif hasattr(policy, "predict_action_chunk"):
        raw_action = policy.predict_action_chunk(batch)
    else:
        raw_action = policy.select_action(batch)

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
    return PredictionTrace(actions=actions, elapsed_ms=elapsed_ms, token_count=token_count, token_ids=token_ids)


@torch.inference_mode()
def _predict_ngram_spec_chunk(
    token_adapter: PI0FastTokenLogitAdapter,
    batch: dict[str, Any],
    postprocessor,
    device: str,
    ngram_drafter: NgramFastTokenDrafter,
    lookahead: int,
    reuse_full_blocks: bool,
) -> PredictionTrace:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    trace = token_adapter.predict_action_chunk_ngram_speculative(
        batch,
        drafter=ngram_drafter,
        lookahead=lookahead,
        reuse_full_blocks=reuse_full_blocks,
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
    ngram_lookahead: int,
    ngram_reuse_full_blocks: bool,
    trajectory_head,
    trajectory_tail_blend: float,
    trajectory_project_smooth: bool,
    guarded_min_acceptance: float,
    guarded_min_tokens_per_forward: float,
    guarded_max_fallback_rate: float,
    device: str,
    use_amp: bool,
) -> EpisodeResult:
    policy.reset()
    controller.reset()
    observation, _info = env.reset(seed=[seed])
    done = False
    steps = 0
    reward_sum = 0.0
    success = False
    control_step_ms: list[float] = []
    wall_start = time.perf_counter()
    autocast_ctx = torch.autocast(device_type="cuda") if use_amp and device.startswith("cuda") else nullcontext()

    while not done and steps < max_steps:
        step_start = time.perf_counter()
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
                        )
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
                    if mode.startswith("target_eos") and token_adapter is None:
                        raise RuntimeError("target_eos mode requires --enable-fast-token-hooks")
                    with autocast_ctx:
                        prediction = _predict_action_chunk(
                            policy,
                            batch,
                            policy_postprocessor,
                            device,
                            token_adapter,
                            early_stop_action_end=mode.startswith("target_eos"),
                        )
                    if mode.startswith("target_eos_validate"):
                        with autocast_ctx:
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
                        controller.stats.exact_verifies += 1
                        controller.stats.action_max_diffs.append(max_diff)
                        controller.stats.action_mean_diffs.append(mean_diff)
                        if max_diff != 0.0:
                            controller.stats.guard_reasons.update(["target_eos_action_diff"])
                    chunk = prediction.actions
                    relaxed_tail = False
                    chunk_extend_handled = False
                    if mode.startswith("target_eos"):
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
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Optional comma-separated task ids to evaluate in one process. Overrides --task-id.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--modes", default="baseline,chunk_m3,chunk_m5_smooth")
    parser.add_argument("--output-dir", default="outputs/pi0fast_chunk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--use-amp", action="store_true")
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
        "--ngram-reuse-full-blocks",
        action="store_true",
        help="Reuse verifier KV on fully accepted n-gram draft blocks. Faster but approximate.",
    )
    parser.add_argument("--guarded-min-acceptance", type=float, default=0.70)
    parser.add_argument("--guarded-min-tokens-per-forward", type=float, default=2.0)
    parser.add_argument("--guarded-max-fallback-rate", type=float, default=0.70)
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
    return parser.parse_args()


def _parse_ids(value: str) -> set[int]:
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True

    selected_task_ids = sorted(_parse_ids(args.task_ids)) if args.task_ids else None
    if selected_task_ids is None and args.task_id is not None:
        selected_task_ids = [args.task_id]

    env_kwargs = {"task": args.task, "control_mode": args.control_mode}
    if selected_task_ids is not None:
        env_kwargs["task_ids"] = selected_task_ids
    env_cfg = LiberoEnv(**env_kwargs)

    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
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
    ngram_drafter = None
    if any(mode.startswith("ngram_sd") or mode.startswith("ngram_extend") or mode.startswith("ngram_traj_tail") for mode in modes):
        records = load_trace_records(args.ngram_data_dir)
        train_task_ids = _parse_ids(args.ngram_train_task_ids)
        ngram_drafter = NgramFastTokenDrafter(
            NgramDraftConfig(
                max_context=args.ngram_max_context,
                min_count=args.ngram_min_count,
                lookahead=args.ngram_lookahead,
            )
        )
        ngram_drafter.fit(record for record in records if record.task_id in train_task_ids)
        if token_adapter is None:
            token_adapter = PI0FastTokenLogitAdapter(policy)
    trajectory_head = None
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
            for ep in range(args.episodes):
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
                    ngram_lookahead=args.ngram_lookahead,
                    ngram_reuse_full_blocks=args.ngram_reuse_full_blocks,
                    trajectory_head=trajectory_head,
                    trajectory_tail_blend=args.trajectory_tail_blend,
                    trajectory_project_smooth=args.trajectory_project_smooth,
                    guarded_min_acceptance=args.guarded_min_acceptance,
                    guarded_min_tokens_per_forward=args.guarded_min_tokens_per_forward,
                    guarded_max_fallback_rate=args.guarded_max_fallback_rate,
                    device=str(device),
                    use_amp=args.use_amp,
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

    summary = summarize(all_results)
    summary["config"] = {
        "policy": args.policy,
        "task": args.task,
        "task_id": args.task_id,
        "task_ids": task_ids,
        "guard": dataclass_dict(guard_cfg),
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
