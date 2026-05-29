#!/usr/bin/env python3
"""Paper-mirrored LIBERO benchmark runner for AR vs SpecVLA decoding.

This runner mirrors the Spec-VLA evaluation shape:
- Suites: libero_goal, libero_object, libero_spatial, libero_10 (paper's "Long")
- Tasks per suite: 10
- Trials per task: 50

It supports:
- smoke runs (subset of tasks/trials)
- full runs
- optional auto-promotion from smoke -> full when gate passes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


SUITE_STEP_CAPS = {
    "libero_goal": 300,
    "libero_object": 280,
    "libero_spatial": 220,
    "libero_10": 520,
}


@dataclass
class GateResult:
    passed: bool
    speedup: float
    success_ar: float
    success_spec: float
    success_drop: float
    reason: str


class ProgressLogger:
    """Writes periodic aggregate checkpoints during long runs."""

    def __init__(self, log_path: Path, interval_seconds: int) -> None:
        self.log_path = log_path
        self.interval_seconds = max(interval_seconds, 1)
        self.next_emit_ts = time.time() + self.interval_seconds

    def maybe_emit(
        self,
        *,
        mode: str,
        decoder: str,
        suite: str,
        rows: list[dict[str, Any]],
        force: bool = False,
    ) -> None:
        now = time.time()
        if not force and now < self.next_emit_ts:
            return
        snapshot = aggregate(rows)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "decoder": decoder,
            "suite": suite,
            "episodes_completed": len(rows),
            "aggregate": snapshot,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.next_emit_ts = now + self.interval_seconds


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def preflight_validate(cfg: dict[str, Any], dry_run: bool) -> list[str]:
    errors: list[str] = []
    inputs = cfg["inputs"]
    paths = [
        Path(cfg["specvla_repo_path"]),
        Path(inputs["pretrained_checkpoint"]),
    ]
    allow_missing_head = bool(inputs.get("trajectory_allow_missing_head", False))
    for suite_name in cfg["benchmark"]["suites"]:
        suite_ckpt = inputs["spec_checkpoint_by_suite"].get(suite_name, "")
        if suite_ckpt or not allow_missing_head:
            paths.append(Path(suite_ckpt))
        smooth_ckpt = inputs.get("smooth_spec_checkpoint_by_suite", {}).get(suite_name, "")
        complex_ckpt = inputs.get("complex_spec_checkpoint_by_suite", {}).get(suite_name, "")
        direct_ckpt = inputs.get("chunk_spec_checkpoint_by_suite", {}).get(suite_name, "")
        smooth_direct_ckpt = inputs.get("smooth_chunk_spec_checkpoint_by_suite", {}).get(suite_name, "")
        complex_direct_ckpt = inputs.get("complex_chunk_spec_checkpoint_by_suite", {}).get(suite_name, "")
        accept_length_gate_ckpt = inputs.get("accept_length_gate_checkpoint_by_suite", {}).get(suite_name, "")
        for phase_ckpt in (
            smooth_ckpt,
            complex_ckpt,
            direct_ckpt,
            smooth_direct_ckpt,
            complex_direct_ckpt,
            accept_length_gate_ckpt,
        ):
            if phase_ckpt or not allow_missing_head:
                paths.append(Path(phase_ckpt))

    for p in paths:
        raw = str(p)
        if not raw:
            errors.append("Found empty required path in config.")
            continue
        # Allow Hugging Face style refs (e.g. org/model-name) as non-local checkpoints.
        is_hf_ref = ("/" in raw) and (not raw.startswith("/")) and (not raw.startswith("./"))
        if is_hf_ref:
            continue
        if not p.exists() and not dry_run:
            errors.append(f"Missing required path: {p}")
    return errors


def make_run_id(mode: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"libero_mirror_{mode}_{stamp}"


def build_model_cfg(
    suite_name: str,
    cfg: dict[str, Any],
    mode: str,
) -> SimpleNamespace:
    inputs = cfg["inputs"]
    return SimpleNamespace(
        model_family=inputs.get("model_family", "openvla"),
        pretrained_checkpoint=inputs["pretrained_checkpoint"],
        load_in_8bit=bool(inputs.get("load_in_8bit", False)),
        load_in_4bit=bool(inputs.get("load_in_4bit", False)),
        center_crop=bool(inputs.get("center_crop", True)),
        use_spec=(mode == "spec"),
        parallel_draft=bool(inputs.get("parallel_draft", False)),
        accept_threshold=int(inputs["accept_threshold_by_suite"][suite_name]),
        spec_checkpoint=inputs["spec_checkpoint_by_suite"][suite_name],
        smooth_spec_checkpoint=inputs.get("smooth_spec_checkpoint_by_suite", {}).get(suite_name, ""),
        complex_spec_checkpoint=inputs.get("complex_spec_checkpoint_by_suite", {}).get(suite_name, ""),
        chunk_spec_checkpoint=inputs.get("chunk_spec_checkpoint_by_suite", {}).get(suite_name, ""),
        smooth_chunk_spec_checkpoint=inputs.get("smooth_chunk_spec_checkpoint_by_suite", {}).get(suite_name, ""),
        complex_chunk_spec_checkpoint=inputs.get("complex_chunk_spec_checkpoint_by_suite", {}).get(suite_name, ""),
        accept_length_gate_checkpoint=inputs.get("accept_length_gate_checkpoint_by_suite", {}).get(suite_name, ""),
        task_suite_name=suite_name,
        num_steps_wait=int(cfg["benchmark"]["num_steps_wait"]),
        num_trials_per_task=0,  # managed externally
        seed=int(cfg["benchmark"]["seed"]),
        unnorm_key=suite_name,
    )


def suite_value(raw: Any, suite_name: str, default: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get(suite_name, default)
    if raw is None:
        return default
    return raw


def task_ids_for_suite(cfg: dict[str, Any], suite_name: str, task_limit: int) -> list[int]:
    raw = suite_value(cfg["benchmark"].get("task_ids_by_suite"), suite_name, None)
    if raw is None:
        return list(range(task_limit))
    if isinstance(raw, str):
        ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        ids = [int(x) for x in raw]
    return ids[:task_limit]


def task_value(raw: Any, suite_name: str, task_id: int, default: Any) -> Any:
    suite_raw = suite_value(raw, suite_name, raw)
    if isinstance(suite_raw, dict):
        return suite_raw.get(str(task_id), suite_raw.get(task_id, suite_raw.get("default", default)))
    if suite_raw is None:
        return default
    return suite_raw


def center_crop_openvla_image(image, *, enabled: bool):
    if not enabled:
        return image
    from PIL import Image
    import numpy as np
    import tensorflow as tf
    from experiments.robot.openvla_utils import crop_and_resize  # type: ignore

    batch_size = 1
    crop_scale = 0.9
    tensor = tf.convert_to_tensor(np.array(image))
    orig_dtype = tensor.dtype
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    tensor = crop_and_resize(tensor, crop_scale, batch_size)
    tensor = tf.clip_by_value(tensor, 0, 1)
    tensor = tf.image.convert_image_dtype(tensor, orig_dtype, saturate=True)
    return Image.fromarray(tensor.numpy()).convert("RGB")


def ensure_openvla_action_prompt_token(inputs, device: str):
    """Match OpenVLA predict_action() prompt-token handling for custom decode."""
    import torch

    input_ids = inputs["input_ids"]
    if torch.all(input_ids[:, -1] == 29871):
        return inputs
    extra = torch.full((input_ids.shape[0], 1), 29871, dtype=input_ids.dtype, device=device)
    inputs["input_ids"] = torch.cat([input_ids, extra], dim=1)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        mask = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat(
            [mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=device)],
            dim=1,
        )
    return inputs


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = sum(r["steps"] for r in rows)
    successes = sum(1 for r in rows if r["success"])
    inference_ms_total = sum(r["inference_ms_total"] for r in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / max(len(rows), 1),
        "steps_total": steps,
        "avg_steps_per_episode": steps / max(len(rows), 1),
        "avg_ms_per_step": inference_ms_total / max(steps, 1),
        "inference_ms_total": inference_ms_total,
    }


def evaluate_gate(cfg: dict[str, Any], ar_rows: list[dict[str, Any]], spec_rows: list[dict[str, Any]]) -> GateResult:
    ar = aggregate(ar_rows)
    spec = aggregate(spec_rows)
    speedup = ar["avg_ms_per_step"] / max(spec["avg_ms_per_step"], 1e-9)
    success_drop = ar["success_rate"] - spec["success_rate"]
    speed_floor = float(cfg["gate"]["speedup_gt"])
    drop_ceiling = float(cfg["gate"]["max_success_drop"])
    passed = speedup > speed_floor and success_drop <= drop_ceiling
    reason = (
        f"speedup={speedup:.3f} (>{speed_floor}) and "
        f"success_drop={success_drop:.4f} (<={drop_ceiling})"
    )
    return GateResult(
        passed=passed,
        speedup=speedup,
        success_ar=ar["success_rate"],
        success_spec=spec["success_rate"],
        success_drop=success_drop,
        reason=reason,
    )


def run_suite_mode(
    *,
    suite_name: str,
    mode: str,
    cfg: dict[str, Any],
    run_dir: Path,
    task_limit: int,
    trials_per_task: int,
    dry_run: bool,
    progress_logger: ProgressLogger | None,
    mode_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if dry_run:
        for task_id in range(task_limit):
            for trial in range(trials_per_task):
                rows.append(
                    {
                        "suite": suite_name,
                        "task_id": task_id,
                        "trial": trial,
                        "mode": mode,
                        "success": bool((task_id + trial) % 2 == 0),
                        "steps": min(15, SUITE_STEP_CAPS[suite_name]),
                        "inference_ms_total": 15 * (200.0 if mode == "ar" else 140.0),
                    }
                )
                if progress_logger:
                    progress_logger.maybe_emit(
                        mode=mode_label,
                        decoder=mode,
                        suite=suite_name,
                        rows=rows,
                    )
        if progress_logger:
            progress_logger.maybe_emit(
                mode=mode_label,
                decoder=mode,
                suite=suite_name,
                rows=rows,
                force=True,
            )
        return rows

    specvla_repo = Path(cfg["specvla_repo_path"]).resolve()
    openvla_pkg_root = specvla_repo / "openvla"
    sys.path.insert(0, str(specvla_repo))
    sys.path.insert(0, str(openvla_pkg_root))

    try:
        from libero.libero import benchmark  # type: ignore
    except ModuleNotFoundError:
        from libero import benchmark  # type: ignore
    from experiments.robot.libero.libero_utils import (  # type: ignore
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (  # type: ignore
        get_action,
        get_image_resize_size,
        get_model,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )
    from experiments.robot.openvla_utils import get_processor  # type: ignore
    from PIL import Image
    import numpy as np
    import torch

    from serving.trajectory_draft_head import TinyTrajectoryHead
    from serving.trajectory_phase import PhaseThresholds
    from serving.trajectory_speculative_decoder import TrajectorySpeculativeDecoder
    from serving.accept_length_gate import AcceptLengthGate

    model_cfg = build_model_cfg(suite_name=suite_name, cfg=cfg, mode=mode)
    set_seed_everywhere(model_cfg.seed)
    model = get_model(model_cfg)
    processor = get_processor(model_cfg)
    resize_size = get_image_resize_size(model_cfg)
    spec_decoder: TrajectorySpeculativeDecoder | None = None
    if mode == "spec":
        draft_head = (
            TinyTrajectoryHead.load(model_cfg.spec_checkpoint, device="cuda")
            if model_cfg.spec_checkpoint and Path(model_cfg.spec_checkpoint).exists()
            else None
        )
        smooth_head = (
            TinyTrajectoryHead.load(model_cfg.smooth_spec_checkpoint, device="cuda")
            if model_cfg.smooth_spec_checkpoint and Path(model_cfg.smooth_spec_checkpoint).exists()
            else None
        )
        complex_head = (
            TinyTrajectoryHead.load(model_cfg.complex_spec_checkpoint, device="cuda")
            if model_cfg.complex_spec_checkpoint and Path(model_cfg.complex_spec_checkpoint).exists()
            else None
        )
        direct_chunk_head = (
            TinyTrajectoryHead.load(model_cfg.chunk_spec_checkpoint, device="cuda")
            if model_cfg.chunk_spec_checkpoint and Path(model_cfg.chunk_spec_checkpoint).exists()
            else None
        )
        smooth_direct_chunk_head = (
            TinyTrajectoryHead.load(model_cfg.smooth_chunk_spec_checkpoint, device="cuda")
            if model_cfg.smooth_chunk_spec_checkpoint and Path(model_cfg.smooth_chunk_spec_checkpoint).exists()
            else None
        )
        complex_direct_chunk_head = (
            TinyTrajectoryHead.load(model_cfg.complex_chunk_spec_checkpoint, device="cuda")
            if model_cfg.complex_chunk_spec_checkpoint and Path(model_cfg.complex_chunk_spec_checkpoint).exists()
            else None
        )
        accept_length_gate = (
            AcceptLengthGate.load(model_cfg.accept_length_gate_checkpoint, device="cuda")
            if model_cfg.accept_length_gate_checkpoint and Path(model_cfg.accept_length_gate_checkpoint).exists()
            else None
        )
        fast_min_confident_tokens = int(
            suite_value(
                cfg["inputs"].get("trajectory_fast_min_confident_tokens_by_suite"),
                suite_name,
                cfg["inputs"].get("trajectory_fast_min_confident_tokens", 5),
            )
        )
        fast_max_draft_calls_cfg = cfg["inputs"].get("trajectory_fast_max_draft_calls")
        fast_max_draft_calls = None if fast_max_draft_calls_cfg is None else int(fast_max_draft_calls_cfg)
        fast_max_action_step_cfg = cfg["inputs"].get("trajectory_fast_max_action_step")
        fast_max_action_step = None if fast_max_action_step_cfg is None else int(fast_max_action_step_cfg)
        stationary_patience_cfg = cfg["inputs"].get("trajectory_fast_stationary_patience")
        stationary_patience = None if stationary_patience_cfg is None else int(stationary_patience_cfg)
        spec_decoder = TrajectorySpeculativeDecoder(
            model=model,
            device="cuda",
            draft_head=draft_head,
            smooth_draft_head=smooth_head,
            complex_draft_head=complex_head,
            direct_chunk_head=direct_chunk_head,
            smooth_direct_chunk_head=smooth_direct_chunk_head,
            complex_direct_chunk_head=complex_direct_chunk_head,
            accept_length_gate=accept_length_gate,
            accept_length_min=int(cfg["inputs"].get("trajectory_accept_length_min", 0)),
            accept_length_max=cfg["inputs"].get("trajectory_accept_length_max"),
            decoder_mode=str(cfg["inputs"].get("trajectory_decoder_mode", "trajectory-spec")),
            band_radius=int(cfg["inputs"].get("trajectory_band_radius", 2)),
            max_residual_bins=float(cfg["inputs"].get("trajectory_max_residual_bins", 8.0)),
            tree_width=int(cfg["inputs"].get("trajectory_tree_width", 8)),
            max_tree_depth=int(cfg["inputs"].get("trajectory_max_tree_depth", 1)),
            allow_approx_tree=bool(cfg["inputs"].get("trajectory_allow_approx_tree", False)),
            fast_draft_only=bool(cfg["inputs"].get("trajectory_fast_draft_only", True)),
            fast_min_confident_tokens=fast_min_confident_tokens,
            fast_max_draft_calls=fast_max_draft_calls,
            fast_max_action_step=fast_max_action_step,
            fast_stop_after_gripper_change=bool(cfg["inputs"].get("trajectory_fast_stop_after_gripper_change", False)),
            fast_stationary_token_delta=float(cfg["inputs"].get("trajectory_fast_stationary_token_delta", 2.0)),
            fast_stationary_patience=stationary_patience,
            use_draft_prefill_hidden=bool(cfg["inputs"].get("trajectory_use_draft_prefill_hidden", True)),
            head_threshold=float(cfg["inputs"].get("trajectory_head_threshold", 0.2)),
            head_top_k=int(cfg["inputs"].get("trajectory_head_top_k", 3)),
            retrieval_top_k=int(cfg["inputs"].get("trajectory_retrieval_top_k", 4)),
            retrieval_min_confidence=float(cfg["inputs"].get("trajectory_retrieval_min_confidence", 0.55)),
            kinematic_threshold=float(cfg["inputs"].get("trajectory_kinematic_threshold", 4.0)),
            smooth_phase_curvature=float(cfg["inputs"].get("trajectory_smooth_phase_curvature", 6.0)),
            smooth_phase_acceleration=float(cfg["inputs"].get("trajectory_smooth_phase_acceleration", 8.0)),
            smooth_phase_min_displacement=float(cfg["inputs"].get("trajectory_smooth_phase_min_displacement", 1.5)),
            relaxed_tolerance=float(cfg["inputs"].get("trajectory_relaxed_tolerance", 2.0)),
            hybrid_max_draft_length=int(cfg["inputs"].get("trajectory_hybrid_max_draft_length", 7)),
            chunk_smooth_len=int(cfg["inputs"].get("trajectory_chunk_smooth_len", 4)),
            chunk_complex_len=int(cfg["inputs"].get("trajectory_chunk_complex_len", 2)),
            chunk_heartbeat=int(cfg["inputs"].get("trajectory_chunk_heartbeat", 6)),
            chunk_min_confident_tokens=int(cfg["inputs"].get("trajectory_chunk_min_confident_tokens", 6)),
            chunk_max_token_delta=float(cfg["inputs"].get("trajectory_chunk_max_token_delta", 32.0)),
            chunk_allow_phase_switch=bool(cfg["inputs"].get("trajectory_chunk_allow_phase_switch", False)),
            chunk_max_kinematic_curvature=(
                None
                if cfg["inputs"].get("trajectory_chunk_max_kinematic_curvature") is None
                else float(cfg["inputs"].get("trajectory_chunk_max_kinematic_curvature"))
            ),
            chunk_kinematic_blend=float(cfg["inputs"].get("trajectory_chunk_kinematic_blend", 0.0)),
            chunk_max_rectification_bins=float(cfg["inputs"].get("trajectory_chunk_max_rectification_bins", 12.0)),
            chunk_dynamic_accept=bool(cfg["inputs"].get("trajectory_chunk_dynamic_accept", False)),
            chunk_late_head_threshold=float(cfg["inputs"].get("trajectory_chunk_late_head_threshold", 0.45)),
            chunk_late_min_confident_tokens=int(cfg["inputs"].get("trajectory_chunk_late_min_confident_tokens", 6)),
            chunk_late_max_token_delta=float(cfg["inputs"].get("trajectory_chunk_late_max_token_delta", 24.0)),
            chunk_late_max_kinematic_curvature=(
                None
                if cfg["inputs"].get("trajectory_chunk_late_max_kinematic_curvature") is None
                else float(cfg["inputs"].get("trajectory_chunk_late_max_kinematic_curvature"))
            ),
            chunk_len_after_step=cfg["inputs"].get("trajectory_chunk_len_after_step"),
            chunk_smooth_len_late=cfg["inputs"].get("trajectory_chunk_smooth_len_late"),
            chunk_complex_len_late=cfg["inputs"].get("trajectory_chunk_complex_len_late"),
            chunk_allow_phase_switch_after_step=cfg["inputs"].get("trajectory_chunk_allow_phase_switch_after_step"),
            chunk_gripper_cooldown_steps=int(cfg["inputs"].get("trajectory_chunk_gripper_cooldown_steps", 0)),
            chunk_precision_guard_after_step=cfg["inputs"].get("trajectory_chunk_precision_guard_after_step"),
            chunk_precision_guard_token_delta=float(
                cfg["inputs"].get("trajectory_chunk_precision_guard_token_delta", 0.0)
            ),
            chunk_precision_guard_history_delta=float(
                cfg["inputs"].get("trajectory_chunk_precision_guard_history_delta", 0.0)
            ),
        )

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    n_tasks = min(task_limit, task_suite.n_tasks)
    task_ids = task_ids_for_suite(cfg, suite_name, n_tasks)
    max_steps = SUITE_STEP_CAPS[suite_name]
    num_steps_wait = int(cfg["benchmark"]["num_steps_wait"])
    log_chunk_decisions = bool(cfg["inputs"].get("trajectory_log_chunk_decisions", False))
    chunk_decision_path = run_dir / f"{mode}_{suite_name}_chunk_decisions.jsonl"
    log_action_trace = bool(cfg["inputs"].get("trajectory_log_action_trace", False))
    action_trace_path = run_dir / f"{mode}_{suite_name}_action_trace.jsonl"

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, model_cfg.model_family, resolution=256)
        for episode_idx in range(trials_per_task):
            if spec_decoder is not None:
                spec_decoder.reset()
                per_task_decoder_mode = task_value(
                    cfg["inputs"].get("trajectory_decoder_mode_by_task"),
                    suite_name,
                    task_id,
                    cfg["inputs"].get("trajectory_decoder_mode", spec_decoder.decoder_mode),
                )
                spec_decoder.set_decoder_mode(str(per_task_decoder_mode))
                per_task_max = task_value(
                    cfg["inputs"].get("trajectory_fast_max_draft_calls_by_task"),
                    suite_name,
                    task_id,
                    spec_decoder.fast_max_draft_calls,
                )
                spec_decoder.fast_max_draft_calls = None if per_task_max is None else int(per_task_max)
                spec_decoder.chunk_smooth_len = int(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_smooth_len_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_smooth_len", spec_decoder.chunk_smooth_len),
                    )
                )
                spec_decoder.chunk_complex_len = int(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_complex_len_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_complex_len", spec_decoder.chunk_complex_len),
                    )
                )
                spec_decoder.chunk_dynamic_accept = bool(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_dynamic_accept_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_dynamic_accept", spec_decoder.chunk_dynamic_accept),
                    )
                )
                len_after_step = task_value(
                    cfg["inputs"].get("trajectory_chunk_len_after_step_by_task"),
                    suite_name,
                    task_id,
                    cfg["inputs"].get("trajectory_chunk_len_after_step", spec_decoder.chunk_len_after_step),
                )
                spec_decoder.chunk_len_after_step = None if len_after_step is None else int(len_after_step)
                smooth_len_late = task_value(
                    cfg["inputs"].get("trajectory_chunk_smooth_len_late_by_task"),
                    suite_name,
                    task_id,
                    cfg["inputs"].get("trajectory_chunk_smooth_len_late", spec_decoder.chunk_smooth_len_late),
                )
                spec_decoder.chunk_smooth_len_late = None if smooth_len_late is None else int(smooth_len_late)
                complex_len_late = task_value(
                    cfg["inputs"].get("trajectory_chunk_complex_len_late_by_task"),
                    suite_name,
                    task_id,
                    cfg["inputs"].get("trajectory_chunk_complex_len_late", spec_decoder.chunk_complex_len_late),
                )
                spec_decoder.chunk_complex_len_late = None if complex_len_late is None else int(complex_len_late)
                spec_decoder.chunk_heartbeat = int(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_heartbeat_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_heartbeat", spec_decoder.chunk_heartbeat),
                    )
                )
                spec_decoder.chunk_min_confident_tokens = int(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_min_confident_tokens_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_min_confident_tokens", spec_decoder.chunk_min_confident_tokens),
                    )
                )
                spec_decoder.chunk_max_token_delta = float(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_max_token_delta_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_max_token_delta", spec_decoder.chunk_max_token_delta),
                    )
                )
                spec_decoder.chunk_allow_phase_switch = bool(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_allow_phase_switch_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_allow_phase_switch", spec_decoder.chunk_allow_phase_switch),
                    )
                )
                allow_phase_after = task_value(
                    cfg["inputs"].get("trajectory_chunk_allow_phase_switch_after_step_by_task"),
                    suite_name,
                    task_id,
                    cfg["inputs"].get(
                        "trajectory_chunk_allow_phase_switch_after_step",
                        spec_decoder.chunk_allow_phase_switch_after_step,
                    ),
                )
                spec_decoder.chunk_allow_phase_switch_after_step = (
                    None if allow_phase_after is None else int(allow_phase_after)
                )
                curvature_cfg = task_value(
                    cfg["inputs"].get("trajectory_chunk_max_kinematic_curvature_by_task"),
                    suite_name,
                    task_id,
                    cfg["inputs"].get(
                        "trajectory_chunk_max_kinematic_curvature",
                        spec_decoder.chunk_max_kinematic_curvature,
                    ),
                )
                spec_decoder.chunk_max_kinematic_curvature = None if curvature_cfg is None else float(curvature_cfg)
                spec_decoder.chunk_kinematic_blend = float(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_kinematic_blend_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_kinematic_blend", spec_decoder.chunk_kinematic_blend),
                    )
                )
                spec_decoder.chunk_max_rectification_bins = float(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_max_rectification_bins_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get(
                            "trajectory_chunk_max_rectification_bins",
                            spec_decoder.chunk_max_rectification_bins,
                        ),
                    )
                )
                spec_decoder.chunk_dynamic_accept = bool(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_dynamic_accept_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_dynamic_accept", spec_decoder.chunk_dynamic_accept),
                    )
                )
                spec_decoder.chunk_late_head_threshold = float(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_late_head_threshold_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_late_head_threshold", spec_decoder.chunk_late_head_threshold),
                    )
                )
                spec_decoder.chunk_late_min_confident_tokens = int(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_late_min_confident_tokens_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get(
                            "trajectory_chunk_late_min_confident_tokens",
                            spec_decoder.chunk_late_min_confident_tokens,
                        ),
                    )
                )
                spec_decoder.chunk_late_max_token_delta = float(
                    task_value(
                        cfg["inputs"].get("trajectory_chunk_late_max_token_delta_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_chunk_late_max_token_delta", spec_decoder.chunk_late_max_token_delta),
                    )
                )
                late_curvature_cfg = task_value(
                    cfg["inputs"].get("trajectory_chunk_late_max_kinematic_curvature_by_task"),
                    suite_name,
                    task_id,
                    cfg["inputs"].get(
                        "trajectory_chunk_late_max_kinematic_curvature",
                        spec_decoder.chunk_late_max_kinematic_curvature,
                    ),
                )
                spec_decoder.chunk_late_max_kinematic_curvature = (
                    None if late_curvature_cfg is None else float(late_curvature_cfg)
                )
                spec_decoder.head_threshold = float(
                    task_value(
                        cfg["inputs"].get("trajectory_head_threshold_by_task"),
                        suite_name,
                        task_id,
                        cfg["inputs"].get("trajectory_head_threshold", spec_decoder.head_threshold),
                    )
                )
                spec_decoder.phase_thresholds = PhaseThresholds(
                    smooth_curvature=float(
                        task_value(
                            cfg["inputs"].get("trajectory_smooth_phase_curvature_by_task"),
                            suite_name,
                            task_id,
                            cfg["inputs"].get("trajectory_smooth_phase_curvature", spec_decoder.phase_thresholds.smooth_curvature),
                        )
                    ),
                    smooth_acceleration=float(
                        task_value(
                            cfg["inputs"].get("trajectory_smooth_phase_acceleration_by_task"),
                            suite_name,
                            task_id,
                            cfg["inputs"].get("trajectory_smooth_phase_acceleration", spec_decoder.phase_thresholds.smooth_acceleration),
                        )
                    ),
                    min_displacement=float(
                        task_value(
                            cfg["inputs"].get("trajectory_smooth_phase_min_displacement_by_task"),
                            suite_name,
                            task_id,
                            cfg["inputs"].get("trajectory_smooth_phase_min_displacement", spec_decoder.phase_thresholds.min_displacement),
                        )
                    ),
                    stationary_displacement=spec_decoder.phase_thresholds.stationary_displacement,
                )
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            inference_ms_total = 0.0
            done = False
            action_trace_rows: list[dict[str, Any]] = []
            while t < max_steps + num_steps_wait:
                if t < num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(model_cfg.model_family))
                    t += 1
                    continue
                if mode == "spec":
                    assert spec_decoder is not None
                    tic = time.perf_counter()
                    action = spec_decoder.try_predict_buffered_action(
                        unnorm_key=model_cfg.unnorm_key,
                        task_key=f"{suite_name}:{task_id}:{task_description}",
                    )
                    if action is not None:
                        inference_ms_total += (time.perf_counter() - tic) * 1000.0
                        action = normalize_gripper_action(action, binarize=True)
                        if model_cfg.model_family == "openvla":
                            action = invert_gripper_action(action)
                        env_action = action.tolist()
                        if log_action_trace:
                            action_trace_rows.append(
                                {
                                    "suite": suite_name,
                                    "task_id": task_id,
                                    "trial": episode_idx,
                                    "mode": mode,
                                    "action_step": max(t - num_steps_wait, 0),
                                    "source": "buffer",
                                    "action": [float(x) for x in env_action],
                                }
                            )
                        obs, _, done, _ = env.step(env_action)
                        t += 1
                        if done:
                            break
                        continue
                img = get_libero_image(obs, resize_size)
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                }
                tic = time.perf_counter()
                if mode == "ar":
                    action = get_action(
                        model_cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )
                else:
                    assert spec_decoder is not None
                    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
                    image = Image.fromarray(img).convert("RGB")
                    image = center_crop_openvla_image(image, enabled=bool(model_cfg.center_crop))
                    inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
                    inputs = ensure_openvla_action_prompt_token(inputs, "cuda")
                    action = spec_decoder.predict_action(
                        inputs,
                        unnorm_key=model_cfg.unnorm_key,
                        max_new_tokens=model.get_action_dim(model_cfg.unnorm_key),
                        task_key=f"{suite_name}:{task_id}:{task_description}",
                    )
                inference_ms_total += (time.perf_counter() - tic) * 1000.0
                action = normalize_gripper_action(action, binarize=True)
                if model_cfg.model_family == "openvla":
                    action = invert_gripper_action(action)
                env_action = action.tolist()
                if log_action_trace:
                    action_trace_rows.append(
                        {
                            "suite": suite_name,
                            "task_id": task_id,
                            "trial": episode_idx,
                            "mode": mode,
                            "action_step": max(t - num_steps_wait, 0),
                            "source": "model",
                            "action": [float(x) for x in env_action],
                        }
                    )
                obs, _, done, _ = env.step(env_action)
                t += 1
                if done:
                    break
            episode_row = {
                "suite": suite_name,
                "task_id": task_id,
                "trial": episode_idx,
                "mode": mode,
                "success": bool(done),
                "steps": max(t - num_steps_wait, 0),
                "inference_ms_total": inference_ms_total,
                "spec_stats": spec_decoder.stats.summary() if spec_decoder is not None else None,
            }
            if log_chunk_decisions and spec_decoder is not None:
                decision_rows = []
                for record in spec_decoder.chunk_decision_log:
                    enriched = dict(record)
                    enriched.update(
                        {
                            "suite": suite_name,
                            "task_id": task_id,
                            "trial": episode_idx,
                            "episode_success": bool(done),
                            "episode_steps": max(t - num_steps_wait, 0),
                        }
                    )
                    decision_rows.append(enriched)
                append_jsonl(chunk_decision_path, decision_rows)
                episode_row["chunk_decision_log"] = str(chunk_decision_path)
                episode_row["chunk_decision_count"] = len(decision_rows)
            if log_action_trace:
                for record in action_trace_rows:
                    record["episode_success"] = bool(done)
                    record["episode_steps"] = max(t - num_steps_wait, 0)
                append_jsonl(action_trace_path, action_trace_rows)
                episode_row["action_trace_log"] = str(action_trace_path)
                episode_row["action_trace_count"] = len(action_trace_rows)
            rows.append(episode_row)
            if progress_logger:
                progress_logger.maybe_emit(
                    mode=mode_label,
                    decoder=mode,
                    suite=suite_name,
                    rows=rows,
                )

    if progress_logger:
        progress_logger.maybe_emit(
            mode=mode_label,
            decoder=mode,
            suite=suite_name,
            rows=rows,
            force=True,
        )
    return rows


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def run_mode(mode: str, cfg: dict[str, Any], dry_run: bool, parent_run_dir: Path) -> dict[str, Any]:
    run_dir = parent_run_dir / mode
    ensure_dir(run_dir)
    progress_interval = int(cfg["benchmark"].get("progress_log_seconds", 3600))
    progress_logger = ProgressLogger(run_dir / "progress.log.jsonl", progress_interval)

    if mode == "smoke":
        task_limit = int(cfg["smoke"]["tasks_per_suite"])
        trials_per_task = int(cfg["smoke"]["trials_per_task"])
    else:
        task_limit = int(cfg["benchmark"]["tasks_per_suite"])
        trials_per_task = int(cfg["benchmark"]["trials_per_task"])

    all_rows: list[dict[str, Any]] = []
    per_decoder_summary: dict[str, Any] = {}
    decoder_modes = tuple(cfg["benchmark"].get("decoder_modes", ("ar", "spec")))
    for decoder_mode in decoder_modes:
        decoder_rows: list[dict[str, Any]] = []
        for suite in cfg["benchmark"]["suites"]:
            rows = run_suite_mode(
                suite_name=suite,
                mode=decoder_mode,
                cfg=cfg,
                run_dir=run_dir,
                task_limit=task_limit,
                trials_per_task=trials_per_task,
                dry_run=dry_run,
                progress_logger=progress_logger,
                mode_label=mode,
            )
            decoder_rows.extend(rows)
        write_jsonl(run_dir / f"{decoder_mode}_episodes.jsonl", decoder_rows)
        per_decoder_summary[decoder_mode] = aggregate(decoder_rows)
        all_rows.extend(decoder_rows)

    if {"ar", "spec"}.issubset(set(decoder_modes)):
        gate = evaluate_gate(cfg, [r for r in all_rows if r["mode"] == "ar"], [r for r in all_rows if r["mode"] == "spec"])
        gate_payload = {
            "passed": gate.passed,
            "speedup": gate.speedup,
            "success_ar": gate.success_ar,
            "success_spec": gate.success_spec,
            "success_drop": gate.success_drop,
            "reason": gate.reason,
        }
    else:
        gate_payload = {
            "passed": None,
            "speedup": None,
            "success_ar": per_decoder_summary.get("ar", {}).get("success_rate"),
            "success_spec": per_decoder_summary.get("spec", {}).get("success_rate"),
            "success_drop": None,
            "reason": f"gate skipped for decoder_modes={decoder_modes}",
        }

    write_json(run_dir / "summary.json", {"mode": mode, "decoder_summary": per_decoder_summary, "gate": gate_payload})
    progress_logger.maybe_emit(
        mode=mode,
        decoder="all",
        suite="all",
        rows=all_rows,
        force=True,
    )
    return {"mode": mode, "run_dir": str(run_dir), "gate": gate_payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-mirrored LIBERO AR-vs-Spec benchmark.")
    parser.add_argument("--config", default="configs/libero_specvla_mirror.yaml")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--auto-full-after-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    errors = preflight_validate(cfg, dry_run=args.dry_run)
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))

    output_root = Path(cfg["output_root"])
    ensure_dir(output_root)
    run_id = make_run_id(args.mode)
    parent_run_dir = output_root / run_id
    ensure_dir(parent_run_dir)
    write_json(parent_run_dir / "manifest.json", {"mode": args.mode, "dry_run": args.dry_run, "config": cfg})

    smoke_result = run_mode(args.mode, cfg, dry_run=args.dry_run, parent_run_dir=parent_run_dir)
    results = [smoke_result]

    if args.mode == "smoke" and args.auto_full_after_smoke and smoke_result["gate"]["passed"]:
        results.append(run_mode("full", cfg, dry_run=args.dry_run, parent_run_dir=parent_run_dir))

    write_json(parent_run_dir / "run_results.json", {"results": results})
    print(json.dumps({"run_id": run_id, "results": results}, indent=2))


if __name__ == "__main__":
    main()
