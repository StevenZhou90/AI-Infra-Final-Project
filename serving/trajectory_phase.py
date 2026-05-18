"""Phase labeling and routing helpers for trajectory draft heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

PhaseLabel = Literal["smooth", "complex"]


@dataclass(frozen=True)
class PhaseMetrics:
    curvature: float
    displacement: float
    acceleration: float
    gripper_changed: bool
    stationary: bool


@dataclass(frozen=True)
class PhaseThresholds:
    smooth_curvature: float = 6.0
    smooth_acceleration: float = 8.0
    min_displacement: float = 1.5
    stationary_displacement: float = 1.0


def _as_np(history_bins: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(history_bins, torch.Tensor):
        history_bins = history_bins.detach().cpu().numpy()
    arr = np.asarray(history_bins, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [steps, action_dim] history, got shape {arr.shape}")
    if arr.shape[0] < 2:
        return arr.copy()
    return arr


def phase_metrics(history_bins: np.ndarray | torch.Tensor) -> PhaseMetrics:
    """Compute simple kinematic statistics over recent action bins.

    Metrics use bin-space deltas so they work for both OpenVLA token-derived bins
    and saved training examples without requiring environment action scaling.
    """
    hist = _as_np(history_bins)
    if hist.shape[0] < 2:
        return PhaseMetrics(
            curvature=0.0,
            displacement=0.0,
            acceleration=0.0,
            gripper_changed=False,
            stationary=True,
        )

    motion = hist[:, :6]
    deltas = np.diff(motion, axis=0)
    displacement = float(np.mean(np.abs(deltas[-1])))
    if hist.shape[0] >= 3:
        second = motion[-1] - 2.0 * motion[-2] + motion[-3]
        curvature = float(np.mean(np.abs(second)))
        acceleration = float(np.mean(np.abs(np.diff(deltas, axis=0)[-1])))
    else:
        curvature = 0.0
        acceleration = 0.0

    gripper_changed = bool(hist[-1, 6] != hist[-2, 6]) if hist.shape[1] >= 7 else False
    return PhaseMetrics(
        curvature=curvature,
        displacement=displacement,
        acceleration=acceleration,
        gripper_changed=gripper_changed,
        stationary=displacement < PhaseThresholds().stationary_displacement,
    )


def label_phase(
    history_bins: np.ndarray | torch.Tensor,
    *,
    thresholds: PhaseThresholds | None = None,
) -> PhaseLabel:
    """Label a training/router example as smooth or complex.

    Smooth means a moving, low-curvature segment with no gripper phase change.
    Complex captures contact, gripper changes, curved approaches, and fine
    stationary control where a conservative head should own the draft.
    """
    th = thresholds or PhaseThresholds()
    metrics = phase_metrics(history_bins)
    stationary = metrics.displacement < th.stationary_displacement
    if metrics.gripper_changed or stationary:
        return "complex"
    if metrics.displacement < th.min_displacement:
        return "complex"
    if metrics.curvature > th.smooth_curvature:
        return "complex"
    if metrics.acceleration > th.smooth_acceleration:
        return "complex"
    return "smooth"
