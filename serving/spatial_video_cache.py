"""Spatial token reuse utilities.

The classes here target robotics/video inference streams where adjacent frames
are mostly stable.  They operate at the patch level:

- ``PatchChangeDetector`` marks image patches that changed enough to refresh.
- ``SpatialKVCache`` keeps per-patch keys/values and reuses unchanged patches.

These utilities are intentionally model-agnostic. A caller supplies fresh
per-patch K/V tensors from its vision/attention stack, and the cache merges them
with cached tensors for patches that did not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchGrid:
    """Square patch layout for BCHW video frames."""

    image_size: int
    patch_size: int

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

    @property
    def side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.side * self.side

    @property
    def patch_area(self) -> int:
        return self.patch_size * self.patch_size


@dataclass
class PatchChangeStats:
    changed_patches: int
    total_patches: int
    changed_ratio: float
    mean_patch_delta: float
    max_patch_delta: float


@dataclass
class SpatialCacheStats:
    frames: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    refreshed_patches: int = 0
    reused_patches: int = 0

    @property
    def total_patches(self) -> int:
        return self.refreshed_patches + self.reused_patches

    @property
    def reuse_ratio(self) -> float:
        if self.total_patches == 0:
            return 0.0
        return self.reused_patches / self.total_patches

    @property
    def hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


def patchify(frames: torch.Tensor, grid: PatchGrid) -> torch.Tensor:
    """Convert BCHW frames to B,N,C,patch,patch patches."""
    if frames.ndim != 4:
        raise ValueError(f"frames must be BCHW, got shape={tuple(frames.shape)}")
    _, _, height, width = frames.shape
    if height != grid.image_size or width != grid.image_size:
        raise ValueError(
            f"expected {grid.image_size}x{grid.image_size} frames, got {height}x{width}"
        )

    patches = F.unfold(frames, kernel_size=grid.patch_size, stride=grid.patch_size)
    batch, flat_dim, num_patches = patches.shape
    channels = flat_dim // grid.patch_area
    return patches.transpose(1, 2).reshape(
        batch,
        num_patches,
        channels,
        grid.patch_size,
        grid.patch_size,
    )


class PatchChangeDetector:
    """Detect changed patches using mean absolute pixel delta."""

    def __init__(self, grid: PatchGrid, threshold: float = 0.03) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.grid = grid
        self.threshold = threshold
        self._prev_patches: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_patches = None

    def update(self, frame: torch.Tensor) -> tuple[torch.Tensor, PatchChangeStats]:
        patches = patchify(frame, self.grid)
        if patches.shape[0] != 1:
            raise ValueError("PatchChangeDetector expects one frame at a time")

        if self._prev_patches is None:
            changed = torch.ones(self.grid.num_patches, dtype=torch.bool, device=frame.device)
            deltas = torch.ones(self.grid.num_patches, dtype=frame.dtype, device=frame.device)
        else:
            deltas = (patches - self._prev_patches).abs().mean(dim=(0, 2, 3, 4))
            changed = deltas > self.threshold

        self._prev_patches = patches.detach().clone()
        changed_count = int(changed.sum().item())
        stats = PatchChangeStats(
            changed_patches=changed_count,
            total_patches=self.grid.num_patches,
            changed_ratio=changed_count / self.grid.num_patches,
            mean_patch_delta=float(deltas.mean().item()),
            max_patch_delta=float(deltas.max().item()),
        )
        return changed, stats


class SpatialKVCache:
    """Patch-indexed K/V cache for slowly changing visual streams."""

    def __init__(self, num_patches: int) -> None:
        if num_patches <= 0:
            raise ValueError("num_patches must be positive")
        self.num_patches = num_patches
        self.stats = SpatialCacheStats()
        self._key: torch.Tensor | None = None
        self._value: torch.Tensor | None = None

    def reset(self) -> None:
        self.stats = SpatialCacheStats()
        self._key = None
        self._value = None

    def update(
        self,
        fresh_key: torch.Tensor,
        fresh_value: torch.Tensor,
        changed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge fresh per-patch K/V tensors with cached unchanged patches.

        Args:
            fresh_key: Tensor shaped ``[num_patches, dim]``.
            fresh_value: Tensor shaped ``[num_patches, dim]``.
            changed_mask: Boolean tensor shaped ``[num_patches]``.
        """
        self._validate_inputs(fresh_key, fresh_value, changed_mask)
        changed_mask = changed_mask.to(device=fresh_key.device, dtype=torch.bool)

        if self._key is None or self._value is None:
            merged_key = fresh_key.detach().clone()
            merged_value = fresh_value.detach().clone()
            refreshed = self.num_patches
            reused = 0
            self.stats.cache_misses += 1
        else:
            merged_key = self._key.to(fresh_key.device).clone()
            merged_value = self._value.to(fresh_value.device).clone()
            merged_key[changed_mask] = fresh_key[changed_mask]
            merged_value[changed_mask] = fresh_value[changed_mask]
            refreshed = int(changed_mask.sum().item())
            reused = self.num_patches - refreshed
            self.stats.cache_hits += 1

        self._key = merged_key.detach().clone()
        self._value = merged_value.detach().clone()
        self.stats.frames += 1
        self.stats.refreshed_patches += refreshed
        self.stats.reused_patches += reused
        return merged_key, merged_value

    def _validate_inputs(
        self,
        fresh_key: torch.Tensor,
        fresh_value: torch.Tensor,
        changed_mask: torch.Tensor,
    ) -> None:
        expected = (self.num_patches,)
        if fresh_key.ndim != 2 or fresh_key.shape[0] != self.num_patches:
            raise ValueError(f"fresh_key must be [num_patches, dim], got {tuple(fresh_key.shape)}")
        if fresh_value.shape != fresh_key.shape:
            raise ValueError("fresh_value must have the same shape as fresh_key")
        if tuple(changed_mask.shape) != expected:
            raise ValueError(f"changed_mask must be {expected}, got {tuple(changed_mask.shape)}")
