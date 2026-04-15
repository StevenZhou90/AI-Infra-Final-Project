from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from common.types import ActionChunk, ActionStep
from server.config import SpecDecodeConfig

logger = logging.getLogger(__name__)


@dataclass
class SpecDecodeResult:
    action_chunk: ActionChunk
    accepted_length: int
    acceptance_rate: float
    drafter_time_ms: float
    verifier_time_ms: float
    total_time_ms: float
    used_speculative: bool


@dataclass
class _SpecDecodeState:
    """Running stats for adaptive fallback."""
    total_proposed: int = 0
    total_accepted: int = 0
    recent_rates: list[float] = field(default_factory=list)
    max_recent: int = 20

    @property
    def acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 1.0
        return self.total_accepted / self.total_proposed

    def record(self, proposed: int, accepted: int) -> None:
        self.total_proposed += proposed
        self.total_accepted += accepted
        rate = accepted / proposed if proposed > 0 else 0.0
        self.recent_rates.append(rate)
        if len(self.recent_rates) > self.max_recent:
            self.recent_rates.pop(0)

    @property
    def recent_rate(self) -> float:
        if not self.recent_rates:
            return 1.0
        return sum(self.recent_rates) / len(self.recent_rates)


class SpeculativeDecoder:
    """Spec-VLA: small drafter proposes action chunks, large VLA verifier accepts/corrects.

    The drafter (small ACT ~80M) runs fast and generates a full chunk of k actions.
    The verifier (7B VLA) evaluates each proposed action and accepts those within
    a relative distance threshold. Rejected positions are replaced by the verifier's
    own predictions, and remaining drafter actions after the first rejection are discarded.
    """

    def __init__(self, config: SpecDecodeConfig) -> None:
        self._config = config
        self._state = _SpecDecodeState()

    @torch.inference_mode()
    def decode(
        self,
        drafter: nn.Module,
        verifier: nn.Module,
        images: torch.Tensor,
        joint_state: torch.Tensor,
        chunk_size: int | None = None,
    ) -> SpecDecodeResult:
        chunk_size = chunk_size or self._config.drafter_chunk_size

        if not self._config.enabled or not self._should_use_speculative():
            return self._standard_decode(verifier, images, joint_state, chunk_size)

        # --- Phase 1: Drafter proposes ---
        t0 = time.perf_counter()
        draft_actions = self._run_drafter(drafter, images, joint_state, chunk_size)
        drafter_ms = (time.perf_counter() - t0) * 1000

        # --- Phase 2: Verifier scores ---
        t1 = time.perf_counter()
        verified_actions, accepted_len = self._run_verifier(
            verifier, images, joint_state, draft_actions, chunk_size
        )
        verifier_ms = (time.perf_counter() - t1) * 1000

        total_ms = drafter_ms + verifier_ms
        rate = accepted_len / chunk_size if chunk_size > 0 else 0.0
        self._state.record(chunk_size, accepted_len)

        chunk = self._tensor_to_chunk(verified_actions, chunk_size)
        chunk.accepted_length = accepted_len
        chunk.confidence = rate

        return SpecDecodeResult(
            action_chunk=chunk,
            accepted_length=accepted_len,
            acceptance_rate=rate,
            drafter_time_ms=drafter_ms,
            verifier_time_ms=verifier_ms,
            total_time_ms=total_ms,
            used_speculative=True,
        )

    def _should_use_speculative(self) -> bool:
        if self._state.total_proposed < 10:
            return True
        return self._state.recent_rate >= self._config.min_acceptance_rate

    # ------------------------------------------------------------------
    # Drafter
    # ------------------------------------------------------------------

    def _run_drafter(
        self,
        drafter: nn.Module,
        images: torch.Tensor,
        joint_state: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """Run drafter to propose an action chunk. Returns (1, chunk_size, action_dim)."""
        try:
            out = drafter(images, joint_state)
        except TypeError:
            out = drafter(images)

        if out.dim() == 2:
            action_dim = out.shape[-1] // chunk_size
            out = out.reshape(1, chunk_size, action_dim)
        return out[:, :chunk_size, :]

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------

    def _run_verifier(
        self,
        verifier: nn.Module,
        images: torch.Tensor,
        joint_state: torch.Tensor,
        draft_actions: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, int]:
        """Verify proposed actions. Returns final actions and accepted count.

        Uses relaxed acceptance: actions within a relative L2 distance threshold
        of the verifier's own prediction are accepted. On first rejection,
        we use the verifier's action from that point onward.
        """
        try:
            verifier_actions = verifier(images, joint_state)
        except TypeError:
            verifier_actions = verifier(images)

        if verifier_actions.dim() == 2:
            action_dim = verifier_actions.shape[-1] // chunk_size
            verifier_actions = verifier_actions.reshape(1, chunk_size, action_dim)

        verifier_actions = verifier_actions[:, :chunk_size, :]
        draft_chunk = draft_actions[:, :chunk_size, :]

        accepted_length = self._relaxed_accept(draft_chunk, verifier_actions)

        final = torch.cat(
            [draft_chunk[:, :accepted_length, :], verifier_actions[:, accepted_length:, :]],
            dim=1,
        )
        return final, accepted_length

    def _relaxed_accept(
        self, draft: torch.Tensor, verified: torch.Tensor
    ) -> int:
        """Find the longest prefix of drafter actions accepted by the verifier.

        Acceptance criterion: relative L2 distance < threshold.
        """
        threshold = self._config.acceptance_threshold
        chunk_size = draft.shape[1]

        for t in range(chunk_size):
            d = draft[0, t]
            v = verified[0, t]
            norm_v = torch.norm(v).item()
            dist = torch.norm(d - v).item()
            relative = dist / (norm_v + 1e-8)
            if relative > threshold:
                return t

        return chunk_size

    # ------------------------------------------------------------------
    # Fallback: standard (no speculation)
    # ------------------------------------------------------------------

    def _standard_decode(
        self,
        verifier: nn.Module,
        images: torch.Tensor,
        joint_state: torch.Tensor,
        chunk_size: int,
    ) -> SpecDecodeResult:
        t0 = time.perf_counter()
        try:
            actions = verifier(images, joint_state)
        except TypeError:
            actions = verifier(images)

        if actions.dim() == 2:
            action_dim = actions.shape[-1] // chunk_size
            actions = actions.reshape(1, chunk_size, action_dim)

        elapsed = (time.perf_counter() - t0) * 1000
        chunk = self._tensor_to_chunk(actions, chunk_size)
        chunk.accepted_length = chunk_size
        chunk.confidence = 1.0

        return SpecDecodeResult(
            action_chunk=chunk,
            accepted_length=chunk_size,
            acceptance_rate=1.0,
            drafter_time_ms=0.0,
            verifier_time_ms=elapsed,
            total_time_ms=elapsed,
            used_speculative=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_chunk(actions: torch.Tensor, chunk_size: int) -> ActionChunk:
        actions_np = actions[0, :chunk_size].cpu().float().numpy()
        steps = []
        for row in actions_np:
            joint = row.tolist()
            gripper = joint[-2:] if len(joint) > 2 else []
            joints = joint[:-2] if len(joint) > 2 else joint
            steps.append(ActionStep(joint_targets=joints, gripper_targets=gripper))
        return ActionChunk(
            steps=steps,
            start_timestamp_ns=time.time_ns(),
            frequency_hz=50.0,
        )

    @property
    def stats(self) -> dict:
        return {
            "total_proposed": self._state.total_proposed,
            "total_accepted": self._state.total_accepted,
            "overall_rate": round(self._state.acceptance_rate, 3),
            "recent_rate": round(self._state.recent_rate, 3),
            "speculative_enabled": self._config.enabled,
            "using_speculative": self._should_use_speculative(),
        }
