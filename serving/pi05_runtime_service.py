"""Service wrapper for PI0.5 deadline-aware serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from serving.pi0fast_serving_runtime import (
    PI0FastRequest,
    PI0FastResponse,
    PI0FastServingConfig,
    PI0FastServingRuntime,
    deadline_ns_from_period,
    now_ns,
)


@dataclass(frozen=True)
class PI05ServiceResult:
    admitted: bool
    responses: list[PI0FastResponse]
    reason: str = ""


class PI05RuntimeService:
    """Small service boundary around the PI0.5 serving runtime.

    The service accepts already-preprocessed observations.  Transport layers
    such as gRPC or HTTP should handle serialization and preprocessing before
    calling this class.
    """

    def __init__(
        self,
        backend,
        *,
        model_id: str = "lerobot/pi05_libero_finetuned_v044",
        config: PI0FastServingConfig | None = None,
    ) -> None:
        self.model_id = model_id
        self.runtime = PI0FastServingRuntime(backend, config)

    def predict(
        self,
        *,
        request_id: str,
        robot_id: str,
        session_id: str,
        observation: Mapping[str, Any],
        enqueued_ns: int | None = None,
        deadline_ms: float = 250.0,
        max_batch_delay_ms: float | None = None,
        force: bool = False,
    ) -> PI05ServiceResult:
        enqueued_ns = now_ns() if enqueued_ns is None else enqueued_ns
        request = PI0FastRequest(
            request_id=request_id,
            robot_id=robot_id,
            session_id=session_id,
            model_id=self.model_id,
            enqueued_ns=enqueued_ns,
            deadline_ns=deadline_ns_from_period(enqueued_ns, deadline_ms),
            control_period_ms=deadline_ms,
            decode_mode="flow",
            observation=observation,
        )
        if not self.runtime.try_submit(request):
            return PI05ServiceResult(admitted=False, responses=[], reason="admission_rejected")

        delay_ms = self.runtime.config.max_batch_delay_ms if max_batch_delay_ms is None else max_batch_delay_ms
        drain_ns = enqueued_ns + int(delay_ms * 1_000_000)
        return PI05ServiceResult(admitted=True, responses=self.runtime.drain_ready(at_ns=drain_ns, force=force))

    def drain(self, *, at_ns: int | None = None, force: bool = False) -> list[PI0FastResponse]:
        return self.runtime.drain_ready(at_ns=at_ns, force=force)

    def status(self) -> dict[str, Any]:
        return self.runtime.stats()
