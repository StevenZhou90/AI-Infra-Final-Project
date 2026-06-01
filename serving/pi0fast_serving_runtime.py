"""PI0-FAST-specific serving runtime primitives.

This module is intentionally independent from the existing gRPC server.  It
captures the systems policy we want to validate first: deadline-aware continuous
batching for many PI0-FAST robot sessions, session/prompt cache accounting, and
per-request telemetry that can later be surfaced through the public API.
"""

from __future__ import annotations

import heapq
import contextlib
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import torch

from serving.kv_cache_manager import KVCacheManager

NS_PER_MS = 1_000_000


def now_ns() -> int:
    return time.time_ns()


def deadline_ns_from_period(enqueued_ns: int, control_period_ms: float) -> int:
    return enqueued_ns + int(control_period_ms * NS_PER_MS)


@dataclass(frozen=True)
class PI0FastServingConfig:
    """Runtime knobs for PI0-FAST multi-robot serving."""

    max_batch_size: int = 8
    max_batch_delay_ms: float = 5.0
    deadline_slack_ms: float = 8.0
    estimated_prefill_ms: float = 80.0
    estimated_decode_ms_per_token: float = 2.0
    default_control_period_ms: float = 200.0
    default_decode_mode: str = "prefix_gate"
    default_max_action_tokens: int = 64
    cache_memory_mb: float = 4096.0
    max_active_sessions: int | None = None
    max_admission_utilization: float | None = None
    default_request_period_ms: float = 1000.0
    estimated_batch_base_ms: float | None = None
    estimated_batch_per_request_ms: float = 0.0

    def max_wait_ns(self, control_period_ms: float) -> int:
        """Wait no longer than a small fraction of the control period."""
        wait_ms = min(self.max_batch_delay_ms, max(control_period_ms, 1.0) * 0.25)
        return int(wait_ms * NS_PER_MS)

    def estimated_runtime_ns(self, token_budget: int) -> int:
        ms = self.estimated_prefill_ms + self.estimated_decode_ms_per_token * max(token_budget, 1)
        return int(ms * NS_PER_MS)

    def estimated_batch_runtime_ns(self, token_budget: int, batch_size: int) -> int:
        if self.estimated_batch_base_ms is not None:
            ms = self.estimated_batch_base_ms + self.estimated_batch_per_request_ms * max(batch_size - 1, 0)
            return int(ms * NS_PER_MS)
        return self.estimated_runtime_ns(token_budget)


@dataclass(frozen=True)
class PI0FastRequest:
    """A single robot-session policy request."""

    request_id: str
    session_id: str
    robot_id: str
    model_id: str = "lerobot/pi0fast-libero"
    priority: int = 1
    enqueued_ns: int = field(default_factory=now_ns)
    deadline_ns: int | None = None
    control_period_ms: float = 200.0
    max_action_tokens: int = 64
    decode_mode: str = "prefix_gate"
    prompt: str = ""
    camera_schema: str = "default"
    safety_mode: str = "conservative"
    observation: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effective_deadline_ns(self) -> int:
        if self.deadline_ns is not None:
            return self.deadline_ns
        return deadline_ns_from_period(self.enqueued_ns, self.control_period_ms)

    @property
    def batch_key(self) -> tuple[str, str, str, str]:
        return (self.model_id, self.decode_mode, self.camera_schema, self.safety_mode)

    @property
    def prompt_hash(self) -> str:
        return KVCacheManager.hash_prefix(self.prompt)


@dataclass
class PI0FastSessionState:
    session_id: str
    robot_id: str
    model_id: str
    prompt_hash: str = ""
    last_access_ns: int = field(default_factory=now_ns)
    requests_served: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    buffered_actions: int = 0
    request_period_ms: float = 1000.0
    token_budget: int = 64


@dataclass
class PI0FastBatch:
    requests: list[PI0FastRequest]
    formed_ns: int
    reason: str

    @property
    def batch_key(self) -> tuple[str, str, str, str]:
        return self.requests[0].batch_key

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def earliest_deadline_ns(self) -> int:
        return min(req.effective_deadline_ns for req in self.requests)

    @property
    def max_action_tokens(self) -> int:
        return max(req.max_action_tokens for req in self.requests)


@dataclass
class PI0FastTelemetry:
    request_id: str
    session_id: str
    robot_id: str
    model_id: str
    decode_mode: str
    queue_ms: float
    runtime_ms: float
    batch_size: int
    batch_reason: str
    cache_hit: bool
    prompt_cache_hit: bool
    deadline_missed: bool
    deadline_slack_ms: float
    action_tokens: int
    actions_returned: int
    accelerator: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PI0FastResponse:
    request_id: str
    actions: np.ndarray
    telemetry: PI0FastTelemetry


@dataclass
class PI0FastBackendResult:
    actions: np.ndarray
    action_tokens: int
    accelerator: str
    extra: dict[str, Any] = field(default_factory=dict)


class PI0FastBatchBackend(Protocol):
    """Backend contract for real PI0-FAST policies and synthetic benchmarks."""

    def predict_batch(
        self,
        batch: PI0FastBatch,
        sessions: Mapping[str, PI0FastSessionState],
    ) -> Sequence[PI0FastBackendResult]:
        ...


def merge_prepared_pi0fast_batches(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge already-preprocessed PI0-FAST observations into one batch.

    LeRobot preprocessors usually produce tensors with a leading batch
    dimension plus a few list/string fields such as task prompts.  This helper
    concatenates tensors on dim 0 and flattens list-like metadata.
    """
    if not batches:
        raise ValueError("Cannot merge an empty PI0-FAST batch list")

    keys = set(batches[0].keys())
    for batch in batches[1:]:
        if set(batch.keys()) != keys:
            missing = sorted(keys.symmetric_difference(batch.keys()))
            raise ValueError(f"Prepared PI0-FAST batches have different keys: {missing}")

    merged: dict[str, Any] = {}
    for key in sorted(keys):
        values = [batch[key] for batch in batches]
        first = values[0]
        if torch.is_tensor(first):
            merged[key] = torch.cat([value if value.ndim > 0 else value.view(1) for value in values], dim=0)
        elif isinstance(first, np.ndarray):
            merged[key] = np.concatenate([value if value.ndim > 0 else value.reshape(1) for value in values], axis=0)
        elif isinstance(first, list):
            out: list[Any] = []
            for value in values:
                out.extend(value)
            merged[key] = out
        elif isinstance(first, tuple):
            out = []
            for value in values:
                out.extend(value)
            merged[key] = tuple(out)
        else:
            merged[key] = values
    return merged


def actions_to_numpy(actions: Any) -> np.ndarray:
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1)
    if arr.ndim == 2:
        return arr.reshape(1, arr.shape[0], arr.shape[1])
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported PI0-FAST action shape: {arr.shape}")


class RealPIBatchBackend:
    """Backend that executes one real LeRobot policy call per request batch."""

    def __init__(
        self,
        policy: Any,
        postprocessor: Callable[[Any], Any] | None = None,
        *,
        accelerator: str = "real_pi_batch",
        autocast_dtype: torch.dtype | None = None,
        inference_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.postprocessor = postprocessor
        self.accelerator = accelerator
        self.autocast_dtype = autocast_dtype
        self.inference_kwargs = dict(inference_kwargs or {})
        self.calls = 0
        self.last_runtime_ms = 0.0

    def predict_batch(
        self,
        batch: PI0FastBatch,
        sessions: Mapping[str, PI0FastSessionState],
    ) -> Sequence[PI0FastBackendResult]:
        prepared = []
        for request in batch.requests:
            if request.observation is None:
                raise ValueError(f"Request {request.request_id} has no prepared PI0-FAST observation")
            prepared.append(request.observation)
        merged = merge_prepared_pi0fast_batches(prepared)

        device = self._policy_device()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            if self.autocast_dtype is not None and device is not None and device.type == "cuda":
                cast_context = torch.autocast(device_type=device.type, dtype=self.autocast_dtype)
            else:
                cast_context = contextlib.nullcontext()
            with cast_context:
                raw_actions = (
                    self.policy.predict_action_chunk(merged, **self.inference_kwargs)
                    if hasattr(self.policy, "predict_action_chunk")
                    else self.policy.select_action(merged)
                )
            processed = self.postprocessor(raw_actions) if self.postprocessor is not None else raw_actions
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        self.last_runtime_ms = (time.perf_counter() - t0) * 1000.0
        self.calls += 1

        action_batch = actions_to_numpy(processed)
        if action_batch.shape[0] != batch.size:
            raise RuntimeError(f"Expected {batch.size} action chunks, got shape {action_batch.shape}")

        token_budget = max(request.max_action_tokens for request in batch.requests)
        return [
            PI0FastBackendResult(
                actions=action_batch[idx],
                action_tokens=token_budget,
                accelerator=self.accelerator,
                extra={
                    "batch_runtime_ms": self.last_runtime_ms,
                    "inference_kwargs": dict(self.inference_kwargs),
                    "autocast_dtype": str(self.autocast_dtype).replace("torch.", "")
                    if self.autocast_dtype is not None
                    else None,
                },
            )
            for idx in range(batch.size)
        ]

    def _policy_device(self) -> torch.device | None:
        try:
            return next(self.policy.parameters()).device
        except Exception:
            try:
                return next(self.policy.model.parameters()).device
            except Exception:
                return None


class RealPI0FastBatchBackend(RealPIBatchBackend):
    """Backward-compatible name for the generic real PI batch backend."""

    def __init__(
        self,
        policy: Any,
        postprocessor: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(policy, postprocessor, accelerator="real_pi0fast_batch")


class PI0FastDeadlineBatchScheduler:
    """Deadline-aware batch former for PI0-FAST requests.

    The scheduler only batches requests that can share the same PI0-FAST decode
    path.  It flushes a group when the oldest request has waited long enough,
    when a deadline is close, or when the group reaches ``max_batch_size``.
    """

    def __init__(self, config: PI0FastServingConfig | None = None) -> None:
        self.config = config or PI0FastServingConfig()
        self._queues: dict[tuple[str, str, str, str], list[PI0FastRequest]] = defaultdict(list)
        self.total_submitted = 0
        self.total_batches = 0

    def submit(self, request: PI0FastRequest) -> None:
        self._queues[request.batch_key].append(request)
        self.total_submitted += 1

    def queue_depth(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def pop_ready_batch(self, at_ns: int | None = None, force: bool = False) -> PI0FastBatch | None:
        at_ns = now_ns() if at_ns is None else at_ns
        candidates: list[tuple[int, int, str, tuple[str, str, str, str]]] = []

        for key, queue in self._queues.items():
            if not queue:
                continue
            queue.sort(key=lambda req: (req.effective_deadline_ns, -req.priority, req.enqueued_ns))
            reason = self._ready_reason(queue, at_ns, force)
            if reason is None:
                continue
            earliest_deadline = queue[0].effective_deadline_ns
            highest_priority = max(req.priority for req in queue)
            heapq.heappush(candidates, (earliest_deadline, -highest_priority, reason, key))

        if not candidates:
            return None

        _deadline, _neg_priority, reason, key = heapq.heappop(candidates)
        queue = self._queues[key]
        selected = self._select_batch_requests(queue, at_ns, reason)
        del queue[: len(selected)]
        if not queue:
            self._queues.pop(key, None)
        self.total_batches += 1
        return PI0FastBatch(requests=selected, formed_ns=at_ns, reason=reason)

    def _select_batch_requests(self, queue: list[PI0FastRequest], at_ns: int, reason: str) -> list[PI0FastRequest]:
        limit = min(len(queue), self.config.max_batch_size)
        if limit <= 1 or reason in {"deadline", "force_flush"}:
            return queue[:1] if not reason == "force_flush" else queue[:limit]

        selected: list[PI0FastRequest] = []
        slack_ns = int(self.config.deadline_slack_ms * NS_PER_MS)
        for candidate in queue[:limit]:
            trial = selected + [candidate]
            token_budget = max(req.max_action_tokens for req in trial)
            runtime_ns = self.config.estimated_batch_runtime_ns(token_budget, len(trial))
            finish_ns = at_ns + runtime_ns + slack_ns
            if selected and any(finish_ns >= req.effective_deadline_ns for req in trial):
                break
            selected = trial
        return selected or queue[:1]

    def _ready_reason(self, queue: list[PI0FastRequest], at_ns: int, force: bool) -> str | None:
        if force:
            return "force_flush"
        if len(queue) >= self.config.max_batch_size:
            return "max_batch"

        oldest = min(queue, key=lambda req: req.enqueued_ns)
        waited_ns = at_ns - oldest.enqueued_ns
        if waited_ns >= self.config.max_wait_ns(oldest.control_period_ms):
            return "batch_delay"

        earliest = min(queue, key=lambda req: req.effective_deadline_ns)
        runtime_ns = self.config.estimated_runtime_ns(earliest.max_action_tokens)
        slack_ns = int(self.config.deadline_slack_ms * NS_PER_MS)
        if at_ns + runtime_ns + slack_ns >= earliest.effective_deadline_ns:
            return "deadline"
        return None


class PI0FastServingRuntime:
    """Coordinates session cache accounting, batching, and PI0-FAST backend calls."""

    def __init__(
        self,
        backend: PI0FastBatchBackend,
        config: PI0FastServingConfig | None = None,
    ) -> None:
        self.config = config or PI0FastServingConfig()
        self.backend = backend
        self.scheduler = PI0FastDeadlineBatchScheduler(self.config)
        self.sessions: dict[str, PI0FastSessionState] = {}
        self.prompt_cache: set[str] = set()
        self.responses: list[PI0FastResponse] = []
        self.rejected_requests = 0
        self.rejection_reasons: dict[str, int] = defaultdict(int)
        self.last_rejection_reason = ""

    def submit(self, request: PI0FastRequest) -> None:
        if not self.try_submit(request):
            raise RuntimeError(f"Admission rejected request {request.request_id} for session {request.session_id}")

    def try_submit(self, request: PI0FastRequest) -> bool:
        reason = self._admission_rejection_reason(request)
        if reason is not None:
            self.rejected_requests += 1
            self.rejection_reasons[reason] += 1
            self.last_rejection_reason = reason
            return False
        self.last_rejection_reason = ""
        self._ensure_session(request)
        self.scheduler.submit(request)
        return True

    def drain_ready(self, at_ns: int | None = None, force: bool = False) -> list[PI0FastResponse]:
        responses: list[PI0FastResponse] = []
        while True:
            batch = self.scheduler.pop_ready_batch(at_ns=at_ns, force=force)
            if batch is None:
                break
            responses.extend(self.execute_batch(batch, at_ns=at_ns))
            if not force:
                break
        return responses

    def execute_batch(self, batch: PI0FastBatch, at_ns: int | None = None) -> list[PI0FastResponse]:
        start_ns = now_ns() if at_ns is None else at_ns
        prompt_hits = {req.request_id: req.prompt_hash in self.prompt_cache for req in batch.requests}
        for req in batch.requests:
            if req.prompt_hash:
                self.prompt_cache.add(req.prompt_hash)

        t0 = time.perf_counter()
        results = list(self.backend.predict_batch(batch, self.sessions))
        measured_runtime_ms = (time.perf_counter() - t0) * 1000.0
        runtime_ms = float(getattr(self.backend, "last_runtime_ms", measured_runtime_ms))
        finish_ns = start_ns + int(runtime_ms * NS_PER_MS)

        if len(results) != len(batch.requests):
            raise RuntimeError(f"Backend returned {len(results)} results for {len(batch.requests)} requests")

        responses: list[PI0FastResponse] = []
        for req, result in zip(batch.requests, results, strict=True):
            session = self._ensure_session(req)
            cache_hit = session.prompt_hash == req.prompt_hash and session.requests_served > 0
            if cache_hit:
                session.cache_hits += 1
            else:
                session.cache_misses += 1
            session.prompt_hash = req.prompt_hash
            session.last_access_ns = finish_ns
            session.requests_served += 1
            session.buffered_actions = int(result.actions.shape[0]) if result.actions.ndim > 1 else 1

            deadline_slack_ms = (req.effective_deadline_ns - finish_ns) / NS_PER_MS
            telemetry = PI0FastTelemetry(
                request_id=req.request_id,
                session_id=req.session_id,
                robot_id=req.robot_id,
                model_id=req.model_id,
                decode_mode=req.decode_mode,
                queue_ms=max(0.0, (batch.formed_ns - req.enqueued_ns) / NS_PER_MS),
                runtime_ms=runtime_ms,
                batch_size=batch.size,
                batch_reason=batch.reason,
                cache_hit=cache_hit,
                prompt_cache_hit=prompt_hits[req.request_id],
                deadline_missed=deadline_slack_ms < 0,
                deadline_slack_ms=deadline_slack_ms,
                action_tokens=result.action_tokens,
                actions_returned=int(result.actions.shape[0]) if result.actions.ndim > 1 else 1,
                accelerator=result.accelerator,
                extra=dict(result.extra),
            )
            responses.append(PI0FastResponse(req.request_id, result.actions, telemetry))

        self.responses.extend(responses)
        return responses

    def stats(self) -> dict[str, Any]:
        total = len(self.responses)
        misses = sum(1 for resp in self.responses if resp.telemetry.deadline_missed)
        latencies = [resp.telemetry.queue_ms + resp.telemetry.runtime_ms for resp in self.responses]
        batch_sizes = [resp.telemetry.batch_size for resp in self.responses]
        cache_hits = sum(1 for resp in self.responses if resp.telemetry.cache_hit)
        prompt_hits = sum(1 for resp in self.responses if resp.telemetry.prompt_cache_hit)
        return {
            "requests": total,
            "deadline_misses": misses,
            "deadline_miss_rate": misses / max(total, 1),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "avg_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
            "batches": self.scheduler.total_batches,
            "session_cache_hit_rate": cache_hits / max(total, 1),
            "prompt_cache_hit_rate": prompt_hits / max(total, 1),
            "sessions": len(self.sessions),
            "rejected_requests": self.rejected_requests,
            "rejection_reasons": dict(self.rejection_reasons),
            "admission_utilization": self.admission_utilization(),
        }

    def admission_utilization(self) -> float:
        return sum(self._session_utilization(session) for session in self.sessions.values())

    def clear_session_state(self) -> None:
        self.sessions.clear()
        self.prompt_cache.clear()

    def _admission_rejection_reason(self, request: PI0FastRequest) -> str | None:
        if request.session_id in self.sessions:
            return None
        if self.config.max_active_sessions is not None and len(self.sessions) >= self.config.max_active_sessions:
            return "max_active_sessions"
        if self.config.max_admission_utilization is None:
            return None
        projected = self.admission_utilization() + self._request_utilization(request)
        if projected > self.config.max_admission_utilization:
            return "projected_utilization"
        return None

    def _session_utilization(self, session: PI0FastSessionState) -> float:
        runtime_ms = self.config.estimated_batch_runtime_ns(session.token_budget, 1) / NS_PER_MS
        return runtime_ms / max(session.request_period_ms, 1.0)

    def _request_utilization(self, request: PI0FastRequest) -> float:
        runtime_ms = self.config.estimated_batch_runtime_ns(request.max_action_tokens, 1) / NS_PER_MS
        return runtime_ms / max(self._request_period_ms(request), 1.0)

    def _request_period_ms(self, request: PI0FastRequest) -> float:
        raw = request.metadata.get("request_period_ms", self.config.default_request_period_ms)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return self.config.default_request_period_ms

    def _ensure_session(self, request: PI0FastRequest) -> PI0FastSessionState:
        session = self.sessions.get(request.session_id)
        if session is None:
            session = PI0FastSessionState(
                session_id=request.session_id,
                robot_id=request.robot_id,
                model_id=request.model_id,
                request_period_ms=self._request_period_ms(request),
                token_budget=request.max_action_tokens,
            )
            self.sessions[request.session_id] = session
        else:
            session.request_period_ms = self._request_period_ms(request)
            session.token_budget = request.max_action_tokens
        return session


class SyntheticPI0FastBackend:
    """Deterministic backend for scheduler and capacity experiments.

    The latency model is deliberately simple: one shared prefill cost per batch,
    plus token decode cost that improves sublinearly with batch size.  It lets us
    test scheduling policy without loading PI0-FAST or a simulator.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 10,
        baseline_tokens: int = 96,
        prefix_gate_tokens: int = 28,
        cutoff16_tokens: int = 16,
        prefill_ms: float = 70.0,
        decode_ms_per_token: float = 2.0,
        batch_efficiency: float = 0.72,
        sleep: bool = False,
    ) -> None:
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.baseline_tokens = baseline_tokens
        self.prefix_gate_tokens = prefix_gate_tokens
        self.cutoff16_tokens = cutoff16_tokens
        self.prefill_ms = prefill_ms
        self.decode_ms_per_token = decode_ms_per_token
        self.batch_efficiency = batch_efficiency
        self.sleep = sleep
        self.calls = 0
        self.last_runtime_ms = 0.0

    def predict_batch(
        self,
        batch: PI0FastBatch,
        sessions: Mapping[str, PI0FastSessionState],
    ) -> Sequence[PI0FastBackendResult]:
        self.calls += 1
        tokens = [self._tokens_for(req) for req in batch.requests]
        decode_tokens = max(tokens) if tokens else 0
        effective_batch = max(batch.size, 1) ** self.batch_efficiency
        self.last_runtime_ms = self.prefill_ms + self.decode_ms_per_token * decode_tokens * effective_batch
        if self.sleep:
            time.sleep(self.last_runtime_ms / 1000.0)

        results: list[PI0FastBackendResult] = []
        for req, token_count in zip(batch.requests, tokens, strict=True):
            seed = abs(hash((req.request_id, req.session_id))) % (2**32)
            rng = np.random.default_rng(seed)
            actions = rng.normal(0.0, 0.02, size=(self.action_horizon, self.action_dim)).astype(np.float32)
            actions[:, -1] = 0.0
            results.append(
                PI0FastBackendResult(
                    actions=actions,
                    action_tokens=token_count,
                    accelerator=req.decode_mode,
                    extra={"synthetic_runtime_ms": self.last_runtime_ms},
                )
            )
        return results

    def _tokens_for(self, request: PI0FastRequest) -> int:
        if request.decode_mode == "full_eos":
            return min(request.max_action_tokens, self.baseline_tokens)
        if request.decode_mode == "cutoff16":
            return min(request.max_action_tokens, self.cutoff16_tokens)
        if request.decode_mode == "prefix_gate":
            return min(request.max_action_tokens, self.prefix_gate_tokens)
        return min(request.max_action_tokens, self.baseline_tokens)


class SyntheticPI05Backend:
    """Synthetic backend calibrated from measured PI0.5 flow serving latencies."""

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 50,
        base_ms: float = 158.0,
        per_request_ms: float = 31.0,
        num_inference_steps: int = 4,
        sleep: bool = False,
    ) -> None:
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.base_ms = base_ms
        self.per_request_ms = per_request_ms
        self.num_inference_steps = num_inference_steps
        self.sleep = sleep
        self.calls = 0
        self.last_runtime_ms = 0.0

    def predict_batch(
        self,
        batch: PI0FastBatch,
        sessions: Mapping[str, PI0FastSessionState],
    ) -> Sequence[PI0FastBackendResult]:
        self.calls += 1
        self.last_runtime_ms = self.base_ms + self.per_request_ms * max(batch.size - 1, 0)
        if self.sleep:
            time.sleep(self.last_runtime_ms / 1000.0)

        results: list[PI0FastBackendResult] = []
        for req in batch.requests:
            seed = abs(hash((req.request_id, req.session_id))) % (2**32)
            rng = np.random.default_rng(seed)
            actions = rng.normal(0.0, 0.02, size=(self.action_horizon, self.action_dim)).astype(np.float32)
            actions[:, -1] = 0.0
            results.append(
                PI0FastBackendResult(
                    actions=actions,
                    action_tokens=self.num_inference_steps,
                    accelerator="synthetic_pi05",
                    extra={
                        "num_inference_steps": self.num_inference_steps,
                        "synthetic_runtime_ms": self.last_runtime_ms,
                    },
                )
            )
        return results
