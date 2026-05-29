from __future__ import annotations

from serving.pi0fast_serving_runtime import (
    NS_PER_MS,
    PI0FastDeadlineBatchScheduler,
    PI0FastRequest,
    PI0FastServingConfig,
    PI0FastServingRuntime,
    SyntheticPI0FastBackend,
    deadline_ns_from_period,
)


def make_request(
    idx: int,
    *,
    at_ns: int = 0,
    session: str | None = None,
    mode: str = "prefix_gate",
    period_ms: float = 200.0,
    model: str = "lerobot/pi0fast-libero",
    deadline_ns: int | None = None,
) -> PI0FastRequest:
    return PI0FastRequest(
        request_id=f"req-{idx}",
        session_id=session or f"session-{idx}",
        robot_id=f"robot-{idx}",
        model_id=model,
        enqueued_ns=at_ns,
        deadline_ns=deadline_ns if deadline_ns is not None else deadline_ns_from_period(at_ns, period_ms),
        control_period_ms=period_ms,
        decode_mode=mode,
        prompt="pick up the object",
    )


def test_scheduler_batches_compatible_requests_after_delay() -> None:
    cfg = PI0FastServingConfig(
        max_batch_size=4,
        max_batch_delay_ms=5.0,
        estimated_prefill_ms=10.0,
        estimated_decode_ms_per_token=0.1,
    )
    scheduler = PI0FastDeadlineBatchScheduler(cfg)
    scheduler.submit(make_request(0, at_ns=0))
    scheduler.submit(make_request(1, at_ns=1 * NS_PER_MS))

    assert scheduler.pop_ready_batch(at_ns=3 * NS_PER_MS) is None
    batch = scheduler.pop_ready_batch(at_ns=6 * NS_PER_MS)

    assert batch is not None
    assert batch.reason == "batch_delay"
    assert [req.request_id for req in batch.requests] == ["req-0", "req-1"]


def test_scheduler_does_not_mix_decode_modes() -> None:
    cfg = PI0FastServingConfig(max_batch_size=8, max_batch_delay_ms=1.0)
    scheduler = PI0FastDeadlineBatchScheduler(cfg)
    scheduler.submit(make_request(0, at_ns=0, mode="prefix_gate"))
    scheduler.submit(make_request(1, at_ns=0, mode="full_eos"))

    first = scheduler.pop_ready_batch(at_ns=2 * NS_PER_MS)
    second = scheduler.pop_ready_batch(at_ns=2 * NS_PER_MS)

    assert first is not None
    assert second is not None
    assert first.batch_key != second.batch_key
    assert first.size == 1
    assert second.size == 1


def test_scheduler_flushes_when_deadline_is_close() -> None:
    cfg = PI0FastServingConfig(
        max_batch_delay_ms=950.0,
        estimated_prefill_ms=30.0,
        estimated_decode_ms_per_token=1.0,
        deadline_slack_ms=5.0,
    )
    scheduler = PI0FastDeadlineBatchScheduler(cfg)
    req = make_request(0, at_ns=0, period_ms=10_000.0, deadline_ns=1_000 * NS_PER_MS)
    scheduler.submit(req)

    batch = scheduler.pop_ready_batch(at_ns=905 * NS_PER_MS)

    assert batch is not None
    assert batch.reason == "deadline"


def test_runtime_tracks_prompt_and_session_cache_hits() -> None:
    runtime = PI0FastServingRuntime(
        SyntheticPI0FastBackend(prefill_ms=10.0, decode_ms_per_token=1.0),
        PI0FastServingConfig(max_batch_delay_ms=1.0),
    )
    runtime.submit(make_request(0, at_ns=0, session="session-a"))
    first = runtime.drain_ready(at_ns=2 * NS_PER_MS)
    runtime.submit(make_request(1, at_ns=10 * NS_PER_MS, session="session-a"))
    second = runtime.drain_ready(at_ns=12 * NS_PER_MS)

    assert first[0].telemetry.prompt_cache_hit is False
    assert first[0].telemetry.cache_hit is False
    assert second[0].telemetry.prompt_cache_hit is True
    assert second[0].telemetry.cache_hit is True
    assert runtime.stats()["session_cache_hit_rate"] == 0.5


def test_synthetic_runtime_reports_batch_telemetry() -> None:
    backend = SyntheticPI0FastBackend(prefill_ms=20.0, decode_ms_per_token=2.0, batch_efficiency=0.5)
    runtime = PI0FastServingRuntime(backend, PI0FastServingConfig(max_batch_size=4, max_batch_delay_ms=1.0))
    for idx in range(4):
        runtime.submit(make_request(idx, at_ns=0))

    responses = runtime.drain_ready(at_ns=0)

    assert len(responses) == 4
    assert {resp.telemetry.batch_size for resp in responses} == {4}
    assert {resp.telemetry.batch_reason for resp in responses} == {"max_batch"}
    assert all(resp.telemetry.runtime_ms == backend.last_runtime_ms for resp in responses)
    assert runtime.stats()["avg_batch_size"] == 4.0
