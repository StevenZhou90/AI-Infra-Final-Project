from __future__ import annotations

import numpy as np
import torch

from serving.pi0fast_serving_runtime import (
    NS_PER_MS,
    PI0FastBackendResult,
    PI0FastDeadlineBatchScheduler,
    PI0FastRequest,
    PI0FastServingConfig,
    PI0FastServingRuntime,
    RealPIBatchBackend,
    SyntheticPI05Backend,
    SyntheticPI0FastBackend,
    deadline_ns_from_period,
    merge_prepared_pi0fast_batches,
)
from serving.pi05_runtime_service import PI05RuntimeService
from serving.pi05_grpc_codec import (
    decode_prepared_observation,
    decode_prepared_observation_fields,
    encode_prepared_observation,
    encode_prepared_observation_fields,
)
from serving.pi05_server import PI05InferenceServicer, PI05QueuedWork


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


def test_scheduler_trims_batch_when_slack_is_tight() -> None:
    cfg = PI0FastServingConfig(
        max_batch_size=4,
        max_batch_delay_ms=1.0,
        deadline_slack_ms=5.0,
        estimated_batch_base_ms=160.0,
        estimated_batch_per_request_ms=45.0,
    )
    scheduler = PI0FastDeadlineBatchScheduler(cfg)
    for idx in range(3):
        scheduler.submit(make_request(idx, at_ns=0, period_ms=250.0, model="lerobot/pi05", mode="flow"))

    batch = scheduler.pop_ready_batch(at_ns=2 * NS_PER_MS)

    assert batch is not None
    assert batch.size == 2


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


def test_runtime_rejects_new_sessions_over_limit() -> None:
    runtime = PI0FastServingRuntime(
        SyntheticPI0FastBackend(prefill_ms=10.0, decode_ms_per_token=1.0),
        PI0FastServingConfig(max_active_sessions=1),
    )

    assert runtime.try_submit(make_request(0, session="session-a")) is True
    assert runtime.try_submit(make_request(1, session="session-b")) is False
    assert runtime.try_submit(make_request(2, session="session-a")) is True
    assert runtime.stats()["rejected_requests"] == 1
    assert runtime.stats()["rejection_reasons"] == {"max_active_sessions": 1}


def test_runtime_rejects_projected_utilization_over_limit() -> None:
    runtime = PI0FastServingRuntime(
        SyntheticPI05Backend(base_ms=160.0, per_request_ms=30.0),
        PI0FastServingConfig(
            estimated_batch_base_ms=160.0,
            max_admission_utilization=0.7,
            default_request_period_ms=1000.0,
        ),
    )

    for idx in range(4):
        req = make_request(
            idx,
            session=f"session-{idx}",
            model="lerobot/pi05_libero_finetuned_v044",
            mode="flow",
        )
        req = PI0FastRequest(**{**req.__dict__, "metadata": {"request_period_ms": 1000.0}})
        assert runtime.try_submit(req) is True

    rejected = make_request(5, session="session-5", model="lerobot/pi05_libero_finetuned_v044", mode="flow")
    rejected = PI0FastRequest(**{**rejected.__dict__, "metadata": {"request_period_ms": 1000.0}})

    assert runtime.try_submit(rejected) is False
    stats = runtime.stats()
    assert stats["rejected_requests"] == 1
    assert stats["rejection_reasons"] == {"projected_utilization": 1}
    assert 0.63 < stats["admission_utilization"] < 0.65


def test_synthetic_runtime_reports_batch_telemetry() -> None:
    backend = SyntheticPI0FastBackend(prefill_ms=20.0, decode_ms_per_token=2.0, batch_efficiency=0.5)
    runtime = PI0FastServingRuntime(
        backend,
        PI0FastServingConfig(
            max_batch_size=4,
            max_batch_delay_ms=1.0,
            estimated_prefill_ms=20.0,
            estimated_decode_ms_per_token=2.0,
        ),
    )
    for idx in range(4):
        runtime.submit(make_request(idx, at_ns=0))

    responses = runtime.drain_ready(at_ns=0)

    assert len(responses) == 4
    assert {resp.telemetry.batch_size for resp in responses} == {4}
    assert {resp.telemetry.batch_reason for resp in responses} == {"max_batch"}
    assert all(resp.telemetry.runtime_ms == backend.last_runtime_ms for resp in responses)
    assert runtime.stats()["avg_batch_size"] == 4.0


def test_merge_prepared_pi0fast_batches_concatenates_batch_fields() -> None:
    first = {
        "image": torch.ones((1, 3, 4, 4)),
        "state": np.ones((1, 7), dtype=np.float32),
        "task": ["pick"],
        "scalar": 1,
    }
    second = {
        "image": torch.zeros((1, 3, 4, 4)),
        "state": np.zeros((1, 7), dtype=np.float32),
        "task": ["place"],
        "scalar": 2,
    }

    merged = merge_prepared_pi0fast_batches([first, second])

    assert merged["image"].shape == (2, 3, 4, 4)
    assert merged["state"].shape == (2, 7)
    assert merged["task"] == ["pick", "place"]
    assert merged["scalar"] == [1, 2]


def test_runtime_carries_backend_extra_telemetry() -> None:
    class ExtraBackend:
        last_runtime_ms = 12.0

        def predict_batch(self, batch, sessions):
            return [
                PI0FastBackendResult(
                    actions=np.zeros((2, 7), dtype=np.float32),
                    action_tokens=128,
                    accelerator="test_backend",
                    extra={"token_count_max": 128, "straggler": True},
                )
            ]

    runtime = PI0FastServingRuntime(ExtraBackend(), PI0FastServingConfig(max_batch_delay_ms=1.0))
    runtime.submit(make_request(0, at_ns=0))

    responses = runtime.drain_ready(at_ns=2 * NS_PER_MS)

    assert responses[0].telemetry.extra == {"token_count_max": 128, "straggler": True}


def test_real_pi_batch_backend_passes_inference_kwargs() -> None:
    class FakePolicy:
        def __init__(self) -> None:
            self.seen_kwargs = None

        def predict_action_chunk(self, batch, **kwargs):
            self.seen_kwargs = kwargs
            return torch.zeros((batch["state"].shape[0], 2, 7), dtype=torch.float32)

    policy = FakePolicy()
    backend = RealPIBatchBackend(
        policy,
        accelerator="real_pi05_batch",
        inference_kwargs={"num_steps": 6},
    )
    batch = type(
        "Batch",
        (),
        {
            "requests": [
                make_request(0, model="lerobot/pi05_libero_finetuned_v044"),
                make_request(1, model="lerobot/pi05_libero_finetuned_v044"),
            ],
            "size": 2,
        },
    )()
    batch.requests[0] = PI0FastRequest(
        **{**batch.requests[0].__dict__, "observation": {"state": torch.ones((1, 7))}}
    )
    batch.requests[1] = PI0FastRequest(
        **{**batch.requests[1].__dict__, "observation": {"state": torch.zeros((1, 7))}}
    )

    results = backend.predict_batch(batch, {})

    assert policy.seen_kwargs == {"num_steps": 6}
    assert len(results) == 2
    assert results[0].accelerator == "real_pi05_batch"
    assert results[0].extra["inference_kwargs"] == {"num_steps": 6}


def test_synthetic_pi05_backend_scales_with_batch_size() -> None:
    backend = SyntheticPI05Backend(base_ms=160.0, per_request_ms=30.0, num_inference_steps=4)
    runtime = PI0FastServingRuntime(
        backend,
        PI0FastServingConfig(
            max_batch_size=4,
            max_batch_delay_ms=1.0,
            estimated_batch_base_ms=160.0,
            estimated_batch_per_request_ms=30.0,
        ),
    )
    for idx in range(4):
        runtime.submit(
            make_request(idx, at_ns=0, period_ms=300.0, model="lerobot/pi05_libero_finetuned_v044", mode="flow")
        )

    responses = runtime.drain_ready(at_ns=0)

    assert len(responses) == 4
    assert {resp.telemetry.runtime_ms for resp in responses} == {250.0}
    assert {resp.telemetry.action_tokens for resp in responses} == {4}
    assert responses[0].telemetry.extra["num_inference_steps"] == 4


def test_pi05_runtime_service_reports_admission_rejection() -> None:
    service = PI05RuntimeService(
        SyntheticPI05Backend(base_ms=10.0, per_request_ms=1.0),
        config=PI0FastServingConfig(max_active_sessions=1, max_batch_delay_ms=1.0),
    )
    observation = {"state": torch.zeros((1, 7))}

    first = service.predict(
        request_id="req-0",
        robot_id="robot-0",
        session_id="session-0",
        observation=observation,
        enqueued_ns=0,
        force=True,
    )
    second = service.predict(
        request_id="req-1",
        robot_id="robot-1",
        session_id="session-1",
        observation=observation,
        enqueued_ns=0,
        force=True,
    )

    assert first.admitted is True
    assert len(first.responses) == 1
    assert second.admitted is False
    assert second.reason == "max_active_sessions"
    assert service.status()["rejected_requests"] == 1


def test_pi05_grpc_codec_round_trips_prepared_observation() -> None:
    observation = {
        "state": torch.ones((1, 7), dtype=torch.float32),
        "task": ["pick up the object"],
    }

    decoded = decode_prepared_observation(encode_prepared_observation(observation))

    assert torch.equal(decoded["state"], observation["state"])
    assert decoded["task"] == ["pick up the object"]


def test_pi05_grpc_codec_round_trips_tensor_fields() -> None:
    observation = {
        "state": torch.ones((1, 7), dtype=torch.float32),
        "task": ["pick up the object"],
    }

    decoded = decode_prepared_observation_fields(encode_prepared_observation_fields(observation))

    assert torch.equal(decoded["state"], observation["state"])
    assert decoded["task"] == ["pick up the object"]


def test_pi05_grpc_worker_batches_queued_work() -> None:
    service = PI05RuntimeService(
        SyntheticPI05Backend(base_ms=10.0, per_request_ms=2.0),
        config=PI0FastServingConfig(
            max_batch_size=2,
            max_batch_delay_ms=1.0,
            estimated_batch_base_ms=10.0,
            estimated_batch_per_request_ms=2.0,
        ),
    )
    servicer = PI05InferenceServicer(service, device=torch.device("cpu"), metrics_path=None)
    works = [
        PI05QueuedWork(
            request_id=f"req-{idx}",
            robot_id=f"robot-{idx}",
            session_id=f"session-{idx}",
            observation={"state": torch.zeros((1, 7))},
            enqueued_ns=0,
            deadline_ms=250.0,
            request_period_ms=1000.0,
        )
        for idx in range(2)
    ]

    servicer._process_worker_batch(works)

    assert all(work.event.is_set() for work in works)
    assert all(work.response is not None and work.response.admitted for work in works)
    assert {work.response.batch_size for work in works if work.response is not None} == {2}
    assert service.status()["requests"] == 2
