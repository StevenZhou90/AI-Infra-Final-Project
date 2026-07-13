from __future__ import annotations

import torch

from serving.pi0fast_serving_runtime import (
    PI0FastRequest,
    RealPI0FastActionEndBatchBackend,
    deadline_ns_from_period,
)


def make_request(
    idx: int,
    *,
    at_ns: int = 0,
    session: str | None = None,
    period_ms: float = 200.0,
) -> PI0FastRequest:
    return PI0FastRequest(
        request_id=f"req-{idx}",
        session_id=session or f"session-{idx}",
        robot_id=f"robot-{idx}",
        model_id="lerobot/pi0fast-libero-v044",
        enqueued_ns=at_ns,
        deadline_ns=deadline_ns_from_period(at_ns, period_ms),
        control_period_ms=period_ms,
        decode_mode="action_end",
        prompt="pick up the object",
    )


def test_real_pi0fast_action_end_backend_uses_token_adapter() -> None:
    class FakePolicy(torch.nn.Module):
        pass

    class FakeTrace:
        def __init__(self) -> None:
            self.actions = torch.zeros((2, 3, 7), dtype=torch.float32)
            self.token_count = 12
            self.stats = {
                "row_token_counts": [11, 12],
                "stopped_on_action_end": 1.0,
                "action_end_token_id": 235371,
            }

    class FakeAdapter:
        def __init__(self) -> None:
            self.seen_batch = None
            self.seen_kwargs = None

        def predict_action_chunk_action_end(self, batch, **kwargs):
            self.seen_batch = batch
            self.seen_kwargs = kwargs
            return FakeTrace()

    adapter = FakeAdapter()
    backend = RealPI0FastActionEndBatchBackend(
        FakePolicy(),
        token_adapter=adapter,
        inference_kwargs={"max_decoding_steps": 128},
    )
    batch = type(
        "Batch",
        (),
        {
            "requests": [
                make_request(0),
                make_request(1),
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

    assert adapter.seen_batch["state"].shape == (2, 7)
    assert adapter.seen_kwargs == {"max_decoding_steps": 128}
    assert [result.action_tokens for result in results] == [11, 12]
    assert results[0].actions.shape == (3, 7)
    assert results[0].accelerator == "real_pi0fast_action_end_batch"
    assert results[0].extra["decode_path"] == "action_end"
    assert results[0].extra["stopped_on_action_end"] == 1.0
