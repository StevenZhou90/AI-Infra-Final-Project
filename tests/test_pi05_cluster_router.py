from __future__ import annotations

import json
import time

from proto import inference_pb2
from serving.pi05_cluster_router import ClusterRouter, WorkerSpec, parse_worker_spec


class FakeWorkerClient:
    def __init__(self, worker_id: str, *, admitted: bool = True) -> None:
        self.worker_id = worker_id
        self.admitted = admitted
        self.requests: list[str] = []

    def predict(self, request, *, timeout_s: float):
        self.requests.append(request.request_id)
        return inference_pb2.PredictResponse(
            request_id=request.request_id,
            admitted=self.admitted,
            robot_id=request.robot_id,
            session_id=request.session_id,
            latency_ms=75.0,
            runtime_ms=70.0,
            queue_ms=5.0,
            telemetry_json=json.dumps({"worker": self.worker_id}),
        )

    def status(self, *, timeout_s: float):
        return inference_pb2.StatusResponse(total_gpus=1, total_models=1)

    def close(self) -> None:
        pass


def make_predict(idx: int, *, session_id: str | None = None, period_ms: float = 1000.0):
    session = session_id or f"session-{idx}"
    return inference_pb2.PredictRequest(
        request_id=f"req-{idx}",
        robot_id=f"robot-{idx}",
        session_id=session,
        deadline_ms=250.0,
        request_period_ms=period_ms,
        observation_format="torch",
    )


def make_router(*specs: WorkerSpec, clients: dict[str, FakeWorkerClient] | None = None) -> ClusterRouter:
    if clients is None:
        clients = {spec.worker_id: FakeWorkerClient(spec.worker_id) for spec in specs}
    return ClusterRouter(list(specs), clients=clients)


def test_router_keeps_session_affinity() -> None:
    router = make_router(
        WorkerSpec(worker_id="w0", address="localhost:50051"),
        WorkerSpec(worker_id="w1", address="localhost:50052"),
    )

    first, first_worker = router.route_predict(make_predict(0, session_id="robot-a"), timeout_s=1.0)
    second, second_worker = router.route_predict(make_predict(1, session_id="robot-a"), timeout_s=1.0)

    assert first.admitted is True
    assert second.admitted is True
    assert first_worker == second_worker
    assert router.stats()["session_to_worker"] == {"robot-a": first_worker}


def test_router_balances_new_sessions_by_projected_utilization() -> None:
    router = make_router(
        WorkerSpec(worker_id="w0", address="localhost:50051"),
        WorkerSpec(worker_id="w1", address="localhost:50052"),
    )

    first = router.route_predict(make_predict(0, session_id="robot-a"), timeout_s=1.0)[1]
    second = router.route_predict(make_predict(1, session_id="robot-b"), timeout_s=1.0)[1]
    third = router.route_predict(make_predict(2, session_id="robot-c"), timeout_s=1.0)[1]

    assert first == "w0"
    assert second == "w1"
    assert third == "w0"
    stats = router.stats()
    assert stats["workers"]["w0"]["sessions"] == 2
    assert stats["workers"]["w1"]["sessions"] == 1


def test_router_rejects_when_cluster_capacity_is_full() -> None:
    router = make_router(
        WorkerSpec(worker_id="w0", address="localhost:50051", max_sessions=1),
        WorkerSpec(worker_id="w1", address="localhost:50052", max_sessions=1),
    )

    assert router.route_predict(make_predict(0, session_id="robot-a"), timeout_s=1.0)[0].admitted
    assert router.route_predict(make_predict(1, session_id="robot-b"), timeout_s=1.0)[0].admitted
    rejected, worker_id = router.route_predict(make_predict(2, session_id="robot-c"), timeout_s=1.0)

    assert worker_id is None
    assert rejected.admitted is False
    assert rejected.rejection_reason == "cluster_admission_rejected"
    assert router.stats()["rejected_requests"] == 1


def test_router_avoids_stale_workers() -> None:
    router = make_router(
        WorkerSpec(worker_id="w0", address="localhost:50051", stale_after_s=0.01),
        WorkerSpec(worker_id="w1", address="localhost:50052", stale_after_s=10.0),
    )
    router.workers["w0"].last_heartbeat_s = time.monotonic() - 1.0

    _response, worker_id = router.route_predict(make_predict(0, session_id="robot-a"), timeout_s=1.0)

    assert worker_id == "w1"


def test_router_releases_session_after_worker_rejection() -> None:
    clients = {
        "w0": FakeWorkerClient("w0", admitted=False),
        "w1": FakeWorkerClient("w1", admitted=True),
    }
    router = make_router(
        WorkerSpec(worker_id="w0", address="localhost:50051", max_sessions=1),
        WorkerSpec(worker_id="w1", address="localhost:50052", max_sessions=1),
        clients=clients,
    )

    response, worker_id = router.route_predict(make_predict(0, session_id="robot-a"), timeout_s=1.0)

    assert worker_id == "w0"
    assert response.admitted is False
    assert router.stats()["sessions"] == 0
    assert router.stats()["workers"]["w0"]["sessions"] == 0


def test_parse_worker_spec_supports_short_keys() -> None:
    spec = parse_worker_spec("id=w0,addr=localhost:50051,gpu=1,max_sessions=24,util=0.8,rt=70")

    assert spec.worker_id == "w0"
    assert spec.address == "localhost:50051"
    assert spec.gpu_id == 1
    assert spec.max_sessions == 24
    assert spec.max_utilization == 0.8
    assert spec.estimated_runtime_ms == 70.0
