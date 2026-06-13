"""Cluster router for PI0.5 gRPC workers.

The router exposes the existing InferenceService API and forwards Predict calls
to per-GPU PI0.5 workers.  It keeps worker registration in memory for now so the
same code can run locally with one GPU and later with one worker per GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import Protocol

import grpc

from proto import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    address: str
    gpu_id: int = 0
    model_id: str = "lerobot/pi05_libero_finetuned_v044"
    max_sessions: int = 12
    max_utilization: float = 1.0
    estimated_runtime_ms: float = 75.0
    stale_after_s: float = 10.0


@dataclass
class WorkerState:
    spec: WorkerSpec
    sessions: dict[str, float] = field(default_factory=dict)
    healthy: bool = True
    last_heartbeat_s: float = field(default_factory=time.monotonic)
    requests: int = 0
    admitted: int = 0
    rejected: int = 0
    rpc_errors: int = 0
    last_error: str = ""

    def is_available(self, now_s: float | None = None) -> bool:
        now_s = time.monotonic() if now_s is None else now_s
        return self.healthy and (now_s - self.last_heartbeat_s) <= self.spec.stale_after_s

    def utilization(self) -> float:
        total = 0.0
        for period_ms in self.sessions.values():
            total += self.spec.estimated_runtime_ms / max(period_ms, 1.0)
        return total

    def projected_utilization(self, session_id: str, request_period_ms: float) -> float:
        if session_id in self.sessions:
            return self.utilization()
        return self.utilization() + self.spec.estimated_runtime_ms / max(request_period_ms, 1.0)

    def can_accept_new_session(self, session_id: str, request_period_ms: float) -> bool:
        if session_id in self.sessions:
            return True
        if len(self.sessions) >= self.spec.max_sessions:
            return False
        return self.projected_utilization(session_id, request_period_ms) <= self.spec.max_utilization


class WorkerClient(Protocol):
    def predict(self, request, *, timeout_s: float):
        ...

    def status(self, *, timeout_s: float):
        ...

    def close(self) -> None:
        ...


class GrpcWorkerClient:
    def __init__(self, address: str) -> None:
        self.channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)

    def predict(self, request, *, timeout_s: float):
        return self.stub.Predict(request, timeout=timeout_s)

    def status(self, *, timeout_s: float):
        return self.stub.GetStatus(inference_pb2.StatusRequest(), timeout=timeout_s)

    def close(self) -> None:
        self.channel.close()


class ClusterRouter:
    def __init__(
        self,
        workers: list[WorkerSpec],
        *,
        clients: dict[str, WorkerClient] | None = None,
        default_request_period_ms: float = 1000.0,
    ) -> None:
        if not workers:
            raise ValueError("ClusterRouter requires at least one worker")
        self.default_request_period_ms = default_request_period_ms
        self.workers = {spec.worker_id: WorkerState(spec=spec) for spec in workers}
        self.clients = clients or {spec.worker_id: GrpcWorkerClient(spec.address) for spec in workers}
        missing = sorted(set(self.workers) - set(self.clients))
        if missing:
            raise ValueError(f"Missing worker clients for: {missing}")
        self.session_to_worker: dict[str, str] = {}
        self.total_requests = 0
        self.rejected_requests = 0
        self.rpc_errors = 0
        self.started_at_s = time.time()
        self._lock = threading.Lock()

    def close(self) -> None:
        for client in self.clients.values():
            client.close()

    def heartbeat_once(self, *, timeout_s: float = 1.0) -> None:
        now_s = time.monotonic()
        for worker_id, client in self.clients.items():
            state = self.workers[worker_id]
            try:
                client.status(timeout_s=timeout_s)
            except grpc.RpcError as exc:
                with self._lock:
                    state.healthy = False
                    state.last_error = exc.details() or exc.code().name
                    state.rpc_errors += 1
                continue
            except Exception as exc:
                with self._lock:
                    state.healthy = False
                    state.last_error = str(exc)
                    state.rpc_errors += 1
                continue
            with self._lock:
                state.healthy = True
                state.last_heartbeat_s = now_s
                state.last_error = ""

    def route_predict(self, request, *, timeout_s: float) -> tuple[inference_pb2.PredictResponse, str | None]:
        session_id = request.session_id or request.robot_id or request.request_id
        request_period_ms = float(request.request_period_ms or self.default_request_period_ms)
        with self._lock:
            self.total_requests += 1
            worker_id = self._select_worker_locked(session_id, request_period_ms)
            if worker_id is None:
                self.rejected_requests += 1
                return self._router_rejection(request, session_id), None
            state = self.workers[worker_id]
            state.requests += 1
            new_session = session_id not in state.sessions
            state.sessions.setdefault(session_id, request_period_ms)
            self.session_to_worker[session_id] = worker_id

        try:
            response = self.clients[worker_id].predict(request, timeout_s=timeout_s)
        except grpc.RpcError as exc:
            with self._lock:
                self.rpc_errors += 1
                state.rpc_errors += 1
                state.last_error = exc.details() or exc.code().name
                if new_session:
                    state.sessions.pop(session_id, None)
                    self.session_to_worker.pop(session_id, None)
            raise

        with self._lock:
            if response.admitted:
                state.admitted += 1
            else:
                state.rejected += 1
                if new_session:
                    state.sessions.pop(session_id, None)
                    self.session_to_worker.pop(session_id, None)
        return self._annotate_response(response, worker_id), worker_id

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "rejected_requests": self.rejected_requests,
                "rpc_errors": self.rpc_errors,
                "sessions": len(self.session_to_worker),
                "workers": {
                    worker_id: {
                        "address": state.spec.address,
                        "gpu_id": state.spec.gpu_id,
                        "healthy": state.is_available(),
                        "sessions": len(state.sessions),
                        "utilization": state.utilization(),
                        "requests": state.requests,
                        "admitted": state.admitted,
                        "rejected": state.rejected,
                        "rpc_errors": state.rpc_errors,
                        "last_error": state.last_error,
                    }
                    for worker_id, state in sorted(self.workers.items())
                },
                "session_to_worker": dict(sorted(self.session_to_worker.items())),
            }

    def _select_worker_locked(self, session_id: str, request_period_ms: float) -> str | None:
        assigned = self.session_to_worker.get(session_id)
        if assigned is not None:
            state = self.workers.get(assigned)
            if state is not None and state.is_available() and state.can_accept_new_session(session_id, request_period_ms):
                return assigned
            self.session_to_worker.pop(session_id, None)

        candidates = [
            state
            for state in self.workers.values()
            if state.is_available() and state.can_accept_new_session(session_id, request_period_ms)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda state: (state.projected_utilization(session_id, request_period_ms), len(state.sessions)))
        return candidates[0].spec.worker_id

    def _router_rejection(self, request, session_id: str) -> inference_pb2.PredictResponse:
        return inference_pb2.PredictResponse(
            request_id=request.request_id,
            admitted=False,
            rejection_reason="cluster_admission_rejected",
            robot_id=request.robot_id or session_id,
            session_id=session_id,
            telemetry_json=json.dumps({"cluster_rejected": True, "cluster_reason": "no_worker_capacity"}),
        )

    def _annotate_response(self, response: inference_pb2.PredictResponse, worker_id: str) -> inference_pb2.PredictResponse:
        telemetry = {}
        if response.telemetry_json:
            try:
                telemetry = json.loads(response.telemetry_json)
            except json.JSONDecodeError:
                telemetry = {"worker_telemetry_json": response.telemetry_json}
        telemetry.update(
            {
                "cluster_worker_id": worker_id,
                "cluster_worker_gpu_id": self.workers[worker_id].spec.gpu_id,
                "cluster_worker_address": self.workers[worker_id].spec.address,
            }
        )
        response.telemetry_json = json.dumps(telemetry, sort_keys=True)
        return response


class PI05ClusterRouterServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, router: ClusterRouter, *, rpc_timeout_s: float = 5.0) -> None:
        self.router = router
        self.rpc_timeout_s = rpc_timeout_s

    def Predict(self, request, context):
        try:
            response, _worker_id = self.router.route_predict(request, timeout_s=self.rpc_timeout_s)
            return response
        except grpc.RpcError as exc:
            context.set_code(exc.code())
            context.set_details(exc.details() or "worker_rpc_error")
            return inference_pb2.PredictResponse(
                request_id=request.request_id,
                admitted=False,
                rejection_reason=f"worker_rpc_{exc.code().name.lower()}",
                robot_id=request.robot_id,
                session_id=request.session_id,
            )

    def GetStatus(self, request, context):
        stats = self.router.stats()
        workers = stats["workers"]
        return inference_pb2.StatusResponse(
            total_gpus=len(workers),
            gpus=[
                inference_pb2.GpuStatus(
                    gpu_id=worker["gpu_id"],
                    name=worker_id,
                    total_memory_mb=0.0,
                    used_memory_mb=0.0,
                    loaded_models=[self.router.workers[worker_id].spec.model_id],
                )
                for worker_id, worker in workers.items()
            ],
            total_models=len(workers),
            total_requests_served=int(stats["total_requests"]),
            uptime_seconds=float(time.time() - self.router.started_at_s),
        )

    def LoadModel(self, request, context):
        return inference_pb2.LoadModelResponse(success=False, message="Cluster router uses static worker registration")

    def UnloadModel(self, request, context):
        return inference_pb2.UnloadModelResponse(success=False, message="Cluster router does not unload worker models")

    def ListModels(self, request, context):
        return inference_pb2.ListModelsResponse(
            models=[
                inference_pb2.ModelInfo(
                    model_id=state.spec.model_id,
                    pretrained_path=state.spec.model_id,
                    gpu_id=state.spec.gpu_id,
                    total_requests=state.requests,
                )
                for state in self.router.workers.values()
            ]
        )


def parse_worker_spec(raw: str) -> WorkerSpec:
    values: dict[str, str] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Invalid worker spec part {part!r}; expected key=value")
        values[key.strip()] = value.strip()
    worker_id = values.pop("id", values.pop("worker_id", ""))
    address = values.pop("addr", values.pop("address", ""))
    if not worker_id or not address:
        raise ValueError("--worker requires id=<id>,addr=<host:port>")
    return WorkerSpec(
        worker_id=worker_id,
        address=address,
        gpu_id=int(values.pop("gpu", values.pop("gpu_id", 0))),
        model_id=values.pop("model", values.pop("model_id", "lerobot/pi05_libero_finetuned_v044")),
        max_sessions=int(values.pop("max_sessions", 12)),
        max_utilization=float(values.pop("max_utilization", values.pop("util", 1.0))),
        estimated_runtime_ms=float(values.pop("estimated_runtime_ms", values.pop("rt", 75.0))),
        stale_after_s=float(values.pop("stale_after_s", 10.0)),
    )


def heartbeat_loop(router: ClusterRouter, *, interval_s: float, timeout_s: float) -> None:
    while True:
        router.heartbeat_once(timeout_s=timeout_s)
        time.sleep(interval_s)


def serve(args: argparse.Namespace) -> None:
    worker_specs = [parse_worker_spec(raw) for raw in args.worker]
    router = ClusterRouter(worker_specs, default_request_period_ms=args.default_request_period_ms)
    if args.heartbeat_interval_s > 0:
        thread = threading.Thread(
            target=heartbeat_loop,
            kwargs={"router": router, "interval_s": args.heartbeat_interval_s, "timeout_s": args.heartbeat_timeout_s},
            daemon=True,
            name="pi05-cluster-heartbeat",
        )
        thread.start()
    servicer = PI05ClusterRouterServicer(router, rpc_timeout_s=args.worker_rpc_timeout_s)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{args.port}")
    server.start()
    logger.info("PI0.5 cluster router listening on port %d with %d workers", args.port, len(worker_specs))
    server.wait_for_termination()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI0.5 cluster router")
    parser.add_argument("--port", type=int, default=50100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--worker", action="append", required=True, help="id=w0,addr=localhost:50051,gpu=0,max_sessions=12")
    parser.add_argument("--default-request-period-ms", type=float, default=1000.0)
    parser.add_argument("--worker-rpc-timeout-s", type=float, default=5.0)
    parser.add_argument("--heartbeat-interval-s", type=float, default=2.0)
    parser.add_argument("--heartbeat-timeout-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s")
    serve(parse_args())


if __name__ == "__main__":
    main()
