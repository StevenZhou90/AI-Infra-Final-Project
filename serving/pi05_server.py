"""PI0.5 gRPC serving entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import grpc
import numpy as np
import torch

from proto import inference_pb2, inference_pb2_grpc
from serving.pi05_grpc_codec import decode_prepared_observation, decode_prepared_observation_fields
from serving.pi05_runtime_service import PI05RuntimeService
from serving.pi0fast_serving_runtime import (
    NS_PER_MS,
    PI0FastRequest,
    PI0FastResponse,
    PI0FastServingConfig,
    RealPIBatchBackend,
    deadline_ns_from_period,
)

logger = logging.getLogger(__name__)


@dataclass
class PI05QueuedWork:
    request_id: str
    robot_id: str
    session_id: str
    observation: dict[str, Any]
    enqueued_ns: int
    deadline_ms: float
    request_period_ms: float | None
    event: threading.Event = field(default_factory=threading.Event)
    response: inference_pb2.PredictResponse | None = None
    error: Exception | None = None


def response_from_telemetry(resp) -> inference_pb2.PredictResponse:
    telemetry = resp.telemetry
    actions = np.asarray(resp.actions, dtype=np.float32)
    action_dim = int(actions.shape[-1]) if actions.ndim else 0
    chunk_size = int(actions.shape[0]) if actions.ndim > 1 else 1
    latency_ms = telemetry.queue_ms + telemetry.runtime_ms
    return inference_pb2.PredictResponse(
        request_id=resp.request_id,
        actions=actions.reshape(-1).tolist(),
        action_dim=action_dim,
        chunk_size=chunk_size,
        inference_time_ms=float(telemetry.runtime_ms),
        admitted=True,
        robot_id=telemetry.robot_id,
        session_id=telemetry.session_id,
        queue_ms=float(telemetry.queue_ms),
        runtime_ms=float(telemetry.runtime_ms),
        latency_ms=float(latency_ms),
        deadline_missed=bool(telemetry.deadline_missed),
        deadline_slack_ms=float(telemetry.deadline_slack_ms),
        batch_size=int(telemetry.batch_size),
        batch_reason=telemetry.batch_reason,
        actions_returned=int(telemetry.actions_returned),
        session_cache_hit=bool(telemetry.cache_hit),
        prompt_cache_hit=bool(telemetry.prompt_cache_hit),
        telemetry_json=json.dumps(telemetry.extra, sort_keys=True),
    )


class PI05InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(
        self,
        service: PI05RuntimeService,
        *,
        device: torch.device,
        metrics_path: Path | None = None,
        worker_timeout_s: float = 30.0,
    ) -> None:
        self.service = service
        self.device = device
        self.metrics_path = metrics_path
        self.started_at = time.time()
        self.total_requests = 0
        self.worker_timeout_s = worker_timeout_s
        self._queue: queue.Queue[PI05QueuedWork] = queue.Queue()
        self._worker = threading.Thread(target=self._gpu_worker_loop, daemon=True, name="pi05-gpu-worker")
        self._worker.start()
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def Predict(self, request, context):
        self.total_requests += 1
        deadline_ms = float(request.deadline_ms or self.service.runtime.config.default_control_period_ms)
        session_id = request.session_id or request.robot_id or request.request_id
        robot_id = request.robot_id or session_id
        try:
            if request.observation_format not in {"", "torch", "tensor_fields"}:
                raise ValueError(f"Unsupported observation_format={request.observation_format!r}")
            if request.observation_format == "tensor_fields":
                observation = decode_prepared_observation_fields(request.prepared_fields, device=self.device)
            else:
                observation = decode_prepared_observation(request.prepared_observation, device=self.device)
            work = PI05QueuedWork(
                request_id=request.request_id,
                robot_id=robot_id,
                session_id=session_id,
                observation=observation,
                enqueued_ns=request.timestamp_ns or time.time_ns(),
                deadline_ms=deadline_ms,
                request_period_ms=float(request.request_period_ms or 0.0) or None,
            )
            self._queue.put(work)
            if not work.event.wait(timeout=self.worker_timeout_s):
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details("PI0.5 GPU worker timeout")
                return inference_pb2.PredictResponse(
                    request_id=request.request_id,
                    admitted=False,
                    rejection_reason="worker_timeout",
                    robot_id=robot_id,
                    session_id=session_id,
                )
            if work.error is not None:
                raise work.error
            response = work.response or inference_pb2.PredictResponse(
                request_id=request.request_id,
                admitted=False,
                rejection_reason="empty_worker_response",
                robot_id=robot_id,
                session_id=session_id,
            )
            self._write_metric(response)
            return response
        except Exception as exc:
            logger.exception("PI0.5 Predict failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return inference_pb2.PredictResponse(
                request_id=request.request_id,
                admitted=False,
                rejection_reason=type(exc).__name__,
                robot_id=robot_id,
                session_id=session_id,
            )

    def GetStatus(self, request, context):
        stats = self.service.status()
        return inference_pb2.StatusResponse(
            total_gpus=1 if self.device.type == "cuda" else 0,
            total_models=1,
            total_requests_served=self.total_requests,
            uptime_seconds=float(time.time() - self.started_at),
        )

    def LoadModel(self, request, context):
        return inference_pb2.LoadModelResponse(success=False, message="PI0.5 server loads one model at startup")

    def UnloadModel(self, request, context):
        return inference_pb2.UnloadModelResponse(success=False, message="PI0.5 server keeps the startup model resident")

    def ListModels(self, request, context):
        return inference_pb2.ListModelsResponse(
            models=[
                inference_pb2.ModelInfo(
                    model_id=self.service.model_id,
                    pretrained_path=self.service.model_id,
                    gpu_id=0 if self.device.type == "cuda" else -1,
                    total_requests=self.total_requests,
                )
            ]
        )

    def _write_metric(self, response: inference_pb2.PredictResponse) -> None:
        if self.metrics_path is None:
            return
        row: dict[str, Any] = {
            "ts": time.time(),
            "request_id": response.request_id,
            "robot_id": response.robot_id,
            "session_id": response.session_id,
            "admitted": response.admitted,
            "rejection_reason": response.rejection_reason,
            "queue_ms": response.queue_ms,
            "runtime_ms": response.runtime_ms,
            "latency_ms": response.latency_ms,
            "deadline_missed": response.deadline_missed,
            "deadline_slack_ms": response.deadline_slack_ms,
            "batch_size": response.batch_size,
            "batch_reason": response.batch_reason,
            "actions_returned": response.actions_returned,
            "session_cache_hit": response.session_cache_hit,
            "prompt_cache_hit": response.prompt_cache_hit,
        }
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _gpu_worker_loop(self) -> None:
        while True:
            first = self._queue.get()
            batch = [first]
            deadline_s = time.monotonic() + max(self.service.runtime.config.max_batch_delay_ms, 0.0) / 1000.0
            while len(batch) < self.service.runtime.config.max_batch_size:
                timeout_s = deadline_s - time.monotonic()
                if timeout_s <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=timeout_s))
                except queue.Empty:
                    break
            self._process_worker_batch(batch)

    def _process_worker_batch(self, works: list[PI05QueuedWork]) -> None:
        response_by_id: dict[str, PI0FastResponse] = {}
        try:
            for work in works:
                request = PI0FastRequest(
                    request_id=work.request_id,
                    robot_id=work.robot_id,
                    session_id=work.session_id,
                    model_id=self.service.model_id,
                    enqueued_ns=work.enqueued_ns,
                    deadline_ns=deadline_ns_from_period(work.enqueued_ns, work.deadline_ms),
                    control_period_ms=work.deadline_ms,
                    decode_mode="flow",
                    observation=work.observation,
                    metadata={"request_period_ms": work.request_period_ms}
                    if work.request_period_ms is not None
                    else {},
                )
                if not self.service.runtime.try_submit(request):
                    work.response = inference_pb2.PredictResponse(
                        request_id=work.request_id,
                        admitted=False,
                        rejection_reason=self.service.runtime.last_rejection_reason or "admission_rejected",
                        robot_id=work.robot_id,
                        session_id=work.session_id,
                    )
            drain_ns = max(work.enqueued_ns for work in works) + int(
                self.service.runtime.config.max_batch_delay_ms * NS_PER_MS
            )
            responses = self.service.runtime.drain_ready(at_ns=drain_ns, force=True)
            response_by_id = {resp.request_id: resp for resp in responses}
            for work in works:
                if work.response is None:
                    runtime_response = response_by_id.get(work.request_id)
                    if runtime_response is None:
                        work.response = inference_pb2.PredictResponse(
                            request_id=work.request_id,
                            admitted=False,
                            rejection_reason="missing_runtime_response",
                            robot_id=work.robot_id,
                            session_id=work.session_id,
                        )
                    else:
                        work.response = response_from_telemetry(runtime_response)
        except Exception as exc:
            logger.exception("PI0.5 GPU worker failed")
            for work in works:
                work.error = exc
        finally:
            for work in works:
                work.event.set()


def load_service(args: argparse.Namespace) -> tuple[PI05RuntimeService, torch.device]:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    amp_dtype = None if args.amp_dtype == "none" else getattr(torch, args.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    policy = PI05Policy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    if hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = int(args.num_inference_steps)
    maybe_compile_policy(policy, args)
    backend = RealPIBatchBackend(
        policy,
        postprocessor=None,
        accelerator="real_pi05_grpc",
        autocast_dtype=amp_dtype if device.type == "cuda" else None,
    )
    config = PI0FastServingConfig(
        max_batch_size=args.max_batch_size,
        max_batch_delay_ms=args.max_batch_delay_ms,
        deadline_slack_ms=args.deadline_slack_ms,
        estimated_batch_base_ms=args.estimated_base_ms,
        estimated_batch_per_request_ms=args.estimated_per_request_ms,
        default_control_period_ms=args.deadline_ms,
        default_request_period_ms=args.default_request_period_ms,
        default_decode_mode="flow",
        max_active_sessions=args.max_active_sessions,
        max_admission_utilization=args.max_admission_utilization,
    )
    return PI05RuntimeService(backend, model_id=args.policy, config=config), device


def maybe_compile_policy(policy: Any, args: argparse.Namespace) -> None:
    if args.compile_target == "none":
        return
    if args.compile_target == "model":
        if not hasattr(policy, "model"):
            raise ValueError("--compile-target model requires policy.model")
        policy.model = torch.compile(policy.model, mode=args.compile_mode)
        logger.info("Compiled policy.model with torch.compile mode=%s", args.compile_mode)
        return
    if args.compile_target == "sample_actions":
        if not hasattr(policy, "model") or not hasattr(policy.model, "sample_actions"):
            raise ValueError("--compile-target sample_actions requires policy.model.sample_actions")
        policy.model.sample_actions = torch.compile(policy.model.sample_actions, mode=args.compile_mode)
        logger.info("Compiled policy.model.sample_actions with torch.compile mode=%s", args.compile_mode)
        return
    raise ValueError(f"Unsupported compile target: {args.compile_target}")


def serve(args: argparse.Namespace) -> None:
    service, device = load_service(args)
    if args.warmup_observation_path is not None:
        warmup_service(
            service,
            device=device,
            observation_path=args.warmup_observation_path,
            requests=args.warmup_requests,
            deadline_ms=args.deadline_ms,
            request_period_ms=args.default_request_period_ms,
        )
    servicer = PI05InferenceServicer(
        service,
        device=device,
        metrics_path=args.metrics_path,
        worker_timeout_s=args.worker_timeout_s,
    )
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
    logger.info("PI0.5 gRPC server listening on port %d", args.port)
    server.wait_for_termination()


def warmup_service(
    service: PI05RuntimeService,
    *,
    device: torch.device,
    observation_path: Path,
    requests: int,
    deadline_ms: float,
    request_period_ms: float,
) -> None:
    payload = observation_path.read_bytes()
    observation = decode_prepared_observation(payload, device=device)
    for idx in range(max(requests, 0)):
        result = service.predict(
            request_id=f"server-warmup-{idx}",
            robot_id="server-warmup",
            session_id="server-warmup",
            observation=observation,
            enqueued_ns=time.time_ns(),
            deadline_ms=deadline_ms,
            request_period_ms=request_period_ms,
            force=True,
        )
        if not result.admitted:
            raise RuntimeError(f"PI0.5 server warmup was rejected: {result.reason}")
    service.runtime.clear_session_state()
    logger.info("Completed %d PI0.5 server warmup request(s)", max(requests, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI0.5 gRPC server")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "none"])
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--compile-target", choices=["none", "model", "sample_actions"], default="none")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--max-active-sessions", type=int, default=4)
    parser.add_argument("--max-admission-utilization", type=float, default=None)
    parser.add_argument("--default-request-period-ms", type=float, default=1000.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-delay-ms", type=float, default=5.0)
    parser.add_argument("--deadline-slack-ms", type=float, default=8.0)
    parser.add_argument("--estimated-base-ms", type=float, default=160.0)
    parser.add_argument("--estimated-per-request-ms", type=float, default=35.0)
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/pi05_grpc_server/metrics.jsonl"))
    parser.add_argument("--worker-timeout-s", type=float, default=30.0)
    parser.add_argument("--warmup-observation-path", type=Path, default=None)
    parser.add_argument("--warmup-requests", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s")
    serve(parse_args())


if __name__ == "__main__":
    main()
