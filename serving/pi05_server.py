"""PI0.5 gRPC serving entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import grpc
import numpy as np
import torch

from proto import inference_pb2, inference_pb2_grpc
from serving.pi05_grpc_codec import decode_prepared_observation
from serving.pi05_runtime_service import PI05RuntimeService
from serving.pi0fast_serving_runtime import PI0FastServingConfig, RealPIBatchBackend

logger = logging.getLogger(__name__)


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
    def __init__(self, service: PI05RuntimeService, *, device: torch.device, metrics_path: Path | None = None) -> None:
        self.service = service
        self.device = device
        self.metrics_path = metrics_path
        self.started_at = time.time()
        self.total_requests = 0
        self._lock = threading.Lock()
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def Predict(self, request, context):
        self.total_requests += 1
        deadline_ms = float(request.deadline_ms or self.service.runtime.config.default_control_period_ms)
        session_id = request.session_id or request.robot_id or request.request_id
        robot_id = request.robot_id or session_id
        try:
            if request.observation_format not in {"", "torch"}:
                raise ValueError(f"Unsupported observation_format={request.observation_format!r}")
            observation = decode_prepared_observation(request.prepared_observation, device=self.device)
            with self._lock:
                result = self.service.predict(
                    request_id=request.request_id,
                    robot_id=robot_id,
                    session_id=session_id,
                    observation=observation,
                    enqueued_ns=request.timestamp_ns or None,
                    deadline_ms=deadline_ms,
                )
                if not result.admitted:
                    response = inference_pb2.PredictResponse(
                        request_id=request.request_id,
                        admitted=False,
                        rejection_reason=result.reason,
                        robot_id=robot_id,
                        session_id=session_id,
                    )
                    self._write_metric(response)
                    return response
                responses = result.responses or self.service.drain(force=True)
            matching = next((resp for resp in responses if resp.request_id == request.request_id), responses[-1])
            response = response_from_telemetry(matching)
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


def load_service(args: argparse.Namespace) -> tuple[PI05RuntimeService, torch.device]:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    amp_dtype = None if args.amp_dtype == "none" else getattr(torch, args.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    policy = PI05Policy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    if hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = int(args.num_inference_steps)
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
        default_decode_mode="flow",
        max_active_sessions=args.max_active_sessions,
    )
    return PI05RuntimeService(backend, model_id=args.policy, config=config), device


def serve(args: argparse.Namespace) -> None:
    service, device = load_service(args)
    servicer = PI05InferenceServicer(
        service,
        device=device,
        metrics_path=args.metrics_path,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI0.5 gRPC server")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "none"])
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--max-active-sessions", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-delay-ms", type=float, default=5.0)
    parser.add_argument("--deadline-slack-ms", type=float, default=8.0)
    parser.add_argument("--estimated-base-ms", type=float, default=160.0)
    parser.add_argument("--estimated-per-request-ms", type=float, default=35.0)
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/pi05_grpc_server/metrics.jsonl"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s")
    serve(parse_args())


if __name__ == "__main__":
    main()
