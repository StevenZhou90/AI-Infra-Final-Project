"""gRPC inference server — receives observations, routes through priority scheduler.

Usage:
    uv run python -m serving.grpc_server
    uv run python -m serving.grpc_server --port 50051
"""

from __future__ import annotations

import argparse
import io
import logging
import threading
import time
from concurrent import futures

import grpc
import numpy as np
import torch
from PIL import Image

from proto import inference_pb2, inference_pb2_grpc
from serving.model_registry import ModelRegistry
from serving.scheduler import PriorityScheduler, SchedulerConfig, QueuedRequest

logger = logging.getLogger(__name__)


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):

    def __init__(self, registry: ModelRegistry, scheduler: PriorityScheduler) -> None:
        self._registry = registry
        self._scheduler = scheduler
        self._start_time = time.time()
        self._total_requests = 0
        self._active_episodes: dict[str, str] = {}

    def Predict(self, request, context):
        self._total_requests += 1

        try:
            self._registry.get_model(request.model_id)
        except ValueError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return inference_pb2.PredictResponse()

        episode_id = request.request_id.rsplit("-", 1)[0] if "-" in request.request_id else request.request_id

        sched_request = {
            "request": request,
            "episode_id": episode_id,
            "priority": request.priority,
            "model_id": request.model_id,
        }

        result = self._scheduler.submit(sched_request, timeout=30.0)

        if result is None:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Scheduler timeout")
            return inference_pb2.PredictResponse()

        return result

    def LoadModel(self, request, context):
        try:
            entry = self._registry.load_model(
                model_id=request.model_id,
                pretrained_path=request.pretrained_path,
                gpu_id=request.gpu_id,
                model_type=request.model_type,
                use_kv_cache=request.use_kv_cache,
                use_speculative_decoding=request.use_speculative_decoding,
            )
            return inference_pb2.LoadModelResponse(
                success=True, message=f"Loaded {request.model_id}",
                model_id=entry.model_id, gpu_id=entry.gpu_id,
                memory_used_mb=entry.memory_mb,
            )
        except Exception as e:
            return inference_pb2.LoadModelResponse(success=False, message=str(e))

    def UnloadModel(self, request, context):
        try:
            self._registry.unload_model(request.model_id)
            return inference_pb2.UnloadModelResponse(success=True, message=f"Unloaded {request.model_id}")
        except ValueError as e:
            return inference_pb2.UnloadModelResponse(success=False, message=str(e))

    def ListModels(self, request, context):
        models = self._registry.list_models()
        return inference_pb2.ListModelsResponse(
            models=[
                inference_pb2.ModelInfo(
                    model_id=m.model_id, pretrained_path=m.pretrained_path,
                    gpu_id=m.gpu_id, memory_used_mb=m.memory_mb,
                    total_requests=m.total_requests,
                )
                for m in models
            ]
        )

    def GetStatus(self, request, context):
        gpus = self._registry.gpu_status()
        models = self._registry.list_models()
        sched_stats = self._scheduler.stats()
        return inference_pb2.StatusResponse(
            total_gpus=len(gpus),
            gpus=[
                inference_pb2.GpuStatus(
                    gpu_id=g["gpu_id"], name=g["name"],
                    total_memory_mb=g["total_memory_mb"],
                    used_memory_mb=g["used_memory_mb"],
                    loaded_models=g["loaded_models"],
                )
                for g in gpus
            ],
            total_models=len(models),
            total_requests_served=self._total_requests,
            uptime_seconds=time.time() - self._start_time,
        )

    def _decode_observation(self, request, device: torch.device) -> dict[str, torch.Tensor]:
        """Decode a PredictRequest into the dict[str, Tensor] format policies expect."""
        obs: dict[str, torch.Tensor] = {}
        for img_frame in request.images:
            if img_frame.encoding == "raw_rgb":
                arr = np.frombuffer(img_frame.data, dtype=np.uint8).reshape(
                    img_frame.height, img_frame.width, 3
                ).astype(np.float32) / 255.0
            else:
                pil_img = Image.open(io.BytesIO(img_frame.data)).convert("RGB")
                arr = np.array(pil_img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
            obs[f"observation.images.{img_frame.camera_name}"] = tensor
        if request.state:
            obs["observation.state"] = torch.tensor(request.state, dtype=torch.float32, device=device)
        return obs


def _inference_worker(
    servicer: InferenceServicer,
    scheduler: PriorityScheduler,
    registry: ModelRegistry,
) -> None:
    """Worker thread: pulls highest-priority request, runs inference, returns result."""
    while True:
        queued = scheduler.wait_next(timeout=1.0)
        if queued is None:
            continue

        try:
            req_data = queued.item
            grpc_request = req_data["request"]
            model_id = req_data["model_id"]
            episode_id = req_data["episode_id"]

            entry = registry.get_model(model_id)

            if servicer._active_episodes.get(model_id) != episode_id:
                entry.policy.reset()
                servicer._active_episodes[model_id] = episode_id

            observation = servicer._decode_observation(grpc_request, entry.policy._device)
            instruction = grpc_request.instruction if grpc_request.instruction else ""
            actions, inference_ms = registry.predict(model_id, observation, instruction=instruction)

            action_dim = len(grpc_request.state) if grpc_request.state else 14
            chunk_size = len(actions) // action_dim if action_dim > 0 else 1

            queued.result = inference_pb2.PredictResponse(
                request_id=grpc_request.request_id,
                actions=actions, action_dim=action_dim,
                chunk_size=chunk_size, inference_time_ms=inference_ms,
            )
        except Exception as e:
            logger.error("Inference worker error: %s", e)
            queued.error = e
        finally:
            queued.event.set()


def serve(port: int = 50051, max_workers: int = 4, num_inference_workers: int = 1) -> None:
    registry = ModelRegistry()
    scheduler = PriorityScheduler(SchedulerConfig())
    servicer = InferenceServicer(registry, scheduler)

    for i in range(num_inference_workers):
        t = threading.Thread(
            target=_inference_worker, args=(servicer, scheduler, registry),
            daemon=True, name=f"inference-worker-{i}",
        )
        t.start()
    logger.info("Started %d inference worker(s)", num_inference_workers)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    logger.info("gRPC server listening on port %d", port)
    server.wait_for_termination()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    )
    parser = argparse.ArgumentParser(description="VLA Inference Server")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.workers)


if __name__ == "__main__":
    main()
