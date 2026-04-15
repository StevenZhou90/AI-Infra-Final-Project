from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

import grpc

from common.types import (
    ActionChunk,
    ActionStep,
    Codec,
    EncodedFrame,
    InferenceRequest,
    InferenceResult,
    JointState,
    Priority,
    VideoBurst,
)
from server.config import ServerConfig
from server.gpu_manager import GpuManager
from server.inference_engine import InferenceEngine
from server.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# proto stubs are generated at build time; use lazy imports
_pb2 = None
_pb2_grpc = None


def _load_proto():
    global _pb2, _pb2_grpc
    if _pb2 is not None:
        return
    try:
        from proto import inference_pb2 as pb2
        from proto import inference_pb2_grpc as pb2_grpc

        _pb2 = pb2
        _pb2_grpc = pb2_grpc
    except ImportError:
        logger.warning("Generated proto stubs not found; run grpc_tools.protoc first")


_CODEC_MAP = {
    0: Codec.RAW,
    1: Codec.JPEG,
    2: Codec.WEBP,
    3: Codec.H264,
    4: Codec.H265,
}

_PRIORITY_MAP = {
    0: Priority.LOW,
    1: Priority.NORMAL,
    2: Priority.HIGH,
    3: Priority.CRITICAL,
}


# ------------------------------------------------------------------
# Convert proto <-> internal types
# ------------------------------------------------------------------

def _proto_to_request(pb_req) -> InferenceRequest:
    frames = [
        EncodedFrame(
            data=f.data,
            codec=_CODEC_MAP.get(f.codec, Codec.RAW),
            width=f.width,
            height=f.height,
            channels=f.channels or 3,
            camera_id=f.camera_id,
            timestamp_ns=f.timestamp_ns,
        )
        for f in pb_req.camera_frames
    ]
    bursts = [
        VideoBurst(
            frames=[
                EncodedFrame(
                    data=f.data,
                    codec=_CODEC_MAP.get(f.codec, Codec.RAW),
                    width=f.width,
                    height=f.height,
                    channels=f.channels or 3,
                    camera_id=f.camera_id,
                    timestamp_ns=f.timestamp_ns,
                )
                for f in b.frames
            ],
            camera_id=b.camera_id,
            start_timestamp_ns=b.start_timestamp_ns,
            end_timestamp_ns=b.end_timestamp_ns,
        )
        for b in pb_req.video_bursts
    ]
    js = None
    if pb_req.HasField("joint_state"):
        js = JointState(
            positions=list(pb_req.joint_state.positions),
            velocities=list(pb_req.joint_state.velocities),
            efforts=list(pb_req.joint_state.efforts),
            timestamp_ns=pb_req.joint_state.timestamp_ns,
        )
    return InferenceRequest(
        request_id=pb_req.request_id,
        model_id=pb_req.model_id,
        priority=_PRIORITY_MAP.get(pb_req.priority, Priority.NORMAL),
        camera_frames=frames,
        video_bursts=bursts,
        joint_state=js,
        chunk_size=pb_req.chunk_size or 90,
        session_id=pb_req.session_id,
        metadata=dict(pb_req.metadata),
    )


def _result_to_proto(result: InferenceResult):
    _load_proto()
    chunk_pb = None
    if result.action_chunk is not None:
        steps_pb = []
        for s in result.action_chunk.steps:
            steps_pb.append(
                _pb2.ActionStep(
                    joint_targets=s.joint_targets,
                    gripper_targets=s.gripper_targets,
                )
            )
        chunk_pb = _pb2.ActionChunk(
            steps=steps_pb,
            start_timestamp_ns=result.action_chunk.start_timestamp_ns,
            frequency_hz=result.action_chunk.frequency_hz,
            accepted_length=result.action_chunk.accepted_length,
            confidence=result.action_chunk.confidence,
        )
    return _pb2.InferResponse(
        request_id=result.request_id,
        action_chunk=chunk_pb,
        used_speculative=result.used_speculative,
        drafter_acceptance_rate=result.drafter_acceptance_rate,
        inference_time_ms=result.inference_time_ms,
        queue_wait_ms=result.queue_wait_ms,
        error=result.error,
    )


# ------------------------------------------------------------------
# gRPC Servicers
# ------------------------------------------------------------------

class InferenceServicer:
    """Implements InferenceService RPCs."""

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine

    async def Infer(self, request, context):
        _load_proto()
        req = _proto_to_request(request)
        result = await self._engine.infer(req)
        return _result_to_proto(result)

    async def StreamInfer(self, request_iterator, context):
        _load_proto()
        async for pb_req in request_iterator:
            req = _proto_to_request(pb_req)
            future = self._engine.submit(req)
            result = await future
            yield _result_to_proto(result)


class ModelServicer:
    """Implements ModelService RPCs."""

    def __init__(self, registry: ModelRegistry, gpu_manager: GpuManager) -> None:
        self._registry = registry
        self._gpu = gpu_manager

    async def ListModels(self, request, context):
        _load_proto()
        models = self._registry.list_models()
        infos = []
        for m in models:
            infos.append(
                _pb2.ModelInfo(
                    model_id=m.model_id,
                    model_name=m.hf_repo,
                    param_count=m.param_count,
                    precision=m.precision.value,
                    gpu_id=m.gpu_id,
                    memory_bytes=m.memory_bytes,
                    is_loaded=m.is_loaded,
                    drafter_model_id=m.drafter_model_id or "",
                )
            )
        return _pb2.ListModelsResponse(models=infos)

    async def LoadModel(self, request, context):
        _load_proto()
        try:
            from common.types import Precision

            precision = None
            if request.precision:
                precision = Precision(request.precision)
            target_gpu = request.target_gpu if request.target_gpu >= 0 else None

            placement = self._registry.load_model(
                model_id=request.model_id,
                hf_repo=request.hf_repo,
                drafter_hf_repo=request.drafter_hf_repo or None,
                target_gpu=target_gpu,
                precision=precision,
            )
            return _pb2.LoadModelResponse(
                model=_pb2.ModelInfo(
                    model_id=placement.model_id,
                    model_name=placement.hf_repo,
                    param_count=placement.param_count,
                    precision=placement.precision.value,
                    gpu_id=placement.gpu_id,
                    memory_bytes=placement.memory_bytes,
                    is_loaded=True,
                    drafter_model_id=placement.drafter_model_id or "",
                )
            )
        except Exception as e:
            return _pb2.LoadModelResponse(error=str(e))

    async def UnloadModel(self, request, context):
        _load_proto()
        try:
            self._registry.unload_model(request.model_id)
            return _pb2.UnloadModelResponse(success=True)
        except Exception as e:
            return _pb2.UnloadModelResponse(success=False, error=str(e))


class HealthServicer:
    """Implements HealthService RPCs."""

    def __init__(
        self,
        gpu_manager: GpuManager,
        model_registry: ModelRegistry,
        kv_cache,
        scheduler,
    ) -> None:
        self._gpu = gpu_manager
        self._registry = model_registry
        self._kv_cache = kv_cache
        self._scheduler = scheduler

    async def Check(self, request, context):
        _load_proto()
        gpu_statuses = []
        for gid in self._gpu.gpu_ids:
            mem = self._gpu.refresh_memory(gid)
            info = self._gpu.get_info(gid)
            util = self._kv_cache.utilization(gid)
            models_on = self._gpu.models_on_gpu(gid)
            gpu_statuses.append(
                _pb2.GpuStatus(
                    gpu_id=gid,
                    gpu_name=info.name,
                    total_memory=info.total_memory,
                    used_memory=mem.used,
                    kv_cache_memory=mem.kv_cache,
                    kv_cache_utilization=util,
                    gpu_utilization=0.0,
                    models_loaded=len(models_on),
                )
            )
        return _pb2.HealthResponse(
            healthy=True,
            gpu_count=self._gpu.gpu_count,
            gpus=gpu_statuses,
            queue_depth=self._scheduler.depth,
            active_models=len(self._registry.list_models()),
        )


# ------------------------------------------------------------------
# Server builder
# ------------------------------------------------------------------

async def serve(
    config: ServerConfig,
    engine: InferenceEngine,
    registry: ModelRegistry,
    gpu_manager: GpuManager,
    kv_cache,
    scheduler,
) -> grpc.aio.Server:
    """Create, configure, and start the gRPC server."""
    _load_proto()

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", config.grpc.max_message_size),
            ("grpc.max_receive_message_length", config.grpc.max_message_size),
            ("grpc.max_concurrent_streams", config.grpc.max_concurrent_rpcs),
        ]
    )

    inference_servicer = InferenceServicer(engine)
    model_servicer = ModelServicer(registry, gpu_manager)
    health_servicer = HealthServicer(gpu_manager, registry, kv_cache, scheduler)

    _pb2_grpc.add_InferenceServiceServicer_to_server(inference_servicer, server)
    _pb2_grpc.add_ModelServiceServicer_to_server(model_servicer, server)
    _pb2_grpc.add_HealthServiceServicer_to_server(health_servicer, server)

    addr = f"{config.grpc.host}:{config.grpc.port}"
    server.add_insecure_port(addr)
    await server.start()
    logger.info("gRPC server listening on %s", addr)
    return server
