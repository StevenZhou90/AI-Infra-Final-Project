from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator, Callable

import numpy as np

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
from client.action_buffer import ActionBuffer
from client.encoder import FrameEncoder
from client.video_burst import VideoBurstBatcher
from client.isaac_sim_bridge import IsaacSimBridge

logger = logging.getLogger(__name__)


class VLAClient:
    """Async gRPC client for the VLA serving platform.

    Handles encoding, burst batching, inference requests, and action buffering.
    """

    def __init__(
        self,
        server_address: str = "localhost:50051",
        model_id: str = "default",
        priority: Priority = Priority.NORMAL,
        codec: Codec = Codec.JPEG,
        burst_size: int = 4,
        chunk_size: int = 90,
        frequency_hz: float = 50.0,
        max_message_size: int = 64 * 1024 * 1024,
    ) -> None:
        self._address = server_address
        self._model_id = model_id
        self._priority = priority
        self._chunk_size = chunk_size
        self._session_id = uuid.uuid4().hex[:12]

        self._encoder = FrameEncoder(codec=codec)
        self._batcher = VideoBurstBatcher(burst_size=burst_size)
        self._action_buffer = ActionBuffer(frequency_hz=frequency_hz)
        self._max_message_size = max_message_size

        self._channel = None
        self._stub = None
        self._connected = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        import grpc

        self._channel = grpc.aio.insecure_channel(
            self._address,
            options=[
                ("grpc.max_send_message_length", self._max_message_size),
                ("grpc.max_receive_message_length", self._max_message_size),
            ],
        )
        try:
            from proto import inference_pb2_grpc

            self._stub = {
                "inference": inference_pb2_grpc.InferenceServiceStub(self._channel),
                "model": inference_pb2_grpc.ModelServiceStub(self._channel),
                "health": inference_pb2_grpc.HealthServiceStub(self._channel),
            }
        except ImportError:
            logger.warning("Proto stubs not available, using raw channel")
            self._stub = None

        self._connected = True
        self._reconnect_delay = 1.0
        logger.info("Connected to %s", self._address)

    async def disconnect(self) -> None:
        if self._channel:
            await self._channel.close()
        self._connected = False
        logger.info("Disconnected from %s", self._address)

    async def _reconnect(self) -> None:
        while not self._connected:
            try:
                logger.info("Reconnecting to %s (delay=%.1fs)...", self._address, self._reconnect_delay)
                await self.connect()
                return
            except Exception:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    # ------------------------------------------------------------------
    # Unary inference
    # ------------------------------------------------------------------

    async def infer(
        self,
        camera_images: dict[str, np.ndarray],
        joint_state: JointState | None = None,
    ) -> InferenceResult:
        """Encode frames, send unary inference request, return result."""
        frames = []
        for cam_id, image in camera_images.items():
            ef = self._encoder.encode(image, camera_id=cam_id)
            frames.append(ef)

        request = InferenceRequest(
            model_id=self._model_id,
            priority=self._priority,
            camera_frames=frames,
            joint_state=joint_state,
            chunk_size=self._chunk_size,
            session_id=self._session_id,
        )

        return await self._send_unary(request)

    async def _send_unary(self, request: InferenceRequest) -> InferenceResult:
        from proto import inference_pb2

        pb_req = self._request_to_proto(request)

        try:
            response = await self._stub["inference"].Infer(pb_req)
            return self._proto_to_result(response)
        except Exception as e:
            self._connected = False
            return InferenceResult(request_id=request.request_id, error=str(e))

    # ------------------------------------------------------------------
    # Streaming inference
    # ------------------------------------------------------------------

    async def stream_infer(
        self,
        frame_source: AsyncIterator[tuple[dict[str, np.ndarray], JointState | None]],
    ) -> AsyncIterator[InferenceResult]:
        """Bidirectional streaming: send frames, receive action chunks."""

        async def request_gen():
            async for camera_images, joint_state in frame_source:
                frames = []
                bursts = []

                for cam_id, image in camera_images.items():
                    ef = self._encoder.encode(image, camera_id=cam_id)
                    burst = self._batcher.add_frame(ef)
                    if burst is not None:
                        bursts.append(burst)
                    else:
                        frames.append(ef)

                remaining = self._batcher.flush_all()
                bursts.extend(remaining)

                from proto import inference_pb2

                req = InferenceRequest(
                    model_id=self._model_id,
                    priority=self._priority,
                    camera_frames=frames,
                    video_bursts=bursts,
                    joint_state=joint_state,
                    chunk_size=self._chunk_size,
                    session_id=self._session_id,
                )
                yield self._request_to_proto(req)

        try:
            responses = self._stub["inference"].StreamInfer(request_gen())
            async for response in responses:
                result = self._proto_to_result(response)
                if result.action_chunk:
                    self._action_buffer.push_chunk(result.action_chunk)
                yield result
        except Exception as e:
            logger.error("Stream error: %s", e)
            self._connected = False

    # ------------------------------------------------------------------
    # Action buffer access
    # ------------------------------------------------------------------

    @property
    def action_buffer(self) -> ActionBuffer:
        return self._action_buffer

    def pop_action(self) -> ActionStep:
        return self._action_buffer.pop_action()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    async def list_models(self) -> list[dict]:
        from proto import inference_pb2

        resp = await self._stub["model"].ListModels(inference_pb2.ListModelsRequest())
        return [
            {
                "model_id": m.model_id,
                "name": m.model_name,
                "params": m.param_count,
                "gpu": m.gpu_id,
                "precision": m.precision,
                "loaded": m.is_loaded,
            }
            for m in resp.models
        ]

    async def health_check(self) -> dict:
        from proto import inference_pb2

        resp = await self._stub["health"].Check(inference_pb2.HealthRequest())
        return {
            "healthy": resp.healthy,
            "gpu_count": resp.gpu_count,
            "queue_depth": resp.queue_depth,
            "active_models": resp.active_models,
        }

    # ------------------------------------------------------------------
    # Proto conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _request_to_proto(request: InferenceRequest):
        from proto import inference_pb2

        frames_pb = [
            inference_pb2.EncodedFrame(
                data=f.data,
                codec=f.codec.value if isinstance(f.codec, Codec) else 0,
                width=f.width,
                height=f.height,
                channels=f.channels,
                camera_id=f.camera_id,
                timestamp_ns=f.timestamp_ns,
            )
            for f in request.camera_frames
        ]

        bursts_pb = []
        for b in request.video_bursts:
            burst_frames = [
                inference_pb2.EncodedFrame(
                    data=f.data,
                    codec=f.codec.value if isinstance(f.codec, Codec) else 0,
                    width=f.width,
                    height=f.height,
                    channels=f.channels,
                    camera_id=f.camera_id,
                    timestamp_ns=f.timestamp_ns,
                )
                for f in b.frames
            ]
            bursts_pb.append(
                inference_pb2.VideoBurst(
                    frames=burst_frames,
                    camera_id=b.camera_id,
                    start_timestamp_ns=b.start_timestamp_ns,
                    end_timestamp_ns=b.end_timestamp_ns,
                )
            )

        js_pb = None
        if request.joint_state:
            js_pb = inference_pb2.JointState(
                positions=request.joint_state.positions,
                velocities=request.joint_state.velocities,
                efforts=request.joint_state.efforts,
                timestamp_ns=request.joint_state.timestamp_ns,
            )

        codec_int_map = {
            Codec.RAW: 0, Codec.JPEG: 1, Codec.WEBP: 2,
            Codec.H264: 3, Codec.H265: 4,
        }

        for fpb in frames_pb:
            pass

        return inference_pb2.InferRequest(
            request_id=request.request_id,
            model_id=request.model_id,
            priority=int(request.priority),
            camera_frames=frames_pb,
            video_bursts=bursts_pb,
            joint_state=js_pb,
            chunk_size=request.chunk_size,
            session_id=request.session_id,
            metadata=request.metadata,
        )

    @staticmethod
    def _proto_to_result(pb_resp) -> InferenceResult:
        chunk = None
        if pb_resp.HasField("action_chunk") and pb_resp.action_chunk.steps:
            steps = [
                ActionStep(
                    joint_targets=list(s.joint_targets),
                    gripper_targets=list(s.gripper_targets),
                )
                for s in pb_resp.action_chunk.steps
            ]
            chunk = ActionChunk(
                steps=steps,
                start_timestamp_ns=pb_resp.action_chunk.start_timestamp_ns,
                frequency_hz=pb_resp.action_chunk.frequency_hz,
                accepted_length=pb_resp.action_chunk.accepted_length,
                confidence=pb_resp.action_chunk.confidence,
            )

        return InferenceResult(
            request_id=pb_resp.request_id,
            action_chunk=chunk,
            used_speculative=pb_resp.used_speculative,
            drafter_acceptance_rate=pb_resp.drafter_acceptance_rate,
            inference_time_ms=pb_resp.inference_time_ms,
            queue_wait_ms=pb_resp.queue_wait_ms,
            error=pb_resp.error,
        )
