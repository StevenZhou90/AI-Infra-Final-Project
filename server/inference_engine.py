from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import torch

from common.types import (
    ActionChunk,
    InferenceRequest,
    InferenceResult,
    Priority,
)
from server.config import ServerConfig
from server.gpu_manager import GpuManager
from server.kv_cache import KVCacheManager
from server.model_registry import ModelRegistry
from server.scheduler import PriorityScheduler
from server.spec_decode import SpeculativeDecoder
from server.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Core inference loop: dequeue requests, decode video, run speculative inference."""

    def __init__(
        self,
        config: ServerConfig,
        gpu_manager: GpuManager,
        model_registry: ModelRegistry,
        scheduler: PriorityScheduler,
        kv_cache: KVCacheManager,
        spec_decoder: SpeculativeDecoder,
    ) -> None:
        self._config = config
        self._gpu = gpu_manager
        self._registry = model_registry
        self._scheduler = scheduler
        self._kv_cache = kv_cache
        self._spec_decoder = spec_decoder
        self._video_proc = VideoProcessor(config.video)
        self._running = False
        self._workers: list[asyncio.Task] = []
        self._current_priority: dict[int, Priority] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, num_workers: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        n = num_workers or max(1, self._gpu.gpu_count)
        for i in range(n):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)
        logger.info("Inference engine started with %d workers", n)

    async def stop(self) -> None:
        self._running = False
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Inference engine stopped")

    # ------------------------------------------------------------------
    # Single-shot inference (bypass scheduler)
    # ------------------------------------------------------------------

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference synchronously (used by unary gRPC)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_request, request
        )

    # ------------------------------------------------------------------
    # Worker loop (async scheduler-driven)
    # ------------------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        logger.info("Worker %d started", worker_id)
        while self._running:
            try:
                request = await self._scheduler.dequeue(timeout=1.0)
                if request is None:
                    continue

                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_request, request
                )

                if request.request_id in self._result_futures:
                    self._result_futures[request.request_id].set_result(result)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Worker %d error", worker_id)

    _result_futures: dict[str, asyncio.Future] = {}

    def submit(self, request: InferenceRequest) -> asyncio.Future[InferenceResult]:
        """Submit request to scheduler and return a future for the result."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future[InferenceResult] = loop.create_future()
        self._result_futures[request.request_id] = future

        gpu_id = self._gpu.get_model_gpu(request.model_id) or -1
        if not self._scheduler.enqueue(request, gpu_id=gpu_id):
            future.set_result(
                InferenceResult(
                    request_id=request.request_id,
                    error="Queue full",
                )
            )
        return future

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_request(self, request: InferenceRequest) -> InferenceResult:
        t_start = time.perf_counter()
        queue_wait_ms = (time.time_ns() - request.enqueue_time_ns) / 1e6

        try:
            loaded = self._registry.get(request.model_id)
            if loaded is None:
                return InferenceResult(
                    request_id=request.request_id,
                    error=f"Model {request.model_id} not loaded",
                    queue_wait_ms=queue_wait_ms,
                )

            gpu_id = loaded.placement.gpu_id
            device = torch.device(f"cuda:{gpu_id}")

            images = self._decode_frames(request, device)
            joint_tensor = self._prepare_joints(request, device)

            if loaded.drafter is not None:
                spec_result = self._spec_decoder.decode(
                    drafter=loaded.drafter,
                    verifier=loaded.model,
                    images=images,
                    joint_state=joint_tensor,
                    chunk_size=request.chunk_size or self._config.spec_decode.drafter_chunk_size,
                )
                action_chunk = spec_result.action_chunk
                used_spec = spec_result.used_speculative
                acceptance_rate = spec_result.acceptance_rate
            else:
                spec_result = self._spec_decoder._standard_decode(
                    loaded.model, images, joint_tensor, request.chunk_size
                )
                action_chunk = spec_result.action_chunk
                used_spec = False
                acceptance_rate = 1.0

            self._kv_cache.apply_sliding_window(request.session_id or request.request_id)

            inference_ms = (time.perf_counter() - t_start) * 1000
            return InferenceResult(
                request_id=request.request_id,
                action_chunk=action_chunk,
                used_speculative=used_spec,
                drafter_acceptance_rate=acceptance_rate,
                inference_time_ms=inference_ms,
                queue_wait_ms=queue_wait_ms,
            )

        except Exception as e:
            logger.exception("Inference error for %s", request.request_id)
            return InferenceResult(
                request_id=request.request_id,
                error=str(e),
                queue_wait_ms=queue_wait_ms,
            )
        finally:
            self._result_futures.pop(request.request_id, None)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _decode_frames(self, request: InferenceRequest, device: torch.device) -> torch.Tensor:
        """Decode all camera frames and video bursts into a batched image tensor."""
        frames_np = []

        for ef in request.camera_frames:
            arr = self._video_proc.decode_frame(ef)
            frames_np.append(arr)

        for burst in request.video_bursts:
            for ef in burst.frames:
                arr = self._video_proc.decode_frame(ef)
                frames_np.append(arr)

        if not frames_np:
            return torch.zeros(1, 3, 480, 640, device=device)

        import numpy as np

        stacked = np.stack(frames_np, axis=0)  # (N, H, W, C)
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
        return tensor.to(device)

    @staticmethod
    def _prepare_joints(request: InferenceRequest, device: torch.device) -> torch.Tensor:
        if request.joint_state is None:
            return torch.zeros(1, 14, device=device)
        positions = request.joint_state.positions
        return torch.tensor(positions, dtype=torch.float32, device=device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "running": self._running,
            "workers": len(self._workers),
            "pending_futures": len(self._result_futures),
            "scheduler": self._scheduler.stats(),
            "spec_decode": self._spec_decoder.stats,
            "kv_cache": self._kv_cache.stats(),
        }
