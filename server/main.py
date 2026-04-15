#!/usr/bin/env python3
"""VLA Serving Platform -- server entrypoint."""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import torch

from server.config import ServerConfig
from server.gpu_manager import GpuManager
from server.inference_engine import InferenceEngine
from server.kv_cache import KVCacheManager
from server.model_registry import ModelRegistry
from server.scheduler import PriorityScheduler
from server.spec_decode import SpeculativeDecoder

logger = logging.getLogger("vla_serving")


async def run(config: ServerConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    logger.info("Initializing VLA Serving Platform")
    logger.info("PyTorch %s  CUDA available: %s", torch.__version__, torch.cuda.is_available())

    # --- Core components ---
    gpu_manager = GpuManager(config.gpu)
    if gpu_manager.gpu_count == 0:
        logger.warning("No GPUs detected -- inference will run on CPU (slow)")

    logger.info("GPU summary: %s", gpu_manager.summary())

    model_registry = ModelRegistry(config, gpu_manager)
    scheduler = PriorityScheduler(config.scheduler)
    kv_cache = KVCacheManager(config.cache, gpu_manager)
    spec_decoder = SpeculativeDecoder(config.spec_decode)

    engine = InferenceEngine(
        config=config,
        gpu_manager=gpu_manager,
        model_registry=model_registry,
        scheduler=scheduler,
        kv_cache=kv_cache,
        spec_decoder=spec_decoder,
    )

    # --- KV cache init (default dims for 7B VLA) ---
    if gpu_manager.gpu_count > 0:
        dtype = torch.bfloat16 if gpu_manager.get_info(gpu_manager.gpu_ids[0]).supports_bf16 else torch.float16
        kv_cache.initialize(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            dtype=dtype,
        )

    # --- Start inference workers ---
    await engine.start()

    # --- gRPC server ---
    from server.grpc_service import serve as start_grpc

    grpc_server = await start_grpc(
        config=config,
        engine=engine,
        registry=model_registry,
        gpu_manager=gpu_manager,
        kv_cache=kv_cache,
        scheduler=scheduler,
    )

    logger.info("VLA Serving Platform ready")
    logger.info("  GPUs: %d", gpu_manager.gpu_count)
    logger.info("  gRPC: %s:%d", config.grpc.host, config.grpc.port)
    logger.info("  KV cache target: %.0f%%", config.gpu.kv_cache_target * 100)
    logger.info("  Spec decode: %s", "enabled" if config.spec_decode.enabled else "disabled")

    # --- Shutdown ---
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()

    logger.info("Shutting down...")
    await engine.stop()
    await grpc_server.stop(grace=5)
    logger.info("Shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA Serving Platform")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument("--port", type=int, default=None, help="gRPC port override")
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated GPU IDs")
    parser.add_argument(
        "--kv-cache-target",
        type=float,
        default=None,
        help="KV cache utilization target (0.0 - 0.99)",
    )
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        config = ServerConfig.from_yaml(args.config)
    else:
        config = ServerConfig.from_env()

    if args.port:
        config.grpc.port = args.port
    if args.gpu_ids:
        config.gpu.target_gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    if args.kv_cache_target is not None:
        config.gpu.kv_cache_target = args.kv_cache_target
    if args.log_level:
        config.log_level = args.log_level

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
