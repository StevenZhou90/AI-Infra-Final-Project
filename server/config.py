from __future__ import annotations

from pydantic import BaseModel, Field


class GpuConfig(BaseModel):
    """Per-GPU overrides (optional). When empty, auto-detect everything."""
    target_gpu_ids: list[int] = Field(
        default_factory=list,
        description="Restrict to these GPU IDs. Empty = use all detected GPUs.",
    )
    kv_cache_target: float = Field(
        default=0.90,
        ge=0.0,
        le=0.99,
        description="Fraction of free memory (after model weights) to allocate for KV cache.",
    )
    memory_overhead_mb: int = Field(
        default=512,
        description="Reserved headroom in MB per GPU for CUDA context, activations, etc.",
    )


class CacheConfig(BaseModel):
    page_size: int = Field(default=16, description="Tokens per KV cache page.")
    sliding_window_size: int = Field(
        default=256,
        description="Max context length in frames before oldest entries are evicted.",
    )
    evict_low_priority_first: bool = Field(
        default=True,
        description="When evicting, prefer caches belonging to LOW priority requests.",
    )


class SchedulerConfig(BaseModel):
    age_boost_ms: float = Field(
        default=500.0,
        description="After this wait time, a request's priority is bumped by one level.",
    )
    max_queue_depth: int = Field(
        default=1024,
        description="Maximum pending requests before rejecting new ones.",
    )
    preempt_enabled: bool = Field(
        default=True,
        description="Allow CRITICAL requests to preempt running NORMAL/LOW inference.",
    )


class SpecDecodeConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Spec-VLA speculative decoding.")
    acceptance_threshold: float = Field(
        default=0.15,
        description="Max relative distance between drafter and verifier actions for acceptance.",
    )
    min_acceptance_rate: float = Field(
        default=0.3,
        description="If drafter acceptance rate falls below this, fall back to standard inference.",
    )
    drafter_chunk_size: int = Field(
        default=90,
        description="Number of actions the drafter proposes per chunk (ACT default = 90).",
    )


class VideoConfig(BaseModel):
    default_codec: str = Field(default="jpeg", description="Default encoding codec.")
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    target_width: int = Field(default=640)
    target_height: int = Field(default=480)
    num_cameras: int = Field(default=4, description="Expected camera count (ALOHA = 4).")
    burst_size: int = Field(default=4, description="Default frames per video burst packet.")
    use_nvdec: bool = Field(default=True, description="Use GPU-accelerated video decode when available.")


class GrpcConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=50051)
    max_message_size: int = Field(
        default=64 * 1024 * 1024,
        description="Max gRPC message size in bytes (64MB default for video bursts).",
    )
    max_concurrent_rpcs: int = Field(default=128)


class ServerConfig(BaseModel):
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    spec_decode: SpecDecodeConfig = Field(default_factory=SpecDecodeConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    grpc: GrpcConfig = Field(default_factory=GrpcConfig)
    log_level: str = Field(default="INFO")

    @classmethod
    def from_yaml(cls, path: str) -> ServerConfig:
        import yaml  # noqa: F811

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Build config picking up env-var overrides (VLA_SERVING_ prefix)."""
        import os

        overrides: dict = {}
        port = os.getenv("VLA_SERVING_PORT")
        if port:
            overrides.setdefault("grpc", {})["port"] = int(port)
        kv_target = os.getenv("VLA_SERVING_KV_CACHE_TARGET")
        if kv_target:
            overrides.setdefault("gpu", {})["kv_cache_target"] = float(kv_target)
        gpu_ids = os.getenv("VLA_SERVING_GPU_IDS")
        if gpu_ids:
            overrides.setdefault("gpu", {})["target_gpu_ids"] = [
                int(x.strip()) for x in gpu_ids.split(",")
            ]
        return cls.model_validate(overrides)
