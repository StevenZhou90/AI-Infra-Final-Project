from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class Priority(enum.IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Codec(enum.Enum):
    RAW = "raw"
    JPEG = "jpeg"
    WEBP = "webp"
    H264 = "h264"
    H265 = "h265"


class Precision(enum.Enum):
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"


@dataclass
class EncodedFrame:
    data: bytes
    codec: Codec
    width: int
    height: int
    channels: int = 3
    camera_id: str = ""
    timestamp_ns: int = 0


@dataclass
class VideoBurst:
    frames: list[EncodedFrame]
    camera_id: str = ""
    start_timestamp_ns: int = 0
    end_timestamp_ns: int = 0


@dataclass
class JointState:
    positions: list[float]
    velocities: list[float] = field(default_factory=list)
    efforts: list[float] = field(default_factory=list)
    timestamp_ns: int = 0


@dataclass
class ActionStep:
    joint_targets: list[float]
    gripper_targets: list[float] = field(default_factory=list)


@dataclass
class ActionChunk:
    steps: list[ActionStep]
    start_timestamp_ns: int = 0
    frequency_hz: float = 50.0
    accepted_length: int = 0
    confidence: float = 1.0


@dataclass
class InferenceRequest:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    priority: Priority = Priority.NORMAL
    camera_frames: list[EncodedFrame] = field(default_factory=list)
    video_bursts: list[VideoBurst] = field(default_factory=list)
    joint_state: JointState | None = None
    chunk_size: int = 90
    session_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    enqueue_time_ns: int = field(default_factory=lambda: time.time_ns())
    _seq: int = 0


@dataclass
class InferenceResult:
    request_id: str = ""
    action_chunk: ActionChunk | None = None
    used_speculative: bool = False
    drafter_acceptance_rate: float = 0.0
    inference_time_ms: float = 0.0
    queue_wait_ms: float = 0.0
    error: str = ""


@dataclass
class GpuInfo:
    gpu_id: int
    name: str
    total_memory: int
    compute_capability: tuple[int, int]
    supports_bf16: bool

    @property
    def preferred_precision(self) -> Precision:
        if self.supports_bf16:
            return Precision.BF16
        return Precision.FP16


@dataclass
class GpuMemoryState:
    gpu_id: int
    total: int
    used: int
    model_weights: int = 0
    kv_cache: int = 0
    overhead: int = 0

    @property
    def free(self) -> int:
        return self.total - self.used

    @property
    def kv_utilization(self) -> float:
        available_for_kv = self.total - self.model_weights - self.overhead
        if available_for_kv <= 0:
            return 0.0
        return self.kv_cache / available_for_kv


@dataclass
class ModelPlacement:
    model_id: str
    gpu_id: int
    precision: Precision
    memory_bytes: int
    drafter_model_id: str | None = None
    hf_repo: str = ""
    param_count: int = 0
    is_loaded: bool = False


@dataclass
class KVCachePage:
    page_id: int
    gpu_id: int
    block_size: int
    key_data: Any = None
    value_data: Any = None
    request_id: str = ""
    seq_position: int = 0
    last_access_ns: int = 0
    is_free: bool = True
