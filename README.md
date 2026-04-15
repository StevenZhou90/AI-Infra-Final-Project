# VLA Serving Platform

GPU-agnostic multi-model serving platform for 7B Vision-Language-Action (VLA) models with ACT-style action chunking, speculative decoding, and priority scheduling.

## Architecture

```
Isaac Sim (client)                          GPU Server
+-----------------------+                   +----------------------------------+
| Camera Capture (4cam) |                   | gRPC Service                     |
| Frame Encoder (JPEG/  | --- gRPC --->     | Priority Scheduler (not FIFO)    |
|   H.264/WebP)         |   bidirectional   | Video Decoder                    |
| Video Burst Batcher   |   streaming       | Spec-VLA Decoder                 |
| Action Buffer         | <--- gRPC ---     |   Drafter (~80M ACT)             |
| Temporal Ensembling   |   action chunks   |   Verifier (7B VLA)              |
+-----------------------+                   | KV Cache (90% util, sliding win) |
                                            | GPU Manager (auto-detect)        |
                                            +----------------------------------+
```

## Features

- **GPU-agnostic**: Auto-detects GPUs (A100, H100, H200, etc.) and adapts precision, memory budgets, and model placement
- **Priority scheduling**: Heap-based queue with CRITICAL/HIGH/NORMAL/LOW levels, age-boost starvation prevention, preemption
- **Spec-VLA speculative decoding**: Small ACT drafter proposes action chunks, large VLA verifier accepts/corrects with relaxed distance threshold
- **KV cache management**: Paged allocation at 90% GPU memory target with sliding window eviction
- **Video/image pipeline**: JPEG, WebP, H.264/H.265 encoding with adaptive quality and burst batching
- **Client-side buffering**: Temporal ensembling of overlapping action chunks with safe-stop fallback
- **Isaac Sim integration**: Camera capture, joint state reading, action application at 50Hz

## Quick Start

### Server

```bash
# Install dependencies
pip install -r requirements.txt

# Generate gRPC stubs
python -m grpc_tools.protoc -Iproto --python_out=proto --grpc_python_out=proto proto/inference.proto

# Run the server (auto-detects all GPUs)
python -m server.main --port 50051

# Or with specific GPUs
python -m server.main --gpu-ids 0,1,2,3 --kv-cache-target 0.90
```

### Docker

```bash
cd docker
docker compose up --build

# With specific GPU configuration
GPU_IDS=0,1,2,3 KV_CACHE_TARGET=0.90 docker compose up --build
```

### Client

```python
import asyncio
from client.client import VLAClient
from common.types import Priority

async def main():
    client = VLAClient(
        server_address="localhost:50051",
        model_id="openvla-7b",
        priority=Priority.HIGH,
    )
    await client.connect()

    # Single inference
    images = bridge.capture_cameras()
    joints = bridge.get_joint_state()
    result = await client.infer(images, joints)

    # Execute the action chunk
    for step in result.action_chunk.steps:
        bridge.apply_action(step)

    await client.disconnect()

asyncio.run(main())
```

## Configuration

The server can be configured via YAML, environment variables, or CLI flags:

```yaml
# config.yaml
gpu:
  target_gpu_ids: []  # empty = all GPUs
  kv_cache_target: 0.90
  memory_overhead_mb: 512

cache:
  page_size: 16
  sliding_window_size: 256

scheduler:
  age_boost_ms: 500
  max_queue_depth: 1024
  preempt_enabled: true

spec_decode:
  enabled: true
  acceptance_threshold: 0.15
  min_acceptance_rate: 0.3
  drafter_chunk_size: 90

video:
  default_codec: jpeg
  jpeg_quality: 85
  burst_size: 4

grpc:
  port: 50051
  max_message_size: 67108864  # 64MB
```

```bash
python -m server.main --config config.yaml
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## GPU Compatibility

| GPU | Memory | Precision | Status |
|-----|--------|-----------|--------|
| A100 | 80 GB HBM2e | FP16 | Supported (dev) |
| H100 | 80 GB HBM3 | BF16 | Supported |
| H200 | 141 GB HBM3e | BF16 | Supported (target) |

The platform auto-detects GPU type and selects optimal precision. Memory budgets are computed dynamically -- no hardcoded GPU assumptions.

## Project Structure

```
proto/inference.proto       # gRPC service definitions
server/
  main.py                   # Server entrypoint
  config.py                 # Pydantic configuration
  gpu_manager.py            # GPU auto-detection + model placement
  model_registry.py         # HF model loading (VLA + ACT drafter)
  kv_cache.py               # Paged KV cache, 90% target, sliding window
  scheduler.py              # Priority heap scheduler
  spec_decode.py            # Spec-VLA speculative decoding
  inference_engine.py       # Core inference loop
  grpc_service.py           # gRPC servicer implementations
  video_processor.py        # Server-side frame decoding
client/
  client.py                 # Async gRPC client
  encoder.py                # Video/image encoding
  video_burst.py            # Burst batching
  action_buffer.py          # Temporal ensembling buffer
  isaac_sim_bridge.py       # Isaac Sim integration
common/types.py             # Shared data types
tests/                      # Unit tests
docker/                     # Docker deployment
```
