# VLA Serving Platform

Multi-model inference serving platform for Vision-Language-Action (VLA) models. Supports [ACT](https://tonyzhaozh.github.io/aloha/) (single-pass) and [OpenVLA](https://openvla.github.io/) 7B (autoregressive) policies with KV cache management and self-speculative decoding. Targeting GH200 for unified CPU/GPU memory.

## Setup (fresh machine)

Requires: Ubuntu with an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/) installed.

```bash
# 1. Clone
git clone https://github.com/StevenZhou90/AI-Infra-Final-Project.git
cd AI-Infra-Final-Project

# 2. System libs for headless rendering
sudo apt-get update && sudo apt-get install -y libegl1 libopengl0 libgl1-mesa-glx

# 3. Python 3.10 + dependencies
uv python install 3.10
uv sync --python 3.10
```

## Running

### Direct evaluation (no server)

```bash
uv run python -m eval.run_rollout
uv run python -m eval.run_rollout --episodes 10 --task AlohaInsertion-v0
```

### Client-server mode (gRPC)

Terminal 1 — start the inference server:
```bash
uv run python -m serving.grpc_server
```

Terminal 2 — run the client (sim + video):
```bash
uv run python -m serving.grpc_client --episodes 5
```

The client runs the sim, encodes camera frames as JPEG, sends them to the server over gRPC, and receives actions back. Works on `localhost` for dev, or across machines in production.

### Loading OpenVLA 7B (with KV cache + speculative decoding)

```bash
uv run python -c "
from serving.grpc_client import InferenceClient
c = InferenceClient()
c.load_model(
    'openvla-7b', 'openvla/openvla-7b',
    model_type='openvla',
    use_kv_cache=True,
    use_speculative_decoding=True,
)
print(c.list_models())
"
```

### Multi-model (ACT + OpenVLA)

```bash
uv run python -c "
from serving.grpc_client import InferenceClient
c = InferenceClient()
c.load_model('act-cube', 'lerobot/act_aloha_sim_transfer_cube_human')
c.load_model('openvla-7b', 'openvla/openvla-7b', model_type='openvla',
             use_kv_cache=True, use_speculative_decoding=True)
print(c.list_models())
"
```

## Project Structure

```
envs/       — Sim environment wrappers (gym-aloha / MuJoCo)
policies/   — Policy wrappers (ACT single-pass, OpenVLA 7B autoregressive)
eval/       — Direct rollout runner (no server needed)
serving/    — gRPC server, client, model registry, KV cache, speculative decoder
proto/      — Protobuf service definitions + generated stubs
configs/    — YAML defaults
scripts/    — Setup helpers
```

## Architecture

```
Client (sim machine)              Server (GPU machine / GH200)
┌─────────────────┐              ┌──────────────────────────────┐
│  Isaac Sim /     │   gRPC      │  Model Registry              │
│  MuJoCo          │────────────▶│  ┌─ ACT (single-pass)       │
│                  │  JPEG imgs  │  └─ OpenVLA 7B (autoregress) │
│  Video Encoder   │  + state    │                              │
│  Action Buffer   │  + instruct │  ┌─ KV Cache Manager ───┐   │
│                  │◀────────────│  │  prefix reuse         │   │
│                  │   actions   │  │  90% mem budget        │   │
└─────────────────┘              │  │  sliding window        │   │
                                 │  └───────────────────────┘   │
                                 │  ┌─ Speculative Decoder ─┐   │
                                 │  │  self-spec (layer skip)│   │
                                 │  │  draft → verify loop   │   │
                                 │  │  greedy acceptance     │   │
                                 │  └───────────────────────┘   │
                                 │  Priority Scheduler          │
                                 └──────────────────────────────┘
```

## What's built

- [x] ACT policy inference with normalization handling
- [x] OpenVLA 7B policy wrapper (autoregressive action token generation)
- [x] KV cache manager (90% utilization target, prefix reuse, sliding window, LRU eviction)
- [x] Self-speculative decoding (layer-skip draft, greedy verification)
- [x] ALOHA sim environment (TransferCube, Insertion)
- [x] Video recording (mp4)
- [x] gRPC serving layer (Predict, LoadModel, UnloadModel, ListModels, GetStatus)
- [x] JPEG video encoding over gRPC
- [x] Model registry with multi-model + GPU memory tracking
- [x] Priority scheduler
- [x] Client-side action buffer
