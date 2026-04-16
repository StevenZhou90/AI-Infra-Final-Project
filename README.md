# VLA Serving Platform

Multi-model inference serving platform for Vision-Language-Action (VLA) models. Currently running [ACT](https://tonyzhaozh.github.io/aloha/) policies in ALOHA sim, designed to scale to 7B VLA models on multi-GPU setups.

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

### Multi-model

```bash
# Load multiple models on the server via the client
uv run python -c "
from serving.grpc_client import InferenceClient
c = InferenceClient()
c.load_model('act-cube', 'lerobot/act_aloha_sim_transfer_cube_human')
c.load_model('act-insert', 'lerobot/act_aloha_sim_insertion_human')
print(c.list_models())
"
```

## Project Structure

```
envs/       — Sim environment wrappers (gym-aloha / MuJoCo)
policies/   — Policy loading + inference (ACT via LeRobot)
eval/       — Direct rollout runner (no server needed)
serving/    — gRPC server, client, model registry
proto/      — Protobuf service definitions + generated stubs
configs/    — YAML defaults
scripts/    — Setup helpers
```

## Architecture

```
Client (sim machine)              Server (GPU machine)
┌─────────────────┐              ┌──────────────────────┐
│  Isaac Sim /     │   gRPC      │  Model Registry      │
│  MuJoCo          │────────────▶│  ┌─ GPU 0: Model A   │
│                  │  JPEG imgs  │  ├─ GPU 1: Model B   │
│  Video Encoder   │  + state    │  └─ GPU N: Model C   │
│  Action Buffer   │◀────────────│                      │
│                  │   actions   │  Priority Scheduler   │
└─────────────────┘              └──────────────────────┘
```

## What's built

- [x] ACT policy inference with normalization handling
- [x] ALOHA sim environment (TransferCube, Insertion)
- [x] Video recording (mp4)
- [x] gRPC serving layer (Predict, LoadModel, UnloadModel, ListModels, GetStatus)
- [x] JPEG video encoding over gRPC
- [x] Model registry with multi-model + GPU memory tracking
- [ ] Priority scheduler (not FIFO)
- [ ] KV cache manager (90% utilization, sliding window)
- [ ] Speculative decoding
- [ ] Client-side action buffer
