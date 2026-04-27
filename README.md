# VLA Serving Platform

Multi-model inference serving platform for Vision-Language-Action (VLA) models. Supports [OpenVLA](https://openvla.github.io/) 7B on [SimplerEnv](https://github.com/simpler-env/SimplerEnv) tasks, plus legacy [ACT](https://tonyzhaozh.github.io/aloha/) ALOHA evaluation. Includes KV cache management, self-speculative decoding, and **EAGLE speculative decoding** for 1.3x inference speedup. Targeting GH200 for unified CPU/GPU memory.

## Setup (fresh machine)

Requires: Ubuntu with an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/) installed.

```bash
# 1. Clone
git clone https://github.com/StevenZhou90/AI-Infra-Final-Project.git
cd AI-Infra-Final-Project

# 2. System libs for headless SimplerEnv/SAPIEN rendering
sudo apt-get update && sudo apt-get install -y libegl1 libopengl0 libgl1-mesa-glx libvulkan1 libglvnd-dev

# 3. Python 3.10 + dependencies
uv python install 3.10
uv sync --python 3.10

# 4. SimplerEnv + ManiSkill2 real-to-sim
git clone https://github.com/simpler-env/SimplerEnv.git --recurse-submodules --depth 1 external/SimplerEnv
uv pip install -e external/SimplerEnv/ManiSkill2_real2sim
uv pip install -e external/SimplerEnv
```

## Running

### OpenVLA SimplerEnv rollout

```bash
uv run python scripts/run_openvla_sim.py --episodes 3 --steps 50
uv run python scripts/run_openvla_sim.py --task widowx_spoon_on_towel --episodes 3 --steps 50
```

Primary zero-shot SimplerEnv tasks:
- `google_robot_pick_coke_can`
- `google_robot_pick_object`
- `google_robot_move_near`
- `google_robot_open_drawer`
- `google_robot_close_drawer`
- `google_robot_place_in_closed_drawer`
- `widowx_spoon_on_towel`
- `widowx_carrot_on_plate`
- `widowx_stack_cube`
- `widowx_put_eggplant_in_basket`

### Legacy ACT ALOHA eval

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

### OpenVLA SimplerEnv rollout with EAGLE speculative decoding

```bash
# Run OpenVLA in SimplerEnv with EAGLE acceleration + video recording
python scripts/run_openvla_sim.py \
    --task google_robot_pick_coke_can --episodes 3 --steps 50 \
    --eagle_dir checkpoints/eagle_openvla_v2

# Baseline only (no EAGLE)
python scripts/run_openvla_sim.py --episodes 3 --steps 50
```

Videos saved to `outputs/openvla_sim/`.

### EAGLE training pipeline

```bash
# 1. Generate training data (hidden states from OpenVLA)
python scripts/generate_eagle_data.py --num_samples 10000 --out_dir data/eagle_train_v2

# 2. Train EAGLE draft head (2-layer, ~570M params)
python scripts/train_eagle.py \
    --data_dir data/eagle_train_v2 \
    --out_dir checkpoints/eagle_openvla_v2 \
    --epochs 50 --batch_size 8 --num_layers 2

# 3. Benchmark
python scripts/benchmark_eagle.py \
    --eagle_dir checkpoints/eagle_openvla_v2 \
    --num_samples 100
```

## Project Structure

```
envs/       — Sim environment wrappers (SimplerEnv primary, gym-aloha legacy)
policies/   — Policy wrappers (ACT single-pass, OpenVLA 7B autoregressive)
eval/       — Direct rollout runner (no server needed)
serving/    — gRPC server, client, model registry, KV cache, speculative decoder,
              EAGLE draft head
proto/      — Protobuf service definitions + generated stubs
configs/    — YAML defaults
scripts/    — EAGLE training pipeline, sim rollout, setup helpers
```

## Architecture

```
Client (sim machine)              Server (GPU machine / GH200)
┌─────────────────┐              ┌──────────────────────────────┐
│  SimplerEnv /    │   gRPC      │  Model Registry              │
│  MuJoCo          │────────────▶│  ┌─ ACT (single-pass)       │
│                  │  JPEG imgs  │  └─ OpenVLA 7B (autoregress) │
│  Video Encoder   │  + state    │                              │
│  Action Buffer   │  + instruct │  ┌─ KV Cache Manager ───┐   │
│                  │◀────────────│  │  prefix reuse         │   │
│                  │   actions   │  │  90% mem budget        │   │
└─────────────────┘              │  │  sliding window        │   │
                                 │  └───────────────────────┘   │
                                 │  ┌─ Speculative Decoder ─┐   │
                                 │  │  EAGLE draft head      │   │
                                 │  │  self-spec (layer skip)│   │
                                 │  │  draft → verify loop   │   │
                                 │  └───────────────────────┘   │
                                 │  Priority Scheduler          │
                                 └──────────────────────────────┘
```

## EAGLE Speculative Decoding Results

Trained an EAGLE-1 draft head on OpenVLA's hidden state distribution to accelerate autoregressive action token generation.

| Method | Latency (ms) | Speedup | Acceptance Rate |
|--------|-------------|---------|-----------------|
| Baseline (autoregressive) | 221 ms | 1.0x | — |
| Layer-skip (16L draft) | 261 ms | 0.85x | 14% |
| Pre-trained EAGLE (Llama-2-Chat) | 190 ms | 0.95x | 14% |
| **Trained EAGLE (OpenVLA)** | **169 ms** | **1.31x** | **60%** |

Key improvements over baseline speculative decoding:
- **Domain-matched training**: EAGLE head trained on OpenVLA's own hidden states (not Llama-2-Chat)
- **Batched hidden states**: Training data uses verify-pass-style batched forwards, eliminating the extra forward pass at inference
- **2-layer draft head**: 570M parameter EAGLE head with increased capacity
- **Correct KV management**: Fixed verify-pass KV trimming to avoid cache corruption on draft rejection

## What's built

- [x] ACT policy inference with normalization handling
- [x] OpenVLA 7B policy wrapper (autoregressive action token generation)
- [x] KV cache manager (90% utilization target, prefix reuse, sliding window, LRU eviction)
- [x] Self-speculative decoding (layer-skip draft, greedy verification)
- [x] **EAGLE speculative decoding** (trained draft head, 1.31x speedup)
- [x] **EAGLE training pipeline** (data generation, training, benchmarking)
- [x] SimplerEnv environment adapter for OpenVLA-compatible 7-DOF tasks
- [x] Legacy ALOHA sim environment (TransferCube, Insertion)
- [x] OpenVLA SimplerEnv rollout with video recording
- [x] Video recording (mp4)
- [x] gRPC serving layer (Predict, LoadModel, UnloadModel, ListModels, GetStatus)
- [x] JPEG video encoding over gRPC
- [x] Model registry with multi-model + GPU memory tracking
- [x] Priority scheduler
- [x] Client-side action buffer
