# VLA Multi-Serving Platform

This repository packages a multi-service inference platform for robotic-arm
Vision-Language-Action (VLA) policies. It combines OpenVLA/SimplerEnv rollout
evaluation, SpecVLA-style trajectory drafting, pi0-FAST chunk execution, gRPC
serving, GPU-aware routing, admission control, and benchmark tooling.

## Highlights

- Multi-model serving for OpenVLA-style policies and pi0-FAST experiments.
- gRPC inference API with protobuf stubs in `proto/`.
- GPU-aware pi0.5 cluster router with warmup, admission control, and profiling.
- Speculative trajectory heads for lower-latency robotic-arm control loops.
- pi0-FAST token/chunk hooks for target-equivalent serving experiments.
- LIBERO and SimplerEnv benchmark runners with reproducible result summaries.

## Verified Results

### SimplerEnv OpenVLA Coke-Can Matrix

Matched benchmark matrix:
- tasks: vertical, horizontal, standing coke-can
- x positions: `-0.3500`, `-0.2925`, `-0.2350`
- episodes: `27`

| Decoder | Success | Avg ms/step | Speedup |
| --- | ---: | ---: | ---: |
| Baseline OpenVLA | `14/27` | `302.5` | `1.00x` |
| Adaptive fast policy | `14/27` | `145.1` | `2.08x` |

The adaptive fast policy keeps the same success count while reducing per-step
latency by about 52%.

### LIBERO Goal SpecVLA-Style Slice

Selected-slice protocol:
- suite: `libero_goal`
- task ids: `0, 1, 3, 5, 7, 9`
- trials per task: `5`
- total episodes: `30`
- base policy: `openvla/openvla-7b-finetuned-libero-goal`

| Decoder | Success | Avg ms/step | Speedup vs AR |
| --- | ---: | ---: | ---: |
| AR OpenVLA | `22/30` | `317.6` | `1.00x` |
| SpecVLA-style tuned chunk | `23/30` | `280.7` | `1.13x` |
| Adaptive direct K=2 | `21/30` | `202.7` | `1.57x` |
| Two-head direct K=3/K=2, strict smooth | `22/30` | `213.5` | `1.49x` |
| Two-head direct K=3/K=2, loose smooth | `23/30` | `202.0` | `1.57x` |

The best no-task-router configuration is the loose-smooth two-head direct chunk
decoder: K=3 smooth head, K=2 complex head, and relaxed smooth-phase thresholds.

## Quick Start

Requirements:
- Ubuntu Linux
- NVIDIA GPU with a working CUDA driver
- Python 3.10
- `uv`

```bash
git clone https://github.com/StevenZhou90/AI-Infra-Final-Project.git
cd AI-Infra-Final-Project

sudo apt-get update
sudo apt-get install -y libegl1 libopengl0 libgl1-mesa-glx libvulkan1 libglvnd-dev

uv python install 3.10
uv sync --python 3.10

git clone https://github.com/simpler-env/SimplerEnv.git --recurse-submodules --depth 1 external/SimplerEnv
uv pip install -e external/SimplerEnv/ManiSkill2_real2sim
uv pip install -e external/SimplerEnv
```

Use GPUs 0-2 for local benchmark and serving runs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
```

## Run Baseline OpenVLA

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_openvla_sim.py \
  --task google_robot_pick_vertical_coke_can \
  --published-eval-setup \
  --episodes 3 \
  --steps 80
```

Videos are written under `outputs/openvla_sim` unless `--output_dir` is set.

## Run Fast Speculative Policy

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_openvla_sim.py \
  --decoder trajectory-spec \
  --trajectory-head-checkpoint checkpoints/traj_head_dagger_r1/best.pt \
  --trajectory-fast-draft-only \
  --trajectory-head-threshold 0.2 \
  --trajectory-fast-min-confident-tokens 5 \
  --task google_robot_pick_vertical_coke_can \
  --published-eval-setup \
  --episodes 3 \
  --steps 80
```

Gate interpretation:
- Lower gate: more aggressive, faster, less reliable.
- Higher gate: more conservative, slower, often more reliable.

## Run gRPC Serving

The pi0.5 gRPC service exposes policy inference over `proto/inference.proto` and
uses the serving runtime, router, and admission-control layers.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python -m serving.pi05_server \
  --host 0.0.0.0 \
  --port 50051 \
  --devices cuda:0,cuda:1,cuda:2 \
  --max-concurrent 12 \
  --warmup-steps 2
```

Load test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python scripts/load_pi05_grpc.py \
  --target localhost:50051 \
  --clients 16 \
  --requests 128
```

Profile serving:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python scripts/profile_pi05_serving.py \
  --requests 64 \
  --concurrency 8
```

## Run LIBERO SpecVLA Benchmark

Smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_libero_specvla_mirror.py \
  --config configs/libero_specvla_mirror.yaml \
  --mode smoke
```

Full run after smoke:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_libero_specvla_mirror.py \
  --config configs/libero_goal_selected_direct_twohead_k3k2_loose_smooth.yaml \
  --mode full
```

Distributed single-node run on GPUs 0-2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  scripts/run_libero_specvla_distributed.py \
  --config configs/libero_specvla_distributed.yaml \
  --mode smoke
```

Summarize:

```bash
uv run python scripts/summarize_libero_mirror.py \
  --run-dir outputs/libero_specvla_mirror/<run_id>/smoke
```

## Run pi0-FAST Chunk Serving Experiments

The pi0-FAST path uses public LeRobot/LIBERO weights and gated PaliGemma
tokenizer access. Authenticate with Hugging Face before model loading.

```bash
python -m venv --system-site-packages .venv-pi
.venv-pi/bin/python -m pip install -U pip setuptools wheel
.venv-pi/bin/python -m pip install -e . --no-deps
.venv-pi/bin/python -m pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@v0.4.4"
.venv-pi/bin/python -m pip install hf-libero==0.1.3 --no-deps
.venv-pi/bin/python -m pip install hydra-core robomimic==0.2.0 robosuite==1.4.0 bddl==1.0.1 easydict thop mujoco tensorboardX imageio-ffmpeg egl_probe numba jupytext pytest
.venv-pi/bin/python -m pip install "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12" "matplotlib>=3.5.3" hf-egl-probe

CUDA_VISIBLE_DEVICES=0 HF_HOME=.hf_cache MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=egl \
.venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
  --policy lerobot/pi0fast-libero \
  --task libero_object \
  --task-id 0 \
  --episodes 3 \
  --steps 300 \
  --modes baseline,chunk_m3,chunk_m5_smooth,relaxed_chunk_retrieval_m3,exact_fast_sd_retrieval \
  --device cuda \
  --dtype bfloat16 \
  --enable-fast-token-hooks \
  --output-dir outputs/pi0fast_chunk/libero_object_task0
```

The runner reports success, model calls per control step, average ms/control
step, accepted execution windows, FAST token counts, fallback reasons, and
speedup versus baseline.

## Training and Data Generation

Collect DAgger data:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_trajectory_head_dagger_data.py \
  --policy-head-checkpoint checkpoints/traj_head_dagger_r1/best.pt \
  --sweep mini \
  --steps 80 \
  --out-dir data/trajectory_head_dagger_mini_r2 \
  --head-threshold 0.2 \
  --fast-min-confident-tokens 5 \
  --device cuda \
  --dtype bfloat16
```

Train the next trajectory head:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_trajectory_head.py \
  --data-dir data/trajectory_head_dagger_mini_r2 \
  --out-dir checkpoints/traj_head_dagger_r2 \
  --epochs 80 \
  --batch-size 128 \
  --hidden-dim 1024 \
  --embed-dim 128 \
  --hidden-fusion-dim 512 \
  --num-layers 3 \
  --lr 2e-4 \
  --dim-weights 1,1,1.5,2,2,2,5 \
  --change-weight 2.0 \
  --gripper-change-weight 8.0 \
  --late-timestep 20 \
  --late-weight 1.5 \
  --device cuda
```

Suite orchestration:

```bash
./scripts/train_spec_head_goal.sh
./scripts/train_spec_heads_all.sh
```

## Useful Scripts

- `scripts/run_openvla_sim.py`: SimplerEnv rollout and benchmark runner.
- `scripts/run_published_sweep.py`: published-evaluation-style sweeps.
- `scripts/run_libero_specvla_mirror.py`: SpecVLA-style LIBERO benchmark.
- `scripts/run_libero_specvla_distributed.py`: multi-GPU LIBERO runner.
- `scripts/run_pi0fast_chunk_eval.py`: pi0-FAST LIBERO chunk experiments.
- `scripts/benchmark_pi0fast_serving_runtime.py`: pi0-FAST runtime benchmark.
- `scripts/load_pi05_grpc.py`: gRPC service load test.
- `scripts/profile_pi05_serving.py`: serving profiler.
- `scripts/train_trajectory_head.py`: draft head training.
- `scripts/check_spec_exactness.py`: speculative exactness checks.
- `scripts/benchmark_spatial_cache_compression.py`: spatial K/V cache benchmark.

## Project Layout

```text
configs/    YAML configuration defaults and benchmark recipes
envs/       Simulation environment wrappers
eval/       Legacy evaluation entrypoints
policies/   Policy wrappers for OpenVLA and ACT
proto/      Protobuf service definitions and generated stubs
scripts/    Data generation, training, evaluation, load, and benchmark tools
serving/    Runtime services, routers, decoders, draft heads, and gRPC stack
tests/      Unit tests for serving, routing, decoding, and benchmark helpers
```

## Packaging Notes

`pyproject.toml` packages `envs`, `policies`, `eval`, `serving`, and `proto`.
Large generated artifacts are intentionally ignored: checkpoints, logs, datasets,
external simulator clones, Hugging Face caches, and local virtual environments.
