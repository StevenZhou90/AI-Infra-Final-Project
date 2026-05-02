# VLA Serving Platform

This repository provides an inference and evaluation stack for Vision-Language-Action
models, centered on OpenVLA in SimplerEnv.

Current focus:
- SimplerEnv-based OpenVLA evaluation
- Trajectory-head speculative decoding with confidence gating
- Reproducible benchmark and video outputs

## Quick start

Requirements:
- Ubuntu Linux
- NVIDIA GPU + working CUDA driver
- [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/StevenZhou90/AI-Infra-Final-Project.git
cd AI-Infra-Final-Project

# System libs for headless SimplerEnv/SAPIEN rendering
sudo apt-get update
sudo apt-get install -y libegl1 libopengl0 libgl1-mesa-glx libvulkan1 libglvnd-dev

# Python + project deps
uv python install 3.10
uv sync --python 3.10

# SimplerEnv
git clone https://github.com/simpler-env/SimplerEnv.git --recurse-submodules --depth 1 external/SimplerEnv
uv pip install -e external/SimplerEnv/ManiSkill2_real2sim
uv pip install -e external/SimplerEnv
```

## Run OpenVLA in SimplerEnv

```bash
uv run python scripts/run_openvla_sim.py \
  --task google_robot_pick_vertical_coke_can \
  --published-eval-setup \
  --episodes 3 \
  --steps 80
```

Videos are written under the `--output_dir` path (default: `outputs/openvla_sim`).

## Run fast speculative policy

This uses the learned trajectory head in fast mode with confidence gating.

```bash
uv run python scripts/run_openvla_sim.py \
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
- Lower gate (for example 4): more aggressive, faster, less reliable
- Higher gate (for example 6): more conservative, slower, often more reliable

## DAgger data + training loop

```bash
# Collect DAgger data from current fast policy
uv run python scripts/generate_trajectory_head_dagger_data.py \
  --policy-head-checkpoint checkpoints/traj_head_dagger_r1/best.pt \
  --sweep mini \
  --steps 80 \
  --out-dir data/trajectory_head_dagger_mini_r2 \
  --head-threshold 0.2 \
  --fast-min-confident-tokens 5 \
  --device cuda \
  --dtype bfloat16

# Train next head
uv run python scripts/train_trajectory_head.py \
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

## Verified benchmark snapshot

Matched benchmark matrix:
- tasks: vertical/horizontal/standing coke-can
- x positions: `-0.3500`, `-0.2925`, `-0.2350`
- 3 episodes per setting (27 episodes total)

Comparison:
- Baseline OpenVLA: **14/27 success, 302.5 ms/step**
- Adaptive fast policy (vertical gate5, horizontal gate6, standing gate6):
  **14/27 success, 145.1 ms/step**

That is approximately **52% lower latency** at the same total success count in this
evaluation slice.

## Useful scripts

- `scripts/run_openvla_sim.py`: main rollout + benchmark runner
- `scripts/run_published_sweep.py`: published-eval-style batch sweeps
- `scripts/run_libero_specvla_mirror.py`: paper-mirrored SpecVLA LIBERO AR-vs-Spec benchmark
- `scripts/run_libero_specvla_distributed.py`: multi-rank/multi-GPU sharded benchmark runner
- `scripts/summarize_libero_mirror.py`: summary/gate report for mirrored LIBERO runs
- `scripts/train_trajectory_heads_all_suites.py`: suite-wise head training orchestration + checkpoint index
- `scripts/train_spec_head_goal.sh`: one-command `libero_goal` head training
- `scripts/train_spec_heads_all.sh`: one-command all-suite head training
- `scripts/generate_trajectory_head_data.py`: teacher rollout dataset generation
- `scripts/generate_trajectory_head_dagger_data.py`: DAgger rollout data collection
- `scripts/train_trajectory_head.py`: draft head training
- `scripts/check_spec_exactness.py`: speculative exactness checks
- `scripts/debug_depth2_verify.py`: debug utility for depth>1 verification mismatch

## SpecVLA-mirrored LIBERO benchmark

This mirrors the Spec-VLA protocol shape while using your own verifier+spec checkpoints.

Protocol target:
- suites: `libero_goal`, `libero_object`, `libero_spatial`, `libero_10` (Long)
- tasks per suite: `10`
- trials per task: `50`

1) Fill checkpoint paths in `configs/libero_specvla_mirror.yaml`.
   - Local filesystem paths and Hugging Face repo IDs are both supported.

2) Smoke run (with gate check):

```bash
uv run python scripts/run_libero_specvla_mirror.py \
  --config configs/libero_specvla_mirror.yaml \
  --mode smoke
```

3) Smoke then auto-launch full when gate passes:

```bash
uv run python scripts/run_libero_specvla_mirror.py \
  --config configs/libero_specvla_mirror.yaml \
  --mode smoke \
  --auto-full-after-smoke
```

4) Summarize a run stage:

```bash
uv run python scripts/summarize_libero_mirror.py \
  --run-dir outputs/libero_specvla_mirror/<run_id>/smoke
```

Default smoke gate:
- speedup > `1.0x`
- success drop (AR - Spec) <= `0.0`

Long-run progress checkpoints:
- The runner writes periodic aggregate snapshots to
  `outputs/libero_specvla_mirror/<run_id>/<mode>/progress.log.jsonl`.
- Interval is controlled by `benchmark.progress_log_seconds` in
  `configs/libero_specvla_mirror.yaml` (default `3600` seconds).

## Distributed single-node benchmark (8xH200-ready)

Config:
- `configs/libero_specvla_distributed.yaml`

Single process smoke dry-run:

```bash
python scripts/run_libero_specvla_distributed.py \
  --config configs/libero_specvla_distributed.yaml \
  --mode smoke \
  --dry-run
```

8-process launch (single node):

```bash
torchrun --standalone --nproc_per_node=8 scripts/run_libero_specvla_distributed.py \
  --config configs/libero_specvla_distributed.yaml \
  --mode smoke
```

Continue same run id with full mode:

```bash
torchrun --standalone --nproc_per_node=8 scripts/run_libero_specvla_distributed.py \
  --config configs/libero_specvla_distributed.yaml \
  --mode full \
  --run-id <run_id_from_smoke>
```

Distributed summary:

```bash
python scripts/summarize_libero_mirror.py \
  --run-dir outputs/libero_specvla_distributed/<run_id>/smoke \
  --distributed
```

Progress/checkpoint logs:
- rank-local: `progress.rank{N}.jsonl`
- merged: `progress.global.jsonl`
- run checkpoint index: `outputs/libero_specvla_distributed/<run_id>/checkpoints/index.json`

## Suite head orchestration

Command-template orchestration (configure templates first in
`configs/libero_specvla_distributed.yaml`):

```bash
python scripts/train_trajectory_heads_all_suites.py \
  --config configs/libero_specvla_distributed.yaml \
  --dry-run
```

This writes an index of generated/trained artifacts at:
- `artifacts/checkpoints/index.json`

Quick-start training launchers:

```bash
# single suite
./scripts/train_spec_head_goal.sh

# all configured suites
./scripts/train_spec_heads_all.sh
```

Auto-upload to Hugging Face:
- Controlled by `hf_upload` in `configs/libero_specvla_distributed.yaml`.
- When enabled, each successful training round uploads checkpoint files from:
  - `artifacts/checkpoints/<suite>/r<round>/`
  into:
  - `<hf_repo_id>/<suite>/r<round>/`

## Project layout

```text
envs/       Sim environment wrappers (SimplerEnv primary)
policies/   Policy wrappers (OpenVLA, ACT)
serving/    Decoders, draft heads, and gRPC serving stack
scripts/    Data generation, training, evaluation, benchmarking
eval/       Legacy evaluation entrypoints
proto/      Protobuf service definitions and stubs
configs/    YAML configuration defaults
```
