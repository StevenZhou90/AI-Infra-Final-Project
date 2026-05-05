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
  --trajectory-head-checkpoint checkpoints/traj_head_dagger_r2/best.pt \
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

The benchmark below uses task-adaptive gates: **vertical=5, horizontal=6, standing=6**.

## DAgger data + training loop

The trajectory head uses standard DAgger. Each round rolls out the current fast
policy, labels those visited states with teacher OpenVLA actions, then retrains on
the aggregated dataset from all rounds. Aggregation matters because training only
on the newest rollout set overfits and forgets earlier teacher behavior.

```bash
./scripts/reproduce_readme_speedup_local.sh
```

This script regenerates:
- `data/trajectory_head_mini_r1`
- `data/trajectory_head_dagger_mini_r1`
- `data/trajectory_head_dagger_mini_r2`
- `data/trajectory_head_dagger_aggr_r2`
- `checkpoints/traj_head_dagger_r2/best.pt`
- `outputs/benchmarks/readme_speedup_<run_id>/summary.json`

## Weights

```bash
# Reuse the saved DAgger r2 head once it has been downloaded or regenerated.
uv run python scripts/run_openvla_sim.py \
  --decoder trajectory-spec \
  --trajectory-head-checkpoint checkpoints/traj_head_dagger_r2/best.pt \
  --trajectory-fast-draft-only \
  --trajectory-head-threshold 0.2 \
  --trajectory-fast-min-confident-tokens 5 \
  --task google_robot_pick_vertical_coke_can \
  --published-eval-setup \
  --episodes 3 \
  --steps 80
```

Upload target should include `best.pt`, `metrics.json`, and a short model card with
the benchmark matrix below. Keep tokens out of committed files and shell history.

## Verified benchmark snapshot

Matched benchmark matrix:
- tasks: vertical/horizontal/standing coke-can
- x positions: `-0.3500`, `-0.2925`, `-0.2350` (`obj_init_y = -0.02`)
- 3 episodes per setting (27 episodes total)
- hardware: 1x A100-SXM4-40GB, bfloat16

Reproduced results (`checkpoints/traj_head_dagger_r2/best.pt`):

| Decoder | Success | Avg ms/step | Speedup |
|---|---|---|---|
| Baseline OpenVLA | 14/27 (51.9%) | 295.2 | 1.00x |
| Adaptive fast policy (gate v=5, h=6, s=6) | **16/27 (59.3%)** | **105.0** | **2.81x** |

Per-task success for the adaptive fast policy:
- vertical: 3/9 (one full success cell at x=-0.2925)
- horizontal: 8/9
- standing: 5/9

Reproduce only the benchmark stage (with checkpoints already present):

```bash
sudo env HF_HOME=$PWD/.cache/huggingface MUJOCO_GL=osmesa \
  ./.venv/bin/python scripts/run_readme_speedup_matrix.py \
  --checkpoint checkpoints/traj_head_dagger_r2/best.pt \
  --output-dir outputs/benchmarks/readme_repro \
  --vertical-gate 5 --horizontal-gate 6 --standing-gate 6 \
  --head-threshold 0.2 \
  --steps 80 --episodes 3 \
  --device cuda --dtype bfloat16
```

## Useful scripts

- `scripts/run_openvla_sim.py`: main rollout + benchmark runner
- `scripts/run_published_sweep.py`: published-eval-style batch sweeps
- `scripts/generate_trajectory_head_data.py`: teacher rollout dataset generation
- `scripts/generate_trajectory_head_dagger_data.py`: DAgger rollout data collection
- `scripts/aggregate_trajectory_head_data.py`: DAgger dataset aggregation
- `scripts/train_trajectory_head.py`: draft head training
- `scripts/reproduce_readme_speedup_local.sh`: end-to-end reproduction script
- `scripts/run_readme_speedup_matrix.py`: benchmark matrix used for the README numbers
- `scripts/check_spec_exactness.py`: speculative exactness checks

## Spatial K/V + video compression benchmark

This benchmark isolates two non-speculative inference optimizations for video or
robot-camera streams:

- spatial K/V reuse: refresh only changed visual patches and reuse cached K/V for unchanged patches
- model-aware video compression: account for bandwidth when sending only changed patches

```bash
python scripts/benchmark_spatial_cache_compression.py \
  --frames 120 \
  --image-size 224 \
  --patch-size 16 \
  --hidden-dim 768 \
  --threshold 0.02 \
  --device cuda
```

Key outputs:
- `speedup`: full patch recomputation time divided by cached patch-refresh time
- `spatial_cache_reuse_ratio`: fraction of visual patches served from cache
- `video_compression_ratio`: raw frame bytes divided by changed-patch bytes
- `max_output_error`: output drift from stale cached patches, useful for threshold tuning

H200 synthetic benchmark snapshot with `--hidden-dim 4096`:
- baseline: `0.984 ms/frame`
- cached patch refresh: `0.688 ms/frame`
- speedup: `1.43x`
- spatial cache reuse: `94.4%`
- video compression ratio: `17.8x`
- max output error: `0.000147`

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
