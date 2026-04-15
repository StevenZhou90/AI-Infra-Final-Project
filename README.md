# ALOHA ACT Sim MVP

Run a pre-trained [ACT](https://tonyzhaozh.github.io/aloha/) policy in the ALOHA bimanual manipulation sim and watch rollout videos.

This is the foundation for a multi-model VLA serving platform with priority queuing, speculative decoding, and KV cache management — that comes next once this pipeline works end-to-end.

## Quick Start

```bash
# One-time setup
bash scripts/setup.sh

# Run 5 episodes with video recording
uv run python -m eval.run_rollout

# Run more episodes, custom model, no video
uv run python -m eval.run_rollout --episodes 50 --no-video --device cuda
```

## Project Structure

```
envs/           Environment wrappers (gym-aloha today, Isaac Lab later)
policies/       Policy loading and inference (ACT via LeRobot)
eval/           Evaluation scripts and rollout runner
configs/        YAML configs for environments and policies
scripts/        Setup and utility scripts
```

## Configuration

Override defaults via CLI flags or YAML config:

```bash
uv run python -m eval.run_rollout \
    --config configs/env/aloha_default.yaml \
    --model lerobot/act_aloha_sim_transfer_cube_human \
    --task AlohaTransferCube-v0 \
    --episodes 10 \
    --device cuda \
    --output-dir outputs/eval
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~4 GB disk for model weights + MuJoCo assets
