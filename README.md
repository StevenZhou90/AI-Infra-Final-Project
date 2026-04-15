# ALOHA ACT Sim MVP

Pre-trained [ACT](https://tonyzhaozh.github.io/aloha/) policy running in the ALOHA bimanual manipulation sim. Foundation for a multi-model VLA serving platform.

## Setup (fresh machine)

Requires: Ubuntu with an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/) installed.

```bash
# 1. Clone the repo
git clone https://github.com/StevenZhou90/AI-Infra-Final-Project.git
cd AI-Infra-Final-Project

# 2. Install system libs for headless rendering
sudo apt-get update && sudo apt-get install -y libegl1 libopengl0 libgl1-mesa-glx

# 3. Install Python 3.10 + all dependencies
uv python install 3.10
uv sync --python 3.10

# 4. Run it (first run downloads the model from HuggingFace, ~200MB)
uv run python -m eval.run_rollout
```

Videos are saved to `outputs/eval/`.

## Options

```bash
uv run python -m eval.run_rollout --episodes 50         # more episodes
uv run python -m eval.run_rollout --episodes 50 --no-video  # skip video recording (faster)
uv run python -m eval.run_rollout --task AlohaInsertion-v0   # harder task
uv run python -m eval.run_rollout --model lerobot/act_aloha_sim_insertion_human  # different model
```

## Project Structure

```
envs/       — Sim environment wrapper (gym-aloha / MuJoCo ALOHA robot)
policies/   — ACT policy loader with normalization handling
eval/       — Rollout runner and video recording
configs/    — YAML defaults for environment and policy settings
scripts/    — Setup helpers
```
