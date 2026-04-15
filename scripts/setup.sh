#!/usr/bin/env bash
set -euo pipefail

echo "=== ALOHA ACT Sim — Setup ==="

# System libs for headless MuJoCo rendering
echo "Installing EGL/OpenGL libraries..."
sudo apt-get update -qq
sudo apt-get install -y -qq libegl1 libopengl0 libgl1-mesa-glx

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Python + deps
echo "Installing Python 3.10 + dependencies..."
uv python install 3.10
uv sync --python 3.10

echo ""
echo "Done. Run with:  uv run python -m eval.run_rollout"
