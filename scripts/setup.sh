#!/usr/bin/env bash
set -euo pipefail

echo "=== ALOHA ACT Sim — Environment Setup ==="

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install system deps for headless MuJoCo rendering
echo "Installing EGL/OpenGL libraries for headless rendering..."
sudo apt-get update -qq
sudo apt-get install -y -qq libegl1 libopengl0 libgl1-mesa-glx

# Pin Python 3.10 and sync
echo "Syncing dependencies with uv (Python 3.10)..."
uv python install 3.10
uv sync --python 3.10

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run evaluation with:"
echo "  MUJOCO_GL=egl uv run python -m eval.run_rollout"
echo ""
echo "Or add to your shell profile:"
echo "  export MUJOCO_GL=egl"
