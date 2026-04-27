#!/usr/bin/env bash
set -euo pipefail

echo "=== VLA Serving Platform — Setup ==="

# System libs for headless rendering and SAPIEN/ManiSkill2.
echo "Installing EGL/OpenGL/Vulkan libraries..."
sudo apt-get update -qq
sudo apt-get install -y -qq libegl1 libopengl0 libgl1-mesa-glx libvulkan1 libglvnd-dev

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

# SimplerEnv is installed from source because it depends on its ManiSkill2
# real-to-sim submodule and assets.
if [ ! -d "external/SimplerEnv" ]; then
    echo "Cloning SimplerEnv..."
    mkdir -p external
    git clone https://github.com/simpler-env/SimplerEnv.git --recurse-submodules --depth 1 external/SimplerEnv
fi

echo "Installing SimplerEnv..."
uv pip install -e external/SimplerEnv/ManiSkill2_real2sim
uv pip install -e external/SimplerEnv

echo ""
echo "Done. Run options:"
echo "  OpenVLA sim:     uv run python scripts/run_openvla_sim.py --task google_robot_pick_coke_can"
echo "  Legacy ACT eval: uv run python -m eval.run_rollout"
echo "  Start server:    uv run python -m serving.grpc_server"
echo "  Run client:      uv run python -m serving.grpc_client --episodes 5"
