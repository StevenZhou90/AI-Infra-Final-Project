#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/AI-Infra-Final-Project

export HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.cache/huggingface
export UV_CACHE_DIR=/home/ubuntu/AI-Infra-Final-Project/.cache/uv
export XDG_RUNTIME_DIR=/tmp
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false

PY=/home/ubuntu/AI-Infra-Final-Project/.venv/bin/python
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="artifacts/logs/readme_repro_${RUN_ID}"
mkdir -p "${LOG_DIR}" data checkpoints outputs/benchmarks

echo "run_id=${RUN_ID}"
echo "logs=${LOG_DIR}"

# Standard hyperparameters used for every head training round.
TRAIN_ARGS=(
  --epochs 80
  --batch-size 128
  --hidden-dim 1024
  --embed-dim 128
  --hidden-fusion-dim 512
  --num-layers 3
  --lr 2e-4
  --dim-weights 1,1,1.5,2,2,2,5
  --change-weight 2.0
  --gripper-change-weight 8.0
  --late-timestep 20
  --late-weight 1.5
  --device cuda
)

echo "[1/8] Generate supervised trajectory-head data"
"${PY}" scripts/generate_trajectory_head_data.py \
  --sweep mini \
  --steps 80 \
  --out-dir data/trajectory_head_mini_r1 \
  --device cuda \
  --dtype bfloat16 \
  --pretrained openvla/openvla-7b \
  2>&1 | tee "${LOG_DIR}/01_generate_supervised.log"

echo "[2/8] Train supervised trajectory head"
"${PY}" scripts/train_trajectory_head.py \
  --data-dir data/trajectory_head_mini_r1 \
  --out-dir checkpoints/traj_head_r1 \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/02_train_supervised.log"

echo "[3/8] Generate DAgger r1 trajectory-head data (rollouts under r1 supervised head)"
"${PY}" scripts/generate_trajectory_head_dagger_data.py \
  --policy-head-checkpoint checkpoints/traj_head_r1/best.pt \
  --sweep mini \
  --steps 80 \
  --out-dir data/trajectory_head_dagger_mini_r1 \
  --head-threshold 0.2 \
  --fast-min-confident-tokens 5 \
  --device cuda \
  --dtype bfloat16 \
  --pretrained openvla/openvla-7b \
  2>&1 | tee "${LOG_DIR}/03_generate_dagger_r1.log"

echo "[4/8] Train DAgger r1 head on aggregated supervised + dagger_r1 data"
"${PY}" scripts/aggregate_trajectory_head_data.py \
  --inputs data/trajectory_head_mini_r1 data/trajectory_head_dagger_mini_r1 \
  --out-dir data/trajectory_head_dagger_aggr_r1 \
  2>&1 | tee "${LOG_DIR}/04a_aggregate_r1.log"
"${PY}" scripts/train_trajectory_head.py \
  --data-dir data/trajectory_head_dagger_aggr_r1 \
  --out-dir checkpoints/traj_head_dagger_r1 \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/04b_train_dagger_r1.log"

echo "[5/8] Generate DAgger r2 data (rollouts under DAgger r1 head)"
"${PY}" scripts/generate_trajectory_head_dagger_data.py \
  --policy-head-checkpoint checkpoints/traj_head_dagger_r1/best.pt \
  --sweep mini \
  --steps 80 \
  --out-dir data/trajectory_head_dagger_mini_r2 \
  --head-threshold 0.2 \
  --fast-min-confident-tokens 5 \
  --device cuda \
  --dtype bfloat16 \
  --pretrained openvla/openvla-7b \
  2>&1 | tee "${LOG_DIR}/05_generate_dagger_r2.log"

echo "[6/8] Train DAgger r2 head on aggregated supervised + dagger_r1 + dagger_r2 data"
"${PY}" scripts/aggregate_trajectory_head_data.py \
  --inputs \
    data/trajectory_head_mini_r1 \
    data/trajectory_head_dagger_mini_r1 \
    data/trajectory_head_dagger_mini_r2 \
  --out-dir data/trajectory_head_dagger_aggr_r2 \
  2>&1 | tee "${LOG_DIR}/06a_aggregate_r2.log"
"${PY}" scripts/train_trajectory_head.py \
  --data-dir data/trajectory_head_dagger_aggr_r2 \
  --out-dir checkpoints/traj_head_dagger_r2 \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/06b_train_dagger_r2.log"

echo "[7/8] Benchmark README matrix (baseline + adaptive fast r2)"
"${PY}" scripts/run_readme_speedup_matrix.py \
  --checkpoint checkpoints/traj_head_dagger_r2/best.pt \
  --output-dir "outputs/benchmarks/readme_speedup_${RUN_ID}" \
  --vertical-gate 5 \
  --horizontal-gate 6 \
  --standing-gate 6 \
  --head-threshold 0.2 \
  --steps 80 \
  --episodes 3 \
  --device cuda \
  --dtype bfloat16 \
  2>&1 | tee "${LOG_DIR}/07_benchmark.log"

echo "[8/8] Done"
echo "checkpoint=checkpoints/traj_head_dagger_r2/best.pt"
echo "benchmark=outputs/benchmarks/readme_speedup_${RUN_ID}"
