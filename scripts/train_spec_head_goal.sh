#!/usr/bin/env bash
set -euo pipefail

# Trains a single speculative draft head for libero_goal
# using the suite orchestrator.

python scripts/train_trajectory_heads_all_suites.py \
  --config configs/libero_specvla_distributed.yaml \
  --suites libero_goal
