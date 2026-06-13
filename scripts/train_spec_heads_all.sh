#!/usr/bin/env bash
set -euo pipefail

# Trains speculative draft heads for all configured suites.

python scripts/train_trajectory_heads_all_suites.py \
  --config configs/libero_specvla_distributed.yaml
