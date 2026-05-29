# PI0.5 and PI0-FAST System Serving Results

This note records the current A100 serving recommendation and the benchmark
commands used to validate it.  The default low-latency path should be PI0.5;
PI0-FAST remains useful for throughput experiments but needs straggler-aware
serving when distinct robot observations are batched together.

## Current Recommendation

- Use `lerobot/pi05_libero_finetuned_v044` with bf16 autocast and
  `num_inference_steps=6` for the primary single-machine serving path.
- Keep PI0-FAST on the `action_end` decode path for serving.  Do not use the
  public fixed-256-token decode path except as a baseline.
- Treat PI0-FAST rows above `96` FAST tokens as stragglers in telemetry.  Use a
  hard serving cap such as `128` tokens for latency experiments, and keep `256`
  only for correctness comparisons.

## Measured A100 Latencies

PI0.5, bf16 autocast, `lerobot/pi05_libero_finetuned_v044`:

| Mode | Mean chunk latency |
| --- | ---: |
| 6 flow steps, batch 1 | ~212 ms |
| 10 flow steps, batch 1 | ~295 ms |
| 6 flow steps, batch 4 | ~287 ms total, ~72 ms/request |

PI0-FAST, bf16, `lerobot/pi0fast-libero`, action-end decode:

| Mode | Mean latency |
| --- | ---: |
| Single request | ~613-658 ms |
| Replicated batch 8 | ~821 ms total, ~103 ms/request |
| Distinct-reset batch 8 | ~4.7 s total when stragglers run near the 256-token cap |

The PI0-FAST result means the implementation should expose token-count
telemetry and straggler warnings rather than presenting distinct batching as a
reliable low-latency path.

## Benchmark Commands

PI0.5 bf16 sweep:

```bash
TORCH_COMPILE_DISABLE=1 HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.hf_cache \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/benchmark_pi0fast_system_components.py \
  --policy-kind pi05 --policy lerobot/pi05_libero_finetuned_v044 \
  --task libero_object --task-id 0 --warmup 1 --steps 3 \
  --batch-sizes 1,2,4 --decode-path public \
  --num-inference-steps 6,10 --kv-modes default \
  --device cuda --dtype bfloat16 \
  --output outputs/pi0fast_system_components/pi05_bf16_autocast_steps3.json
```

PI0-FAST action-end replicated batch:

```bash
HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.hf_cache \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/benchmark_pi0fast_system_components.py \
  --policy-kind pi0fast --policy lerobot/pi0fast-libero \
  --task libero_object --task-id 0 --warmup 1 --steps 2 \
  --batch-sizes 1,2,4,8 --decode-path action_end \
  --max-decoding-steps default --kv-modes default \
  --batch-source replicated --device cuda --dtype bfloat16 \
  --output outputs/pi0fast_system_components/pi0fast_action_end_compaction_replicated_steps2.json
```

PI0-FAST distinct-reset straggler check:

```bash
HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.hf_cache \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/benchmark_pi0fast_system_components.py \
  --policy-kind pi0fast --policy lerobot/pi0fast-libero \
  --task libero_object --task-id 0 --warmup 1 --steps 1 \
  --batch-sizes 1,2,4,8 --decode-path action_end \
  --max-decoding-steps default --kv-modes default \
  --batch-source distinct-reset --device cuda --dtype bfloat16 \
  --action-token-warn-threshold 96 \
  --output outputs/pi0fast_system_components/pi0fast_action_end_compaction_distinct_steps1_tokenmean.json
```
