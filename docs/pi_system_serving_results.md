# PI0.5 and PI0-FAST System Serving Results

This note records the current A100 serving recommendation and the benchmark
commands used to validate it.  The default low-latency path should be PI0.5;
PI0-FAST remains useful for throughput experiments but needs straggler-aware
serving when distinct robot observations are batched together.

## Current Recommendation

- Use `lerobot/pi05_libero_finetuned_v044` with bf16 autocast and
  `num_inference_steps=4` for the primary single-machine latency path.
- Serve PI0.5 through deadline-aware admission control.  On one A100, start
  with `max_active_sessions=4` for 1000 ms chunk requests or `8` for 1500 ms
  chunk requests.
- Use action-buffer mode for robot control loops.  With a 50-action chunk,
  20 ms control period, and 5-action low watermark, each robot should request a
  new chunk about every 900 ms or slower.
- Do not keep `num_inference_steps=6` as the default fallback for task 7.  The
  same task/seed failed at 4, 6, and 10 steps, so the failure is not explained
  by the lower latency setting.
- Keep PI0-FAST on the `action_end` decode path for serving.  Do not use the
  public fixed-256-token decode path except as a baseline.
- Treat PI0-FAST rows above `96` FAST tokens as stragglers in telemetry.  Use a
  hard serving cap such as `128` tokens for latency experiments, and keep `256`
  only for correctness comparisons.

## Measured A100 Latencies

PI0.5, bf16 autocast, `lerobot/pi05_libero_finetuned_v044`:

| Mode | Mean chunk latency |
| --- | ---: |
| 4 flow steps, batch 1 | ~169 ms |
| 5 flow steps, batch 1 | ~190 ms |
| 6 flow steps, batch 1 | ~211 ms |
| 8 flow steps, batch 1 | ~254 ms |
| 10 flow steps, batch 1 | ~298 ms |
| 4 flow steps, replicated batch 8 | ~357 ms total, ~44.6 ms/request |
| 4 flow steps, distinct batch 8 | ~405 ms total, ~50.7 ms/request |

PI0.5 rollout smoke, `libero_object` task 0, baseline chunk execution:

| Mode | Result |
| --- | ---: |
| 4 flow steps, 1 episode | 1/1 success, ~155.9 ms/control step |
| 6 flow steps, 1 episode | 1/1 success, ~159.3 ms/control step |
| 4 flow steps, 3 episodes | 3/3 success, ~159.2 ms/control step |
| 4 flow steps, tasks 0-4, 2 episodes each | 10/10 success, ~157.3 ms/control step |
| 4 flow steps, tasks 0-9, 3 episodes each | 29/30 success, ~156.0 ms/control step |
| 6 flow steps, task 7, 3 episodes | 2/3 success, ~150.6 ms/control step |
| 10 flow steps, task 7, 3 episodes | 2/3 success, ~153.3 ms/control step |

The one 4-step failure was `libero_object` task 7, seed 43.  The same episode
also failed at 6 and 10 flow steps, so this result does not appear to be caused
by the 4-step latency setting.

PI0.5 synthetic serving capacity, calibrated from 4-step bf16 latency and
staggered robot chunk requests:

| Chunk request period | 200 ms deadline | 250 ms deadline |
| --- | ---: | ---: |
| 800 ms | 4 robots | 4 robots |
| 1000 ms | 4 robots | 4 robots |
| 1500 ms | 8 robots | 8 robots |
| 2000 ms | 12 robots | 12 robots |

These are zero-deadline-miss estimates using single-request execution after
staggering robot phases.  Synchronous requests batch better for throughput but
miss strict 200-250 ms per-request deadlines once batch latency exceeds the
deadline.

PI0.5 real serving-runtime smoke, bf16 autocast, 4 flow steps, TorchDynamo
disabled, staggered robot chunk requests:

| Mode | Result |
| --- | ---: |
| 1 robot, 1000 ms request period, 250 ms deadline | 0/5 misses, p95 ~166.9 ms |
| 4 robots, 1000 ms request period, 250 ms deadline | 0/20 misses, p95 ~169.4 ms |
| 8 robots, 1000 ms request period, 250 ms deadline | 22/24 misses, p95 ~1120 ms |
| 8 robots, 1500 ms request period, 250 ms deadline | 0/24 misses, p95 ~223.0 ms |
| 4 robots, 10 s soak, action-buffer mode, 1000 ms request period, 250 ms deadline | 0/40 misses, p95 ~182.8 ms |
| gRPC server, 1 robot, 3 s warm smoke, 1000 ms request period, 250 ms deadline | 0/3 misses, p95 ~171.9 ms |

The real serving smoke confirms that the practical 250 ms single-GPU boundary is
around 4 robots at a 1000 ms chunk request period, or 8 robots at 1500 ms.  The
8-robot/1500 ms case has only ~18 ms worst-case slack in this short run.
The serving runtime now trims batches when estimated runtime would consume
deadline slack, and it can reject new sessions with `--max-active-sessions`.

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
  --batch-sizes 1,2,4,8 --decode-path public \
  --num-inference-steps 4,5,6,8,10 --kv-modes default \
  --device cuda --dtype bfloat16 --target-latency-ms 250 \
  --output outputs/pi0fast_system_components/pi05_bf16_steps4_10_batch8.json
```

PI0.5 distinct-reset batch check:

```bash
TORCH_COMPILE_DISABLE=1 HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.hf_cache \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/benchmark_pi0fast_system_components.py \
  --policy-kind pi05 \
  --task libero_object --task-id 0 --warmup 1 --steps 1 \
  --batch-sizes 1,4,8 --batch-source distinct-reset \
  --decode-path public --num-inference-steps 4,6 \
  --kv-modes default --device cuda --dtype bfloat16 \
  --target-latency-ms 250 \
  --output outputs/pi0fast_system_components/pi05_bf16_distinct_steps4_6_batch8.json
```

PI0.5 rollout smoke:

```bash
TORCH_COMPILE_DISABLE=1 HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.hf_cache \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
  --policy-kind pi05 --task libero_object --task-ids 0 \
  --episodes 3 --steps 300 --modes baseline \
  --summary-baseline-mode baseline --device cuda \
  --dtype bfloat16 --amp-dtype bfloat16 \
  --num-inference-steps 4 \
  --output-dir outputs/pi05_rollout_steps4_task0_ep3
```

Broader PI0.5 rollout check:

```bash
TORCH_COMPILE_DISABLE=1 HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.hf_cache \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
  --policy-kind pi05 --task libero_object --task-ids 0,1,2,3,4 \
  --episodes 2 --steps 300 --modes baseline \
  --summary-baseline-mode baseline --device cuda \
  --dtype bfloat16 --amp-dtype bfloat16 \
  --num-inference-steps 4 \
  --output-dir outputs/pi05_rollout_steps4_object0_4_ep2
```

Full PI0.5 object-task rollout:

```bash
TORCH_COMPILE_DISABLE=1 HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.hf_cache \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
  --policy-kind pi05 --task libero_object --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --episodes 3 --steps 300 --modes baseline \
  --summary-baseline-mode baseline --device cuda \
  --dtype bfloat16 --amp-dtype bfloat16 \
  --num-inference-steps 4 \
  --output-dir outputs/pi05_rollout_steps4_object0_9_ep3
```

PI0.5 synthetic serving capacity:

```bash
.venv-pi/bin/python scripts/benchmark_pi0fast_serving_runtime.py \
  --backend pi05 --robots 8 --steps 50 \
  --request-period-ms 1500 --deadline-ms 250 \
  --mode flow --max-batch-size 8 --max-batch-delay-ms 5 \
  --stagger-arrivals --pi05-base-ms 158 --pi05-per-request-ms 31 \
  --output outputs/pi05_serving_capacity_staggered/pi05_r8_req1500_d250.json
```

PI0.5 real serving-runtime smoke:

```bash
TORCHDYNAMO_DISABLE=1 \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/benchmark_pi05_real_serving_runtime.py \
  --robots 8 --steps 3 --warmup 2 \
  --request-period-ms 1500 --deadline-ms 250 \
  --max-batch-size 8 --max-batch-delay-ms 5 \
  --num-inference-steps 4 --stagger-arrivals \
  --output outputs/pi05_real_serving_runtime/stagger_r8_s3_req1500_d250.json
```

PI0.5 load sweep / soak driver:

```bash
TORCHDYNAMO_DISABLE=1 \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/load_pi05_serving_runtime.py \
  --robots 4,8 --request-period-ms 1000,1500 \
  --deadline-ms 250 --soak-seconds 60 \
  --max-active-sessions 8 --action-buffer-mode \
  --output outputs/pi05_real_serving_runtime/load_soak_60s.json
```

PI0.5 gRPC server:

```bash
TORCHDYNAMO_DISABLE=1 \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python -m serving.pi05_server \
  --port 50051 \
  --max-active-sessions 4 \
  --deadline-ms 250 \
  --num-inference-steps 4 \
  --metrics-path outputs/pi05_grpc_server/metrics.jsonl
```

PI0.5 gRPC load test:

```bash
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/load_pi05_grpc.py \
  --server localhost:50051 \
  --robots 4 \
  --duration-seconds 300 \
  --request-period-ms 1000 \
  --deadline-ms 250 \
  --stagger-arrivals \
  --warmup-requests 1 \
  --output outputs/pi05_grpc_load/r4_300s_req1000_d250.json
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
