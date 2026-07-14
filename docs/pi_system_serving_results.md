# PI0.5 and PI0-FAST System Serving Results

This note records the current A100 serving recommendation and the benchmark
commands used to validate it.  The default low-latency path should be PI0.5;
PI0-FAST remains useful for throughput experiments but needs straggler-aware
serving when distinct robot observations are batched together.

## Current Recommendation

- Use `lerobot/pi05_libero_finetuned_v044` with bf16 autocast and
  `num_inference_steps=4` for the primary single-machine latency path.
- For experimental low-latency deployments, compile `sample_actions` with
  `--compile-mode default` and `TORCHINDUCTOR_USE_CUDAGRAPHS=0`.  Do not use
  `reduce-overhead` yet; CUDA graph capture failed on this PI0.5 path.
- Serve PI0.5 through deadline-aware admission control.  On one A100 with the
  compiled `sample_actions` path, 12 robots at 1000 ms chunk requests were clean
  under a 250 ms deadline; 16 open sessions overloaded the single-worker queue.
- Prefer load-based admission once request periods are known.  With the eager
  runtime, `--max-admission-utilization 0.7` admits about four 1000 ms sessions.
  With compiled `sample_actions`, cap 1000 ms sessions around 12 until a longer
  soak validates more aggressive settings.
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

PI0-FAST v044 accuracy sanity checks:

| Path | Slice | Result |
| --- | --- | ---: |
| Official LeRobot eval with camera `rename_map` | HF task list, object/spatial/goal/10, 1 episode each | 30/40 success, 75.0%, 139.8 s/episode |
| Official LeRobot eval with camera `rename_map` | `libero_object`, task 1, 2 episodes | 2/2 success, 88.9 s/episode |
| Custom fixed-budget runner | `libero_object`, task 1, episode 0 | 1/1 success, 602.6 ms/control |
| Custom `action_end` runner | `libero_object`, task 1, episode 0 | 1/1 success, 224.1 ms/control |
| Custom `action_end` runner | `libero_object`, tasks 0-9, episode 0 | 8/10 success, 206.1 ms/control |
| Custom `action_end` runner | `libero_spatial`, tasks 0-9, episode 0 | 9/10 success, 330.0 ms/control |
| Custom `action_end` runner | `libero_goal`, tasks 0-9, episode 0 | 7/10 success, 282.6 ms/control |
| Custom `action_end` runner | object + spatial + goal, tasks 0-9, episode 0 | 24/30 success, 272.9 ms/control |
| Custom `action_end` runner | object + spatial + goal, tasks 0-9, episodes 0-3 | 93/120 success, 269.2 ms/control |
| Custom `action_end` runner | `libero_10`, tasks 0-9, episode 0 | 1/10 success, 182.5 ms/control |
| Custom `action_end` exact validator | `libero_object`, task 1, episode 0 | 1/1 success, 14 exact verifies, max action diff 0.0 |
| Custom `action_end` exact validator | weak `libero_goal` rows, task ids 0/6/9, episode 0 | 0/3 success, 90 exact verifies, max action diff 0.0 |
| Custom `action_end`, absolute control | `libero_goal`, task 0, episode 0 | 0/1 success, 222.0 ms/control |
| Official LeRobot eval, `env.init_states=false`, seed 1000 | `libero_goal`, task 0, episode 0 | 1/1 success, 95.7 s/episode |
| Custom `action_end`, `--no-init-states --seed 1000` | `libero_goal`, task 0, episode 0 | 1/1 success, 271.2 ms/control |
| Official LeRobot eval, `env.init_states=false`, seed 1000 | `libero_object`, task 0, episode 0 | 1/1 success, 91.1 s/episode |
| Custom baseline, `--no-init-states --seed 1000` | `libero_object`, task 0, episode 0 | 0/1 success, 559.2 ms/control |

The older 7/120 strict PI0-FAST rows used `lerobot/pi0fast-libero`, which is
not the HF-carded `lerobot/pi0fast-libero-v044` checkpoint that reports 82.5%
LIBERO SR. Treat those rows as historical latency/equivalence artifacts only.
The official LeRobot command on this v0.4.4 install, using the carded v044
checkpoint, the HF task list, `eval.n_episodes=1`, and the camera `rename_map`,
is `30/40 = 75.0%`: object `9/10`, spatial `8/10`, goal `8/10`, and
`libero_10` `5/10`. Failed task ids are `libero_object_0`,
`libero_spatial_1`, `libero_spatial_9`, `libero_goal_3`, `libero_goal_9`, and
`libero_10_{0,2,4,6,9}`. This is below the HF card's `82.5%` table, but it is
not the old `7/120` baseline, and it used fixed 256-token decode, so the gap is
not caused by action-end early stopping. Artifact:
`outputs/eval/2026-07-14/00-35-17_libero_pi0_fast/eval_info.json`.

Follow-up HF/cache check: current HF docs are for LeRobot main/v0.6.0, while
this environment uses LeRobot v0.4.4. The current `lerobot/pi0fast-libero`
snapshot is not a drop-in replacement for the v044 card here: its config
expects `observation.images.image` and `observation.images.image2`, so the docs
`rename_map` fails with missing image features. Without the rename map, this
install failed early `libero_object` probes. Re-running the v044 carded
checkpoint on `libero_object_0` with the required camera `rename_map` failed
again (`0/1`), artifact
`outputs/eval/2026-07-14/02-46-06_v044_object0_rerun/eval_info.json`. Treat
the remaining `75.0%` versus `82.5%` delta as a LeRobot/LIBERO version or
protocol reproduction gap pending a v0.6.0-stack rerun, not as a token-stopping
accuracy regression.

The current v044 30-row smoke is in the same broad accuracy regime, and the
v044 120-row custom run is `93/120 = 77.5%` with object `38/40`, spatial
`31/40`, and goal `24/40`. Exact validation on representative failed goal rows
matched the fixed 256-token decode exactly. Read the `93/120` row as a fixed
LIBERO-init-state stress test, not as an exact HF-card protocol reproduction. A
sentinel row exposed a remaining custom-runner gap: official LeRobot eval
succeeds on `libero_object` task 0 seed 1000, while the custom baseline loop
still fails after adding the official camera `rename_map`, global seeding, and
config-first policy load. For HF-card accuracy claims, use official
`lerobot-eval`; use the custom runner for token-level latency/equivalence
diagnostics until the rollout mismatch is fully reconciled.

PI0-FAST v044 model-serving component benchmark, bf16, action-end decode,
`outputs/pi0fast_system_components/v044_action_end_replicated_task1_steps5.json`:

| Mode | Chunk mean | Per request | Per action |
| --- | ---: | ---: | ---: |
| Single request | 605.1 ms | 605.1 ms | 60.5 ms |
| Replicated batch 2 | 636.2 ms | 318.1 ms | 31.8 ms |
| Replicated batch 4 | 676.1 ms | 169.0 ms | 16.9 ms |
| Replicated batch 8 | 788.6 ms | 98.6 ms | 9.9 ms |

The benchmark recommendation marks batch 8 as meeting a 100 ms/request target,
and the single-request path already meets a 100 ms/action target by amortizing
one policy call over the 10-action PI0-FAST chunk. These rows exclude simulator
render/step overhead; rollout wall-clock rows above include LIBERO/OSMesa
observation cost.

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
| gRPC worker queue + server warmup, 1 robot, 3 s smoke, 1000 ms request period, 250 ms deadline | 0/3 misses, p95 server ~176.4 ms |
| gRPC compiled `sample_actions`, no CUDA graphs, 1 robot, 10 s, 1000 ms request period, 250 ms deadline | 0/10 misses, p95 server ~76.3 ms |
| gRPC compiled `sample_actions`, no CUDA graphs, 2 robots, 5 s, 1000 ms request period, 250 ms deadline | 0/10 misses, p95 server ~79.1 ms |
| gRPC compiled `sample_actions`, no CUDA graphs, 4 robots, 10 s, 1000 ms request period, 250 ms deadline | 0/40 misses, p95 server ~76.5 ms |
| gRPC compiled `sample_actions`, no CUDA graphs, 8 robots, 10 s, 1000 ms request period, 250 ms deadline | 0/80 misses, p95 server ~67.7 ms |
| gRPC compiled `sample_actions`, no CUDA graphs, 12 robots, 10 s, 1000 ms request period, 250 ms deadline, open cap | 0/120 misses, p95 server ~73.1 ms |
| gRPC compiled `sample_actions`, no CUDA graphs, 16 robots, 10 s, 1000 ms request period, 250 ms deadline, open cap | 157/160 misses, 157 RPC timeouts |

The compiled `sample_actions` path is the first large model-runtime speedup.
`reduce-overhead` mode failed during Inductor CUDA graph capture with a cuDNN
device-allocation error, but `default` mode with `TORCHINDUCTOR_USE_CUDAGRAPHS=0`
was stable in short smoke tests.  The first compile/warmup can take minutes on a
cold cache, so this should remain an explicit server startup option until longer
soaks validate it.

The compiled capacity cliff is admission-related, not a slow admitted request:
the successful 16-robot responses still ran in ~78-88 ms server latency, but
open admission allowed a timeout storm.  The load tester now records RPC errors
as deadline misses and can reuse `--prepared-observation-path` to avoid rebuilding
LIBERO observations between capacity runs.

The PyTorch profiler run adds substantial overhead and should not be used for
latency SLO numbers.  It did show many pageable host-to-device copies under
profile, so the next transport/runtime focus should be keeping prepared tensors
resident or moving to shared-memory/local tensor transport instead of rebuilding
payload tensors for every request.

The real serving smoke confirms that the practical 250 ms single-GPU boundary is
around 4 robots at a 1000 ms chunk request period, or 8 robots at 1500 ms.  The
8-robot/1500 ms case has only ~18 ms worst-case slack in this short run.
The serving runtime now trims batches when estimated runtime would consume
deadline slack.  It can reject new sessions with either `--max-active-sessions`
or projected utilization from `--max-admission-utilization`.  The gRPC server
also has a dedicated GPU worker queue so request handler threads only decode and
enqueue work, and it can run startup warmup from a saved prepared observation.

PI0-FAST, bf16, `lerobot/pi0fast-libero`, action-end decode
(historical uncarded-checkpoint measurement; use `lerobot/pi0fast-libero-v044`
for HF-carded accuracy reproduction):

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
  --max-admission-utilization 0.7 \
  --deadline-ms 250 \
  --num-inference-steps 4 \
  --warmup-observation-path outputs/pi05_grpc_load/warmup_observation.pt \
  --warmup-requests 1 \
  --metrics-path outputs/pi05_grpc_server/metrics.jsonl
```

PI0.5 compiled gRPC server:

```bash
TORCHDYNAMO_DISABLE=0 TORCHINDUCTOR_USE_CUDAGRAPHS=0 \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python -m serving.pi05_server \
  --port 50051 \
  --max-active-sessions 4 \
  --max-admission-utilization 0.7 \
  --deadline-ms 250 \
  --num-inference-steps 4 \
  --compile-target sample_actions \
  --compile-mode default \
  --warmup-observation-path outputs/pi05_grpc_load/warmup_observation.pt \
  --warmup-requests 1 \
  --metrics-path outputs/pi05_grpc_server/compiled_metrics.jsonl
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
  --save-warmup-observation outputs/pi05_grpc_load/warmup_observation.pt \
  --output outputs/pi05_grpc_load/r4_300s_req1000_d250.json
```

For repeated capacity checks, reuse the saved prepared observation so the client
does not rebuild the LeRobot policy and LIBERO simulator before every run:

```bash
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/load_pi05_grpc.py \
  --server localhost:50051 \
  --robots 12 \
  --duration-seconds 10 \
  --request-period-ms 1000 \
  --deadline-ms 250 \
  --stagger-arrivals \
  --warmup-requests 0 \
  --prepared-observation-path outputs/pi05_grpc_load/warmup_observation.pt \
  --output outputs/pi05_grpc_load/r12_cached_10s_req1000_d250.json
```

PI0.5 cluster router, local dev mode:

```bash
.venv-pi/bin/python -m serving.pi05_cluster_router \
  --port 50100 \
  --worker id=w0,addr=localhost:50051,gpu=0,max_sessions=12,util=1.0,rt=75
```

The router exposes the same `InferenceService/Predict` API as the worker.  For
single-GPU development, run one compiled PI0.5 worker and route through
`localhost:50100`; for multi-GPU deployment, launch one worker per GPU and add
one `--worker` entry per address.  Router telemetry is added to
`telemetry_json`, and `scripts/load_pi05_grpc.py` reports
`cluster_worker_counts` when pointed at the router.

PI0-FAST action-end replicated batch:

```bash
HF_HOME=/home/ubuntu/AI-Infra-Final-Project/.hf_cache \
LIBERO_CONFIG_PATH=/home/ubuntu/AI-Infra-Final-Project/.libero_config \
MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
.venv-pi/bin/python scripts/benchmark_pi0fast_system_components.py \
  --policy-kind pi0fast --policy lerobot/pi0fast-libero-v044 \
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
  --policy-kind pi0fast --policy lerobot/pi0fast-libero-v044 \
  --task libero_object --task-id 0 --warmup 1 --steps 1 \
  --batch-sizes 1,2,4,8 --decode-path action_end \
  --max-decoding-steps default --kv-modes default \
  --batch-source distinct-reset --device cuda --dtype bfloat16 \
  --action-token-warn-threshold 96 \
  --output outputs/pi0fast_system_components/pi0fast_action_end_compaction_distinct_steps1_tokenmean.json
```
