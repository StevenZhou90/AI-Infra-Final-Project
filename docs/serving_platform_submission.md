# Robotic-Arm Multi-Serving Platform Submission

## What This Delivers

This branch packages the repository as a working robotic-arm serving platform:

- OpenVLA/SimplerEnv rollout evaluation.
- SpecVLA-style trajectory speculative decoding.
- pi0-FAST chunk execution and target-equivalent token experiments.
- pi0.5 gRPC serving with GPU-aware routing and admission control.
- Multi-GPU benchmark entrypoints for GPUs 0-2.
- Unit tests for serving, routing, chunking, and speculative draft behavior.

## Architecture

```text
client
  -> proto/inference.proto
  -> serving/pi05_server.py
  -> serving/pi05_cluster_router.py
  -> serving/pi05_runtime_service.py
  -> policy/runtime backends
```

The serving layer is intentionally separate from benchmark scripts:

- `serving/`: runtime services, router, codecs, decoders, and draft heads.
- `scripts/`: benchmark, training, profiling, load, and data-generation tools.
- `configs/`: reproducible benchmark and training recipes.
- `tests/`: package-level checks for serving and speculative decoding.

## GPU 0-2 Launches

Start a three-GPU serving process:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python -m serving.pi05_server \
  --host 0.0.0.0 \
  --port 50051 \
  --devices cuda:0,cuda:1,cuda:2 \
  --max-concurrent 12 \
  --warmup-steps 2
```

Run gRPC load:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python scripts/load_pi05_grpc.py \
  --target localhost:50051 \
  --clients 16 \
  --requests 128
```

Run distributed LIBERO smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  scripts/run_libero_specvla_distributed.py \
  --config configs/libero_specvla_distributed.yaml \
  --mode smoke
```

## Result Snapshot

| Evaluation | Baseline | Best Serving Path |
| --- | ---: | ---: |
| SimplerEnv coke-can latency | `302.5 ms/step` | `145.1 ms/step` |
| SimplerEnv coke-can success | `14/27` | `14/27` |
| LIBERO Goal latency | `317.6 ms/step` | `202.0 ms/step` |
| LIBERO Goal success | `22/30` | `23/30` |

## Validation

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m pytest
```

Current local result:

```text
53 passed, 1 warning
```

The warning is from PyTorch transformer nested-tensor configuration in the
pi0-FAST block drafter test and does not indicate a test failure.
