#!/usr/bin/env python3
"""Train a compact Medusa-style future-token head for PI0-FAST traces.

This is an offline probe for speculative FAST-token decoding. It collects
target PI0-FAST hidden states while decoding real LIBERO observations, trains
small heads to predict future generated FAST tokens, then reports exact
future-token and speculative-prefix acceptance estimates on held-out traces.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pi0fast_chunk_eval import (  # noqa: E402
    _env_step,
    _import_lerobot,
    _prepare_observation,
    _to_numpy_action,
)
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

logger = logging.getLogger("train_pi0fast_sd_head")


@dataclass
class TraceSample:
    hidden_states: torch.Tensor
    token_ids: torch.Tensor


class FutureTokenHeads(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, lookahead: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(lookahead)])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        return [head(hidden) for head in self.heads]


def _action_from_trace(trace, policy_postprocessor) -> np.ndarray:
    try:
        processed = policy_postprocessor(trace.actions)
    except Exception:
        processed = trace.actions
    return _to_numpy_action(processed)


@torch.inference_mode()
def collect_traces(args) -> list[TraceSample]:
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)

    env_kwargs: dict[str, Any] = {"task": args.task, "control_mode": args.control_mode}
    if args.task_id is not None:
        env_kwargs["task_ids"] = [args.task_id]
    env_cfg = LiberoEnv(**env_kwargs)

    logger.info("Loading policy %s on %s", args.policy, device)
    policy = PI0FastPolicy.from_pretrained(args.policy).to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    task_id = args.task_id if args.task_id is not None else min(env_map[args.task])
    env = env_map[args.task][task_id]

    traces: list[TraceSample] = []
    observation, _info = env.reset(seed=[args.seed])
    steps = 0
    try:
        while steps < args.steps and len(traces) < args.max_traces:
            batch = _prepare_observation(
                observation,
                env,
                env_preprocessor,
                policy_preprocessor,
                preprocess_observation,
            )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            trace = adapter.predict_action_chunk_with_trace(batch, return_hidden_states=True)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            logger.info(
                "trace=%d decode_ms=%.1f tokens=%d",
                len(traces),
                (time.perf_counter() - t0) * 1000,
                trace.token_count,
            )
            if trace.hidden_states is None:
                raise RuntimeError("Adapter did not return hidden states")
            traces.append(
                TraceSample(
                    hidden_states=trace.hidden_states[0].detach().to("cpu", dtype=torch.float32),
                    token_ids=trace.token_ids[0].detach().to("cpu", dtype=torch.long),
                )
            )

            actions = _action_from_trace(trace, policy_postprocessor)
            for action in actions[: args.execute_actions_per_trace]:
                observation, _reward, terminated, truncated, _info = _env_step(env, action, env_postprocessor)
                steps += 1
                if terminated or truncated or steps >= args.steps:
                    observation, _info = env.reset(seed=[args.seed + steps + 1])
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass

    return traces


def build_dataset(
    traces: list[TraceSample],
    lookahead: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, dict[int, int], list[int]]:
    token_values = sorted({int(t) for trace in traces for t in trace.token_ids.tolist()})
    token_to_class = {token_id: idx for idx, token_id in enumerate(token_values)}

    hidden_rows: list[torch.Tensor] = []
    trace_ids: list[int] = []
    labels_by_head: list[list[int]] = [[] for _ in range(lookahead)]
    for trace_idx, trace in enumerate(traces):
        hidden = trace.hidden_states
        tokens = trace.token_ids
        usable = len(tokens) - lookahead
        for pos in range(max(usable, 0)):
            hidden_rows.append(hidden[pos])
            trace_ids.append(trace_idx)
            for offset in range(1, lookahead + 1):
                labels_by_head[offset - 1].append(token_to_class[int(tokens[pos + offset])])

    if not hidden_rows:
        raise RuntimeError("No trainable rows collected; increase --steps or reduce --lookahead")
    X = torch.stack(hidden_rows)
    ys = [torch.tensor(labels, dtype=torch.long) for labels in labels_by_head]
    return X, ys, torch.tensor(trace_ids, dtype=torch.long), token_to_class, token_values


def split_row_indices(n_rows: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_rows, generator=generator)
    n_val = max(1, int(round(n_rows * val_fraction)))
    return perm[n_val:], perm[:n_val]


def split_trace_indices(trace_ids: torch.Tensor, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    unique = torch.unique(trace_ids)
    generator = torch.Generator().manual_seed(seed)
    perm = unique[torch.randperm(len(unique), generator=generator)]
    n_val_traces = max(1, int(round(len(unique) * val_fraction)))
    if len(unique) > 1:
        n_val_traces = min(n_val_traces, len(unique) - 1)
    val_traces = set(int(t) for t in perm[:n_val_traces].tolist())
    val_mask = torch.tensor([int(t) in val_traces for t in trace_ids.tolist()], dtype=torch.bool)
    all_idx = torch.arange(len(trace_ids))
    return all_idx[~val_mask], all_idx[val_mask]


def train_heads(
    X: torch.Tensor,
    ys: list[torch.Tensor],
    trace_ids: torch.Tensor,
    args,
) -> tuple[FutureTokenHeads, dict[str, Any]]:
    if args.split == "trace":
        train_idx, val_idx = split_trace_indices(trace_ids, args.val_fraction, args.seed)
    else:
        train_idx, val_idx = split_row_indices(len(X), args.val_fraction, args.seed)
    device = torch.device(args.train_device if torch.cuda.is_available() or not args.train_device.startswith("cuda") else "cpu")
    model = FutureTokenHeads(X.shape[-1], int(max(y.max().item() for y in ys) + 1), args.lookahead).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    X_train = X[train_idx].to(device)
    y_train = [y[train_idx].to(device) for y in ys]
    X_val = X[val_idx].to(device)
    y_val = [y[val_idx].to(device) for y in ys]

    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        total_loss = 0.0
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            logits = model(X_train[idx])
            loss = sum(F.cross_entropy(logit, y[idx]) for logit, y in zip(logits, y_train)) / len(logits)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)

        metrics = evaluate_heads(model, X_val, y_val)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(len(X_train), 1)
        history.append(metrics)
        logger.info("epoch=%d train_loss=%.4f val=%s", epoch, metrics["train_loss"], metrics)

    return model, {
        "split": args.split,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "train_traces": sorted(set(int(t) for t in trace_ids[train_idx].tolist())),
        "val_traces": sorted(set(int(t) for t in trace_ids[val_idx].tolist())),
        "history": history,
    }


@torch.inference_mode()
def evaluate_heads(model: FutureTokenHeads, X: torch.Tensor, ys: list[torch.Tensor]) -> dict[str, Any]:
    model.eval()
    logits = model(X)
    accuracies = []
    preds = []
    for logit, y in zip(logits, ys):
        pred = logit.argmax(dim=-1)
        preds.append(pred)
        accuracies.append(float((pred == y).float().mean().item()))

    accepted = []
    for row in range(X.shape[0]):
        n = 0
        for head_idx, pred in enumerate(preds):
            if int(pred[row]) != int(ys[head_idx][row]):
                break
            n += 1
        accepted.append(n)

    return {
        "head_accuracy": accuracies,
        "mean_spec_accept": float(np.mean(accepted)),
        "p_accept_ge_1": float(np.mean([a >= 1 for a in accepted])),
        "p_accept_ge_2": float(np.mean([a >= 2 for a in accepted])),
        "p_accept_full": float(np.mean([a == len(ys) for a in accepted])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate PI0-FAST speculative future-token heads")
    parser.add_argument("--policy", default="lerobot/pi0fast-libero")
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--max-traces", type=int, default=8)
    parser.add_argument("--execute-actions-per-trace", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--split", choices=["trace", "row"], default="trace")
    parser.add_argument("--output-dir", default="outputs/pi0fast_sd_head")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = collect_traces(args)
    logger.info("collected %d traces", len(traces))
    X, ys, trace_ids, token_to_class, token_values = build_dataset(traces, args.lookahead)
    model, train_summary = train_heads(X, ys, trace_ids, args)
    metrics = evaluate_heads(model, X.to(next(model.parameters()).device), [y.to(next(model.parameters()).device) for y in ys])

    summary = {
        "config": vars(args),
        "num_traces": len(traces),
        "rows": int(len(X)),
        "hidden_dim": int(X.shape[-1]),
        "candidate_vocab_size": len(token_values),
        "train": train_summary,
        "all_rows_metrics": metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "token_values": token_values,
            "summary": summary,
        },
        out_dir / "sd_heads.pt",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
