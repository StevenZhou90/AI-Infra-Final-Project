"""Lightweight PI0-FAST token trace shard export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _flatten_scalar_stats(stats: dict[str, Any], prefix: str = "") -> dict[str, int | float | str | bool]:
    flattened: dict[str, int | float | str | bool] = {}
    for key, value in stats.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_scalar_stats(value, name))
        elif isinstance(value, (bool, int, float, str)):
            flattened[name] = value
        elif hasattr(value, "item"):
            try:
                item = value.item()
            except Exception:
                continue
            if isinstance(item, (bool, int, float, str)):
                flattened[name] = item
    return flattened


def parse_token_trace_modes(value: str) -> set[str] | None:
    modes = {part.strip() for part in value.split(",") if part.strip()}
    if not modes or modes & {"all", "*"}:
        return None
    return modes


@dataclass
class TokenTraceSink:
    output_dir: Path
    modes: set[str] | None = None
    max_rows_per_shard: int = 512

    def __post_init__(self) -> None:
        if self.max_rows_per_shard <= 0:
            raise ValueError("max_rows_per_shard must be positive")
        self.rows: list[dict[str, Any]] = []
        self.shards_written = 0
        self.total_rows = 0

    def record(
        self,
        prediction: Any,
        *,
        mode: str,
        task: str,
        task_id: int | None,
        episode: int,
        seed: int,
        step: int,
    ) -> None:
        if self.modes is not None and mode not in self.modes:
            return
        if getattr(prediction, "token_ids", None) is None:
            return
        token_ids = prediction.token_ids.detach().cpu().to(dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.numel() == 0:
            return
        stats = getattr(prediction, "stats", None) or {}
        stop_token_ids: list[int] = []
        if isinstance(stats, dict):
            raw_stop_tokens = stats.get("stop_token_ids")
            if raw_stop_tokens is not None:
                if isinstance(raw_stop_tokens, int):
                    stop_token_ids = [int(raw_stop_tokens)]
                elif isinstance(raw_stop_tokens, str):
                    stop_token_ids = [int(part.strip()) for part in raw_stop_tokens.split(",") if part.strip()]
                else:
                    stop_token_ids = [int(token) for token in raw_stop_tokens]
            elif stats.get("action_end_token_id") is not None:
                stop_token_ids = [int(stats["action_end_token_id"])]
        for batch_idx, row_tokens in enumerate(token_ids):
            row = {
                "token_ids": row_tokens.clone(),
                "task": task,
                "task_id": -1 if task_id is None else int(task_id),
                "episode": int(episode),
                "seed": int(seed),
                "step": int(step),
                "mode": mode,
                "batch_idx": int(batch_idx),
                "token_count": int(row_tokens.numel()),
                "trace_id": (
                    f"{task}:task{(-1 if task_id is None else int(task_id))}:"
                    f"seed{int(seed)}:episode{int(episode)}:step{int(step)}:"
                    f"mode{mode}:batch{int(batch_idx)}"
                ),
            }
            if stop_token_ids:
                row["stop_token_ids"] = list(stop_token_ids)
            if isinstance(stats, dict):
                flattened_stats = _flatten_scalar_stats(stats)
                if flattened_stats:
                    row["stats"] = flattened_stats
            self.rows.append(row)
            self.total_rows += 1
            if len(self.rows) >= self.max_rows_per_shard:
                self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = self.output_dir / f"shard_{self.shards_written:05d}.pt"
        torch.save(self.rows, shard_path)
        self.rows = []
        self.shards_written += 1
