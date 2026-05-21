"""PI0-FAST EAGLE-style FAST-token drafter utilities.

This module keeps the PI0-FAST experiment offline-friendly: traces contain
target hidden states and generated FAST token ids; the EAGLE head learns to
predict the next FAST token under teacher forcing; evaluation simulates exact
speculative-prefix acceptance on held-out traces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from serving.eagle_draft import EagleDraftHead


@dataclass
class PI0FastEagleConfig:
    hidden_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 8192
    num_hidden_layers: int = 1
    rms_norm_eps: float = 1e-6
    train_embeddings: bool = False


@dataclass
class PI0FastTraceRecord:
    hidden_states: torch.Tensor
    token_ids: torch.Tensor
    task_id: int
    seed: int
    trace_id: str
    decode_ms: float | None = None
    success: bool | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PI0FastTraceRecord":
        return cls(
            hidden_states=row["hidden_states"].to(dtype=torch.float32),
            token_ids=row["token_ids"].to(dtype=torch.long),
            task_id=int(row.get("task_id", -1)),
            seed=int(row.get("seed", -1)),
            trace_id=str(row.get("trace_id", "")),
            decode_ms=None if row.get("decode_ms") is None else float(row["decode_ms"]),
            success=None if row.get("success") is None else bool(row["success"]),
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["hidden_states"] = self.hidden_states.cpu()
        row["token_ids"] = self.token_ids.cpu()
        return row


class CompactTokenMap:
    """Map sparse PaliGemma FAST token ids to compact class ids."""

    def __init__(self, token_values: Iterable[int]) -> None:
        values = sorted({int(t) for t in token_values})
        if not values:
            raise ValueError("CompactTokenMap requires at least one token")
        self.token_values = values
        self.token_to_class = {token_id: idx for idx, token_id in enumerate(values)}

    def __len__(self) -> int:
        return len(self.token_values)

    def encode_tensor(self, token_ids: torch.Tensor, unknown_value: int = -100) -> torch.Tensor:
        flat = token_ids.detach().cpu().reshape(-1).tolist()
        encoded = [self.token_to_class.get(int(t), unknown_value) for t in flat]
        return torch.tensor(encoded, dtype=torch.long).reshape(token_ids.shape)

    def decode_tensor(self, class_ids: torch.Tensor) -> torch.Tensor:
        values = torch.tensor(self.token_values, dtype=torch.long, device=class_ids.device)
        return values[class_ids.to(dtype=torch.long)]

    def to_dict(self) -> dict[str, Any]:
        return {"token_values": self.token_values}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CompactTokenMap":
        return cls(row["token_values"])


class PI0FastEagleHead(nn.Module):
    """EAGLE recurrent head with compact FAST-token logits."""

    def __init__(self, config: PI0FastEagleConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.eagle, self.eagle_config = EagleDraftHead.from_config(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            vocab_size=vocab_size,
            num_hidden_layers=config.num_hidden_layers,
            rms_norm_eps=config.rms_norm_eps,
            device="cpu",
            dtype=torch.float32,
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def reset_kv(self) -> None:
        self.eagle.reset_kv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_class_ids: torch.Tensor,
        *,
        use_cache: bool = False,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        draft_hidden = self.eagle(
            hidden_states,
            input_class_ids,
            use_cache=use_cache,
            train_embeddings=self.config.train_embeddings,
        )
        logits = self.output_head(self.norm(draft_hidden))
        if return_hidden:
            return logits, draft_hidden
        return logits

    def initialize_embeddings_from_target(
        self,
        target_embedding_weight: torch.Tensor,
        token_map: CompactTokenMap,
    ) -> None:
        """Copy target PaliGemma embeddings for compact FAST-token ids."""

        token_ids = torch.tensor(token_map.token_values, dtype=torch.long, device=target_embedding_weight.device)
        with torch.no_grad():
            copied = target_embedding_weight.index_select(0, token_ids).to(
                device=self.eagle.embed_tokens.weight.device,
                dtype=self.eagle.embed_tokens.weight.dtype,
            )
            self.eagle.embed_tokens.weight.copy_(copied)
        self.eagle.embed_tokens.weight.requires_grad = self.config.train_embeddings


def load_trace_records(data_dir: str | Path) -> list[PI0FastTraceRecord]:
    data_path = Path(data_dir)
    shard_files = sorted(data_path.glob("shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.pt files found in {data_path}")
    records: list[PI0FastTraceRecord] = []
    for shard in shard_files:
        rows = torch.load(shard, map_location="cpu", weights_only=False)
        records.extend(PI0FastTraceRecord.from_dict(row) for row in rows)
    return records


def save_trace_shard(records: list[PI0FastTraceRecord], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save([record.to_dict() for record in records], out)


def split_trace_records(
    records: list[PI0FastTraceRecord],
    *,
    split: str,
    val_fraction: float,
    seed: int,
    heldout_task_id: int | None = None,
) -> tuple[list[int], list[int]]:
    if len(records) < 2:
        raise ValueError("Need at least two traces for a train/validation split")

    if split == "task":
        if heldout_task_id is None:
            heldout_task_id = max(record.task_id for record in records)
        train = [idx for idx, record in enumerate(records) if record.task_id != heldout_task_id]
        val = [idx for idx, record in enumerate(records) if record.task_id == heldout_task_id]
    elif split == "trace":
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(records), generator=generator).tolist()
        n_val = max(1, int(round(len(records) * val_fraction)))
        n_val = min(n_val, len(records) - 1)
        val_set = set(perm[:n_val])
        train = [idx for idx in range(len(records)) if idx not in val_set]
        val = [idx for idx in range(len(records)) if idx in val_set]
    else:
        raise ValueError(f"Unsupported split: {split}")

    if not train or not val:
        raise ValueError(f"Split produced empty train or val set: train={len(train)} val={len(val)}")
    return train, val


def build_compact_token_map(records: list[PI0FastTraceRecord], indices: Iterable[int]) -> CompactTokenMap:
    tokens: list[int] = []
    for idx in indices:
        tokens.extend(int(t) for t in records[idx].token_ids.tolist())
    return CompactTokenMap(tokens)


def make_teacher_forcing_rows(
    records: list[PI0FastTraceRecord],
    indices: Iterable[int],
    token_map: CompactTokenMap,
    *,
    drop_oov: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_rows: list[torch.Tensor] = []
    input_classes: list[int] = []
    target_classes: list[int] = []
    trace_indices: list[int] = []

    for idx in indices:
        record = records[idx]
        if record.token_ids.numel() < 2:
            continue
        encoded = token_map.encode_tensor(record.token_ids)
        for pos in range(record.token_ids.numel() - 1):
            inp = int(encoded[pos])
            tgt = int(encoded[pos + 1])
            if drop_oov and (inp < 0 or tgt < 0):
                continue
            hidden_rows.append(record.hidden_states[pos].to(dtype=torch.float32))
            input_classes.append(inp)
            target_classes.append(tgt)
            trace_indices.append(idx)

    if not hidden_rows:
        raise ValueError("No teacher-forcing rows; check trace length and compact vocab coverage")
    return (
        torch.stack(hidden_rows),
        torch.tensor(input_classes, dtype=torch.long),
        torch.tensor(target_classes, dtype=torch.long),
        torch.tensor(trace_indices, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate_offline_acceptance(
    model: PI0FastEagleHead,
    records: list[PI0FastTraceRecord],
    indices: Iterable[int],
    token_map: CompactTokenMap,
    *,
    lookahead: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    model.eval()
    device = torch.device(device)
    model = model.to(device=device, dtype=dtype)

    accepted_counts: list[int] = []
    next_correct = 0
    next_total = 0
    total_oov_targets = 0
    total_future_targets = 0
    per_task: dict[int, list[int]] = {}

    for idx in indices:
        record = records[idx]
        tokens = record.token_ids
        if tokens.numel() < 2:
            continue
        encoded = token_map.encode_tensor(tokens)
        positions = max(0, tokens.numel() - 1)
        for pos in range(positions):
            max_k = min(lookahead, tokens.numel() - pos - 1)
            future = encoded[pos + 1 : pos + 1 + max_k]
            total_future_targets += int(future.numel())
            total_oov_targets += int((future < 0).sum().item())

            current_class = int(encoded[pos])
            if current_class < 0:
                accepted_counts.append(0)
                per_task.setdefault(record.task_id, []).append(0)
                continue

            hidden = record.hidden_states[pos : pos + 1].to(device=device, dtype=dtype).unsqueeze(0)
            input_class = torch.tensor([[current_class]], dtype=torch.long, device=device)
            model.reset_kv()
            accepted = 0
            for offset in range(max_k):
                logits, draft_hidden = model(hidden, input_class, use_cache=True, return_hidden=True)
                pred_class = int(logits[:, -1].argmax(dim=-1).item())
                target_class = int(future[offset])
                if offset == 0 and target_class >= 0:
                    next_total += 1
                    next_correct += int(pred_class == target_class)
                if target_class < 0 or pred_class != target_class:
                    break
                accepted += 1
                hidden = draft_hidden[:, -1:, :]
                input_class = torch.tensor([[pred_class]], dtype=torch.long, device=device)
            accepted_counts.append(accepted)
            per_task.setdefault(record.task_id, []).append(accepted)

    if not accepted_counts:
        raise ValueError("No positions evaluated")

    accepted_tensor = torch.tensor(accepted_counts, dtype=torch.float32)
    metrics: dict[str, Any] = {
        "positions": len(accepted_counts),
        "lookahead": int(lookahead),
        "next_token_accuracy": next_correct / max(next_total, 1),
        "mean_spec_accept": float(accepted_tensor.mean().item()),
        "p_accept_ge_1": float((accepted_tensor >= 1).float().mean().item()),
        "p_accept_ge_2": float((accepted_tensor >= 2).float().mean().item()),
        "p_accept_full": float((accepted_tensor >= lookahead).float().mean().item()),
        "future_oov_rate": total_oov_targets / max(total_future_targets, 1),
        "per_task": {},
    }
    for task_id, values in sorted(per_task.items()):
        task_tensor = torch.tensor(values, dtype=torch.float32)
        metrics["per_task"][str(task_id)] = {
            "positions": len(values),
            "mean_spec_accept": float(task_tensor.mean().item()),
            "p_accept_ge_2": float((task_tensor >= 2).float().mean().item()),
        }
    return metrics


def save_checkpoint(
    model: PI0FastEagleHead,
    out_dir: str | Path,
    *,
    config: PI0FastEagleConfig,
    token_map: CompactTokenMap,
    summary: dict[str, Any],
) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": cpu_state,
            "config": asdict(config),
            "token_map": token_map.to_dict(),
            "summary": summary,
        },
        path / "pi0fast_eagle.pt",
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[PI0FastEagleHead, CompactTokenMap, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = PI0FastEagleConfig(**ckpt["config"])
    token_map = CompactTokenMap.from_dict(ckpt["token_map"])
    model = PI0FastEagleHead(config, len(token_map))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, token_map, ckpt.get("summary", {})
