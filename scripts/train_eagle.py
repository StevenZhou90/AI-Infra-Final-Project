#!/usr/bin/env python3
"""Train an EAGLE-1 draft head on OpenVLA hidden-state data.

Reads shards produced by generate_eagle_data.py and trains the lightweight
EAGLE head (fc + 1 decoder layer) to predict next tokens from the target
model's second-to-last layer hidden states.

Usage:
    python scripts/train_eagle.py \
        --data_dir data/eagle_train \
        --out_dir checkpoints/eagle_openvla \
        --pretrained openvla/openvla-7b \
        --epochs 15 --lr 3e-4 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving.eagle_draft import EagleDraftHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EagleShardDataset(Dataset):
    """Loads all .pt shards from a directory into memory.

    Each sample has:
      hidden_states: [7, 4096]  — target model second-to-last layer output per decode step
      token_ids:     [7]        — token fed at each step
      target_ids:    [7]        — greedy next token at each step
    """

    def __init__(self, data_dir: str | Path):
        self.samples = []
        data_dir = Path(data_dir)
        shard_files = sorted(data_dir.glob("shard_*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {data_dir}")
        for sf in shard_files:
            self.samples.extend(torch.load(sf, map_location="cpu"))
        logger.info("Loaded %d samples from %d shards", len(self.samples), len(shard_files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["hidden_states"], s["token_ids"], s["target_ids"]


def collate_fn(batch):
    hidden_states = torch.stack([b[0] for b in batch])  # [B, 7, 4096]
    token_ids = torch.stack([b[1] for b in batch])      # [B, 7]
    target_ids = torch.stack([b[2] for b in batch])     # [B, 7]
    return hidden_states, token_ids, target_ids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_target_head(pretrained: str, device: str, dtype: torch.dtype):
    """Load just norm + lm_head from OpenVLA (frozen, for computing logits)."""
    from transformers import AutoModelForVision2Seq

    logger.info("Loading OpenVLA for norm + lm_head: %s", pretrained)
    model = AutoModelForVision2Seq.from_pretrained(
        pretrained,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    llm = model.language_model
    norm = llm.model.norm
    lm_head = llm.lm_head
    embed_tokens = llm.model.embed_tokens

    logger.info(
        "Extracted norm (%s), lm_head (%s), embed_tokens (%s)",
        type(norm).__name__, type(lm_head).__name__, type(embed_tokens).__name__,
    )
    return model, norm, lm_head, embed_tokens


def build_eagle_head(embed_tokens: nn.Embedding, device: str, dtype: torch.dtype,
                     num_layers: int = 2):
    """Create a randomly-initialized EAGLE head and copy OpenVLA's embeddings."""
    vocab_size, hidden_size = embed_tokens.weight.shape

    head, config = EagleDraftHead.from_config(
        hidden_size=hidden_size,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        vocab_size=vocab_size,
        num_hidden_layers=num_layers,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        head.embed_tokens.weight.copy_(embed_tokens.weight)
    head.embed_tokens.weight.requires_grad = False
    logger.info("Copied embed_tokens from OpenVLA (frozen in EAGLE head)")

    return head, config


def train_loop(
    eagle_head: EagleDraftHead,
    norm: nn.Module,
    lm_head: nn.Module,
    dataset: EagleShardDataset,
    args,
    config: dict,
):
    """Train the EAGLE head with teacher-forcing.

    At each of the 7 positions, feed the real hidden state + real token from the
    target model, predict the next token.  Pure teacher-forcing is stable and
    converges reliably; the EAGLE head learns to approximate the target model's
    final two layers (layer[-1] + norm + lm_head).
    """
    device = args.device
    dtype = getattr(torch, args.dtype)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    trainable = [p for p in eagle_head.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("Trainable parameters: %.1fM", n_trainable / 1e6)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    total_steps = len(loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        eagle_head.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (hidden_states, token_ids, target_ids) in enumerate(loader):
            hidden_states = hidden_states.to(device, dtype=dtype)  # [B, 7, 4096]
            token_ids = token_ids.to(device)                       # [B, 7]
            target_ids = target_ids.to(device)                     # [B, 7]

            eagle_head.reset_kv()

            with torch.amp.autocast("cuda", dtype=dtype):
                eagle_out = eagle_head(
                    hidden_states, token_ids, use_cache=False, train_embeddings=False,
                )
                normed = norm(eagle_out)
                logits = lm_head(normed)  # [B, 7, vocab]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction="mean",
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == target_ids).sum().item()
                total = target_ids.numel()
                epoch_correct += correct
                epoch_total += total
                epoch_loss += loss.item()

            global_step += 1

            if global_step % args.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                acc = epoch_correct / max(epoch_total, 1)
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "step %d | epoch %d/%d | loss %.4f | acc %.1f%% | lr %.2e | %.0fs",
                    global_step, epoch + 1, args.epochs, avg_loss, acc * 100, lr_now, elapsed,
                )

        avg_epoch_loss = epoch_loss / max(len(loader), 1)
        epoch_acc = epoch_correct / max(epoch_total, 1)
        logger.info(
            "=== Epoch %d/%d complete: loss=%.4f  acc=%.1f%% ===",
            epoch + 1, args.epochs, avg_epoch_loss, epoch_acc * 100,
        )

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(eagle_head, config, out_dir)
            logger.info("Saved best checkpoint (loss=%.4f)", best_loss)

    save_checkpoint(eagle_head, config, out_dir, suffix="_final")
    elapsed = time.perf_counter() - t_start
    logger.info("Training complete in %.1fs. Best loss: %.4f", elapsed, best_loss)


def save_checkpoint(head: EagleDraftHead, config: dict, out_dir: Path, suffix: str = ""):
    ckpt_dir = out_dir if not suffix else out_dir / suffix.strip("_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_path = ckpt_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    weight_path = ckpt_dir / "pytorch_model.bin"
    torch.save(head.state_dict(), weight_path)
    logger.info("Checkpoint saved to %s", ckpt_dir)


def main():
    parser = argparse.ArgumentParser(description="Train EAGLE head on OpenVLA data")
    parser.add_argument("--data_dir", default="data/eagle_train")
    parser.add_argument("--out_dir", default="checkpoints/eagle_openvla")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_layers", type=int, default=2, help="Number of decoder layers in EAGLE head")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    dataset = EagleShardDataset(args.data_dir)
    full_model, norm, lm_head, embed_tokens = load_target_head(args.pretrained, args.device, dtype)
    eagle_head, config = build_eagle_head(embed_tokens, args.device, dtype, args.num_layers)

    del full_model.language_model.model.layers
    torch.cuda.empty_cache()
    logger.info("Freed target model decoder layers to save memory")

    train_loop(eagle_head, norm, lm_head, dataset, args, config)


if __name__ == "__main__":
    main()
