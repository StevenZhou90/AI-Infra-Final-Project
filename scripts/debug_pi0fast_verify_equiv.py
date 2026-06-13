#!/usr/bin/env python3
"""Debug PI0-FAST cached multi-token verify equivalence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_pi0fast_eagle_data import _ensure_libero_config  # noqa: E402
from scripts.run_pi0fast_chunk_eval import _import_lerobot, _prepare_observation  # noqa: E402
from serving.kv_cache_manager import clone_kv, trim_kv  # noqa: E402
from serving.pi0fast_token_hooks import PI0FastTokenLogitAdapter  # noqa: E402

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default="outputs/pi0fast_verify_equiv.json")
    parser.add_argument("--variant", choices=["triangular", "full", "prefix_only", "all"], default="all")
    parser.add_argument("--libero-config-path", default=os.environ.get("LIBERO_CONFIG_PATH"))
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    _ensure_libero_config(args.libero_config_path)
    make_env, make_env_pre_post_processors, preprocess_observation, LiberoEnv, make_pre_post_processors, PI0FastPolicy = _import_lerobot()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = getattr(torch, args.dtype)
    env_cfg = LiberoEnv(task=args.task, task_ids=[args.task_id], control_mode="relative")
    policy = PI0FastPolicy.from_pretrained("lerobot/pi0fast-libero").to(device=device, dtype=dtype).eval()
    adapter = PI0FastTokenLogitAdapter(policy)
    policy_preprocessor, _policy_postprocessor = make_pre_post_processors(
        policy.config,
        "lerobot/pi0fast-libero",
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = env_map[args.task][args.task_id]
    obs, _ = env.reset(seed=[args.seed])
    batch = _prepare_observation(obs, env, env_preprocessor, policy_preprocessor, preprocess_observation)

    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = adapter._language_tokens(batch)
    model = policy.model
    lm_head = model.paligemma_with_expert.paligemma.lm_head
    bsize = tokens.shape[0]
    bos = torch.full((bsize, 1), model._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device)
    tokens_in = torch.cat([tokens, bos], dim=1)
    masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
    prefix_embs, prefix_pad_masks, prefix_att_masks, *_ = model.embed_prefix_fast(
        images, img_masks, tokens_in, masks_in, fast_action_tokens=None, fast_action_masks=None
    )
    prefix_embs = adapter._match_model_precision(prefix_embs)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_4d = model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)
    (prefix_out, _), past = adapter._forward_prefix_language_model(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=prefix_embs,
        use_cache=True,
        cache_position=torch.arange(prefix_embs.shape[1], device=device, dtype=torch.long),
    )
    prev_logits = lm_head(prefix_out[:, -1:, :])
    seq_tokens: list[int] = []
    seq_logits = []
    current_pad = prefix_pad_masks
    seq_past = clone_kv(past)
    next_token = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)
    for _ in range(12):
        seq_tokens.append(int(next_token.item()))
        seq_logits.append(prev_logits)
        emb = model.paligemma_with_expert.embed_language_tokens(next_token)
        emb = emb * math.sqrt(emb.shape[-1])
        emb = emb.to(dtype=prefix_embs.dtype)
        current_pad = torch.cat([current_pad, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
        pos = (torch.sum(current_pad, dim=1, keepdim=True) - 1).long()
        mask = model._prepare_attention_masks_4d(current_pad.unsqueeze(1), dtype=emb.dtype)
        (out, _), seq_past = adapter._forward_prefix_language_model(
            attention_mask=mask,
            position_ids=pos,
            past_key_values=seq_past,
            inputs_embeds=emb,
            use_cache=True,
            cache_position=torch.tensor([current_pad.shape[1] - 1], device=device, dtype=torch.long),
        )
        prev_logits = lm_head(out[:, -1:, :])
        next_token = torch.argmax(prev_logits[:, -1], dim=-1, keepdim=True)

    draft = torch.tensor([seq_tokens[:6]], dtype=torch.long, device=device)
    old_len = prefix_pad_masks.shape[1]
    draft_emb = model.paligemma_with_expert.embed_language_tokens(draft)
    draft_emb = draft_emb * math.sqrt(draft_emb.shape[-1])
    draft_emb = draft_emb.to(dtype=prefix_embs.dtype)
    verify_pad = torch.cat([prefix_pad_masks, torch.ones((bsize, draft.shape[1]), dtype=torch.bool, device=device)], dim=1)
    pos_start = int(torch.sum(prefix_pad_masks, dim=1).item())
    pos_ids = torch.arange(pos_start, pos_start + draft.shape[1], device=device, dtype=torch.long).unsqueeze(0)

    masks_by_name = {}
    triangular = torch.zeros((bsize, draft.shape[1], old_len + draft.shape[1]), dtype=torch.bool, device=device)
    for row in range(draft.shape[1]):
        triangular[:, row, : old_len + row + 1] = verify_pad[:, : old_len + row + 1]
    masks_by_name["triangular"] = triangular
    full = torch.zeros_like(triangular)
    full[:, :, :] = verify_pad[:, None, :]
    masks_by_name["full"] = full
    prefix_only = torch.zeros_like(triangular)
    prefix_only[:, :, :old_len] = prefix_pad_masks[:, None, :]
    masks_by_name["prefix_only"] = prefix_only

    variants = {}
    selected = masks_by_name if args.variant == "all" else {args.variant: masks_by_name[args.variant]}
    for name, mask2d in selected.items():
        (out, _), variant_kv = adapter._forward_prefix_language_model(
            attention_mask=model._prepare_attention_masks_4d(mask2d, dtype=draft_emb.dtype),
            position_ids=pos_ids,
            past_key_values=clone_kv(past),
            inputs_embeds=draft_emb,
            use_cache=True,
            cache_position=pos_ids.squeeze(0),
        )
        logits = lm_head(out)
        logits_all = torch.cat([seq_logits[0], logits], dim=1)
        predictions = [int(x) for x in torch.argmax(logits_all[:, : draft.shape[1]], dim=-1)[0].tolist()]
        logit_diffs = [
            float((logits_all[:, idx : idx + 1].float() - seq_logits[idx].float()).abs().max().item())
            for idx in range(draft.shape[1])
        ]
        partial_accept = min(3, draft.shape[1] - 1)
        trimmed_kv = trim_kv(variant_kv, old_len + partial_accept)
        trimmed_pad = verify_pad[:, : old_len + partial_accept]
        trimmed_prev_logits = logits_all[:, partial_accept : partial_accept + 1, :]
        correction = torch.tensor([[seq_tokens[partial_accept]]], dtype=torch.long, device=device)
        correction_emb = model.paligemma_with_expert.embed_language_tokens(correction)
        correction_emb = correction_emb * math.sqrt(correction_emb.shape[-1])
        correction_emb = correction_emb.to(dtype=prefix_embs.dtype)
        after_correction_pad = torch.cat(
            [trimmed_pad, torch.ones((bsize, 1), dtype=torch.bool, device=device)],
            dim=1,
        )
        correction_pos = (torch.sum(after_correction_pad, dim=1, keepdim=True) - 1).long()
        correction_mask = model._prepare_attention_masks_4d(
            after_correction_pad.unsqueeze(1),
            dtype=correction_emb.dtype,
        )
        (correction_out, _), _ = adapter._forward_prefix_language_model(
            attention_mask=correction_mask,
            position_ids=correction_pos,
            past_key_values=trimmed_kv,
            inputs_embeds=correction_emb,
            use_cache=True,
            cache_position=torch.tensor([after_correction_pad.shape[1] - 1], device=device, dtype=torch.long),
        )
        after_correction_logits = lm_head(correction_out[:, -1:, :])
        variants[name] = {
            "predictions": predictions,
            "matches_sequential_tokens": [p == int(seq_tokens[idx]) for idx, p in enumerate(predictions)],
            "max_logit_diff_by_position": logit_diffs,
            "partial_accept": partial_accept,
            "trimmed_next_prediction": int(torch.argmax(trimmed_prev_logits[:, -1], dim=-1).item()),
            "sequential_next_at_trim": int(seq_tokens[partial_accept]),
            "after_correction_prediction": int(torch.argmax(after_correction_logits[:, -1], dim=-1).item()),
            "sequential_after_correction": int(seq_tokens[partial_accept + 1]),
            "after_correction_logit_diff": float(
                (after_correction_logits.float() - seq_logits[partial_accept + 1].float()).abs().max().item()
            ),
        }

    # Reproduce the SD state after emitting the first token, then verify the next block.
    first = torch.tensor([[seq_tokens[0]]], dtype=torch.long, device=device)
    first_emb = model.paligemma_with_expert.embed_language_tokens(first)
    first_emb = first_emb * math.sqrt(first_emb.shape[-1])
    first_emb = first_emb.to(dtype=prefix_embs.dtype)
    after_first_pad = torch.cat([prefix_pad_masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)
    first_pos = (torch.sum(after_first_pad, dim=1, keepdim=True) - 1).long()
    first_mask = model._prepare_attention_masks_4d(after_first_pad.unsqueeze(1), dtype=first_emb.dtype)
    (first_out, _), after_first_past = adapter._forward_prefix_language_model(
        attention_mask=first_mask,
        position_ids=first_pos,
        past_key_values=clone_kv(past),
        inputs_embeds=first_emb,
        use_cache=True,
        cache_position=torch.tensor([after_first_pad.shape[1] - 1], device=device, dtype=torch.long),
    )
    after_first_prev = lm_head(first_out[:, -1:, :])
    after_first_draft = torch.tensor([seq_tokens[1:5]], dtype=torch.long, device=device)
    after_first_emb = model.paligemma_with_expert.embed_language_tokens(after_first_draft)
    after_first_emb = after_first_emb * math.sqrt(after_first_emb.shape[-1])
    after_first_emb = after_first_emb.to(dtype=prefix_embs.dtype)
    old_len2 = after_first_pad.shape[1]
    verify_pad2 = torch.cat(
        [after_first_pad, torch.ones((bsize, after_first_draft.shape[1]), dtype=torch.bool, device=device)], dim=1
    )
    tri2 = torch.zeros((bsize, after_first_draft.shape[1], old_len2 + after_first_draft.shape[1]), dtype=torch.bool, device=device)
    for row in range(after_first_draft.shape[1]):
        tri2[:, row, : old_len2 + row + 1] = verify_pad2[:, : old_len2 + row + 1]
    pos_start2 = int(torch.sum(after_first_pad, dim=1).item())
    pos_ids2 = torch.arange(pos_start2, pos_start2 + after_first_draft.shape[1], device=device, dtype=torch.long).unsqueeze(0)
    (after_first_verify_out, _), _ = adapter._forward_prefix_language_model(
        attention_mask=model._prepare_attention_masks_4d(tri2, dtype=after_first_emb.dtype),
        position_ids=pos_ids2,
        past_key_values=clone_kv(after_first_past),
        inputs_embeds=after_first_emb,
        use_cache=True,
        cache_position=pos_ids2.squeeze(0),
    )
    after_first_logits = lm_head(after_first_verify_out)
    after_first_logits_all = torch.cat([after_first_prev, after_first_logits], dim=1)

    result = {
        "sequential_tokens": seq_tokens[:12],
        "sequential_next_after_inputs": [int(torch.argmax(l[:, -1], dim=-1).item()) for l in seq_logits[:8]],
        "draft": [int(x) for x in draft[0].tolist()],
        "verify_argmax_by_variant": variants,
        "after_first_prev": int(torch.argmax(after_first_prev[:, -1], dim=-1).item()),
        "after_first_draft": [int(x) for x in after_first_draft[0].tolist()],
        "after_first_verify_argmax": [
            int(x) for x in torch.argmax(after_first_logits_all[:, : after_first_draft.shape[1]], dim=-1)[0].tolist()
        ],
        "after_first_matches": [
            int(x) == int(seq_tokens[idx + 1])
            for idx, x in enumerate(torch.argmax(after_first_logits_all[:, : after_first_draft.shape[1]], dim=-1)[0].tolist())
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
