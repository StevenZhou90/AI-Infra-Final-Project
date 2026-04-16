"""OpenVLA 7B policy wrapper with KV-cache management and speculative decoding.

Usage (standalone):
    from policies.openvla_policy import OpenVLAPolicyWrapper, OpenVLAPolicyConfig
    cfg = OpenVLAPolicyConfig(pretrained_path="openvla/openvla-7b")
    policy = OpenVLAPolicyWrapper(cfg)
    action = policy.predict(observation, instruction="pick up the cube")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from serving.kv_cache_manager import KVCacheManager, PastKV

logger = logging.getLogger(__name__)


@dataclass
class OpenVLAPolicyConfig:
    pretrained_path: str = "openvla/openvla-7b"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    # KV cache
    use_kv_cache: bool = True
    max_cache_memory_mb: float = 4096
    sliding_window: int | None = None
    # Speculative decoding
    use_speculative_decoding: bool = False
    speculative_lookahead: int = 7
    draft_layers_fraction: float = 0.5
    # Task
    instruction: str = "pick up the object"
    unnorm_key: str | None = None


class OpenVLAPolicyWrapper:
    """Loads OpenVLA from HuggingFace, provides predict() with KV cache + speculative decoding."""

    ACTION_DIM = 7  # x, y, z, rx, ry, rz, gripper

    def __init__(self, config: OpenVLAPolicyConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._dtype = getattr(torch, config.torch_dtype)
        self._processor, self._model = self._load_model(config)

        self._kv_mgr = KVCacheManager(
            max_memory_mb=config.max_cache_memory_mb,
            sliding_window=config.sliding_window,
            device=self._device,
        )

        # Speculative decoder (optional)
        self._spec_decoder = None
        if config.use_speculative_decoding:
            from serving.speculative_decoder import SpeculativeDecoder

            n_draft = None
            if config.draft_layers_fraction < 1.0:
                total = self._count_decoder_layers()
                n_draft = max(int(total * config.draft_layers_fraction), 1)

            # OpenVLA's top-level forward() only supports single-token decode
            # with KV cache; pass the underlying LLM for multi-token verify.
            decode_model = getattr(self._model, "language_model", self._model)

            self._spec_decoder = SpeculativeDecoder(
                model=self._model,
                decode_model=decode_model,
                lookahead=config.speculative_lookahead,
                draft_layers=n_draft,
                device=self._device,
            )

        self._prefix_cache_key: str | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self, config: OpenVLAPolicyConfig):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info("Loading OpenVLA from: %s", config.pretrained_path)

        processor = AutoProcessor.from_pretrained(
            config.pretrained_path, trust_remote_code=True,
        )

        model = AutoModelForVision2Seq.from_pretrained(
            config.pretrained_path,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self._device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info(
            "OpenVLA loaded on %s  params=%.2fB  dtype=%s",
            self._device, n_params, config.torch_dtype,
        )
        return processor, model

    def _count_decoder_layers(self) -> int:
        from serving.speculative_decoder import _LayerSkipDraft

        root, attr = _LayerSkipDraft._find_layers(self._model)
        layers = root
        for part in attr.split("."):
            layers = getattr(layers, part)
        return len(layers)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear cached state between episodes."""
        self._kv_mgr.clear_all()
        self._prefix_cache_key = None
        if self._spec_decoder is not None:
            self._spec_decoder.stats = type(self._spec_decoder.stats)()

    def predict(
        self,
        observation: dict[str, torch.Tensor],
        instruction: str | None = None,
    ) -> np.ndarray:
        """Generate a 7-DOF action from an image observation + language instruction."""
        instruction = instruction or self._config.instruction

        image_tensor = self._get_image_tensor(observation)
        pil_image = self._tensor_to_pil(image_tensor)

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self._processor(prompt, pil_image).to(self._device, dtype=self._dtype)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")

        prefill_kwargs: dict = {}
        if pixel_values is not None:
            prefill_kwargs["pixel_values"] = pixel_values
        if attention_mask is not None:
            prefill_kwargs["attention_mask"] = attention_mask

        # Try to reuse a cached instruction prefix from a previous timestep.
        # The image changes each step, so the prefix only covers the text
        # instruction tokens.  When the instruction is identical across steps
        # (very common in robot control), this avoids re-computing those KVs.
        cached_kv = None
        prefix_hash = ""
        if self._config.use_kv_cache:
            prefix_hash = KVCacheManager.hash_prefix(instruction)
            cached_kv = self._kv_mgr.get_by_prefix(prefix_hash)

        with torch.no_grad():
            if self._spec_decoder is not None:
                generated_ids, final_kv = self._spec_decoder.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.ACTION_DIM,
                    prefill_kwargs=prefill_kwargs,
                )
            else:
                generated_ids, final_kv = self._generate_with_kv_cache(
                    input_ids, prefill_kwargs,
                )

        if self._config.use_kv_cache and final_kv is not None:
            self._kv_mgr.put("latest", final_kv, prefix_hash=prefix_hash)

        action = self._decode_actions(generated_ids, input_ids)
        return action

    def _generate_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        prefill_kwargs: dict,
    ) -> tuple[torch.Tensor, PastKV | None]:
        """Standard autoregressive generation with HuggingFace use_cache.

        Returns (generated_ids, final_kv) so the caller can store the KV
        in the cache manager for cross-timestep prefix reuse.
        """
        out = self._model.generate(
            input_ids,
            max_new_tokens=self.ACTION_DIM,
            do_sample=False,
            use_cache=self._config.use_kv_cache,
            return_dict_in_generate=True,
            **prefill_kwargs,
        )
        final_kv = getattr(out, "past_key_values", None)
        return out.sequences, final_kv

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_actions(
        self,
        generated_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> np.ndarray:
        """Convert generated token IDs into continuous action values."""
        action_ids = generated_ids[0, input_ids.shape[1]:]
        action_strs = self._processor.batch_decode(
            action_ids.unsqueeze(1) if action_ids.dim() == 1 else action_ids,
            skip_special_tokens=True,
        )

        discrete: list[int] = []
        for s in action_strs:
            s = s.strip()
            if s.isdigit():
                discrete.append(min(int(s), 255))
            else:
                # Fallback: try to extract a number
                nums = [c for c in s if c.isdigit()]
                discrete.append(int("".join(nums)) if nums else 128)

        while len(discrete) < self.ACTION_DIM:
            discrete.append(128)  # neutral default (mid-bin)

        discrete = discrete[: self.ACTION_DIM]
        raw = np.array(discrete, dtype=np.float32)

        # If the model exposes per-dataset unnormalization, use it
        if (
            self._config.unnorm_key
            and hasattr(self._model, "norm_stats")
            and self._config.unnorm_key in self._model.norm_stats
        ):
            stats = self._model.norm_stats[self._config.unnorm_key]["action"]
            lo = np.array(stats.get("q01", stats.get("min", [0] * self.ACTION_DIM)), dtype=np.float32)
            hi = np.array(stats.get("q99", stats.get("max", [1] * self.ACTION_DIM)), dtype=np.float32)
            actions = lo + (raw / 255.0) * (hi - lo)
        else:
            actions = (raw / 255.0) * 2 - 1  # default: map [0,255] → [-1, 1]

        return actions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_image_tensor(observation: dict[str, torch.Tensor]) -> torch.Tensor:
        for key in observation:
            if "image" in key:
                return observation[key]
        raise ValueError(f"No image key in observation: {list(observation.keys())}")

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        arr = (tensor.cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _param_count(model: torch.nn.Module) -> float:
        return sum(p.numel() for p in model.parameters()) / 1e6
