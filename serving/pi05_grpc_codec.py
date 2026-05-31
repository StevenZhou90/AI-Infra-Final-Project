"""Serialization helpers for PI0.5 gRPC serving."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import torch


def encode_prepared_observation(observation: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(observation), buffer)
    return buffer.getvalue()


def decode_prepared_observation(payload: bytes, *, device: torch.device | None = None) -> dict[str, Any]:
    if not payload:
        raise ValueError("prepared_observation payload is empty")
    loaded = torch.load(io.BytesIO(payload), map_location=device or "cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected prepared observation dict, got {type(loaded).__name__}")
    return loaded
