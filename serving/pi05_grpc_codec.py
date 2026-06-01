"""Serialization helpers for PI0.5 gRPC serving."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from proto import inference_pb2


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


def encode_prepared_observation_fields(observation: Mapping[str, Any]) -> list[inference_pb2.ObservationField]:
    fields: list[inference_pb2.ObservationField] = []
    for name, value in observation.items():
        if torch.is_tensor(value):
            cpu_value = value.detach().contiguous().cpu()
            if cpu_value.dtype is torch.bfloat16:
                cpu_value = cpu_value.float()
            fields.append(
                inference_pb2.ObservationField(
                    name=name,
                    dtype=str(cpu_value.dtype).replace("torch.", ""),
                    shape=list(cpu_value.shape),
                    data=cpu_value.numpy().tobytes(),
                )
            )
        elif isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            fields.append(
                inference_pb2.ObservationField(
                    name=name,
                    dtype=str(arr.dtype),
                    shape=list(arr.shape),
                    data=arr.tobytes(),
                )
            )
        else:
            fields.append(inference_pb2.ObservationField(name=name, json_value=json.dumps(value)))
    return fields


def decode_prepared_observation_fields(
    fields: list[inference_pb2.ObservationField] | Any,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for field in fields:
        if field.json_value:
            observation[field.name] = json.loads(field.json_value)
            continue
        dtype = _torch_dtype(field.dtype)
        np_dtype = _numpy_dtype(field.dtype)
        arr = np.frombuffer(field.data, dtype=np_dtype).copy().reshape(tuple(field.shape))
        observation[field.name] = torch.from_numpy(arr).to(device=device or torch.device("cpu"), dtype=dtype)
    return observation


def _torch_dtype(name: str) -> torch.dtype:
    if name in {"float32", "float"}:
        return torch.float32
    if name in {"float16", "half"}:
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name in {"int64", "long"}:
        return torch.int64
    if name in {"int32", "int"}:
        return torch.int32
    if name == "uint8":
        return torch.uint8
    if name == "bool":
        return torch.bool
    raise ValueError(f"Unsupported tensor field dtype: {name}")


def _numpy_dtype(name: str) -> np.dtype:
    if name == "bfloat16":
        return np.float32
    if name in {"float", "float32"}:
        return np.float32
    if name in {"half", "float16"}:
        return np.float16
    if name in {"long", "int64"}:
        return np.int64
    if name in {"int", "int32"}:
        return np.int32
    if name == "uint8":
        return np.uint8
    if name == "bool":
        return np.bool_
    raise ValueError(f"Unsupported tensor field dtype: {name}")
