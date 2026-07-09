#!/usr/bin/env python3
"""Build an empirical PI0-FAST candidate-token whitelist from trace shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def _parse_modes(value: str | None) -> set[str] | None:
    if value is None:
        return None
    modes = {part.strip() for part in value.split(",") if part.strip()}
    if not modes or modes & {"*", "all"}:
        return None
    return modes


def _load_shard(path: Path) -> list[dict[str, Any]]:
    try:
        rows = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        rows = torch.load(path, map_location="cpu")
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of trace rows")
    return rows


def _row_tokens(row: dict[str, Any], *, include_stop_tokens: bool) -> list[int]:
    raw_tokens = row.get("token_ids")
    if raw_tokens is None:
        return []
    if torch.is_tensor(raw_tokens):
        token_ids = [int(token_id) for token_id in raw_tokens.flatten().tolist()]
    else:
        token_ids = [int(token_id) for token_id in raw_tokens]
    if include_stop_tokens:
        raw_stop_tokens = row.get("stop_token_ids") or []
        if isinstance(raw_stop_tokens, int):
            token_ids.append(int(raw_stop_tokens))
        else:
            token_ids.extend(int(token_id) for token_id in raw_stop_tokens)
    return token_ids


def collect_token_counts(
    data_dir: Path,
    *,
    modes: set[str] | None = None,
    include_stop_tokens: bool = False,
) -> tuple[Counter[int], dict[str, int]]:
    counts: Counter[int] = Counter()
    row_count = 0
    total_tokens = 0
    shard_count = 0
    for shard_path in sorted(data_dir.rglob("shard_*.pt")):
        shard_count += 1
        for row in _load_shard(shard_path):
            if not isinstance(row, dict):
                continue
            if modes is not None and str(row.get("mode", "")) not in modes:
                continue
            token_ids = _row_tokens(row, include_stop_tokens=include_stop_tokens)
            if not token_ids:
                continue
            counts.update(token_ids)
            row_count += 1
            total_tokens += len(token_ids)
    return counts, {"shards": shard_count, "rows": row_count, "total_tokens": total_tokens}


def select_token_ids(counts: Counter[int], *, min_count: int = 1, top_k: int = 0) -> list[int]:
    min_count = max(1, int(min_count))
    selected = [(token_id, count) for token_id, count in counts.items() if count >= min_count]
    if top_k > 0:
        selected = sorted(selected, key=lambda item: (-item[1], item[0]))[: int(top_k)]
    return sorted(token_id for token_id, _count in selected)


def expand_token_ids(
    token_ids: list[int],
    *,
    radius: int = 0,
    min_token_id: int | None = None,
    max_token_id: int | None = None,
) -> list[int]:
    """Return token ids plus optional local neighborhoods around eligible ids."""

    radius = max(0, int(radius))
    expanded = {int(token_id) for token_id in token_ids}
    if radius <= 0:
        return sorted(expanded)
    for token_id in token_ids:
        token_id = int(token_id)
        if min_token_id is not None and token_id < int(min_token_id):
            continue
        if max_token_id is not None and token_id > int(max_token_id):
            continue
        lo = token_id - radius
        hi = token_id + radius
        if min_token_id is not None:
            lo = max(lo, int(min_token_id))
        if max_token_id is not None:
            hi = min(hi, int(max_token_id))
        expanded.update(range(lo, hi + 1))
    return sorted(expanded)


def build_vocab_payload(
    data_dir: Path,
    *,
    modes: set[str] | None = None,
    min_count: int = 1,
    top_k: int = 0,
    include_stop_tokens: bool = False,
    expand_radius: int = 0,
    expand_min_token_id: int | None = None,
    expand_max_token_id: int | None = None,
) -> dict[str, Any]:
    counts, metadata = collect_token_counts(
        data_dir,
        modes=modes,
        include_stop_tokens=include_stop_tokens,
    )
    base_token_ids = select_token_ids(counts, min_count=min_count, top_k=top_k)
    token_ids = expand_token_ids(
        base_token_ids,
        radius=expand_radius,
        min_token_id=expand_min_token_id,
        max_token_id=expand_max_token_id,
    )
    selected_counts = {str(token_id): int(counts[token_id]) for token_id in base_token_ids}
    return {
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "base_token_count": len(base_token_ids),
        "expanded_token_count": len(token_ids) - len(base_token_ids),
        "source_dir": str(data_dir),
        "shards": metadata["shards"],
        "rows": metadata["rows"],
        "total_tokens": metadata["total_tokens"],
        "min_count": max(1, int(min_count)),
        "top_k": max(0, int(top_k)),
        "mode_filter": None if modes is None else sorted(modes),
        "include_stop_tokens": bool(include_stop_tokens),
        "expand_radius": max(0, int(expand_radius)),
        "expand_min_token_id": None if expand_min_token_id is None else int(expand_min_token_id),
        "expand_max_token_id": None if expand_max_token_id is None else int(expand_max_token_id),
        "counts": selected_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing TokenTraceSink shard_*.pt files.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the empirical vocab JSON.")
    parser.add_argument("--modes", default=None, help="Comma-separated trace modes to include, or all/*.")
    parser.add_argument("--min-count", type=int, default=1, help="Keep tokens seen at least this many times.")
    parser.add_argument("--top-k", type=int, default=0, help="Keep only the k most frequent tokens after min-count.")
    parser.add_argument(
        "--expand-radius",
        type=int,
        default=0,
        help="Add +/- radius token-id neighborhoods around selected tokens.",
    )
    parser.add_argument(
        "--expand-min-token-id",
        type=int,
        default=None,
        help="Only expand neighborhoods for selected tokens at or above this id.",
    )
    parser.add_argument(
        "--expand-max-token-id",
        type=int,
        default=None,
        help="Only expand neighborhoods for selected tokens at or below this id.",
    )
    parser.add_argument(
        "--include-stop-tokens",
        action="store_true",
        help="Also count explicit stop_token_ids metadata when present.",
    )
    args = parser.parse_args()

    payload = build_vocab_payload(
        args.data_dir,
        modes=_parse_modes(args.modes),
        min_count=args.min_count,
        top_k=args.top_k,
        include_stop_tokens=args.include_stop_tokens,
        expand_radius=args.expand_radius,
        expand_min_token_id=args.expand_min_token_id,
        expand_max_token_id=args.expand_max_token_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {payload['token_count']} token ids from {payload['rows']} rows "
        f"across {payload['shards']} shards to {args.output}"
    )


if __name__ == "__main__":
    main()
