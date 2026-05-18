#!/usr/bin/env python3
"""Run short LIBERO spec-only speed sweeps from a base mirror config."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def read_summary(run_root: Path) -> dict[str, Any]:
    runs = sorted(run_root.glob("libero_mirror_smoke_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return {}
    summary = runs[-1] / "smoke" / "summary.json"
    return json.loads(summary.read_text()) if summary.exists() else {}


def set_common(cfg: dict[str, Any], output_root: Path, trials_per_task: int) -> None:
    cfg["output_root"] = str(output_root)
    cfg["benchmark"]["decoder_modes"] = ["spec"]
    cfg["benchmark"]["trials_per_task"] = trials_per_task
    cfg["smoke"]["trials_per_task"] = trials_per_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short LIBERO two-head spec speed sweep.")
    parser.add_argument("--base-config", default="configs/libero_goal_selected_eval.yaml")
    parser.add_argument("--out-root", default="outputs/libero_goal_speed_sweep")
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_yaml(Path(args.base_config))
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.out_root) / sweep_id
    variants = [
        {
            "name": "current",
            "trajectory_fast_max_draft_calls": 50,
            "trajectory_fast_min_confident_tokens": 5,
            "trajectory_head_threshold": 0.2,
        },
        {
            "name": "more_calls_150",
            "trajectory_fast_max_draft_calls": 150,
            "trajectory_fast_min_confident_tokens": 5,
            "trajectory_head_threshold": 0.2,
        },
        {
            "name": "calls_300_min4_t015",
            "trajectory_fast_max_draft_calls": 300,
            "trajectory_fast_min_confident_tokens": 4,
            "trajectory_head_threshold": 0.15,
        },
        {
            "name": "calls_300_min3_t010",
            "trajectory_fast_max_draft_calls": 300,
            "trajectory_fast_min_confident_tokens": 3,
            "trajectory_head_threshold": 0.1,
        },
    ]

    rows = []
    for variant in variants:
        cfg = copy.deepcopy(base)
        out_root = root / variant["name"]
        set_common(cfg, out_root, args.trials_per_task)
        for key, value in variant.items():
            if key != "name":
                cfg["inputs"][key] = value
        cfg_path = root / f"{variant['name']}.yaml"
        write_yaml(cfg_path, cfg)
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        cmd = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/run_libero_specvla_mirror.py",
            "--config",
            str(cfg_path),
            "--mode",
            "smoke",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, text=True, capture_output=True, check=False)
        (root / f"{variant['name']}.log").write_text((proc.stdout or "") + (proc.stderr or ""))
        summary = read_summary(out_root)
        spec = summary.get("decoder_summary", {}).get("spec", {})
        rows.append(
            {
                "variant": variant["name"],
                "returncode": proc.returncode,
                "success_rate": spec.get("success_rate"),
                "successes": spec.get("successes"),
                "episodes": spec.get("episodes"),
                "avg_ms_per_step": spec.get("avg_ms_per_step"),
                "summary": summary,
                "config": str(cfg_path),
            }
        )
        if proc.returncode != 0:
            break

    (root / "sweep_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({"sweep_root": str(root), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

