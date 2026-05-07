#!/usr/bin/env python3
"""Sweep LIBERO goal trajectory-spec settings on the mirror smoke benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def parse_csv_int(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_float(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_optional_int(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "unlimited"} else int(item))
    return values


def latest_run_dir(output_root: Path) -> Path:
    runs = sorted(output_root.glob("libero_mirror_smoke_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No smoke run dirs found under {output_root}")
    return runs[-1]


def run_one(
    *,
    base_cfg: dict[str, Any],
    base_config_path: Path,
    sweep_dir: Path,
    gate: int,
    threshold: float,
    top_k: int,
    fast_max_draft_calls: int | None,
    fast_max_action_step: int | None,
    use_prefill_hidden: bool,
    fast_draft_only: bool,
    trials: int,
    dry_run: bool,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["benchmark"]["suites"] = ["libero_goal"]
    cfg["smoke"]["tasks_per_suite"] = 1
    cfg["smoke"]["trials_per_task"] = trials
    cfg["output_root"] = str((sweep_dir / "runs").resolve())
    cfg["inputs"]["trajectory_fast_draft_only"] = fast_draft_only
    cfg["inputs"]["trajectory_use_draft_prefill_hidden"] = use_prefill_hidden
    cfg["inputs"]["trajectory_fast_min_confident_tokens_by_suite"] = {"libero_goal": gate}
    if fast_max_draft_calls is None:
        cfg["inputs"].pop("trajectory_fast_max_draft_calls", None)
    else:
        cfg["inputs"]["trajectory_fast_max_draft_calls"] = fast_max_draft_calls
    if fast_max_action_step is None:
        cfg["inputs"].pop("trajectory_fast_max_action_step", None)
    else:
        cfg["inputs"]["trajectory_fast_max_action_step"] = fast_max_action_step
    cfg["inputs"]["trajectory_head_threshold"] = threshold
    cfg["inputs"]["trajectory_head_top_k"] = top_k

    mode_tag = "fast" if fast_draft_only else "verified"
    budget_tag = "unlimited" if fast_max_draft_calls is None else str(fast_max_draft_calls)
    step_tag = "unlimited" if fast_max_action_step is None else str(fast_max_action_step)
    tag = f"{mode_tag}_gate{gate}_thr{threshold:g}_topk{top_k}_budget{budget_tag}_step{step_tag}"
    config_path = sweep_dir / "configs" / f"{tag}.yaml"
    write_yaml(config_path, cfg)

    env = os.environ.copy()
    workspace = base_config_path.resolve().parent.parent
    env.setdefault("LIBERO_CONFIG_PATH", str(workspace / ".libero"))
    env.setdefault("MPLCONFIGDIR", str(workspace / ".cache" / "matplotlib"))
    env.setdefault("HF_HOME", str(workspace / ".cache" / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", str(workspace / ".cache" / "huggingface" / "hub"))
    env.setdefault("HF_HUB_CACHE", str(workspace / ".cache" / "huggingface" / "hub"))

    cmd = [
        sys.executable,
        "scripts/run_libero_specvla_mirror.py",
        "--config",
        str(config_path),
        "--mode",
        "smoke",
    ]
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(cmd, cwd=str(workspace), env=env, text=True, capture_output=True, check=False)
    log_path = sweep_dir / "logs" / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""))

    row = {
        "tag": tag,
        "gate": gate,
        "threshold": threshold,
        "top_k": top_k,
        "fast_max_draft_calls": fast_max_draft_calls,
        "fast_max_action_step": fast_max_action_step,
        "use_prefill_hidden": use_prefill_hidden,
        "fast_draft_only": fast_draft_only,
        "trials": trials,
        "returncode": proc.returncode,
        "config_path": str(config_path),
        "log_path": str(log_path),
    }
    if proc.returncode == 0:
        run_dir = latest_run_dir(Path(cfg["output_root"]))
        summary_path = run_dir / "smoke" / "summary.json"
        summary = json.loads(summary_path.read_text())
        ar = summary["decoder_summary"]["ar"]
        spec = summary["decoder_summary"]["spec"]
        row.update(
            {
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "ar_success_rate": ar["success_rate"],
                "spec_success_rate": spec["success_rate"],
                "success_drop": ar["success_rate"] - spec["success_rate"],
                "ar_avg_ms": ar["avg_ms_per_step"],
                "spec_avg_ms": spec["avg_ms_per_step"],
                "speedup": ar["avg_ms_per_step"] / max(spec["avg_ms_per_step"], 1e-9),
                "gate_passed": summary["gate"]["passed"],
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep LIBERO goal trajectory-spec settings.")
    parser.add_argument("--config", default="configs/libero_specvla_mirror.yaml")
    parser.add_argument("--gates", default="4,5,6")
    parser.add_argument("--thresholds", default="0.1,0.2")
    parser.add_argument("--top-k", default="1,3")
    parser.add_argument("--fast-budgets", default="none", help="Comma-separated max direct-draft calls; use none for unlimited")
    parser.add_argument("--fast-action-steps", default="none", help="Comma-separated max action step for direct drafts; use none for unlimited")
    parser.add_argument("--use-prefill-hidden", action="store_true", help="Feed OpenVLA prefill hidden state into the draft head")
    parser.add_argument("--spec-modes", default="fast,verified", help="Comma-separated: fast,verified")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config)
    cfg = load_yaml(base_config_path)
    sweep_dir = Path("outputs/libero_goal_spec_sweeps") / datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for gate in parse_csv_int(args.gates):
        for threshold in parse_csv_float(args.thresholds):
            for top_k in parse_csv_int(args.top_k):
                for fast_max_draft_calls in parse_csv_optional_int(args.fast_budgets):
                    for fast_max_action_step in parse_csv_optional_int(args.fast_action_steps):
                        for spec_mode in [x.strip() for x in args.spec_modes.split(",") if x.strip()]:
                            if spec_mode not in {"fast", "verified"}:
                                raise ValueError(f"Unknown spec mode: {spec_mode}")
                            row = run_one(
                                base_cfg=cfg,
                                base_config_path=base_config_path,
                                sweep_dir=sweep_dir,
                                gate=gate,
                                threshold=threshold,
                                top_k=top_k,
                                fast_max_draft_calls=fast_max_draft_calls,
                                fast_max_action_step=fast_max_action_step,
                                use_prefill_hidden=args.use_prefill_hidden,
                                fast_draft_only=(spec_mode == "fast"),
                                trials=args.trials,
                                dry_run=args.dry_run,
                            )
                            rows.append(row)
                            with (sweep_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(row) + "\n")
                            print(json.dumps(row), flush=True)

    successful = [row for row in rows if row["returncode"] == 0]
    successful.sort(key=lambda row: (row.get("success_drop", 999), -row.get("speedup", 0)))
    write_json(sweep_dir / "summary.json", {"rows": rows, "ranked": successful})
    print(json.dumps({"sweep_dir": str(sweep_dir), "best": successful[:5]}, indent=2))


if __name__ == "__main__":
    main()
