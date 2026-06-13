#!/usr/bin/env python3
"""Run paper-aligned LIBERO benchmark configs and summarize the frontier.

This is an orchestration layer over ``run_libero_specvla_distributed.py``. Each
entry in a suite config points at a normal LIBERO runner YAML, so AR baselines,
Spec-VLA-style relaxed thresholds, and future hybrid configs can be run as
separate, reproducible jobs while sharing one final comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkJob:
    name: str
    config: Path
    mode: str
    enabled: bool = True
    nproc_per_node: int = 1


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_jobs(cfg: dict[str, Any], base_dir: Path) -> list[BenchmarkJob]:
    jobs = []
    for raw in cfg.get("jobs", []):
        config = Path(raw["config"])
        if not config.is_absolute():
            config = (base_dir / config).resolve()
        jobs.append(
            BenchmarkJob(
                name=str(raw["name"]),
                config=config,
                mode=str(raw.get("mode", cfg.get("mode", "smoke"))),
                enabled=bool(raw.get("enabled", True)),
                nproc_per_node=int(raw.get("nproc_per_node", cfg.get("nproc_per_node", 1))),
            )
        )
    return jobs


def run_job(job: BenchmarkJob, run_id: str, dry_run: bool) -> dict[str, Any]:
    cmd: list[str]
    runner = [
        "scripts/run_libero_specvla_distributed.py",
        "--config",
        str(job.config),
        "--mode",
        job.mode,
        "--run-id",
        run_id,
    ]
    if dry_run:
        runner.append("--dry-run")

    if job.nproc_per_node > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={job.nproc_per_node}",
            *runner,
        ]
    else:
        cmd = [sys.executable, *runner]

    workspace = Path.cwd()
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(workspace / ".cache" / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", str(workspace / ".cache" / "huggingface" / "hub"))
    env.setdefault("HF_HUB_CACHE", str(workspace / ".cache" / "huggingface" / "hub"))
    env.setdefault("MPLCONFIGDIR", str(workspace / ".cache" / "matplotlib"))
    env.setdefault("LIBERO_CONFIG_PATH", str(workspace / ".cache" / "libero"))
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    return {
        "name": job.name,
        "config": str(job.config),
        "mode": job.mode,
        "nproc_per_node": job.nproc_per_node,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def find_summary(config_path: Path, run_id: str, mode: str) -> Path | None:
    cfg = load_yaml(config_path)
    path = Path(cfg["output_root"]) / run_id / mode / "summary.global.json"
    return path if path.exists() else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(max(int(round((len(ordered) - 1) * pct)), 0), len(ordered) - 1)
    return ordered[idx]


def ci(values: list[float]) -> list[float | None]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def success_rate(rows: list[dict[str, Any]]) -> float:
    return sum(1 for row in rows if row.get("success")) / max(len(rows), 1)


def avg_ms_per_step(rows: list[dict[str, Any]]) -> float:
    steps = sum(int(row.get("steps", 0)) for row in rows)
    infer_ms = sum(float(row.get("inference_ms_total", 0.0)) for row in rows)
    return infer_ms / max(steps, 1)


def bootstrap_intervals(
    ar_rows: list[dict[str, Any]],
    spec_rows: list[dict[str, Any]],
    *,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    if not ar_rows or not spec_rows:
        return {}
    rng = random.Random(seed)
    spec_success = []
    spec_ms = []
    speedups = []
    success_drops = []
    for _ in range(samples):
        ar_sample = [ar_rows[rng.randrange(len(ar_rows))] for _ in ar_rows]
        spec_sample = [spec_rows[rng.randrange(len(spec_rows))] for _ in spec_rows]
        ar_sr = success_rate(ar_sample)
        spec_sr = success_rate(spec_sample)
        ar_ms = avg_ms_per_step(ar_sample)
        sm = avg_ms_per_step(spec_sample)
        spec_success.append(spec_sr)
        spec_ms.append(sm)
        speedups.append(ar_ms / max(sm, 1e-9))
        success_drops.append(ar_sr - spec_sr)
    return {
        "success_rate_ci95": ci(spec_success),
        "avg_ms_per_step_ci95": ci(spec_ms),
        "speedup_vs_ar_ci95": ci(speedups),
        "success_drop_vs_ar_ci95": ci(success_drops),
    }


def decoder_row(name: str, summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    ar = summary["decoder_summary"]["ar"]
    spec = summary["decoder_summary"]["spec"]
    speedup = ar["avg_ms_per_step"] / max(spec["avg_ms_per_step"], 1e-9)
    success_drop = ar["success_rate"] - spec["success_rate"]
    ar_rows = read_jsonl(summary_path.parent / "ar_episodes.global.jsonl")
    spec_rows = read_jsonl(summary_path.parent / "spec_episodes.global.jsonl")
    return {
        "name": name,
        "episodes": spec["episodes"],
        "successes": spec["successes"],
        "success_rate": spec["success_rate"],
        "avg_ms_per_step": spec["avg_ms_per_step"],
        "ar_success_rate": ar["success_rate"],
        "ar_avg_ms_per_step": ar["avg_ms_per_step"],
        "speedup_vs_ar": speedup,
        "success_drop_vs_ar": success_drop,
        **bootstrap_intervals(ar_rows, spec_rows),
    }


def summarize(results: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    rows = []
    for result in results:
        if result["returncode"] != 0:
            continue
        summary_path = find_summary(Path(result["config"]), run_id, result["mode"])
        if summary_path is None:
            continue
        summary = json.loads(summary_path.read_text())
        rows.append({**decoder_row(result["name"], summary, summary_path), "summary_path": str(summary_path)})

    rows.sort(key=lambda row: (row["success_drop_vs_ar"], -row["speedup_vs_ar"]))
    pareto = []
    best_success = -1.0
    for row in sorted(rows, key=lambda row: row["speedup_vs_ar"], reverse=True):
        if row["success_rate"] > best_success:
            pareto.append(row)
            best_success = row["success_rate"]
    pareto.sort(key=lambda row: row["speedup_vs_ar"])
    return {"run_id": run_id, "rows": rows, "pareto_frontier": pareto}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run apples-to-apples LIBERO benchmark jobs.")
    parser.add_argument("--suite-config", default="configs/libero_apples_to_apples.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--only", default=None, help="Comma-separated job name filter")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suite_path = Path(args.suite_config)
    cfg = load_yaml(suite_path)
    run_id = args.run_id or f"libero_apples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    allowed = {x.strip() for x in args.only.split(",")} if args.only else None
    output_root = Path(cfg.get("output_root", "outputs/libero_apples_to_apples"))
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [job for job in load_jobs(cfg, suite_path.parent) if job.enabled]
    if allowed is not None:
        jobs = [job for job in jobs if job.name in allowed]
    if not jobs:
        raise SystemExit("No enabled jobs selected.")

    results = []
    for job in jobs:
        result = run_job(job, run_id=run_id, dry_run=args.dry_run)
        results.append(result)
        job_log = output_dir / f"{job.name}.log"
        job_log.write_text((result["stdout"] or "") + (result["stderr"] or ""))
        if result["returncode"] != 0:
            write_json(output_dir / "job_results.json", {"results": results})
            raise SystemExit(f"Job failed: {job.name}. See {job_log}")

    frontier = summarize(results, run_id=run_id)
    write_json(output_dir / "job_results.json", {"results": results})
    write_json(output_dir / "frontier_summary.json", frontier)
    print(json.dumps({"run_id": run_id, "output_dir": str(output_dir), **frontier}, indent=2))


if __name__ == "__main__":
    main()
