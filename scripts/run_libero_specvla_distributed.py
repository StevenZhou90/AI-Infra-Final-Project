#!/usr/bin/env python3
"""Distributed paper-mirrored LIBERO benchmark runner (single-node multi-GPU).

Shards the full (decoder, suite, task, trial) grid across ranks and merges
rank-local artifacts into a global summary on rank 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover
    torch = None
    dist = None


SUITE_STEP_CAPS = {
    "libero_goal": 300,
    "libero_object": 280,
    "libero_spatial": 220,
    "libero_10": 520,
}


@dataclass
class GateResult:
    passed: bool
    speedup: float
    success_ar: float
    success_spec: float
    success_drop: float
    reason: str


class ProgressLogger:
    def __init__(self, path: Path, interval_seconds: int, rank: int) -> None:
        self.path = path
        self.interval_seconds = max(interval_seconds, 1)
        self.next_emit_ts = time.time() + self.interval_seconds
        self.rank = rank

    def maybe_emit(
        self,
        *,
        mode: str,
        decoder: str,
        suite: str,
        rows: list[dict[str, Any]],
        force: bool = False,
    ) -> None:
        now = time.time()
        if not force and now < self.next_emit_ts:
            return
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "rank": self.rank,
            "mode": mode,
            "decoder": decoder,
            "suite": suite,
            "episodes_completed": len(rows),
            "aggregate": aggregate(rows),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.next_emit_ts = now + self.interval_seconds


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = sum(r["steps"] for r in rows)
    successes = sum(1 for r in rows if r["success"])
    infer_ms = sum(r["inference_ms_total"] for r in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / max(len(rows), 1),
        "steps_total": steps,
        "avg_steps_per_episode": steps / max(len(rows), 1),
        "avg_ms_per_step": infer_ms / max(steps, 1),
        "inference_ms_total": infer_ms,
    }


def evaluate_gate(cfg: dict[str, Any], ar_rows: list[dict[str, Any]], spec_rows: list[dict[str, Any]]) -> GateResult:
    ar = aggregate(ar_rows)
    spec = aggregate(spec_rows)
    speedup = ar["avg_ms_per_step"] / max(spec["avg_ms_per_step"], 1e-9)
    success_drop = ar["success_rate"] - spec["success_rate"]
    speed_floor = float(cfg["gate"]["speedup_gt"])
    drop_ceiling = float(cfg["gate"]["max_success_drop"])
    passed = speedup > speed_floor and success_drop <= drop_ceiling
    reason = f"speedup={speedup:.3f} (>{speed_floor}) and success_drop={success_drop:.4f} (<={drop_ceiling})"
    return GateResult(
        passed=passed,
        speedup=speedup,
        success_ar=ar["success_rate"],
        success_spec=spec["success_rate"],
        success_drop=success_drop,
        reason=reason,
    )


def preflight_validate(cfg: dict[str, Any], dry_run: bool) -> list[str]:
    errors: list[str] = []
    inputs = cfg["inputs"]
    paths = [Path(cfg["specvla_repo_path"]), Path(inputs["pretrained_checkpoint"])]
    for suite in cfg["benchmark"]["suites"]:
        paths.append(Path(inputs["spec_checkpoint_by_suite"].get(suite, "")))
    for p in paths:
        raw = str(p)
        if not raw:
            errors.append("Found empty required path in config.")
            continue
        is_hf_ref = ("/" in raw) and (not raw.startswith("/")) and (not raw.startswith("./"))
        if is_hf_ref:
            continue
        if not p.exists() and not dry_run:
            errors.append(f"Missing required path: {p}")
    return errors


def build_model_cfg(suite_name: str, cfg: dict[str, Any], decoder_mode: str) -> SimpleNamespace:
    inputs = cfg["inputs"]
    return SimpleNamespace(
        model_family=inputs.get("model_family", "openvla"),
        pretrained_checkpoint=inputs["pretrained_checkpoint"],
        load_in_8bit=bool(inputs.get("load_in_8bit", False)),
        load_in_4bit=bool(inputs.get("load_in_4bit", False)),
        center_crop=bool(inputs.get("center_crop", True)),
        use_spec=(decoder_mode == "spec"),
        parallel_draft=bool(inputs.get("parallel_draft", False)),
        accept_threshold=int(inputs["accept_threshold_by_suite"][suite_name]),
        spec_checkpoint=inputs["spec_checkpoint_by_suite"][suite_name],
        task_suite_name=suite_name,
        num_steps_wait=int(cfg["benchmark"]["num_steps_wait"]),
        num_trials_per_task=0,
        seed=int(cfg["benchmark"]["seed"]),
        unnorm_key=suite_name,
    )


def build_work_items(cfg: dict[str, Any], mode: str) -> list[tuple[str, str, int, int]]:
    if mode == "smoke":
        task_limit = int(cfg["smoke"]["tasks_per_suite"])
        trials_per_task = int(cfg["smoke"]["trials_per_task"])
    else:
        task_limit = int(cfg["benchmark"]["tasks_per_suite"])
        trials_per_task = int(cfg["benchmark"]["trials_per_task"])

    items: list[tuple[str, str, int, int]] = []
    for decoder in ("ar", "spec"):
        for suite in cfg["benchmark"]["suites"]:
            for task_id in range(task_limit):
                for trial in range(trials_per_task):
                    items.append((decoder, suite, task_id, trial))
    return items


def run_suite_decoder_real(
    *,
    cfg: dict[str, Any],
    suite_name: str,
    decoder_mode: str,
    task_trials: dict[int, list[int]],
) -> list[dict[str, Any]]:
    specvla_repo = Path(cfg["specvla_repo_path"]).resolve()
    openvla_pkg_root = specvla_repo / "openvla"
    if str(openvla_pkg_root) not in sys.path:
        sys.path.insert(0, str(openvla_pkg_root))

    from libero.libero import benchmark  # type: ignore
    from experiments.robot.libero.libero_utils import (  # type: ignore
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (  # type: ignore
        get_action,
        get_image_resize_size,
        get_model,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )
    from experiments.robot.openvla_utils import get_processor  # type: ignore
    import numpy as np

    model_cfg = build_model_cfg(suite_name=suite_name, cfg=cfg, decoder_mode=decoder_mode)
    set_seed_everywhere(model_cfg.seed)
    model = get_model(model_cfg)
    processor = get_processor(model_cfg)
    resize_size = get_image_resize_size(model_cfg)
    max_steps = SUITE_STEP_CAPS[suite_name]
    num_steps_wait = int(cfg["benchmark"]["num_steps_wait"])

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    rows: list[dict[str, Any]] = []
    for task_id, trials in sorted(task_trials.items()):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, model_cfg.model_family, resolution=256)
        for episode_idx in sorted(trials):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            done = False
            t = 0
            infer_ms = 0.0
            while t < max_steps + num_steps_wait:
                if t < num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(model_cfg.model_family))
                    t += 1
                    continue
                img = get_libero_image(obs, resize_size)
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }
                tic = time.perf_counter()
                action = get_action(
                    model_cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    generate_mode=("AR" if decoder_mode == "ar" else "speculative"),
                )
                infer_ms += (time.perf_counter() - tic) * 1000.0
                action = normalize_gripper_action(action, binarize=True)
                if model_cfg.model_family == "openvla":
                    action = invert_gripper_action(action)
                obs, _, done, _ = env.step(action.tolist())
                t += 1
                if done:
                    break
            rows.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "trial": episode_idx,
                    "mode": decoder_mode,
                    "success": bool(done),
                    "steps": max(t - num_steps_wait, 0),
                    "inference_ms_total": infer_ms,
                }
            )
        env.close()
    return rows


def run_rank_work(
    *,
    cfg: dict[str, Any],
    mode: str,
    dry_run: bool,
    rank: int,
    world_size: int,
    mode_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_items = build_work_items(cfg, mode)
    local_items = [item for idx, item in enumerate(all_items) if idx % world_size == rank]
    local_items_by_ds: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for dec, suite, task, trial in local_items:
        local_items_by_ds[(dec, suite)][task].append(trial)

    progress_interval = int(cfg["benchmark"].get("progress_log_seconds", 3600))
    progress = ProgressLogger(mode_dir / f"progress.rank{rank}.jsonl", progress_interval, rank=rank)

    ar_rows: list[dict[str, Any]] = []
    spec_rows: list[dict[str, Any]] = []
    for (decoder, suite), task_trials in sorted(local_items_by_ds.items()):
        if dry_run:
            rows: list[dict[str, Any]] = []
            for task_id, trials in task_trials.items():
                for trial in trials:
                    rows.append(
                        {
                            "suite": suite,
                            "task_id": task_id,
                            "trial": trial,
                            "mode": decoder,
                            "success": bool((task_id + trial + rank) % 2 == 0),
                            "steps": min(15, SUITE_STEP_CAPS[suite]),
                            "inference_ms_total": 15 * (200.0 if decoder == "ar" else 140.0),
                        }
                    )
                    progress.maybe_emit(mode=mode, decoder=decoder, suite=suite, rows=rows)
            progress.maybe_emit(mode=mode, decoder=decoder, suite=suite, rows=rows, force=True)
        else:
            rows = run_suite_decoder_real(cfg=cfg, suite_name=suite, decoder_mode=decoder, task_trials=task_trials)
            progress.maybe_emit(mode=mode, decoder=decoder, suite=suite, rows=rows, force=True)

        if decoder == "ar":
            ar_rows.extend(rows)
        else:
            spec_rows.extend(rows)

    progress.maybe_emit(mode=mode, decoder="all", suite="all", rows=ar_rows + spec_rows, force=True)
    return ar_rows, spec_rows


def maybe_init_dist(rank: int, world_size: int) -> None:
    if world_size <= 1 or dist is None or dist.is_initialized():
        return
    backend = "nccl" if torch and torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def maybe_barrier(world_size: int) -> None:
    if world_size > 1 and dist is not None and dist.is_initialized():
        dist.barrier()


def finalize_dist() -> None:
    if dist is not None and dist.is_initialized():
        dist.destroy_process_group()


def merge_rank_outputs(mode_dir: Path, world_size: int) -> dict[str, Any]:
    all_ar: list[dict[str, Any]] = []
    all_spec: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    for r in range(world_size):
        rank_dir = mode_dir / f"rank_{r}"
        all_ar.extend(read_jsonl(rank_dir / "ar_episodes.jsonl"))
        all_spec.extend(read_jsonl(rank_dir / "spec_episodes.jsonl"))
        progress_rows.extend(read_jsonl(mode_dir / f"progress.rank{r}.jsonl"))
    progress_rows.sort(key=lambda x: (x.get("timestamp", ""), x.get("rank", 0)))
    write_jsonl(mode_dir / "ar_episodes.global.jsonl", all_ar)
    write_jsonl(mode_dir / "spec_episodes.global.jsonl", all_spec)
    write_jsonl(mode_dir / "progress.global.jsonl", progress_rows)
    return {"ar_rows": all_ar, "spec_rows": all_spec}


def update_checkpoint_index(parent_run_dir: Path, mode: str, summary: dict[str, Any]) -> None:
    idx_path = parent_run_dir / "checkpoints" / "index.json"
    ensure_dir(idx_path.parent)
    if idx_path.exists():
        index = json.loads(idx_path.read_text())
    else:
        index = {"entries": []}
    index["entries"].append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "summary_path": str((parent_run_dir / mode / "summary.global.json").resolve()),
            "gate": summary.get("gate", {}),
        }
    )
    write_json(idx_path, index)


def make_run_id(mode: str) -> str:
    return f"libero_dist_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed paper-mirrored LIBERO benchmark.")
    parser.add_argument("--config", default="configs/libero_specvla_distributed.yaml")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--auto-full-after-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    cfg = load_yaml(Path(args.config))
    errors = preflight_validate(cfg, dry_run=args.dry_run)
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))

    run_id = args.run_id or make_run_id(args.mode)
    output_root = Path(cfg["output_root"])
    parent_run_dir = output_root / run_id
    mode_dir = parent_run_dir / args.mode
    rank_dir = mode_dir / f"rank_{rank}"
    ensure_dir(rank_dir)
    if rank == 0:
        ensure_dir(parent_run_dir)
        write_json(parent_run_dir / "manifest.json", {"config": cfg, "run_id": run_id, "world_size": world_size})

    maybe_init_dist(rank, world_size)
    ar_rows, spec_rows = run_rank_work(
        cfg=cfg,
        mode=args.mode,
        dry_run=args.dry_run,
        rank=rank,
        world_size=world_size,
        mode_dir=mode_dir,
    )
    write_jsonl(rank_dir / "ar_episodes.jsonl", ar_rows)
    write_jsonl(rank_dir / "spec_episodes.jsonl", spec_rows)
    write_json(rank_dir / "summary.rank.json", {"rank": rank, "ar": aggregate(ar_rows), "spec": aggregate(spec_rows)})

    maybe_barrier(world_size)
    if rank == 0:
        merged = merge_rank_outputs(mode_dir, world_size)
        gate = evaluate_gate(cfg, merged["ar_rows"], merged["spec_rows"])
        summary = {
            "mode": args.mode,
            "world_size": world_size,
            "decoder_summary": {
                "ar": aggregate(merged["ar_rows"]),
                "spec": aggregate(merged["spec_rows"]),
            },
            "gate": {
                "passed": gate.passed,
                "speedup": gate.speedup,
                "success_ar": gate.success_ar,
                "success_spec": gate.success_spec,
                "success_drop": gate.success_drop,
                "reason": gate.reason,
            },
        }
        write_json(mode_dir / "summary.global.json", summary)
        update_checkpoint_index(parent_run_dir, args.mode, summary)
        results = [{"mode": args.mode, "run_dir": str(mode_dir), "gate": summary["gate"]}]
        if args.mode == "smoke" and args.auto_full_after_smoke and summary["gate"]["passed"]:
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "next_step": (
                            "Smoke passed. Re-run with --mode full --run-id "
                            f"{run_id} to continue with same artifact root."
                        ),
                        "results": results,
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps({"run_id": run_id, "results": results}, indent=2))
    maybe_barrier(world_size)
    finalize_dist()


if __name__ == "__main__":
    main()
