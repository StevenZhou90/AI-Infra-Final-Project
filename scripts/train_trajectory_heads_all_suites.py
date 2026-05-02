#!/usr/bin/env python3
"""Orchestrate suite-wise trajectory head training and checkpoint indexing.

This script is intentionally command-template driven so teams can plug in their
exact data generation/training commands per environment (SimplerEnv/LIBERO).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover
    HfApi = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_index(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"entries": []}


def append_index(index_path: Path, entry: dict[str, Any]) -> None:
    ensure_dir(index_path.parent)
    idx = load_index(index_path)
    idx["entries"].append(entry)
    write_json(index_path, idx)


def run_command(cmd: str, cwd: Path, dry_run: bool) -> tuple[int, str]:
    if dry_run:
        return 0, f"[dry-run] {cmd}"
    proc = subprocess.run(cmd, cwd=str(cwd), shell=True, text=True, capture_output=True, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def maybe_upload_checkpoint(
    *,
    enabled: bool,
    repo_id: str,
    suite: str,
    round_id: int,
    suite_ckpt_root: Path,
    dry_run: bool,
) -> tuple[bool, str]:
    if not enabled:
        return True, "hf_upload disabled"
    ckpt_dir = suite_ckpt_root / f"r{round_id}"
    if dry_run:
        return True, f"[dry-run] upload {ckpt_dir} -> {repo_id}/{suite}/r{round_id}"
    if not ckpt_dir.exists():
        return False, f"checkpoint directory missing: {ckpt_dir}"
    if HfApi is None:
        return False, "huggingface_hub is not installed"
    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(ckpt_dir),
        path_in_repo=f"{suite}/r{round_id}",
        commit_message=f"Upload {suite} round {round_id} spec head checkpoints",
    )
    return True, f"uploaded {ckpt_dir} to {repo_id}/{suite}/r{round_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train trajectory heads for all suites with indexing.")
    parser.add_argument("--config", default="configs/libero_specvla_distributed.yaml")
    parser.add_argument("--suites", default=None, help="Comma-separated suite filter")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    suites = cfg["head_training"]["suites"]
    if args.suites:
        allowed = {s.strip() for s in args.suites.split(",") if s.strip()}
        suites = [s for s in suites if s in allowed]

    artifact_root = Path(cfg["artifacts_root"])
    check_root = artifact_root / "checkpoints"
    logs_root = artifact_root / "logs"
    ensure_dir(check_root)
    ensure_dir(logs_root)
    index_path = check_root / "index.json"

    cmd_tpls = cfg["head_training"]["command_templates"]
    rounds = int(cfg["head_training"].get("dagger_rounds", 0))
    workspace = Path(cfg["workspace_root"])
    hf_cfg = cfg.get("hf_upload", {})
    hf_enabled = bool(hf_cfg.get("enabled", False))
    hf_repo_id = str(hf_cfg.get("repo_id", "")).strip()

    pretrained_by_suite = cfg["head_training"].get("pretrained_by_suite", {})
    sweep_by_suite = cfg["head_training"].get("sweep_by_suite", {})

    for suite in suites:
        suite_ckpt_root = check_root / suite
        ensure_dir(suite_ckpt_root)

        # Round 1: supervised
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_log = logs_root / f"{suite}_{run_tag}.log"
        fmt = {
            "suite": suite,
            "artifact_root": str(artifact_root),
            "suite_ckpt_root": str(suite_ckpt_root),
            "round": 1,
            "pretrained": pretrained_by_suite.get(suite, "openvla/openvla-7b"),
            "sweep": sweep_by_suite.get(suite, "mini"),
        }
        for stage in ("generate_data", "train"):
            cmd = cmd_tpls[stage].format(**fmt)
            rc, out = run_command(cmd, workspace, args.dry_run)
            suite_log.write_text((suite_log.read_text() if suite_log.exists() else "") + out)
            append_index(
                index_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "suite": suite,
                    "round": 1,
                    "stage": stage,
                    "command": cmd,
                    "status": "ok" if rc == 0 else "failed",
                    "log_path": str(suite_log),
                },
            )
            if rc != 0:
                raise SystemExit(f"Stage failed for suite={suite} stage={stage}\nCommand: {cmd}")
            if stage == "train" and hf_enabled:
                ok, msg = maybe_upload_checkpoint(
                    enabled=hf_enabled,
                    repo_id=hf_repo_id,
                    suite=suite,
                    round_id=1,
                    suite_ckpt_root=suite_ckpt_root,
                    dry_run=args.dry_run,
                )
                suite_log.write_text((suite_log.read_text() if suite_log.exists() else "") + msg + "\n")
                append_index(
                    index_path,
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "suite": suite,
                        "round": 1,
                        "stage": "hf_upload",
                        "command": f"upload {suite_ckpt_root / 'r1'} to {hf_repo_id}/{suite}/r1",
                        "status": "ok" if ok else "failed",
                        "log_path": str(suite_log),
                    },
                )
                if not ok:
                    raise SystemExit(f"HF upload failed for suite={suite}, round=1: {msg}")

        # DAgger rounds
        for r in range(2, rounds + 2):
            fmt = {
                "suite": suite,
                "artifact_root": str(artifact_root),
                "suite_ckpt_root": str(suite_ckpt_root),
                "round": r,
                "prev_round": r - 1,
                "pretrained": pretrained_by_suite.get(suite, "openvla/openvla-7b"),
                "sweep": sweep_by_suite.get(suite, "mini"),
            }
            for stage in ("dagger_generate_data", "train"):
                template_key = stage
                if template_key not in cmd_tpls:
                    continue
                cmd = cmd_tpls[template_key].format(**fmt)
                rc, out = run_command(cmd, workspace, args.dry_run)
                suite_log.write_text((suite_log.read_text() if suite_log.exists() else "") + out)
                append_index(
                    index_path,
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "suite": suite,
                        "round": r,
                        "stage": stage,
                        "command": cmd,
                        "status": "ok" if rc == 0 else "failed",
                        "log_path": str(suite_log),
                    },
                )
                if rc != 0:
                    raise SystemExit(f"Stage failed for suite={suite} stage={stage}\nCommand: {cmd}")
                if stage == "train" and hf_enabled:
                    ok, msg = maybe_upload_checkpoint(
                        enabled=hf_enabled,
                        repo_id=hf_repo_id,
                        suite=suite,
                        round_id=r,
                        suite_ckpt_root=suite_ckpt_root,
                        dry_run=args.dry_run,
                    )
                    suite_log.write_text((suite_log.read_text() if suite_log.exists() else "") + msg + "\n")
                    append_index(
                        index_path,
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "suite": suite,
                            "round": r,
                            "stage": "hf_upload",
                            "command": f"upload {suite_ckpt_root / f'r{r}'} to {hf_repo_id}/{suite}/r{r}",
                            "status": "ok" if ok else "failed",
                            "log_path": str(suite_log),
                        },
                    )
                    if not ok:
                        raise SystemExit(f"HF upload failed for suite={suite}, round={r}: {msg}")

    print(
        json.dumps(
            {
                "status": "ok",
                "suites": suites,
                "index_path": str(index_path),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
